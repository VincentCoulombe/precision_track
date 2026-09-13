"""Track a video with a trained SLEAP model and write PrecisionTrack CSVs.

This is the format bridge: it runs ``sleap_nn`` inference plus tracking, times it, then
converts the predicted ``.slp`` into the two files PrecisionTrack itself produces:

* ``tracked_kpts.csv``   -> ``frame_id,class_id,instance_id,x0,y0,score0,x1,y1,score1,...``
* ``tracked_bboxes.csv`` -> ``frame_id,class_id,instance_id,cx,cy,w,h,score``

SLEAP predicts keypoints, not boxes, so each box is the tightest one enclosing that
instance's detected keypoints (the same derivation ``keypoints_cxcywh`` uses inside
PrecisionTrack) and its score is the mean keypoint confidence. ``dlc/track.py`` derives
its boxes identically, which is what makes the two baselines comparable.

Because the file names match those in ``configs/tasks/tracking.py``, pointing
``saving_directory`` at ``--out-dir`` lets PrecisionTrack's own visualizer render these
results:

    cd tools && python visualize.py <video> <annotated.mp4>

Example:
    python track.py <video>.mp4 --model models/bottomup \\
        --out-dir ../work_dir/sleap_bottomup --max-instances 5 --img-size 640 640
"""

import argparse
import os
import os.path as osp
from time import perf_counter
from typing import List, Optional, Sequence, Tuple

import numpy as np
from pt_format import CsvBoundingBoxes, CsvKeypoints, keypoints_cxcywh
from slp_io import read_predictions, resolve_model_paths
from tabulate import tabulate
from video_utils import ensure_resolution, frame_count

BBOX_FORMATS = {"cxcywh": ["cx", "cy", "w", "h"], "xywh": ["x", "y", "w", "h"]}

SCORING_METHODS = ["oks", "iou", "cosine_sim", "euclidean_dist", "mask_iou"]
CANDIDATES_METHODS = ["fixed_window", "local_queues"]

# Each scoring function consumes a specific feature shape, and ``TrackerConfig`` does not
# enforce the pairing: ``--scoring-method iou`` with the default ``features="keypoints"``
# hands ``compute_iou`` a (n_nodes, 2) array where it expects (xmin, ymin, xmax, ymax) and
# dies with "too many values to unpack" once tracking is already under way. ``--features``
# is therefore derived from the scoring method unless it is set explicitly.
SCORING_FEATURES = {
    "oks": "keypoints",
    "iou": "bboxes",
    "mask_iou": "masks",
    "euclidean_dist": "centroids",
    "cosine_sim": "keypoints",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Path to the video to track")
    parser.add_argument("--model", required=True, help="Trained model directory, or a model_group.json for a two-model top-down setup.")
    parser.add_argument("--out-dir", required=True, help="Directory receiving tracked_kpts.csv and tracked_bboxes.csv")
    parser.add_argument("--max-instances", type=int, default=None, help="Maximum animals per frame. Set it to the real animal count.")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size. Defaults to 4.")
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Rescale the video before tracking, e.g. --img-size 640 640.",
    )
    parser.add_argument("--slp-out", default=None, help="Where the predicted .slp is written. Defaults to <out-dir>/predictions.slp.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, cpu, mps. Defaults to auto.")

    detection = parser.add_argument_group("detection")
    detection.add_argument(
        "--peak-threshold",
        type=float,
        default=None,
        help="Minimum confidence for a keypoint peak. Lowering it trades precision for recall and is the main knob "
        "when animals go undetected. Defaults to the model's trained value.",
    )
    detection.add_argument("--centroid-threshold", type=float, default=None, help="Minimum centroid confidence (top-down only).")
    detection.add_argument("--min-instance-peaks", type=float, default=None, help="Drop instances assembled from fewer than this many peaks.")
    detection.add_argument("--integral-refinement", default=None, help="Sub-pixel peak refinement, e.g. integral.")

    tracking = parser.add_argument_group("tracking")
    tracking.add_argument("--no-tracking", action="store_true", help="Run pose estimation only; every instance gets its own id.")
    tracking.add_argument("--candidates-method", choices=CANDIDATES_METHODS, default="fixed_window", help="Candidate source. Defaults to fixed_window.")
    tracking.add_argument("--scoring-method", choices=SCORING_METHODS, default="oks", help="Instance-to-track similarity. Defaults to oks.")
    tracking.add_argument("--scoring-reduction", default="mean", help="How per-candidate scores are reduced, e.g. mean, max, robust_quantile.")
    tracking.add_argument("--window-size", type=int, default=None, help="Frames of history used for association. Defaults to sleap_nn's value.")
    tracking.add_argument("--track-matching-method", default="hungarian", help="Assignment solver: hungarian or greedy. Defaults to hungarian.")
    tracking.add_argument(
        "--features",
        default=None,
        choices=["keypoints", "centroids", "bboxes", "masks"],
        help="What is compared between instances. Defaults to whatever the chosen --scoring-method requires "
        "(oks->keypoints, iou->bboxes, euclidean_dist->centroids).",
    )
    tracking.add_argument("--robust-best-instance", type=float, default=1.0, help="Quantile for robust score reduction. 1.0 (default) means plain max.")
    tracking.add_argument("--min-new-track-points", type=int, default=0, help="Visible keypoints required before a new track may be spawned.")
    tracking.add_argument("--min-match-points", type=int, default=0, help="Visible keypoints required for an instance to match an existing track.")
    tracking.add_argument("--use-kalman", action="store_true", help="Add a Kalman filter over track motion. Requires --max-instances.")
    tracking.add_argument("--use-flow", action="store_true", help="Shift candidate instances with optical flow before matching.")
    tracking.add_argument(
        "--post-connect-single-breaks",
        action="store_true",
        help="Bridge single-frame identity breaks after tracking. Requires --max-instances.",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--bbox-format",
        choices=sorted(BBOX_FORMATS),
        default="cxcywh",
        help="On-disk bbox layout. Keep cxcywh for evaluate_mot, which reformats predictions itself; xywh is for ground-truth-shaped files.",
    )
    output.add_argument("--pcutoff", type=float, default=0.0, help="Drop keypoints whose score falls below this. Defaults to 0 (keep everything).")
    output.add_argument(
        "--bbox-exclude-keypoints",
        nargs="*",
        default=[],
        metavar="NODE",
        help="Node names (or indices) to leave out of the bounding box, e.g. --bbox-exclude-keypoints tailstart. "
        "Use it when your ground-truth boxes do not cover an extremity the skeleton does. Keypoint output is unaffected.",
    )
    output.add_argument("--skip-inference", action="store_true", help="Reuse the .slp already present instead of re-running SLEAP.")
    return parser.parse_args()


def build_tracker_config(args):
    """Assemble a ``TrackerConfig``, or ``None`` when ``--no-tracking`` is set."""
    from sleap_nn.inference.tracking import TrackerConfig

    if args.no_tracking:
        return None

    expected = SCORING_FEATURES[args.scoring_method]
    features = args.features if args.features is not None else expected
    if features != expected:
        print(f"Warning: --scoring-method {args.scoring_method} normally needs --features {expected}, but '{features}' was requested.")

    kwargs = dict(
        candidates_method=args.candidates_method,
        scoring_method=args.scoring_method,
        scoring_reduction=args.scoring_reduction,
        track_matching_method=args.track_matching_method,
        features=features,
        robust_best_instance=args.robust_best_instance,
        min_new_track_points=args.min_new_track_points,
        min_match_points=args.min_match_points,
        use_kalman=args.use_kalman,
        use_flow=args.use_flow,
        post_connect_single_breaks=args.post_connect_single_breaks,
    )
    if args.window_size is not None:
        kwargs["window_size"] = args.window_size
    if args.max_instances is not None:
        # max_tracks caps how many identities may be created; the target count is what
        # the Kalman filter and the single-break bridging need, and neither accepts
        # max_tracks as a substitute.
        kwargs["max_tracks"] = args.max_instances
        kwargs["tracking_target_instance_count"] = args.max_instances
    elif args.use_kalman or args.post_connect_single_breaks:
        raise SystemExit("--use-kalman and --post-connect-single-breaks both need --max-instances.")

    if args.max_instances is not None and args.candidates_method == "fixed_window":
        print("Note: max_tracks is ignored by fixed_window, so SLEAP will switch to local_queues to honour --max-instances.")
    return TrackerConfig(**kwargs)


def predictor_kwargs(args) -> dict:
    """Only the inference overrides the user actually set, so the rest stay trained values."""
    kwargs = {}
    for flag, name in (
        ("peak_threshold", "peak_threshold"),
        ("centroid_threshold", "centroid_threshold"),
        ("min_instance_peaks", "min_instance_peaks"),
        ("integral_refinement", "integral_refinement"),
    ):
        value = getattr(args, flag)
        if value is not None:
            kwargs[name] = value
    return kwargs


def infer(video: str, model_paths: List[str], slp_out: str, args) -> Optional[float]:
    """Run SLEAP inference and tracking on ``video``, returning the elapsed seconds."""
    from sleap_nn.inference import predict

    kwargs = dict(
        source=video,
        model_paths=model_paths,
        device=args.device,
        batch_size=args.batch_size,
        max_instances=args.max_instances,
        tracker_config=build_tracker_config(args),
        output_path=slp_out,
        output_format="slp",
    )
    kwargs.update(predictor_kwargs(args))

    os.makedirs(osp.dirname(osp.abspath(slp_out)), exist_ok=True)
    start = perf_counter()
    predict(**kwargs)
    return perf_counter() - start


def report_latency(elapsed: Optional[float], n_frames: int) -> None:
    if elapsed is None:
        print(f"\nReused an existing SLEAP prediction for {n_frames} frames; no timing available.")
        return
    table = [
        ["Frames", f"{n_frames:d}"],
        ["Total time (s)", f"{elapsed:.3f}"],
        ["Latency per frame (ms)", f"{1000 * elapsed / max(n_frames, 1):.3f}"],
        ["Throughput (FPS)", f"{n_frames / elapsed:.3f}"],
    ]
    print("\n" + tabulate(table, headers=["SLEAP: predict + track", "Value"], tablefmt="github", stralign="left"))


def resolve_box_keypoints(nodes: List[str], excluded: Sequence[str]) -> np.ndarray:
    """Boolean mask over ``nodes`` selecting those that define the bounding box.

    Excluded entries may be node names or integer indices. Excluding extremities the
    ground truth does not cover -- a tail, most often -- keeps the keypoint hull
    comparable to annotated boxes; keypoints are still written to ``tracked_kpts.csv``
    in full.
    """
    mask = np.ones(len(nodes), dtype=bool)
    for entry in excluded:
        if entry in nodes:
            mask[nodes.index(entry)] = False
        elif entry.lstrip("-").isdigit() and 0 <= int(entry) < len(nodes):
            mask[int(entry)] = False
        else:
            raise SystemExit(f"--bbox-exclude-keypoints: '{entry}' is not one of {nodes} nor a valid index.")
    if not mask.any():
        raise SystemExit("--bbox-exclude-keypoints would exclude every keypoint; the bounding box would be undefined.")
    return mask


def to_precision_track_csvs(slp_path: str, out_dir: str, bbox_format: str, pcutoff: float, bbox_exclude: Sequence[str] = ()):
    """Convert a predicted SLEAP ``.slp`` into the two PrecisionTrack writers."""
    frame_ids, poses, nodes, track_names = read_predictions(slp_path)
    box_mask = resolve_box_keypoints(nodes, bbox_exclude)
    if not box_mask.all():
        dropped = [node for node, keep in zip(nodes, box_mask) if not keep]
        print(f"Bounding boxes derived from {int(box_mask.sum())}/{len(nodes)} keypoints (excluding {dropped}).")

    kpts_output = CsvKeypoints(path=osp.join(out_dir, "tracked_kpts.csv"), instance_data="pred_track_instances", precision=32)
    bboxes_output = CsvBoundingBoxes(
        path=osp.join(out_dir, "tracked_bboxes.csv"),
        subtype="tracked_bboxes",
        instance_data="pred_track_instances",
        precision=64,
        save_bbox_format=BBOX_FORMATS[bbox_format],
    )

    n_instances, n_fallbacks = 0, 0
    for frame_id, frame_poses in zip(frame_ids, poses):
        ids, labels, bboxes, scores, keypoints, keypoint_scores = [], [], [], [], [], []
        for instance_id, pose in enumerate(frame_poses):
            xy, likelihood = pose[:, :2].copy(), pose[:, 2].copy()
            invalid = np.isnan(xy).any(axis=1) | np.isnan(likelihood) | (likelihood < pcutoff)
            if invalid.all():
                continue
            xy[invalid] = np.nan
            likelihood[invalid] = 0.0

            box_xy = xy[box_mask]
            if np.isnan(box_xy).all():
                box_xy = xy  # none of the box keypoints survived; fall back to the full hull
                n_fallbacks += 1

            ids.append(instance_id)
            labels.append(0)
            bboxes.append(keypoints_cxcywh(box_xy))
            scores.append(float(np.mean(likelihood[~invalid])))
            keypoints.append(np.nan_to_num(xy, nan=0.0))
            keypoint_scores.append(likelihood)

        if not ids:
            continue
        n_instances += len(ids)
        instance_data = dict(
            labels=np.asarray(labels, dtype=np.int64),
            instances_id=np.asarray(ids, dtype=np.int64),
            bboxes=np.stack(bboxes),
            scores=np.asarray(scores, dtype=np.float64),
            keypoints=np.stack(keypoints),
            keypoint_scores=np.stack(keypoint_scores),
        )
        data_sample = {"img_id": int(frame_id), "pred_track_instances": instance_data}
        kpts_output(data_sample)
        bboxes_output(data_sample)

    print(f"Converted {n_instances} instances over {len(frame_ids)} frames ({len(track_names)} track(s) x {len(nodes)} nodes).")
    if n_fallbacks:
        print(f"{n_fallbacks} instance(s) had no surviving box keypoint and fell back to the full keypoint hull.")
    return kpts_output, bboxes_output


def rescale_outputs(outputs, tracked_res: Tuple[int, int], original_res: Tuple[int, int]) -> None:
    """Map coordinates from the (possibly downsized) tracked video back to source pixels."""
    if tracked_res == original_res:
        return
    tracked_wh = (tracked_res[1], tracked_res[0])
    original_wh = (original_res[1], original_res[0])
    print(f"Rescaling outputs from {tracked_res[0]}x{tracked_res[1]} back to {original_res[0]}x{original_res[1]}")
    for output in outputs:
        output.scale(tracked_wh, original_wh)


def main(args):
    model_paths = resolve_model_paths(args.model)
    print(f"Model(s): {model_paths}")

    video, original_res, tracked_res = ensure_resolution(osp.abspath(osp.expanduser(args.video)), tuple(args.img_size) if args.img_size else None)
    out_dir = osp.abspath(osp.expanduser(args.out_dir))
    slp_out = osp.abspath(osp.expanduser(args.slp_out)) if args.slp_out else osp.join(out_dir, "predictions.slp")

    if args.skip_inference:
        if not osp.isfile(slp_out):
            raise SystemExit(f"--skip-inference was passed but {slp_out} does not exist.")
        elapsed = None
    else:
        elapsed = infer(video, model_paths, slp_out, args)
    report_latency(elapsed, frame_count(video))

    print(f"Reading SLEAP tracks from {slp_out}")
    outputs = to_precision_track_csvs(slp_out, out_dir, args.bbox_format, args.pcutoff, args.bbox_exclude_keypoints)

    rescale_outputs(outputs, tracked_res, original_res)
    for output in outputs:
        output.save()


if __name__ == "__main__":
    main(parse_args())
