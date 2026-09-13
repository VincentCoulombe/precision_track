"""Track a video with a trained DeepLabCut model and write PrecisionTrack CSVs.

This is the format bridge: it runs ``deeplabcut.analyze_videos(..., auto_track=True)``,
times it, then converts DLC's tracked ``.h5`` into the two files PrecisionTrack itself
produces during tracking:

* ``tracked_kpts.csv``   -> ``frame_id,class_id,instance_id,x0,y0,score0,x1,y1,score1,...``
* ``tracked_bboxes.csv`` -> ``frame_id,class_id,instance_id,cx,cy,w,h,score``

DLC has no notion of a bounding box, so each box is the tightest one enclosing that
individual's detected keypoints (the same derivation ``keypoints_cxcywh`` uses inside
PrecisionTrack) and its score is the mean keypoint likelihood.

Because the file names match those in ``configs/tasks/tracking.py``, pointing
``saving_directory`` at ``--out-dir`` lets PrecisionTrack's own visualizer render these
results:

    cd tools && python visualize.py <video> <annotated.mp4>

Example:
    python track.py ../assets/20mice.avi --project projects/mice-coco-2026-09-08/config.yaml \\
        --n-tracks 20 --out-dir ../work_dir/20mice --img-size 640 640
"""

import argparse
import os.path as osp
from time import perf_counter
from typing import List, Optional, Sequence, Tuple

import numpy as np
from dlc_io import find_tracked_h5, project_dest_folder, read_project_config, read_tracked_h5, tracked_frames
from pt_format import CsvBoundingBoxes, CsvKeypoints, keypoints_cxcywh
from tabulate import tabulate
from video_utils import ensure_resolution, frame_count

BBOX_FORMATS = {"cxcywh": ["cx", "cy", "w", "h"], "xywh": ["x", "y", "w", "h"]}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Path to the video to track")
    parser.add_argument("--project", required=True, help="Path to the DLC project's config.yaml")
    parser.add_argument("--out-dir", required=True, help="Directory receiving tracked_kpts.csv and tracked_bboxes.csv")
    parser.add_argument("--n-tracks", type=int, default=None, help="Number of animals to stitch. Defaults to the project's individuals count.")
    parser.add_argument("--shuffle", type=int, default=1, help="Shuffle index of the trained model. Defaults to 1.")
    parser.add_argument("--batch-size", type=int, default=30, help="Inference batch size. Defaults to 30.")
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Rescale the video before tracking, e.g. --img-size 640 640.",
    )
    parser.add_argument("--dest-folder", default=None, help="Where DLC writes its own artifacts. Defaults to the video's directory.")
    parser.add_argument(
        "--bbox-format",
        choices=sorted(BBOX_FORMATS),
        default="cxcywh",
        help="On-disk bbox layout. Keep cxcywh for evaluate_mot, which reformats predictions itself; xywh is for ground-truth-shaped files.",
    )
    parser.add_argument("--pcutoff", type=float, default=0.0, help="Drop keypoints whose likelihood falls below this. Defaults to 0 (keep everything).")
    parser.add_argument(
        "--bbox-exclude-keypoints",
        nargs="*",
        default=[],
        metavar="BODYPART",
        help="Bodypart names (or indices) to leave out of the bounding box, e.g. --bbox-exclude-keypoints tail tailend. "
        "Use it when your ground-truth boxes do not cover an extremity the skeleton does. Keypoint output is unaffected.",
    )
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--skip-analysis", action="store_true", help="Reuse the tracked .h5 already present instead of re-running DLC.")
    return parser.parse_args()


def analyze(video: str, config_path: str, args, n_tracks: int, dest_folder: str) -> Optional[float]:
    """Run DLC's pose estimation + tracking on ``video``, returning the elapsed seconds."""
    import deeplabcut

    kwargs = dict(
        shuffle=args.shuffle,
        batchsize=args.batch_size,
        destfolder=dest_folder,
        auto_track=True,
        n_tracks=n_tracks,
        save_as_csv=False,
    )
    if args.device:
        kwargs["device"] = args.device

    start = perf_counter()
    deeplabcut.analyze_videos(config_path, [video], **kwargs)
    return perf_counter() - start


def report_latency(elapsed: Optional[float], n_frames: int) -> None:
    if elapsed is None:
        print(f"\nReused an existing DLC analysis for {n_frames} frames; no timing available.")
        return
    table = [
        ["Frames", f"{n_frames:d}"],
        ["Total time (s)", f"{elapsed:.3f}"],
        ["Latency per frame (ms)", f"{1000 * elapsed / max(n_frames, 1):.3f}"],
        ["Throughput (FPS)", f"{n_frames / elapsed:.3f}"],
    ]
    print("\n" + tabulate(table, headers=["DeepLabCut: analyze_videos", "Value"], tablefmt="github", stralign="left"))


def resolve_box_keypoints(bodyparts: List[str], excluded: Sequence[str]) -> np.ndarray:
    """Boolean mask over ``bodyparts`` selecting those that define the bounding box.

    Excluded entries may be bodypart names or integer indices. Excluding extremities the
    ground truth does not cover -- a tail, most often -- keeps the keypoint hull
    comparable to annotated boxes; keypoints are still written to ``tracked_kpts.csv``
    in full.
    """
    mask = np.ones(len(bodyparts), dtype=bool)
    for entry in excluded:
        if entry in bodyparts:
            mask[bodyparts.index(entry)] = False
        elif entry.lstrip("-").isdigit() and 0 <= int(entry) < len(bodyparts):
            mask[int(entry)] = False
        else:
            raise SystemExit(f"--bbox-exclude-keypoints: '{entry}' is not one of {bodyparts} nor a valid index.")
    if not mask.any():
        raise SystemExit("--bbox-exclude-keypoints would exclude every keypoint; the bounding box would be undefined.")
    return mask


def to_precision_track_csvs(h5_path: str, out_dir: str, bbox_format: str, pcutoff: float, bbox_exclude: Sequence[str] = ()):
    """Convert a tracked DLC ``.h5`` into the two PrecisionTrack writers."""
    df, individuals, bodyparts = read_tracked_h5(h5_path)
    frame_ids, poses = tracked_frames(df, individuals, bodyparts)
    box_mask = resolve_box_keypoints(bodyparts, bbox_exclude)
    if not box_mask.all():
        dropped = [bp for bp, keep in zip(bodyparts, box_mask) if not keep]
        print(f"Bounding boxes derived from {int(box_mask.sum())}/{len(bodyparts)} keypoints (excluding {dropped}).")

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

    print(f"Converted {n_instances} instances over {len(frame_ids)} frames ({len(individuals)} individuals x {len(bodyparts)} bodyparts).")
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
    config_path = osp.abspath(args.project)
    config = read_project_config(config_path)
    n_tracks = args.n_tracks or len(config.get("individuals", []))
    if not n_tracks:
        raise SystemExit("Could not determine the number of animals to track; pass --n-tracks.")

    video, original_res, tracked_res = ensure_resolution(osp.abspath(args.video), tuple(args.img_size) if args.img_size else None)
    dest_folder = project_dest_folder(video, args.dest_folder)

    elapsed = None if args.skip_analysis else analyze(video, config_path, args, n_tracks, dest_folder)
    report_latency(elapsed, frame_count(video))

    h5_path = find_tracked_h5(video, dest_folder)
    print(f"Reading DLC tracks from {h5_path}")
    outputs = to_precision_track_csvs(h5_path, osp.abspath(args.out_dir), args.bbox_format, args.pcutoff, args.bbox_exclude_keypoints)

    rescale_outputs(outputs, tracked_res, original_res)
    for output in outputs:
        output.save()


if __name__ == "__main__":
    main(parse_args())
