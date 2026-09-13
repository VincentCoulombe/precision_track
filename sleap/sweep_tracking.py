"""Sweep SLEAP tracking parameters over one video, reusing a single inference pass.

**Run this in the ``sleap`` environment.** It writes one ``tracked_bboxes.csv`` /
``tracked_kpts.csv`` pair per configuration; ``sweep_evaluate.py`` then scores them in the
``precision_track`` environment.

The saving that makes a broad sweep practical: ``sleap_nn`` separates ``predict()`` from
``apply_tracking()``, so pose estimation runs **once per peak threshold** and every
tracking configuration re-runs only the association pass. Only ``peak_threshold`` changes
the detections, so the sweep is
``len(--peak-thresholds)`` inference passes plus ``len(thresholds) x len(grid)`` cheap
association passes -- not one full re-inference per cell. The DLC sweep had no equivalent
shortcut and had to re-infer for every assembly change.

Each configuration reloads the cached untracked ``.slp`` rather than reusing the in-memory
``Labels``: ``apply_tracking`` assigns ``.track`` on the instances it is given, so sharing
one object between configurations would let the first run contaminate the rest.

Example:
    python sweep_tracking.py <video>.mp4 --model models/bottomup_s2 \\
        --out-dir ../work_dir/sleap_sweep/bottomup --max-instances 5 --img-size 640 640
"""

import argparse
import json
import os
import os.path as osp
from time import perf_counter
from typing import Dict, List, Optional

from slp_io import resolve_model_paths
from track import rescale_outputs, to_precision_track_csvs

# Tracking configurations, swept against every --peak-thresholds value.
#
# ``fixed_window_w10`` deliberately drops max_instances: ``max_tracks`` is ignored by the
# fixed_window candidate maker and sleap_nn silently switches to local_queues to honour it
# (tracking/tracker.py:246). Measuring fixed_window at all therefore requires giving up the
# animal-count cap, which makes it a different condition rather than a comparable cell.
TRACKING_GRID: List[Dict] = [
    dict(name="oks_w5", scoring_method="oks", window_size=5),
    dict(name="oks_w10", scoring_method="oks", window_size=10),
    dict(name="oks_w20", scoring_method="oks", window_size=20),
    dict(name="iou_w10", scoring_method="iou", window_size=10),
    dict(name="euclid_w10", scoring_method="euclidean_dist", window_size=10),
    dict(name="oks_w10_kalman", scoring_method="oks", window_size=10, use_kalman=True),
    dict(name="oks_w5_kalman", scoring_method="oks", window_size=5, use_kalman=True),
    dict(name="oks_w20_kalman", scoring_method="oks", window_size=20, use_kalman=True),
    dict(name="iou_w10_kalman", scoring_method="iou", window_size=10, use_kalman=True),
    dict(name="oks_w10_breaks", scoring_method="oks", window_size=10, post_connect_single_breaks=True),
    dict(name="oks_w10_robust", scoring_method="oks", window_size=10, robust_best_instance=0.95, min_new_track_points=3),
    dict(name="fixed_window_w10", scoring_method="oks", window_size=10, candidates_method="fixed_window", drop_max_instances=True),
]

# Keys in a grid entry that are ours, not TrackerConfig's.
LOCAL_KEYS = {"name", "drop_max_instances"}

# Each scoring function consumes a specific feature shape, and ``TrackerConfig`` does not
# enforce the pairing: ``scoring_method="iou"`` with the default ``features="keypoints"``
# hands ``compute_iou`` a (n_nodes, 2) array where it expects (xmin, ymin, xmax, ymax) and
# dies with "too many values to unpack" partway through a sweep. Pair them here instead.
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
    parser.add_argument("--model", required=True, help="Trained model directory, or a model_group.json.")
    parser.add_argument("--out-dir", required=True, help="Directory receiving one subdirectory per configuration")
    parser.add_argument("--max-instances", type=int, default=None, help="Animal count. Applied to every config except the fixed_window arm.")
    parser.add_argument(
        "--peak-thresholds",
        type=float,
        nargs="+",
        default=[0.2, 0.1, 0.05],
        help="Detection thresholds to sweep. Each one costs a full inference pass. Defaults to 0.2 0.1 0.05.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size. Defaults to 4.")
    parser.add_argument("--img-size", type=int, nargs=2, default=None, metavar=("HEIGHT", "WIDTH"), help="Rescale before tracking, e.g. --img-size 640 640.")
    parser.add_argument("--device", default="auto", help="Torch device. Defaults to auto.")
    parser.add_argument("--only", nargs="*", default=None, metavar="NAME", help="Run only these grid entries by name.")
    parser.add_argument("--bbox-format", choices=["cxcywh", "xywh"], default="cxcywh", help="On-disk bbox layout. Defaults to cxcywh.")
    parser.add_argument("--pcutoff", type=float, default=0.0, help="Drop keypoints below this score. Defaults to 0.")
    parser.add_argument("--bbox-exclude-keypoints", nargs="*", default=[], metavar="NODE", help="Nodes to leave out of the bounding box.")
    parser.add_argument("--reuse-inference", action="store_true", default=True, help="Reuse a cached untracked .slp when present (default).")
    parser.add_argument("--force-inference", dest="reuse_inference", action="store_false", help="Re-run inference even if a cache exists.")
    return parser.parse_args()


def grid_for(args) -> List[Dict]:
    entries = TRACKING_GRID if not args.only else [e for e in TRACKING_GRID if e["name"] in set(args.only)]
    if not entries:
        raise SystemExit(f"--only matched nothing. Available: {[e['name'] for e in TRACKING_GRID]}")
    return entries


def tracker_config_for(entry: Dict, max_instances: Optional[int]):
    """Build a ``TrackerConfig`` from one grid entry."""
    from sleap_nn.inference.tracking import TrackerConfig

    kwargs = {k: v for k, v in entry.items() if k not in LOCAL_KEYS}
    kwargs.setdefault("candidates_method", "local_queues")

    scoring = kwargs.get("scoring_method", "oks")
    expected = SCORING_FEATURES.get(scoring)
    if expected is None:
        raise SystemExit(f"Config '{entry['name']}': unknown scoring_method '{scoring}'. Known: {sorted(SCORING_FEATURES)}")
    if kwargs.setdefault("features", expected) != expected:
        raise SystemExit(
            f"Config '{entry['name']}': scoring_method '{scoring}' needs features='{expected}', "
            f"but the entry asks for '{kwargs['features']}'. The two must match or the scoring function "
            "receives the wrong array shape."
        )

    if max_instances is not None and not entry.get("drop_max_instances"):
        kwargs["max_tracks"] = max_instances
        kwargs["tracking_target_instance_count"] = max_instances
    elif entry.get("post_connect_single_breaks") or entry.get("use_kalman"):
        raise SystemExit(f"Config '{entry['name']}' needs --max-instances for kalman/single-break bridging.")
    return TrackerConfig(**kwargs)


def run_inference(video: str, model_paths: List[str], threshold: float, cache_path: str, args) -> float:
    """Pose estimation only, cached to ``cache_path``. Returns elapsed seconds (0 if cached)."""
    import sleap_io as sio
    from sleap_nn.inference import predict

    if args.reuse_inference and osp.isfile(cache_path):
        print(f"  reusing cached detections: {cache_path}")
        return 0.0

    start = perf_counter()
    labels = predict(
        source=video,
        model_paths=model_paths,
        device=args.device,
        batch_size=args.batch_size,
        max_instances=args.max_instances,
        peak_threshold=threshold,
        tracker_config=None,
    )
    elapsed = perf_counter() - start
    sio.save_slp(labels, cache_path)
    return elapsed


def diagnostics(csv_path: str, n_frames: int) -> Dict:
    """Cheap per-config health numbers, the analogue of the DLC sweep's phase1_summary.json."""
    import pandas as pd

    if not osp.isfile(csv_path):
        return dict(rows=0, inst_per_frame=0.0, empty_frames_prcnt=100.0, n_ids=0)
    df = pd.read_csv(csv_path)
    if df.empty:
        return dict(rows=0, inst_per_frame=0.0, empty_frames_prcnt=100.0, n_ids=0)
    populated = df["frame_id"].nunique()
    return dict(
        rows=int(len(df)),
        inst_per_frame=round(len(df) / max(n_frames, 1), 3),
        empty_frames_prcnt=round(100.0 * (n_frames - populated) / max(n_frames, 1), 1),
        n_ids=int(df["instance_id"].nunique()),
    )


def main(args):
    import sleap_io as sio
    from sleap_nn.inference.tracking import apply_tracking
    from video_utils import ensure_resolution, frame_count

    model_paths = resolve_model_paths(args.model)
    out_dir = osp.abspath(osp.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    video, original_res, tracked_res = ensure_resolution(osp.abspath(osp.expanduser(args.video)), tuple(args.img_size) if args.img_size else None)
    n_frames = frame_count(video)
    entries = grid_for(args)

    print(f"Model(s): {model_paths}")
    print(f"Video: {video} ({n_frames} frames)")
    print(f"Sweep: {len(args.peak_thresholds)} threshold(s) x {len(entries)} config(s) = {len(args.peak_thresholds) * len(entries)} runs\n")

    manifest = []
    for threshold in args.peak_thresholds:
        tag = f"pt{threshold:g}".replace(".", "")
        cache_path = osp.join(out_dir, f"untracked_{tag}.slp")
        print(f"[{tag}] pose estimation @ peak_threshold={threshold}")
        infer_seconds = run_inference(video, model_paths, threshold, cache_path, args)
        if infer_seconds:
            print(f"  {infer_seconds:.1f}s ({n_frames / infer_seconds:.1f} fps)")

        detections = sio.load_slp(cache_path)
        n_detections = sum(len(f.instances) for f in detections.labeled_frames)
        print(f"  {n_detections} detections ({n_detections / max(n_frames, 1):.2f}/frame)")

        for entry in entries:
            name = f"{tag}_{entry['name']}"
            config_dir = osp.join(out_dir, name)
            os.makedirs(config_dir, exist_ok=True)

            # Reload per config: apply_tracking stamps .track onto the instances it is
            # handed, so a shared Labels would leak identities between configurations.
            labels = sio.load_slp(cache_path)
            start = perf_counter()
            tracked = apply_tracking(labels, tracker_config_for(entry, args.max_instances))
            track_seconds = perf_counter() - start

            tracked_slp = osp.join(config_dir, "predictions.slp")
            sio.save_slp(tracked, tracked_slp)
            outputs = to_precision_track_csvs(tracked_slp, config_dir, args.bbox_format, args.pcutoff, args.bbox_exclude_keypoints)
            rescale_outputs(outputs, tracked_res, original_res)
            for output in outputs:
                output.save()

            diag = diagnostics(osp.join(config_dir, "tracked_bboxes.csv"), n_frames)
            manifest.append(
                dict(
                    name=name,
                    peak_threshold=threshold,
                    params={k: v for k, v in entry.items() if k != "name"},
                    max_instances=None if entry.get("drop_max_instances") else args.max_instances,
                    n_detections=n_detections,
                    tracking_seconds=round(track_seconds, 2),
                    **diag,
                )
            )
            print(
                f"  {entry['name']:20s} {track_seconds:6.1f}s  rows={diag['rows']:6d}  "
                f"inst/frame={diag['inst_per_frame']:5.2f}  empty={diag['empty_frames_prcnt']:5.1f}%  ids={diag['n_ids']}"
            )
        print()

    # Merge into any existing manifest rather than replacing it: a sweep is often run in
    # several passes (extra thresholds, extra configs), and overwriting would discard the
    # provenance of every config from the earlier passes while their CSVs stayed on disk.
    manifest_path = osp.join(out_dir, "sweep_manifest.json")
    merged = {}
    if osp.isfile(manifest_path):
        with open(manifest_path, "r") as f:
            merged = {entry["name"]: entry for entry in json.load(f)}
    merged.update({entry["name"]: entry for entry in manifest})
    with open(manifest_path, "w") as f:
        json.dump(sorted(merged.values(), key=lambda e: e["name"]), f, indent=2)

    best_coverage = max(manifest, key=lambda m: m["inst_per_frame"]) if manifest else None
    print(f"Wrote {manifest_path} ({len(manifest)} configs)")
    if best_coverage:
        print(f"Highest coverage: {best_coverage['name']} at {best_coverage['inst_per_frame']} inst/frame")
    print(
        "\nCoverage is not quality -- score these in the precision_track environment:\n"
        f"  python sweep_evaluate.py --sweep-dir {out_dir} --gt <gt>.csv --metainfo ../configs/metadata/stripedmice.py"
    )
    return manifest


if __name__ == "__main__":
    main(parse_args())
