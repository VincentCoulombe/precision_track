"""Run SLEAP's pipeline stages separately, timing each, and emit a throughput CSV.

``track.py`` hands ``predict()`` a tracker config, which fuses pose estimation and
tracking into one call -- so it can only report an end-to-end figure. This script runs
the stages apart to measure each, and writes ``mean_throughput_per_substep.csv`` in the
same schema as ``TrackingTestingLoop._save_throughput_csv``: ``substep,throughput_fps``
plus a final ``end_to_end`` row.

This is simpler than ``dlc/profile_stages.py``, which had to back up, patch and restore
``pytorch_config.yaml`` to reach DLC's predictor settings. ``sleap_nn`` exposes
``predict()`` and ``apply_tracking()`` as separate functions over a ``Labels``, so the
split needs no config surgery at all.

**The substep names are SLEAP's own** (``pose_estimation``, ``tracking``,
``saving_results``) and deliberately do not mimic PrecisionTrack's (``detection``,
``init_frame``, ``tracking``, ...), because the two pipelines do not share stages. The
rows are not comparable one-for-one; only ``end_to_end`` is.

Example:
    python profile_stages.py <video>.mp4 --model models/bottomup \\
        --out-dir ../work_dir/sleap_profile --max-instances 5 --img-size 640 640
"""

import argparse
import os
import os.path as osp
from collections import OrderedDict
from time import perf_counter

import pandas as pd
from slp_io import resolve_model_paths
from track import build_tracker_config, predictor_kwargs, rescale_outputs, to_precision_track_csvs
from video_utils import ensure_resolution, frame_count

SUBSTEP_ORDER = ["pose_estimation", "tracking", "saving_results"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Path to the video to track")
    parser.add_argument("--model", required=True, help="Trained model directory, or a model_group.json.")
    parser.add_argument("--out-dir", required=True, help="Directory receiving the CSVs")
    parser.add_argument("--max-instances", type=int, default=None, help="Maximum animals per frame.")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size. Defaults to 4.")
    parser.add_argument("--img-size", type=int, nargs=2, default=None, metavar=("HEIGHT", "WIDTH"), help="Rescale before tracking, e.g. --img-size 640 640.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, cpu. Defaults to auto.")
    parser.add_argument("--slp-out", default=None, help="Where the tracked .slp is written. Defaults to <out-dir>/predictions.slp.")
    parser.add_argument("--profile-output", default=None, help="Where to write mean_throughput_per_substep.csv. Defaults to <out-dir>/.")

    parser.add_argument("--peak-threshold", type=float, default=None, help="Minimum keypoint peak confidence.")
    parser.add_argument("--centroid-threshold", type=float, default=None, help="Minimum centroid confidence (top-down only).")
    parser.add_argument("--min-instance-peaks", type=float, default=None, help="Drop instances with fewer peaks than this.")
    parser.add_argument("--integral-refinement", default=None, help="Sub-pixel peak refinement, e.g. integral.")

    parser.add_argument("--candidates-method", choices=["fixed_window", "local_queues"], default="fixed_window", help="Candidate source.")
    parser.add_argument("--scoring-method", default="oks", help="Instance-to-track similarity. Defaults to oks.")
    parser.add_argument("--scoring-reduction", default="mean", help="Score reduction. Defaults to mean.")
    parser.add_argument("--window-size", type=int, default=None, help="Association history, in frames.")
    parser.add_argument("--track-matching-method", default="hungarian", help="Assignment solver. Defaults to hungarian.")
    parser.add_argument("--features", default="keypoints", help="What is compared between instances. Defaults to keypoints.")
    parser.add_argument("--robust-best-instance", type=float, default=1.0, help="Quantile for robust score reduction.")
    parser.add_argument("--min-new-track-points", type=int, default=0, help="Keypoints required to spawn a track.")
    parser.add_argument("--min-match-points", type=int, default=0, help="Keypoints required to match a track.")
    parser.add_argument("--use-kalman", action="store_true", help="Kalman filter over track motion.")
    parser.add_argument("--use-flow", action="store_true", help="Optical-flow candidate shifting.")
    parser.add_argument("--post-connect-single-breaks", action="store_true", help="Bridge single-frame identity breaks.")

    parser.add_argument("--bbox-format", choices=["cxcywh", "xywh"], default="cxcywh", help="On-disk bbox layout. Defaults to cxcywh.")
    parser.add_argument("--pcutoff", type=float, default=0.0, help="Drop keypoints below this score. Defaults to 0.")
    parser.add_argument("--bbox-exclude-keypoints", nargs="*", default=[], metavar="NODE", help="Nodes to leave out of the bounding box.")
    parser.add_argument("--no-tracking", action="store_true", help="Skip the tracking stage entirely.")
    return parser.parse_args()


def save_throughput_csv(timings: "OrderedDict[str, float]", n_frames: int, path: str) -> pd.DataFrame:
    """Mirror of ``TrackingTestingLoop._save_throughput_csv`` for SLEAP's own substeps."""
    rows = []
    total_mean_latency = 0.0
    for substep in SUBSTEP_ORDER:
        if substep not in timings:
            continue
        mean_latency = timings[substep] / max(n_frames, 1)
        total_mean_latency += mean_latency
        rows.append(dict(substep=substep, throughput_fps=1.0 / mean_latency if mean_latency > 0 else float("inf")))
    if total_mean_latency > 0:
        rows.append(dict(substep="end_to_end", throughput_fps=1.0 / total_mean_latency))
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def main(args):
    import sleap_io as sio
    from sleap_nn.inference import predict
    from sleap_nn.inference.tracking import apply_tracking

    model_paths = resolve_model_paths(args.model)
    out_dir = osp.abspath(osp.expanduser(args.out_dir))
    slp_out = osp.abspath(osp.expanduser(args.slp_out)) if args.slp_out else osp.join(out_dir, "predictions.slp")
    os.makedirs(out_dir, exist_ok=True)

    video, original_res, tracked_res = ensure_resolution(osp.abspath(osp.expanduser(args.video)), tuple(args.img_size) if args.img_size else None)
    n_frames = frame_count(video)
    timings = OrderedDict()

    # Stage 1: pose estimation only. Passing tracker_config=None keeps tracking out of
    # this measurement, and returns the untracked Labels in memory.
    kwargs = dict(
        source=video,
        model_paths=model_paths,
        device=args.device,
        batch_size=args.batch_size,
        max_instances=args.max_instances,
        tracker_config=None,
    )
    kwargs.update(predictor_kwargs(args))

    start = perf_counter()
    labels = predict(**kwargs)
    timings["pose_estimation"] = perf_counter() - start

    # Stage 2: tracking over those predictions.
    tracker_config = build_tracker_config(args)
    if tracker_config is not None:
        start = perf_counter()
        labels = apply_tracking(labels, tracker_config)
        timings["tracking"] = perf_counter() - start

    sio.save_slp(labels, slp_out)

    # Stage 3: conversion to PrecisionTrack's CSVs.
    start = perf_counter()
    outputs = to_precision_track_csvs(slp_out, out_dir, args.bbox_format, args.pcutoff, args.bbox_exclude_keypoints)
    rescale_outputs(outputs, tracked_res, original_res)
    for output in outputs:
        output.save()
    timings["saving_results"] = perf_counter() - start

    profile_output = args.profile_output or osp.join(out_dir, "mean_throughput_per_substep.csv")
    df = save_throughput_csv(timings, n_frames, osp.abspath(osp.expanduser(profile_output)))

    print(f"\n{n_frames} frames")
    for substep, seconds in timings.items():
        print(f"  {substep:22s} {seconds:8.2f} s")
    print(f"\nWrote {osp.abspath(osp.expanduser(profile_output))}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main(parse_args())
