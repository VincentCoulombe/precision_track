"""Run DLC's three tracking stages explicitly, timing each, and emit a throughput CSV.

``track.py`` calls ``analyze_videos(auto_track=True)``, which fuses pose estimation,
tracklet generation and stitching into one call -- so it can only report an end-to-end
figure. This script runs the stages separately to measure each, and writes
``mean_throughput_per_substep.csv`` in the same schema as
``TrackingTestingLoop._save_throughput_csv``: ``substep,throughput_fps`` plus a final
``end_to_end`` row.

**The substep names are DLC's own** (``pose_estimation``, ``tracklet_generation``,
``stitching``, ``saving_results``) and deliberately do not mimic PrecisionTrack's
(``detection``, ``init_frame``, ``tracking``, ...), because the two pipelines do not share
stages. The rows are not comparable one-for-one; only ``end_to_end`` is.

Runs in the ``dlc`` conda environment. Example:
    python profile_stages.py <video>.mp4 --project projects/<proj>/config.yaml \\
        --n-tracks 5 --min-affinity 0.01 --num-animals 5 \\
        --out-dir ../work_dir/dlc_profile --img-size 640 640
"""

import argparse
import os
import os.path as osp
import shutil
from collections import OrderedDict
from time import perf_counter

import pandas as pd
import yaml

from dlc_io import find_tracked_h5, read_project_config
from track import rescale_outputs, to_precision_track_csvs
from video_utils import ensure_resolution, frame_count

SUBSTEP_ORDER = ["pose_estimation", "tracklet_generation", "stitching", "saving_results"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Path to the video to track")
    parser.add_argument("--project", required=True, help="Path to the DLC project's config.yaml")
    parser.add_argument("--out-dir", required=True, help="Directory receiving the CSVs")
    parser.add_argument("--dest-folder", default=None, help="Where DLC writes its own artifacts. Defaults to <out-dir>/dlc_artifacts.")
    parser.add_argument("--n-tracks", type=int, default=None, help="Number of animals. Defaults to the project's individuals count.")
    parser.add_argument("--shuffle", type=int, default=1, help="Shuffle index. Defaults to 1.")
    parser.add_argument("--batch-size", type=int, default=30, help="Inference batch size. Defaults to 30.")
    parser.add_argument("--img-size", type=int, nargs=2, default=None, metavar=("HEIGHT", "WIDTH"), help="Rescale before tracking, e.g. --img-size 640 640.")
    parser.add_argument("--track-method", choices=["box", "ellipse", "skeleton"], default="box", help="Tracklet generator. Defaults to box.")
    parser.add_argument("--split-tracklets", action="store_true", default=True, help="Pass split_tracklets=True to stitch_tracklets (default).")
    parser.add_argument("--no-split-tracklets", dest="split_tracklets", action="store_false", help="Disable tracklet splitting.")
    parser.add_argument("--min-length", type=int, default=10, help="stitch_tracklets min_length. Defaults to 10.")
    parser.add_argument("--min-affinity", type=float, default=None, help="Override the PAF predictor's min_affinity (needs re-inference).")
    parser.add_argument("--num-animals", type=int, default=None, help="Override the PAF predictor's num_animals (needs re-inference).")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0.")
    parser.add_argument("--profile-output", default=None, help="Where to write mean_throughput_per_substep.csv. Defaults to <out-dir>/.")
    return parser.parse_args()


def predictor_overrides(args) -> dict:
    over = {}
    if args.min_affinity is not None:
        over["min_affinity"] = args.min_affinity
    if args.num_animals is not None:
        over["num_animals"] = args.num_animals
    return over


def model_config_path(config_path: str, shuffle: int) -> str:
    """Locate the trained shuffle's pytorch_config.yaml."""
    project_dir = osp.dirname(osp.abspath(config_path))
    root = osp.join(project_dir, "dlc-models-pytorch", "iteration-0")
    candidates = [osp.join(root, d, "train", "pytorch_config.yaml") for d in sorted(os.listdir(root)) if d.endswith(f"shuffle{shuffle}")]
    existing = [c for c in candidates if osp.isfile(c)]
    if not existing:
        raise SystemExit(f"No pytorch_config.yaml found for shuffle {shuffle} under {root}.")
    return existing[0]


def save_throughput_csv(timings: "OrderedDict[str, float]", n_frames: int, path: str) -> pd.DataFrame:
    """Mirror of ``TrackingTestingLoop._save_throughput_csv`` for DLC's own substeps."""
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
    import deeplabcut

    config_path = osp.abspath(osp.expanduser(args.project))
    project_cfg = read_project_config(config_path)
    n_tracks = args.n_tracks or len(project_cfg.get("individuals", []))
    if not n_tracks:
        raise SystemExit("Could not determine the number of animals; pass --n-tracks.")

    out_dir = osp.abspath(osp.expanduser(args.out_dir))
    dest_folder = osp.abspath(osp.expanduser(args.dest_folder)) if args.dest_folder else osp.join(out_dir, "dlc_artifacts")
    os.makedirs(dest_folder, exist_ok=True)

    video, original_res, tracked_res = ensure_resolution(osp.abspath(osp.expanduser(args.video)), tuple(args.img_size) if args.img_size else None)
    n_frames = frame_count(video)

    pcfg_path = model_config_path(config_path, args.shuffle)
    overrides = predictor_overrides(args)
    backup = pcfg_path + ".profilebak"
    timings = OrderedDict()

    if overrides:
        shutil.copy(pcfg_path, backup)
    try:
        if overrides:
            cfg = yaml.safe_load(open(pcfg_path))
            cfg["model"]["heads"]["bodypart"]["predictor"].update(overrides)
            yaml.safe_dump(cfg, open(pcfg_path, "w"), sort_keys=False)
            print(f"Applied predictor overrides {overrides}")

        analyze_kwargs = dict(shuffle=args.shuffle, batchsize=args.batch_size, destfolder=dest_folder, auto_track=False, save_as_csv=False)
        if args.device:
            analyze_kwargs["device"] = args.device

        start = perf_counter()
        deeplabcut.analyze_videos(config_path, [video], **analyze_kwargs)
        timings["pose_estimation"] = perf_counter() - start

        start = perf_counter()
        deeplabcut.convert_detections2tracklets(
            config_path,
            [video],
            shuffle=args.shuffle,
            destfolder=dest_folder,
            track_method=args.track_method,
            overwrite=True,
        )
        timings["tracklet_generation"] = perf_counter() - start

        start = perf_counter()
        deeplabcut.stitch_tracklets(
            config_path,
            [video],
            shuffle=args.shuffle,
            n_tracks=n_tracks,
            destfolder=dest_folder,
            track_method=args.track_method,
            min_length=args.min_length,
            split_tracklets=args.split_tracklets,
        )
        timings["stitching"] = perf_counter() - start
    finally:
        if overrides:
            shutil.copy(backup, pcfg_path)
            os.remove(backup)
            print(f"Restored {pcfg_path}")

    h5_path = find_tracked_h5(video, dest_folder)
    print(f"Reading DLC tracks from {h5_path}")

    start = perf_counter()
    outputs = to_precision_track_csvs(h5_path, out_dir, "cxcywh", 0.0, ())
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
