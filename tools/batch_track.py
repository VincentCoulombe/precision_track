import argparse
import multiprocessing as mp
from logging import WARNING
from pathlib import Path

import psutil
from mmengine import Config
from mmengine.logging import print_log
from train_detection import str2bool

from precision_track import PipelinedTracker, Tracker
from precision_track.utils import VideoReader, load_user_configs, load_validation_config
from precision_track.utils.io import SUPPORTED_VIDEO_BACKEND


def parse_args():
    parser = argparse.ArgumentParser(description="Batch track every video of a MOT dataset and save the tracked bounding boxes.")
    parser.add_argument(
        "--mot_data_root",
        default=None,
        help="Path to the MOT dataset root (with videos/{train,val}). Defaults to the 'mot_data_root' of the user configs.",
    )
    parser.add_argument(
        "--force",
        type=str2bool,
        default=False,
        help="Re-process and overwrite videos that already have a bounding boxes file. Default to False.",
    )
    args = parser.parse_args()
    return args


def main(args):

    system_configs_path = "../configs/tasks/tracking.py"
    user_system_configs_path = "../configs/user_configs.yaml"
    load_user_configs(user_system_configs_path, system_configs_path)

    config = Config.fromfile(system_configs_path)
    load_validation_config(config)

    mot_data_root = Path(args.mot_data_root or config.mot_data_root)

    nb_cpu_cores = psutil.cpu_count(logical=False)
    pipelined = config.pipelined
    if pipelined and nb_cpu_cores < 3:
        print_log(
            f"The tracking can not be pipelined on this machine since it only have {nb_cpu_cores} CPU cores. This will slow down inference significantly.",
            logger="current",
            level=WARNING,
        )
        pipelined = False

    for split in ("train", "val"):
        videos_dir = mot_data_root / "videos" / split
        if not videos_dir.is_dir():
            print_log(
                f"Skipping the '{split}' split since it has no videos directory: {videos_dir}",
                logger="current",
                level=WARNING,
            )
            continue

        bboxes_dir = mot_data_root / "bboxes" / split
        bboxes_dir.mkdir(parents=True, exist_ok=True)

        video_files = sorted(f for f in videos_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_BACKEND)

        for video_file in video_files:
            out_csv = bboxes_dir / f"{video_file.stem}.csv"
            if out_csv.exists() and not args.force:
                print_log(
                    f"Skipping '{video_file.stem}' since a bounding boxes file already exists: {out_csv}",
                    logger="current",
                    level=WARNING,
                )
                continue

            print_log(f"Tracking '{video_file.name}' ({split}) -> {out_csv}", logger="current")
            video = VideoReader(str(video_file))
            outputs = [
                dict(
                    type="CsvBoundingBoxes",
                    path=str(out_csv),
                    instance_data="pred_track_instances",
                    subtype="tracked_bboxes",
                    bbox_format="cxcywh",
                    save_bbox_format=["x", "y", "w", "h"],
                    precision=64,
                )
            ]

            if pipelined:
                tracker = PipelinedTracker(
                    detector=config.get("detector"),
                    assigner=config.get("assigner"),
                    validator=config.get("validator"),
                    analyzer=config.get("analyzer"),
                    outputs=outputs,
                    expected_resolution=(video.resolution[1], video.resolution[0], 3),
                    batch_size=config.get("batch_size"),
                    verbose=True,
                )
            else:
                tracker = Tracker(
                    detector=config.get("detector"),
                    assigner=config.get("assigner"),
                    validator=config.get("validator"),
                    analyzer=config.get("analyzer"),
                    outputs=outputs,
                    batch_size=config.get("batch_size"),
                    verbose=True,
                )
            tracker(video=video)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(parse_args())
