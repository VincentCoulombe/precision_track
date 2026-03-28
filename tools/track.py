import argparse
from datetime import datetime
import multiprocessing as mp
from logging import WARNING
import os
import psutil
from mmengine import Config
from mmengine.logging import print_log

from precision_track import PipelinedTracker, Tracker
from precision_track.utils import VideoReader, load_user_configs, load_validation_config
from train_detection import str2bool


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("video", help="Path to the video to process")
    parser.add_argument("--profile", type=str2bool, default=False, help="To profile the tracker. Default to False.")
    args = parser.parse_args()
    return args


def main(args):

    system_configs_path = "../configs/tasks/tracking.py"
    user_system_configs_path = "../configs/user_configs.yaml"
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    load_user_configs(user_system_configs_path, system_configs_path, dynamic_work_dir_subdir=video_name)

    config = Config.fromfile(system_configs_path)
    load_validation_config(config)
    video = VideoReader(args.video)
    nb_cpu_cores = psutil.cpu_count(logical=False)
    pipelined = config.pipelined
    if pipelined and nb_cpu_cores < 3:
        print_log(
            f"The tracking can not be pipelined on this machine since it only have {nb_cpu_cores} CPU cores. This will slow down inference significantly.",
            logger="current",
            level=WARNING,
        )
        pipelined = False
    if pipelined:
        if args.profile:
            print_log(
                f"The pipelined tracker does not support profiling.",
                logger="current",
                level=WARNING,
            )
        tracker = PipelinedTracker(
            detector=config.get("detector"),
            assigner=config.get("assigner"),
            validator=config.get("validator"),
            analyzer=config.get("analyzer"),
            outputs=config.get("outputs"),
            expected_resolution=(video.resolution[1], video.resolution[0], 3),
            batch_size=config.get("batch_size"),
            verbose=True,
        )
        tracker(video=video)
    else:
        if args.profile:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            profile = os.path.join(config.saving_directory, f"profile_{timestamp}.json")
        else:
            profile = ""
        tracker = Tracker(
            detector=config.get("detector"),
            assigner=config.get("assigner"),
            validator=config.get("validator"),
            analyzer=config.get("analyzer"),
            outputs=config.get("outputs"),
            batch_size=config.get("batch_size"),
            verbose=True,
            profile=profile,
        )
        tracker(video=video)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(parse_args())
