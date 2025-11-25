import argparse
import multiprocessing as mp
from logging import WARNING
import yaml
import psutil
from mmengine import Config
from mmengine.logging import print_log

from precision_track import PipelinedTracker, Tracker
from precision_track.utils import VideoReader, load_user_configs


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("video", help="Path to the video to process")
    args = parser.parse_args()
    return args


def main(args):
    system_configs_path = "../configs/tasks/tracking.py"
    with open("../configs/user_configs.yaml", "r") as f:
        user_configs = yaml.safe_load(f)
    load_user_configs(user_configs, system_configs_path)
    config = Config.fromfile(system_configs_path)
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
        tracker = Tracker(
            detector=config.get("detector"),
            assigner=config.get("assigner"),
            validator=config.get("validator"),
            analyzer=config.get("analyzer"),
            outputs=config.get("outputs"),
            batch_size=config.get("batch_size"),
            verbose=True,
        )
        tracker(video=video)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(parse_args())
