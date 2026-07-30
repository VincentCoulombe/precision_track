import argparse
import multiprocessing as mp
import os
from datetime import datetime
from logging import WARNING

import psutil
from mmengine import Config
from mmengine.logging import print_log
from train_detection import str2bool

from precision_track import PipelinedTracker, Tracker
from precision_track.registry import TRACKING
from precision_track.utils import VideoReader, load_user_configs, load_validation_config, refine_corrections_offline


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("video", help="Path to the video to process")
    parser.add_argument("--profile", type=str2bool, default=False, help="To profile the tracker. Default to False.")
    args = parser.parse_args()
    return args


def build_video_config(
    video_path,
    system_configs_path="../configs/tasks/tracking.py",
    user_system_configs_path="../configs/user_configs.yaml",
    tool="track",
    validate=True,
):
    """Compute the per-video tracking config (with output paths pointing at this video's
    per-video sub-directory). Kept separate from tracker construction so a shared tracker
    instance can be re-pointed at each video's outputs in batch mode.

    ``validate=False`` skips the config check for callers that already validated once (batch
    mode revalidates nothing per video)."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    load_user_configs(user_system_configs_path, system_configs_path, dynamic_work_dir_subdir=video_name, tool=tool, validate=validate)
    config = Config.fromfile(system_configs_path)
    load_validation_config(config)
    return config


def decide_pipelined(config):
    """Whether tracking should run pipelined, given the config and the machine's core count."""
    nb_cpu_cores = psutil.cpu_count(logical=False)
    pipelined = config.pipelined
    if pipelined and nb_cpu_cores < 3:
        print_log(
            f"The tracking can not be pipelined on this machine since it only have {nb_cpu_cores} CPU cores. This will slow down inference significantly.",
            logger="current",
            level=WARNING,
        )
        pipelined = False
    return pipelined


def main(args):

    config = build_video_config(
        args.video,
        tool=getattr(args, "tool", "track"),
        validate=getattr(args, "validate", True),
    )
    video = VideoReader(args.video)
    pipelined = decide_pipelined(config)
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

    if config.get("with_offline_correction_refinement") and config.get("validator") is not None:
        validator = getattr(tracker, "validator", None)
        if validator is None or not hasattr(validator, "identities"):
            validator = TRACKING.build(config.get("validator"))
        refine_corrections_offline(config.get("outputs"), validator)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(parse_args())
