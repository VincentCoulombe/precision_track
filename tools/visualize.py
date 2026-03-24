import argparse
import os
from mmengine.config import Config

from precision_track import Result, Visualizer
from precision_track.utils import load_user_configs, load_writers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Path to the video to process")
    parser.add_argument("sink", help="Path to the saved annotated video")
    args = parser.parse_args()
    return args


def main(args):
    system_configs_path = "../configs/tasks/tracking.py"
    user_system_configs_path = "../configs/user_configs.yaml"
    source_name = os.path.splitext(os.path.basename(args.source))[0]
    load_user_configs(user_system_configs_path, system_configs_path, dynamic_work_dir_subdir=source_name)

    cfg = Config.fromfile(system_configs_path)
    load_writers(cfg)

    result = Result(outputs=cfg.get("outputs"))
    result.read(not_exists_ok=True)
    visualizer = Visualizer(**cfg.get("visualizer"))
    visualizer(args.source, result, args.sink)


if __name__ == "__main__":
    main(parse_args())
