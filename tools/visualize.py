import argparse
import yaml
from mmengine.config import Config

from precision_track import Result, Visualizer
from precision_track.utils import load_user_configs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Path to the video to process")
    parser.add_argument("sink", help="Path to the saved annotated video")
    args = parser.parse_args()
    return args


def main(args):
    system_configs_path = "../configs/tasks/tracking.py"
    with open("../configs/user_configs.yaml", "r") as f:
        user_configs = yaml.safe_load(f)
    load_user_configs(user_configs, system_configs_path)

    cfg = Config.fromfile(system_configs_path)

    result = Result(outputs=cfg.get("outputs"))
    result.read()
    visualizer = Visualizer(**cfg.get("visualizer"))
    visualizer(args.source, result, args.sink)


if __name__ == "__main__":
    main(parse_args())
