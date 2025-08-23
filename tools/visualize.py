import argparse

from mmengine.config import Config

from precision_track import Result, Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("source", help="Path to the video to process")
    parser.add_argument("sink", help="Path to the saved annotated video")
    args = parser.parse_args()
    return args


def main(args):
    cfg = Config.fromfile("../configs/tasks/tracking.py")
    result = Result(outputs=cfg.get("outputs"))
    result.read()
    visualizer = Visualizer(**cfg.get("visualizer"))
    visualizer(args.source, result, args.sink)


if __name__ == "__main__":
    main(parse_args())
