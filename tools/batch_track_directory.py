import argparse
import multiprocessing as mp
from logging import WARNING
from pathlib import Path
from types import SimpleNamespace

from mmengine.logging import print_log
from track import main as track_main
from train_detection import str2bool

from precision_track.utils.io import SUPPORTED_VIDEO_BACKEND


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch track every video inside a directory, running the full configured tracking pipeline per video."
    )
    parser.add_argument("directory", help="Path to the directory containing the videos to track.")
    parser.add_argument("--recursive", type=str2bool, default=False, help="Recurse into sub-directories. Default to False.")
    args = parser.parse_args()
    return args


def main(args):
    directory = Path(args.directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory.")

    entries = directory.rglob("*") if args.recursive else directory.iterdir()
    videos = sorted(f for f in entries if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_BACKEND)

    if not videos:
        print_log(f"No video files found in {directory}.", logger="current", level=WARNING)
        return

    print_log(f"Found {len(videos)} video(s) to track in {directory}.", logger="current")
    for i, video in enumerate(videos, 1):
        print_log(f"[{i}/{len(videos)}] Tracking {video.name}", logger="current")
        track_main(SimpleNamespace(video=str(video), profile=False))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(parse_args())
