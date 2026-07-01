import argparse
import multiprocessing as mp
from logging import WARNING
from pathlib import Path
from types import SimpleNamespace

from mmengine.logging import print_log
from track import build_video_config, decide_pipelined
from track import main as track_main
from train_detection import str2bool

from precision_track import PipelinedTracker, Tracker
from precision_track.registry import TRACKING
from precision_track.utils import VideoReader, refine_corrections_offline
from precision_track.utils.io import SUPPORTED_VIDEO_BACKEND


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch track every video inside a directory, running the full configured tracking pipeline per video."
    )
    parser.add_argument("directory", help="Path to the directory containing the videos to track.")
    parser.add_argument("--recursive", type=str2bool, default=False, help="Recurse into sub-directories. Default to False.")
    parser.add_argument(
        "--restart-tracker-instance",
        dest="restart_tracker_instance",
        type=str2bool,
        default=True,
        help=(
            "If True (default), a fresh tracker is built for every video (IDs restart per video). "
            "If False, a single tracker instance tracks the whole directory so track IDs stay "
            "consistent across the videos (useful when one long recording is split into sub-videos)."
        ),
    )
    args = parser.parse_args()
    return args


def _maybe_refine(outputs, validator_cfg, tracker=None):
    """Run offline correction refinement, mirroring track.main's validator fallback."""
    validator = getattr(tracker, "validator", None) if tracker is not None else None
    if validator is None or not hasattr(validator, "identities"):
        validator = TRACKING.build(validator_cfg)
    refine_corrections_offline(outputs, validator)


def track_directory_shared(videos):
    """Track every video with a SINGLE tracker instance so IDs persist across videos. Outputs
    are saved then reset after each video (into that video's per-video sub-directory) to keep
    RAM flat; only the outputs are reset between videos, never the association/validator state."""
    first_config = build_video_config(str(videos[0]))
    pipelined = decide_pipelined(first_config)

    assigner = first_config.get("assigner")
    assigner["auto_reset"] = False  # keep the ID registry alive across videos
    detector = first_config.get("detector")
    validator = first_config.get("validator")
    analyzer = first_config.get("analyzer")
    batch_size = first_config.get("batch_size")

    # A shared tracker assumes one continuous recording, so every video must share a resolution.
    first_res = VideoReader(str(videos[0])).resolution
    for video in videos[1:]:
        res = VideoReader(str(video)).resolution
        if res != first_res:
            msg = (
                f"{video.name} has resolution {res}, but the first video is {first_res}. "
                "Shared-instance mode requires a constant resolution across videos."
            )
            if pipelined:
                raise ValueError(msg)
            print_log(msg + " Proceeding, but ID continuity may be unreliable.", logger="current", level=WARNING)

    if pipelined:
        tracker = PipelinedTracker(
            detector=detector,
            assigner=assigner,
            validator=validator,
            analyzer=analyzer,
            outputs=first_config.get("outputs"),
            expected_resolution=(first_res[1], first_res[0], 3),
            batch_size=batch_size,
            verbose=True,
        )
        pending_refinements = []
        try:
            for i, video in enumerate(videos):
                config_v = first_config if i == 0 else build_video_config(str(video))
                print_log(f"[{i + 1}/{len(videos)}] Tracking {video.name}", logger="current")
                if i > 0:
                    tracker._advance_to_next_video(config_v.get("outputs"))
                tracker.stream(VideoReader(str(video)))
                if config_v.get("with_offline_correction_refinement") and config_v.get("validator") is not None:
                    pending_refinements.append((config_v.get("outputs"), config_v.get("validator")))
        finally:
            tracker.close()
        # Outputs are only on disk once close() has flushed the last video, so refine afterwards.
        for outputs, validator_cfg in pending_refinements:
            _maybe_refine(outputs, validator_cfg)
    else:
        tracker = Tracker(
            detector=detector,
            assigner=assigner,
            validator=validator,
            analyzer=analyzer,
            outputs=first_config.get("outputs"),
            batch_size=batch_size,
            verbose=True,
        )
        for i, video in enumerate(videos):
            config_v = first_config if i == 0 else build_video_config(str(video))
            print_log(f"[{i + 1}/{len(videos)}] Tracking {video.name}", logger="current")
            if i > 0:
                # Keep IDs/Kalman state, but rebase the frame clock since img_id restarts at 0.
                tracker.association_step.rebase_for_new_video()
            tracker.set_outputs(config_v.get("outputs"))
            tracker.predict(VideoReader(str(video)), save=True, reset_after_save=True)
            if config_v.get("with_offline_correction_refinement") and config_v.get("validator") is not None:
                _maybe_refine(config_v.get("outputs"), config_v.get("validator"), tracker=tracker)


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
    if args.restart_tracker_instance:
        for i, video in enumerate(videos, 1):
            print_log(f"[{i}/{len(videos)}] Tracking {video.name}", logger="current")
            track_main(SimpleNamespace(video=str(video), profile=False))
    else:
        track_directory_shared(videos)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(parse_args())
