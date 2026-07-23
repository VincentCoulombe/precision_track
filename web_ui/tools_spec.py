"""Declarative spec of the runnable tools and their CLI arguments.

Mirrors the actual ``argparse`` definitions in ``tools/*.py`` (infrastructure
flags ``--launcher`` / ``--local_rank`` are intentionally omitted). Bool flags
are emitted in the ``--flag=true`` form the tools' ``str2bool`` expects.
"""

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"]

TOOLS = {
    "train_detection.py": {
        "label": "Train detection",
        "description": "Train and deploy a detection model on your COCO-style dataset.",
        "positionals": [],
        "flags": [
            {"name": "format_dataset", "type": "bool", "default": True, "help": "Resize/format the dataset before training."},
            {"name": "test", "type": "bool", "default": True, "help": "Run test_detection after training."},
            {"name": "feature_extraction", "type": "bool", "default": True, "help": "Train the feature-extraction head."},
            {"name": "calibrate", "type": "bool", "default": True, "help": "Calibrate the trained model."},
            {"name": "deploy", "type": "bool", "default": True, "help": "Deploy optimized runtime checkpoints."},
            {"name": "optimize_hyperparams", "type": "bool", "default": False, "help": "Optimize tracking hyperparameters."},
        ],
    },
    "train_action_recognition.py": {
        "label": "Train action recognition",
        "description": "Train and deploy a MART (& GMART if group action recognition is enabled) on your MOT-style dataset.",
        "positionals": [],
        "flags": [
            {"name": "test", "type": "bool", "default": True, "help": "Run test_action_recognition after training."},
            {"name": "deploy", "type": "bool", "default": True, "help": "Deploy optimized runtime checkpoints."},
            {
                "name": "config",
                "type": "path",
                "default": "../configs/tasks/training_action_recognition.py",
                "picker": {"mode": "file", "exts": [".py"]},
                "help": "Path to the training config.",
            },
        ],
    },
    "test_detection.py": {
        "label": "Test detection",
        "description": "Evaluate a detection model on your COCO-style dataset.",
        "positionals": [],
        "flags": [],
    },
    "test_tracking.py": {
        "label": "Test tracking",
        "description": "Evaluate tracking on your MOT-style benchmark.",
        "positionals": [],
        "flags": [],
    },
    "test_action_recognition.py": {
        "label": "Test action recognition",
        "description": "Evaluate a MART/GMART model on your MOT-style dataset.",
        "positionals": [],
        "flags": [
            {
                "name": "config",
                "type": "path",
                "default": "../configs/tasks/testing_action_recognition.py",
                "picker": {"mode": "file", "exts": [".py"]},
                "help": "Path to the testing config.",
            }
        ],
    },
    "track.py": {
        "label": "Track a video",
        "description": "Run tracking (and downstream tasks) on a pre-recorded video.",
        "positionals": [
            {"name": "video", "type": "path", "picker": {"mode": "file", "exts": VIDEO_EXTS}, "help": "Path to the video to process."},
        ],
        "flags": [
            {"name": "profile", "type": "bool", "default": False, "help": "Record per-frame timing data."},
        ],
    },
    "create_mot_dataset.py": {
        "label": "Batch-track a dataset",
        "description": "Auto-generate MOT bounding-box annotations for every video of a MOT dataset.",
        "positionals": [],
        "flags": [
            {"name": "force", "type": "bool", "default": False, "help": "Overwrite videos that already have a bboxes file."},
        ],
    },
    "batch_track_directory.py": {
        "label": "Batch-track a directory",
        "description": "Run the 'Track a video' tool on every video inside a directory.",
        "warning": (
            "Turning OFF “restart-tracker-instance” makes a single tracker span the whole "
            "directory, and takes for granted that every video is a frame-by-frame follow-up of "
            "the previous one (one long recording split into consecutive parts). If the videos "
            "are not consecutive fragments of the same recording, the results can be really bad. "
            "It also requires all videos to share the same resolution, track IDs keep counting up "
            "across videos, and (when tracking runs pipelined) offline correction refinement runs "
            "per video without cross-video identity state."
        ),
        "positionals": [
            {"name": "directory", "type": "path", "picker": {"mode": "dir"}, "help": "Directory containing the videos to track."},
        ],
        "flags": [
            {"name": "recursive", "type": "bool", "default": False, "help": "Recurse into sub-directories."},
            {
                "name": "restart-tracker-instance",
                "type": "bool",
                "default": True,
                "help": (
                    "Keep ON unless the videos are consecutive fragments of ONE continuous recording. "
                    "When OFF, a single tracker spans the whole directory so track IDs stay consistent "
                    "across videos (see the warning above)."
                ),
            },
        ],
    },
    "visualize.py": {
        "label": "Visualize outputs",
        "description": "Render tracking & action-recognition outputs into an annotated video.",
        "positionals": [
            {"name": "source", "type": "path", "picker": {"mode": "file", "exts": VIDEO_EXTS}, "help": "Path to the source video."},
            {"name": "sink", "type": "path", "picker": {"mode": "save", "exts": VIDEO_EXTS}, "help": "Path of the annotated video to write."},
        ],
        "flags": [],
    },
    "plot_profiles.py": {
        "label": "Plot timing profiles",
        "description": "Plot per-frame timing charts from a track.py profiling JSON.",
        "positionals": [
            {"name": "json_file", "type": "path", "picker": {"mode": "file", "exts": [".json"]}, "help": "Path to the profile_<ts>.json file."},
        ],
        "flags": [
            {"name": "std-threshold", "type": "float", "default": 2.0, "help": "Std-devs above mean to flag a peak."},
        ],
    },
}


def build_argv(tool: str, values: dict) -> list:
    """Turn submitted values into an argv list for the tool (excluding 'python tool')."""
    spec = TOOLS[tool]
    argv = []
    for pos in spec["positionals"]:
        val = values.get(pos["name"], "")
        if val == "" or val is None:
            raise ValueError(f"Missing required argument: {pos['name']}")
        argv.append(str(val))
    for flag in spec["flags"]:
        name = flag["name"]
        if name not in values or values[name] is None or values[name] == "":
            continue
        val = values[name]
        if flag["type"] == "bool":
            argv.append(f"--{name}={'true' if val in (True, 'true', 'True', 1) else 'false'}")
        else:
            argv.append(f"--{name}={val}")
    return argv
