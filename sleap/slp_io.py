"""Read SLEAP ``.slp`` files and resolve trained-model directories.

The SLEAP counterpart of ``dlc/dlc_io.py``, and far smaller than it: ``sleap-io`` can
both read *and* write its own format, so there is no need to hand-build the annotation
container the way ``dlc_io.build_collected_data`` had to for DeepLabCut.

``read_predictions`` deliberately returns the same ``(frame_ids, poses, node_names,
track_names)`` shape that ``dlc_io.tracked_frames`` produces, so the conversion code in
``track.py`` is identical in both benchmark directories.
"""

import json
import os.path as osp
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MODEL_GROUP_FILE = "model_group.json"


def load_labels(slp_path: str):
    """Load a ``.slp`` file, deferring the ``sleap_io`` import to call time."""
    import sleap_io as sio

    return sio.load_slp(slp_path)


def frame_filenames(labels) -> List[str]:
    """Per-frame source image names for an ``ImageVideo``-backed ``Labels``.

    ``sio.load_coco`` represents an image folder as a single ``Video`` whose
    ``filename`` is a *list* of paths, one per frame, so a frame's own name is that list
    indexed by ``frame_idx`` rather than the video's filename. Videos backed by an
    actual movie file expose a plain string, which is returned for every frame.
    """
    names = []
    for labeled_frame in labels.labeled_frames:
        filename = labeled_frame.video.filename
        if isinstance(filename, (list, tuple)):
            names.append(osp.basename(filename[labeled_frame.frame_idx]))
        else:
            names.append(osp.basename(filename))
    return names


def identity_coverage(labels) -> Dict[str, int]:
    """Count how much of ``labels`` carries usable identity annotations.

    SLEAP's ``multi_class_*`` heads classify each instance into one of the project's
    tracks, which needs *every* instance in a frame to be identified -- a frame with one
    unlabelled animal teaches the head that the animal belongs to no class at all. This
    reports the partially- and fully-labelled frame counts separately so ``train.py``
    can refuse a run that would train on almost nothing.
    """
    frames = labels.labeled_frames
    with_instances = [f for f in frames if len(f.instances) > 0]
    any_tracked = [f for f in with_instances if any(i.track is not None for i in f.instances)]
    all_tracked = [f for f in with_instances if all(i.track is not None for i in f.instances)]
    return dict(
        n_frames=len(frames),
        n_empty_frames=len(frames) - len(with_instances),
        n_instances=sum(len(f.instances) for f in frames),
        n_tracked_instances=sum(1 for f in frames for i in f.instances if i.track is not None),
        n_frames_any_tracked=len(any_tracked),
        n_frames_all_tracked=len(all_tracked),
        n_tracks=len(labels.tracks),
    )


def fully_identified_frames(labels):
    """The subset of ``labels``' frames where every instance carries a track."""
    return [f for f in labels.labeled_frames if len(f.instances) > 0 and all(i.track is not None for i in f.instances)]


def read_predictions(slp_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Vectorize a tracked ``.slp`` into the array layout ``track.py`` consumes.

    Returns ``(frame_ids, poses, node_names, track_names)`` where ``poses`` has shape
    ``(n_frames, n_tracks, n_nodes, 3)`` holding ``(x, y, score)`` and missing
    instances are all-NaN -- the same contract as ``dlc_io.tracked_frames``.

    Instances whose ``track`` is ``None`` (tracking disabled, or a detection the tracker
    refused to assign) are appended to the trailing slots so untracked predictions are
    still converted rather than silently dropped.
    """
    labels = load_labels(slp_path)
    if not labels.labeled_frames:
        raise SystemExit(f"{slp_path} contains no labeled frames.")

    node_names = list(labels.skeletons[0].node_names)
    track_names = [track.name for track in labels.tracks]
    slot_of = {track.name: index for index, track in enumerate(labels.tracks)}

    # Untracked instances need slots of their own; size them by the worst frame so a
    # frame with more detections than tracks never overflows.
    max_untracked = max((sum(1 for i in f.instances if i.track is None) for f in labels.labeled_frames), default=0)
    n_slots = len(track_names) + max_untracked
    if n_slots == 0:
        raise SystemExit(f"{slp_path} contains no instances to convert.")

    frame_ids = np.array([f.frame_idx for f in labels.labeled_frames], dtype=np.int64)
    poses = np.full((len(frame_ids), n_slots, len(node_names), 3), np.nan, dtype=np.float64)

    for frame_position, labeled_frame in enumerate(labels.labeled_frames):
        next_free = len(track_names)
        for instance in labeled_frame.instances:
            if instance.track is not None:
                slot = slot_of[instance.track.name]
            else:
                slot = next_free
                next_free += 1
            poses[frame_position, slot] = _instance_array(instance, len(node_names))

    order = np.argsort(frame_ids, kind="stable")
    return frame_ids[order], poses[order], node_names, track_names


def _instance_array(instance, n_nodes: int) -> np.ndarray:
    """``(n_nodes, 3)`` array of ``(x, y, score)`` for one instance.

    User-labelled instances (``Instance``) have no per-point score, so their visible
    points score 1.0 -- a hand annotation is as confident as it gets.
    """
    try:
        array = instance.numpy(scores=True)
    except TypeError:  # pragma: no cover - a user Instance, not a PredictedInstance
        array = None

    if array is None or array.shape[1] < 3:
        xy = instance.numpy() if array is None else array[:, :2]
        scores = np.where(np.isnan(xy).any(axis=1), np.nan, 1.0)
        array = np.column_stack([xy, scores])

    if array.shape[0] != n_nodes:  # pragma: no cover - skeleton mismatch
        raise SystemExit(f"Instance has {array.shape[0]} nodes but the skeleton declares {n_nodes}.")
    return array[:, :3]


def write_model_group(model_dirs: Sequence[str], path: str) -> str:
    """Record a multi-model setup in the order ``sleap_nn`` expects ``model_paths``.

    Top-down inference needs the centroid model *before* the centered-instance model.
    Persisting that order next to the checkpoints means ``track.py`` can be handed one
    path whether the setup is one model or two.
    """
    payload = dict(model_paths=[osp.abspath(osp.expanduser(d)) for d in model_dirs])
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def resolve_model_paths(spec: str) -> List[str]:
    """Turn a user-supplied ``--model`` into the list ``predict()`` wants.

    Accepts a ``model_group.json`` (multi-model), a directory containing one (a
    top-down run root), or a single model directory.
    """
    spec = osp.abspath(osp.expanduser(spec))
    if osp.isfile(spec) and spec.endswith(".json"):
        with open(spec, "r") as f:
            return list(json.load(f)["model_paths"])

    grouped = osp.join(spec, MODEL_GROUP_FILE)
    if osp.isfile(grouped):
        with open(grouped, "r") as f:
            return list(json.load(f)["model_paths"])

    if not osp.isdir(spec):
        raise SystemExit(f"--model: {spec} is neither a model directory nor a {MODEL_GROUP_FILE}.")
    return [spec]


def read_training_config(model_dir: str) -> Optional[Dict]:
    """Load a trained model's ``training_config.yaml``, or ``None`` if absent."""
    import yaml

    path = osp.join(osp.abspath(osp.expanduser(model_dir)), "training_config.yaml")
    if not osp.isfile(path):
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f)


def model_node_names(model_dirs: Sequence[str]) -> Optional[List[str]]:
    """Skeleton node names recorded by the first model that declares them."""
    for model_dir in model_dirs:
        config = read_training_config(model_dir)
        if not config:
            continue
        nodes = (config.get("data_config", {}) or {}).get("skeletons")
        if isinstance(nodes, dict) and nodes:
            first = next(iter(nodes.values()))
            if isinstance(first, dict) and first.get("nodes"):
                return [n["name"] if isinstance(n, dict) else str(n) for n in first["nodes"]]
    return None
