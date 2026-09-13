"""Convert a PrecisionTrack COCO pose dataset into a multi-animal DeepLabCut project.

The COCO side is parsed with ``sleap-io`` (``sio.load_coco``), which yields a ``Labels``
object carrying the skeleton (node names and edges taken from the COCO ``categories``)
and one instance per annotation with NaN for unlabelled keypoints. The DLC side --
``config.yaml`` plus ``labeled-data/<name>/CollectedData_<scorer>.h5`` -- is written by
``dlc/dlc_io.py``, because sleap-io can read DLC projects but cannot write them.

Example:
    python coco2dlc.py ~/Documents/datasets/MICE/pose-estimation_640x640 \\
        --name mice --scorer coco

PrecisionTrack COCO datasets carry no track IDs, so instances are assigned to
``individual1..individualN`` in annotation order and ``identity`` is left ``false``:
multi-animal DLC assembles animals from part-affinity fields, which does not require
consistent identities across frames, but the identity head must not be trained on
arbitrary assignments.
"""

import argparse
import json
import os
import os.path as osp
import shutil
from collections import Counter
from typing import List, Tuple

import numpy as np
from dlc_io import build_collected_data, individual_names, multianimal_config_updates, write_collected_data

IMAGE_SUBDIR = "images"
ANNOTATION_SUBDIR = "annotations"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", help="Root of the COCO dataset (must contain annotations/ and images/)")
    parser.add_argument("--name", required=True, help="Name of the DLC project (Task)")
    parser.add_argument("--scorer", default="coco", help="Annotator name used in the DLC column index. Defaults to 'coco'.")
    parser.add_argument(
        "--individuals",
        default="auto",
        help="Number of animals. 'auto' (default) uses the maximum number of annotations found in a single image.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Annotation files to convert. Defaults to train and val.")
    parser.add_argument("--projects-dir", default=osp.join(osp.dirname(osp.abspath(__file__)), "projects"), help="Where the DLC project is created.")
    parser.add_argument("--videos", nargs="*", default=[], help="Optional videos to register in the project's video_sets.")
    parser.add_argument("--symlink", action="store_true", help="Symlink images into labeled-data/ instead of copying them.")
    parser.add_argument("--identity", action="store_true", help="Set identity: true in config.yaml. Only meaningful if your COCO data has real track IDs.")
    return parser.parse_args()


def _skeleton_info(skeleton) -> Tuple[List[str], List[List[str]]]:
    """Extract ``(node_names, edges_as_name_pairs)`` from a sleap-io ``Skeleton``."""
    try:
        nodes = list(skeleton.node_names)
    except AttributeError:
        nodes = [node.name for node in skeleton.nodes]

    try:
        edges = [[nodes[src], nodes[dst]] for src, dst in skeleton.edge_inds]
    except AttributeError:
        edges = [[edge.source.name, edge.destination.name] for edge in skeleton.edges]
    return nodes, edges


def _frame_filename(labeled_frame) -> str:
    """Image file name backing a ``LabeledFrame`` loaded from COCO."""
    filename = getattr(labeled_frame.video, "filename", None)
    if isinstance(filename, (list, tuple)):
        idx = labeled_frame.frame_idx if labeled_frame.frame_idx < len(filename) else 0
        filename = filename[idx]
    if filename is None:
        raise ValueError("A labeled frame has no backing image file; is this really a COCO dataset?")
    return osp.basename(str(filename))


def _source_image(labeled_frame, dataset_root: str, filename: str) -> str:
    candidate = getattr(labeled_frame.video, "filename", None)
    if isinstance(candidate, (list, tuple)):
        idx = labeled_frame.frame_idx if labeled_frame.frame_idx < len(candidate) else 0
        candidate = candidate[idx]
    if candidate and osp.isfile(str(candidate)):
        return str(candidate)
    return osp.join(dataset_root, IMAGE_SUBDIR, filename)


def load_splits(dataset_root: str, splits: List[str]):
    """Load each requested COCO split, returning ``(frames, split_map, skeleton)``.

    ``frames`` is the ordered ``(filename, keypoints, source_path)`` list across every
    split; ``split_map`` maps a split name to its image file names.
    """
    import sleap_io as sio

    frames, split_map, skeleton = [], {}, None
    seen = set()

    for split in splits:
        json_path = osp.join(dataset_root, ANNOTATION_SUBDIR, f"{split}.json")
        if not osp.isfile(json_path):
            print(f"Skipping split '{split}': {json_path} does not exist.")
            continue

        labels = sio.load_coco(json_path, dataset_root=dataset_root)
        if not labels.skeletons:
            raise ValueError(f"{json_path} defines no keypoints; PrecisionTrack datasets need a 'categories[*].keypoints' entry.")
        if skeleton is None:
            skeleton = labels.skeletons[0]
        elif _skeleton_info(labels.skeletons[0])[0] != _skeleton_info(skeleton)[0]:
            raise ValueError(f"Split '{split}' uses a different skeleton than the previous splits.")

        split_files, empty = [], 0
        for labeled_frame in labels:
            filename = _frame_filename(labeled_frame)
            if filename in seen:
                raise ValueError(f"Image '{filename}' appears in more than one split; DLC cannot hold it twice.")
            seen.add(filename)

            keypoints = [instance.numpy()[:, :2] for instance in labeled_frame.instances]
            keypoints = [kp for kp in keypoints if not np.isnan(kp).all()]
            if not keypoints:
                # An image with no labelled keypoint is an all-NaN target row, which gives a
                # PAF model nothing to learn from and can destabilise the loss.
                empty += 1
                continue

            frames.append((filename, np.stack(keypoints), _source_image(labeled_frame, dataset_root, filename)))
            split_files.append(filename)

        split_map[split] = split_files
        skipped = f" ({empty} skipped: no labelled keypoint)" if empty else ""
        print(f"Loaded split '{split}': {len(split_files)} images{skipped}.")

    if skeleton is None:
        raise ValueError(f"No annotation file found under {osp.join(dataset_root, ANNOTATION_SUBDIR)} for splits {splits}.")
    return frames, split_map, skeleton


def resolve_individuals(frames, requested: str) -> int:
    counts = Counter(len(keypoints) for _, keypoints, _ in frames)
    observed_max = max(counts) if counts else 0
    if requested == "auto":
        if observed_max == 0:
            raise ValueError("Every image is empty; cannot infer the number of individuals.")
        print(f"Inferred {observed_max} individuals (instances per image: {dict(sorted(counts.items()))}).")
        return observed_max

    n_individuals = int(requested)
    if n_individuals < observed_max:
        dropped = sum(count for n, count in counts.items() if n > n_individuals)
        print(f"WARNING: {dropped} image(s) hold more than {n_individuals} annotations; the extra instances will be dropped.")
    return n_individuals


def create_project(name: str, scorer: str, individuals: List[str], videos: List[str], projects_dir: str) -> str:
    """Create the DLC project skeleton, preferring DLC's own project factory.

    ``create_new_project`` deletes the project and returns ``"nothingcreated"`` when it
    cannot open a single video, so a dataset conversion without ``--videos`` falls back
    to writing the project from DLC's own config template.
    """
    try:
        import deeplabcut
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit("deeplabcut is not importable. Create and activate the DLC environment first (see dlc/README.md), then re-run this script.") from exc

    os.makedirs(projects_dir, exist_ok=True)
    if videos:
        config_path = deeplabcut.create_new_project(
            name,
            scorer,
            [osp.abspath(osp.expanduser(video)) for video in videos],
            working_directory=projects_dir,
            copy_videos=False,
            multianimal=True,
            individuals=individuals,
        )
        if config_path != "nothingcreated":
            return config_path
        print("WARNING: DeepLabCut could not read any of the provided videos; creating the project without a video_sets entry.")
    return create_project_without_videos(name, scorer, individuals, projects_dir)


def create_project_without_videos(name: str, scorer: str, individuals: List[str], projects_dir: str) -> str:
    """Mirror of ``deeplabcut.create_new_project`` for projects that register no video.

    DLC's ``create_config_template`` is reused so the generated ``config.yaml`` stays in
    sync with the installed DeepLabCut version.
    """
    from datetime import datetime

    from deeplabcut.core.engine import Engine
    from deeplabcut.utils import auxiliaryfunctions

    today = datetime.today()
    project_dir = osp.join(projects_dir, f"{name}-{scorer}-{today.strftime('%Y-%m-%d')}")
    config_path = osp.join(project_dir, "config.yaml")
    if osp.isfile(config_path):
        print(f'Project "{project_dir}" already exists!')
        return config_path

    for subdir in ("videos", "labeled-data", "training-datasets", "dlc-models"):
        os.makedirs(osp.join(project_dir, subdir), exist_ok=True)

    config, _ = auxiliaryfunctions.create_config_template(True)
    config["multianimalproject"] = True
    config["identity"] = False
    config["individuals"] = list(individuals)
    config["uniquebodyparts"] = []
    config["bodyparts"] = "MULTI!"
    if config.get("engine") in Engine.PYTORCH.aliases:
        config["default_augmenter"] = "albumentations"
        config["default_net_type"] = "resnet_50"
    else:
        config["default_augmenter"] = "multi-animal-imgaug"
        config["default_net_type"] = "dlcrnet_ms5"
    config["default_track_method"] = "ellipse"

    config["Task"] = name
    config["scorer"] = scorer
    config["video_sets"] = {}
    config["project_path"] = project_dir
    config["date"] = f"{today.strftime('%b')}{today.day}"
    config["cropping"] = False
    config["start"] = 0
    config["stop"] = 1
    config["numframes2pick"] = 20
    config["TrainingFraction"] = [0.95]
    config["iteration"] = 0
    config["snapshotindex"] = -1
    config["detector_snapshotindex"] = -1
    config["x1"], config["x2"], config["y1"], config["y2"] = 0, 640, 277, 624
    config["batch_size"] = 8
    config["detector_batch_size"] = 1
    config["corner2move2"] = (50, 50)
    config["move2corner"] = True
    config["skeleton_color"] = "black"
    config["pcutoff"] = 0.6
    config["dotsize"] = 12
    config["alphavalue"] = 0.7
    config["colormap"] = "rainbow"

    auxiliaryfunctions.write_config(config_path, config)
    print(f'Generated "{config_path}"')
    return config_path


def patch_project_config(config_path: str, individuals: List[str], bodyparts: List[str], skeleton_edges, identity: bool) -> None:
    from deeplabcut.utils import auxiliaryfunctions

    config = auxiliaryfunctions.read_config(config_path)
    config.update(multianimal_config_updates(individuals, bodyparts, skeleton_edges, identity=identity))
    auxiliaryfunctions.write_config(config_path, config)


def stage_images(frames, dest_dir: str, symlink: bool) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    missing = []
    for filename, _, source in frames:
        if not osp.isfile(source):
            missing.append(source)
            continue
        destination = osp.join(dest_dir, filename)
        if osp.exists(destination) or osp.islink(destination):
            continue
        if symlink:
            os.symlink(osp.abspath(source), destination)
        else:
            shutil.copy2(source, destination)
    if missing:
        raise FileNotFoundError(f"{len(missing)} source image(s) could not be found, e.g. {missing[:3]}")
    print(f"Staged {len(frames)} images into {dest_dir}")


def main(args):
    dataset_root = osp.abspath(osp.expanduser(args.dataset_root))
    frames, split_map, skeleton = load_splits(dataset_root, args.splits)
    bodyparts, skeleton_edges = _skeleton_info(skeleton)
    n_individuals = resolve_individuals(frames, args.individuals)
    individuals = individual_names(n_individuals)

    print(f"Creating DLC project '{args.name}' with {len(individuals)} individuals and {len(bodyparts)} bodyparts: {bodyparts}")
    config_path = create_project(args.name, args.scorer, individuals, args.videos, osp.abspath(args.projects_dir))
    patch_project_config(config_path, individuals, bodyparts, skeleton_edges, args.identity)

    project_dir = osp.dirname(config_path)
    labeled_dir = osp.join(project_dir, "labeled-data", args.name)
    stage_images(frames, labeled_dir, args.symlink)

    collected = build_collected_data(
        [(filename, keypoints) for filename, keypoints, _ in frames],
        individuals=individuals,
        bodyparts=bodyparts,
        scorer=args.scorer,
        folder_name=args.name,
    )
    h5_path, csv_path = write_collected_data(collected, labeled_dir, args.scorer)
    print(f"Wrote {collected.shape[0]} annotated frames x {collected.shape[1]} columns to:\n  {h5_path}\n  {csv_path}")

    split_path = osp.join(project_dir, "training-datasets", "coco_split.json")
    os.makedirs(osp.dirname(split_path), exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({"folder": args.name, "splits": split_map}, f, indent=2)
    print(f"Recorded the original COCO split in {split_path}")
    print(f"\nProject ready: {config_path}\nNext: python train.py --project {config_path}")


if __name__ == "__main__":
    main(parse_args())
