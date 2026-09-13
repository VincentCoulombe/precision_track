"""Convert a PrecisionTrack COCO pose dataset into SLEAP ``.slp`` label files.

``sleap-io`` reads COCO and writes ``.slp``, so this is a thin, honest conversion:
one ``.slp`` per COCO split, keeping PrecisionTrack's own train/val boundary intact.

That is the whole reason this file is short where ``dlc/coco2dlc.py`` is long. DLC has
no writer in ``sleap-io``, so its converter had to build a ``CollectedData`` HDF5 by
hand and record the split in a side-car JSON for ``train.py`` to re-impose through
DLC's index-based API. Here the split *is* the two files.

Example:
    python coco2sleap.py ~/Documents/datasets/stripedmice/july_2026_640x640 \\
        --name july_2026_640x640
"""

import argparse
import json
import os
import os.path as osp
from typing import Dict, List, Optional

import numpy as np
from slp_io import fully_identified_frames, identity_coverage

DEFAULT_DATASETS_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "datasets")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", help="COCO dataset root holding annotations/ and images/")
    parser.add_argument("--name", default=None, help="Output dataset name. Defaults to the dataset root's basename.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="COCO splits to convert. Defaults to train val.")
    parser.add_argument("--out-dir", default=None, help=f"Where the .slp files land. Defaults to {DEFAULT_DATASETS_DIR}/<name>.")
    parser.add_argument("--annotations-subdir", default="annotations", help="Subdirectory holding <split>.json. Defaults to annotations.")
    parser.add_argument("--images-subdir", default="images", help="Subdirectory holding the images. Defaults to images.")
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Bake the images into the .slp instead of referencing them on disk. Makes the file self-contained but much larger.",
    )
    parser.add_argument(
        "--require-identity",
        action="store_true",
        help="Additionally emit <split>_id.slp holding only frames where every instance carries an identity, "
        "which is what the multi_class_* heads need. See section 5 of README.md.",
    )
    parser.add_argument("--keep-empty", action="store_true", help="Keep frames that have no usable instance. They are dropped by default.")
    return parser.parse_args()


def split_json(dataset_root: str, annotations_subdir: str, split: str) -> str:
    path = osp.join(dataset_root, annotations_subdir, f"{split}.json")
    if not osp.isfile(path):
        raise SystemExit(f"No COCO annotations at {path}.")
    return path


def load_split(json_path: str, images_root: str):
    """Read one COCO split into a ``Labels``."""
    import sleap_io as sio

    return sio.load_coco(json_path, dataset_root=images_root)


def drop_unusable(labels, keep_empty: bool) -> Dict[str, int]:
    """Remove all-NaN instances and, unless asked otherwise, the frames left empty.

    Mirrors the guard in ``dlc/coco2dlc.py``: an annotation whose every keypoint is
    missing carries no signal but still counts as an instance, which inflates the
    instance-per-frame statistics the architecture choice depends on.
    """
    removed_instances, removed_frames = 0, 0
    for labeled_frame in labels.labeled_frames:
        usable = [i for i in labeled_frame.instances if not np.isnan(i.numpy()[:, :2]).all()]
        removed_instances += len(labeled_frame.instances) - len(usable)
        labeled_frame.instances = usable

    if not keep_empty:
        kept = [f for f in labels.labeled_frames if len(f.instances) > 0]
        removed_frames = len(labels.labeled_frames) - len(kept)
        labels.labeled_frames = kept

    return dict(removed_instances=removed_instances, removed_frames=removed_frames)


def identity_only(labels):
    """A ``Labels`` restricted to frames where every instance is identified."""
    import sleap_io as sio

    frames = fully_identified_frames(labels)
    if not frames:
        return None
    return sio.Labels(labeled_frames=frames, videos=labels.videos, skeletons=labels.skeletons, tracks=labels.tracks)


def save(labels, path: str, embed: bool) -> str:
    import sleap_io as sio

    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    sio.save_slp(labels, path, embed="all" if embed else False)
    return path


def skeleton_info(labels) -> Dict:
    skeleton = labels.skeletons[0]
    return dict(
        nodes=list(skeleton.node_names),
        edges=[[edge.source.name, edge.destination.name] for edge in skeleton.edges],
    )


def describe(split: str, coverage: Dict[str, int], dropped: Dict[str, int]) -> None:
    print(
        f"  {split:5s}: {coverage['n_frames']} frames, {coverage['n_instances']} instances "
        f"({coverage['n_instances'] / max(coverage['n_frames'], 1):.2f}/frame); "
        f"dropped {dropped['removed_instances']} empty instance(s) and {dropped['removed_frames']} empty frame(s)"
    )
    print(
        f"         identities: {coverage['n_tracked_instances']}/{coverage['n_instances']} instances, "
        f"{coverage['n_frames_all_tracked']} fully-identified frame(s), "
        f"{coverage['n_frames_any_tracked']} partially-identified, {coverage['n_tracks']} track(s)"
    )


def identity_warning(per_split: Dict[str, Dict]) -> Optional[str]:
    """Explain, with counts, whether the ``multi_class_*`` heads are trainable.

    Called here rather than only in ``train.py`` so the blocker surfaces at conversion
    time, before anyone commits to a training run.
    """
    trainable = {s: info["coverage"]["n_frames_all_tracked"] for s, info in per_split.items()}
    if not any(trainable.values()):
        return "No frame has every instance identified, so the multi_class_* heads cannot be trained on this dataset."
    if min(v for v in trainable.values() if v is not None) < 200:
        counts = ", ".join(f"{s}={n}" for s, n in trainable.items())
        return (
            f"Only {counts} frame(s) have every instance identified. That is very little for an identity "
            "classification head, and COCO identities are numbered per source clip, so the same id in two "
            "clips is two different animals. train.py will refuse multi_class_* unless --allow-thin-identity "
            "is passed. Section 5 of README.md explains what unblocks it."
        )
    return None


def main(args):
    dataset_root = osp.abspath(osp.expanduser(args.dataset_root))
    name = args.name or osp.basename(dataset_root.rstrip("/"))
    out_dir = osp.abspath(osp.expanduser(args.out_dir)) if args.out_dir else osp.join(DEFAULT_DATASETS_DIR, name)
    images_root = osp.join(dataset_root, args.images_subdir)
    if not osp.isdir(images_root):
        raise SystemExit(f"No images directory at {images_root}; pass --images-subdir.")

    os.makedirs(out_dir, exist_ok=True)
    print(f"Converting '{name}' from {dataset_root}\n")

    per_split: Dict[str, Dict] = {}
    skeleton: Optional[Dict] = None
    written: List[str] = []

    for split in args.splits:
        labels = load_split(split_json(dataset_root, args.annotations_subdir, split), images_root)
        dropped = drop_unusable(labels, args.keep_empty)
        coverage = identity_coverage(labels)
        describe(split, coverage, dropped)

        if skeleton is None:
            skeleton = skeleton_info(labels)
        elif skeleton != skeleton_info(labels):
            raise SystemExit(f"Split '{split}' declares a different skeleton than the previous split.")

        written.append(save(labels, osp.join(out_dir, f"{split}.slp"), args.embed))
        entry = dict(coverage=coverage, dropped=dropped, slp=f"{split}.slp")

        if args.require_identity:
            subset = identity_only(labels)
            if subset is None:
                print(f"         --require-identity: no fully-identified frame in '{split}', skipping {split}_id.slp")
            else:
                written.append(save(subset, osp.join(out_dir, f"{split}_id.slp"), args.embed))
                entry["slp_identity"] = f"{split}_id.slp"
                print(f"         --require-identity: wrote {split}_id.slp with {len(subset.labeled_frames)} frame(s)")

        per_split[split] = entry

    info = dict(name=name, dataset_root=dataset_root, skeleton=skeleton, splits=per_split)
    info_path = osp.join(out_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nSkeleton: {len(skeleton['nodes'])} nodes, {len(skeleton['edges'])} edges")
    print(f"  {skeleton['nodes']}")
    warning = identity_warning(per_split)
    if warning:
        print(f"\nNote on identities: {warning}")

    print("\nWrote:")
    for path in written + [info_path]:
        print(f"  {path}")
    print(f"\nTrain with:\n  python train.py --dataset-dir {out_dir} --head-config bottomup")


if __name__ == "__main__":
    main(parse_args())
