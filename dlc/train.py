"""Train a multi-animal DeepLabCut model on a project built by ``coco2dlc.py``.

Runs the three standard DLC steps in order: ``create_training_dataset`` (honouring the
original COCO train/val split recorded by ``coco2dlc.py``), ``train_network`` and
``evaluate_network``.

Example:
    python train.py --project projects/mice-coco-2026-09-08/config.yaml \\
        --net-type resnet_50 --epochs 200
"""

import argparse
import json
import os.path as osp
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dlc_io import read_project_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True, help="Path to the DLC project's config.yaml")
    parser.add_argument("--net-type", default="resnet_50", help="Backbone passed to create_training_dataset. Defaults to resnet_50.")
    parser.add_argument("--shuffle", type=int, default=1, help="Shuffle index. Defaults to 1.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs (PyTorch engine). Defaults to 200.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size. Defaults to 8.")
    parser.add_argument("--save-epochs", type=int, default=25, help="Snapshot interval, in epochs. Defaults to 25.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu. Defaults to DLC's auto-detection.")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the optimizer's base learning rate. DLC's maDLC default (5e-4) can diverge on larger datasets.",
    )
    parser.add_argument(
        "--lr-milestones",
        type=int,
        nargs="*",
        default=None,
        metavar="EPOCH",
        help="Epochs at which the learning rate steps down, e.g. --lr-milestones 60 120.",
    )
    parser.add_argument(
        "--lr-decay",
        type=float,
        nargs="*",
        default=None,
        metavar="LR",
        help="Learning rate to use after each milestone, e.g. --lr-decay 2e-5 2e-6.",
    )
    parser.add_argument("--random-split", action="store_true", help="Ignore the recorded COCO split and let DLC sample its own train/test split.")
    parser.add_argument("--skip-training-dataset", action="store_true", help="Reuse the existing training dataset for this shuffle.")
    parser.add_argument("--skip-evaluation", action="store_true", help="Do not run evaluate_network after training.")
    return parser.parse_args()


def coco_split_indices(config_path: str, scorer: str) -> Optional[Tuple[List[int], List[int]]]:
    """Map the COCO split recorded by ``coco2dlc.py`` onto ``CollectedData`` row indices.

    Returns ``None`` when no split file exists, in which case DLC picks its own split.
    """
    project_dir = osp.dirname(osp.abspath(config_path))
    split_path = osp.join(project_dir, "training-datasets", "coco_split.json")
    if not osp.isfile(split_path):
        print(f"No {split_path}; falling back to DLC's own random train/test split.")
        return None

    with open(split_path, "r") as f:
        recorded = json.load(f)

    folder = recorded["folder"]
    collected = pd.read_hdf(osp.join(project_dir, "labeled-data", folder, f"CollectedData_{scorer}.h5"))
    row_of = {index[-1]: position for position, index in enumerate(collected.index)}

    train = [row_of[name] for name in recorded["splits"].get("train", []) if name in row_of]
    test = [row_of[name] for name in recorded["splits"].get("val", []) if name in row_of]
    if not train or not test:
        print("The recorded COCO split does not cover both train and val; falling back to DLC's own split.")
        return None

    print(f"Reusing the COCO split: {len(train)} training and {len(test)} test frames.")
    return sorted(train), sorted(test)


def record_training_fraction(config_path: str, n_train: int, n_test: int) -> float:
    """Put the split's own training fraction first in ``config.yaml``'s TrainingFraction.

    ``create_training_dataset`` derives the fraction from the indices it is handed and
    names the shuffle after it, but never writes it back to the config. ``train_network``
    and ``evaluate_network`` then look the shuffle up through ``TrainingFraction[0]``,
    which would still hold the project template's default and match nothing.
    """
    from deeplabcut.utils import auxiliaryfunctions

    fraction = round(n_train / (n_train + n_test), 2)
    config = auxiliaryfunctions.read_config(config_path)
    fractions = list(config.get("TrainingFraction") or [])
    if not fractions or fractions[0] != fraction:
        config["TrainingFraction"] = [fraction] + [f for f in fractions if f != fraction]
        auxiliaryfunctions.write_config(config_path, config)
        print(f"Set TrainingFraction to {config['TrainingFraction']} to match the recorded split.")
    return fraction


def learning_rate_updates(args) -> Dict:
    """Dot-notation overrides for ``train_network``'s ``pytorch_cfg_updates``.

    DLC's multi-animal PyTorch default holds AdamW at 5e-4 until epoch 90, which diverges
    on larger datasets: the heatmaps degenerate, peak detection returns tens of thousands
    of candidates, and the PAF predictor's pairing tensor then exhausts the GPU.
    """
    updates = {}
    if args.lr is not None:
        updates["runner.optimizer.params.lr"] = args.lr
    if args.lr_milestones is not None:
        updates["runner.scheduler.params.milestones"] = list(args.lr_milestones)
    if args.lr_decay is not None:
        updates["runner.scheduler.params.lr_list"] = [[lr] for lr in args.lr_decay]
    if args.lr_milestones is not None and args.lr_decay is not None and len(args.lr_milestones) != len(args.lr_decay):
        raise SystemExit(f"--lr-milestones ({len(args.lr_milestones)} values) and --lr-decay ({len(args.lr_decay)} values) must have the same length.")
    return updates


def build_training_dataset(deeplabcut, config_path: str, args, split) -> None:
    """Call the training-dataset builder that the installed DLC version exposes.

    DLC 3.x routes multi-animal projects through ``create_training_dataset``; older
    releases require the dedicated ``create_multianimaltraining_dataset``.

    ``Shuffles=[args.shuffle]`` is passed explicitly. Without it ``num_shuffles=1`` builds
    shuffle 1 every time, so training a second architecture in an existing project
    silently overwrites the first model's training dataset -- and ``--shuffle`` would then
    point ``train_network`` at data that belongs to a different net type.
    """
    kwargs = dict(num_shuffles=1, Shuffles=[args.shuffle], net_type=args.net_type)
    if split is not None:
        train_indices, test_indices = split
        kwargs.update(trainIndices=[train_indices], testIndices=[test_indices])
        record_training_fraction(config_path, len(train_indices), len(test_indices))

    try:
        from deeplabcut.core.engine import Engine

        kwargs["engine"] = Engine.PYTORCH
    except ImportError:
        print("This DeepLabCut version has no PyTorch engine; using its default backend.")

    if hasattr(deeplabcut, "create_training_dataset"):
        deeplabcut.create_training_dataset(config_path, **kwargs)
    else:  # pragma: no cover - legacy DLC
        kwargs.pop("engine", None)
        deeplabcut.create_multianimaltraining_dataset(config_path, **kwargs)


def main(args):
    import deeplabcut

    config_path = osp.abspath(args.project)
    config = read_project_config(config_path)
    print(
        f"Project '{config.get('Task')}' by '{config.get('scorer')}': "
        f"{len(config.get('individuals', []))} individuals, "
        f"{len(config.get('multianimalbodyparts', []))} bodyparts."
    )

    if not args.skip_training_dataset:
        split = None if args.random_split else coco_split_indices(config_path, config["scorer"])
        build_training_dataset(deeplabcut, config_path, args, split)

    train_kwargs = dict(shuffle=args.shuffle, epochs=args.epochs, batch_size=args.batch_size, save_epochs=args.save_epochs)
    if args.device:
        train_kwargs["device"] = args.device
    schedule = learning_rate_updates(args)
    if schedule:
        train_kwargs["pytorch_cfg_updates"] = schedule
        print(f"Overriding the PyTorch training config with {schedule}")
    deeplabcut.train_network(config_path, **train_kwargs)

    if not args.skip_evaluation:
        deeplabcut.evaluate_network(config_path, Shuffles=[args.shuffle], plotting=False)

    print(f"\nDone. Track a video with:\n  python track.py <video> --project {config_path} --shuffle {args.shuffle}")


if __name__ == "__main__":
    main(parse_args())
