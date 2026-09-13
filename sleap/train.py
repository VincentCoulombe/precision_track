"""Train a SLEAP pose-estimation model on a dataset built by ``coco2sleap.py``.

Wraps ``sleap_nn.train`` (SLEAP 1.6's PyTorch backend) and handles the one thing that
API leaves to the caller: **top-down setups are two models**, a centroid detector and a
pose model that runs on its crops. Those are two independent training runs whose order
matters at inference time, so this script trains them in sequence and records the order
in ``model_group.json`` -- letting ``track.py`` accept a single ``--model`` argument
whether the architecture needs one checkpoint or two.

Every architecture SLEAP offers is selectable through ``--head-config``:

    bottomup              confmaps + part-affinity fields, grouped after inference.
                          The direct analogue of the maDLC baseline.
    top-down              centroid  ->  centered_instance   (two models)
    single_instance       one animal per frame, no grouping
    multi_class_bottomup  bottomup plus a head that classifies identity
    multi_class_topdown   centroid  ->  multi_class_topdown  (two models)

The two ``multi_class_*`` heads learn identity from appearance, so they need every
instance in a frame labelled with a consistent identity. That is checked before
training starts rather than discovered from a useless model afterwards -- see
``check_identity_support``.

Example:
    python train.py --dataset-dir datasets/july_2026_640x640 \\
        --head-config bottomup --max-epochs 200 --batch-size 4
"""

import argparse
import os
import os.path as osp
from typing import Dict, List, Optional, Sequence, Tuple

from slp_io import MODEL_GROUP_FILE, identity_coverage, load_labels, write_model_group

DEFAULT_MODELS_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "models")

# Which sleap_nn head config(s) each architecture needs, in the order that
# ``predict()`` expects them in ``model_paths``.
ARCHITECTURES: Dict[str, Tuple[str, ...]] = {
    "bottomup": ("bottomup",),
    "single_instance": ("single_instance",),
    "top-down": ("centroid", "centered_instance"),
    "multi_class_bottomup": ("multi_class_bottomup",),
    "multi_class_topdown": ("centroid", "multi_class_topdown"),
}

# Heads that classify identity, and therefore need identity-labelled frames.
IDENTITY_HEADS = ("multi_class_bottomup", "multi_class_topdown")

# Heads that crop around a centroid and so honour --crop-size.
CROPPING_HEADS = ("centered_instance", "multi_class_topdown")

# Every head emits ``confmaps``; some emit a second *spatial* map under a different key.
# ``multi_class_topdown``'s second part is ``class_vectors``, a globally-pooled vector head
# whose stride means something else, so it is deliberately absent here.
SECONDARY_MAP_PART = {"bottomup": "pafs", "multi_class_bottomup": "class_maps"}

MIN_IDENTITY_FRAMES = 200

# 1771 training frames / batch 8 = 221 steps, which clears sleap_nn's
# ``min_train_steps_per_epoch`` floor of 200. Above batch 8 the step count falls below the
# floor and sleap_nn pads the epoch by oversampling, so an "epoch" stops being one pass
# over the data (model_trainer.py:1816-1822).
MAX_HONEST_BATCH_SIZE = 8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default=None, help="Directory produced by coco2sleap.py, holding train.slp and val.slp.")
    parser.add_argument("--train-slp", default=None, help="Training .slp. Overrides --dataset-dir.")
    parser.add_argument("--val-slp", default=None, help="Validation .slp. Overrides --dataset-dir.")
    parser.add_argument("--head-config", choices=sorted(ARCHITECTURES), default="bottomup", help="Architecture to train. Defaults to bottomup.")
    parser.add_argument(
        "--backbone",
        default="unet_medium_rf",
        help="Backbone passed to sleap_nn.train, e.g. unet, unet_medium_rf, unet_large_rf, convnext, swint. "
        "Defaults to unet_medium_rf, which is what sleap_nn's own ConfigGenerator recommends for this dataset.",
    )
    parser.add_argument("--max-epochs", type=int, default=200, help="Training epochs. Defaults to 200.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=f"Training batch size. Defaults to 8; above {MAX_HONEST_BATCH_SIZE} an epoch may stop being one full pass (see --help notes).",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate. Defaults to sleap_nn's 1e-3.")
    parser.add_argument("--optimizer", default="Adam", help="Optimizer name. Defaults to Adam.")
    parser.add_argument("--scale", type=float, default=1.0, help="Resize factor applied to the images. Defaults to 1.0.")
    parser.add_argument("--crop-size", type=int, default=None, help="Instance crop size for the cropping heads. Defaults to sleap_nn's estimate.")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers. Defaults to 8.")

    strides = parser.add_argument_group("output strides (the dominant cost of training)")
    strides.add_argument(
        "--confmap-stride",
        type=int,
        default=2,
        help="Confidence-map output stride. sleap_nn's string shorthand implies 1, i.e. full-resolution maps "
        "generated on CPU for every sample, which makes training ~30x slower than it needs to be. Defaults to 2.",
    )
    strides.add_argument(
        "--paf-stride",
        type=int,
        default=4,
        help="Part-affinity-field (or class-map) output stride, for the heads that emit one. Defaults to 4.",
    )
    strides.add_argument(
        "--backbone-output-stride",
        type=int,
        default=None,
        help="Backbone output stride. Defaults to --confmap-stride, since the finest head cannot be finer than the features it reads.",
    )
    strides.add_argument("--sigma", type=float, default=None, help="Confidence-map sigma. Defaults to sleap_nn's value (5.0).")
    strides.add_argument(
        "--paf-sigma",
        type=float,
        default=None,
        help="Part-affinity-field sigma, i.e. how wide the limb lines are. Defaults to sleap_nn's value (15.0). "
        "On small animals the default leaves PAF targets so sparse (~0.2%% of cells) that the head cannot learn.",
    )
    strides.add_argument("--confmap-loss-weight", type=float, default=None, help="Weight on the confidence-map loss. Defaults to 1.0.")
    strides.add_argument(
        "--paf-loss-weight",
        type=float,
        default=None,
        help="Weight on the PAF (or class-map) loss. Defaults to 1.0, which on sparse targets lets the head settle at "
        "predicting zeros everywhere -- a near-optimal solution that makes bottom-up grouping impossible.",
    )

    pipeline = parser.add_argument_group("data pipeline")
    pipeline.add_argument(
        "--data-pipeline-fw",
        choices=["torch_dataset", "torch_dataset_cache_img_memory", "torch_dataset_cache_img_disk"],
        default="torch_dataset_cache_img_memory",
        help="Image caching strategy. Defaults to caching decoded images in RAM (~2.2 GB for this dataset).",
    )
    pipeline.add_argument("--cache-img-path", default=None, help="Where torch_dataset_cache_img_disk writes its cache. Defaults to the run directory.")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR, help=f"Where run directories are created. Defaults to {DEFAULT_MODELS_DIR}.")
    parser.add_argument("--run-name", default=None, help="Run directory name. Defaults to <head-config>.")
    parser.add_argument("--device", default=None, help="Accelerator: cuda, cpu, mps, or auto. Defaults to sleap_nn's auto-detection.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Defaults to 42.")
    parser.add_argument("--identity-slp-suffix", default="_id", help="Suffix of the identity-only .slp files coco2sleap.py --require-identity writes.")
    parser.add_argument(
        "--allow-thin-identity",
        action="store_true",
        help="Train a multi_class_* head even though too few frames carry a full set of identities. The model will be near-useless; "
        "this exists so the refusal can be overridden deliberately.",
    )
    parser.add_argument("--no-augmentation", action="store_true", help="Disable training-time augmentation.")
    parser.add_argument("--skip-evaluation", action="store_true", help="Do not evaluate on the validation split after training.")
    return parser.parse_args()


def resolve_labels(args) -> Tuple[str, str]:
    """Locate the train/val ``.slp`` pair, preferring the identity-only files when needed.

    A ``multi_class_*`` head trained on the full split would see mostly unidentified
    instances, so when ``coco2sleap.py --require-identity`` has produced
    ``train_id.slp``/``val_id.slp`` those are used instead.
    """
    if args.train_slp and args.val_slp:
        return osp.abspath(osp.expanduser(args.train_slp)), osp.abspath(osp.expanduser(args.val_slp))
    if not args.dataset_dir:
        raise SystemExit("Pass --dataset-dir, or both --train-slp and --val-slp.")

    dataset_dir = osp.abspath(osp.expanduser(args.dataset_dir))
    needs_identity = any(head in IDENTITY_HEADS for head in ARCHITECTURES[args.head_config])
    suffix = args.identity_slp_suffix if needs_identity else ""

    train = osp.join(dataset_dir, f"train{suffix}.slp")
    val = osp.join(dataset_dir, f"val{suffix}.slp")
    if needs_identity and not (osp.isfile(train) and osp.isfile(val)):
        print(f"No train{suffix}.slp/val{suffix}.slp in {dataset_dir}; falling back to the full split. Re-run coco2sleap.py --require-identity to build them.")
        train, val = osp.join(dataset_dir, "train.slp"), osp.join(dataset_dir, "val.slp")

    for path in (train, val):
        if not osp.isfile(path):
            raise SystemExit(f"No {path}. Run coco2sleap.py first.")
    return train, val


def check_identity_support(head_configs: Sequence[str], train_slp: str, val_slp: str, allow_thin: bool) -> None:
    """Refuse a multi-class run whose identity labels cannot support it.

    SLEAP's identity heads classify each instance into one of the project's tracks, so a
    frame containing an unlabelled animal actively teaches the head that the animal
    belongs to no class. Training regardless produces a model that looks trained and
    tracks worse than no identity head at all, which is why this blocks rather than
    warns.
    """
    if not any(head in IDENTITY_HEADS for head in head_configs):
        return

    report = {}
    for split, path in (("train", train_slp), ("val", val_slp)):
        coverage = identity_coverage(load_labels(path))
        report[split] = coverage
        print(
            f"  {split}: {coverage['n_frames_all_tracked']}/{coverage['n_frames']} frame(s) fully identified, "
            f"{coverage['n_tracked_instances']}/{coverage['n_instances']} instance(s), {coverage['n_tracks']} track(s)"
        )

    usable = min(coverage["n_frames_all_tracked"] for coverage in report.values())
    if usable == 0:
        raise SystemExit(
            "No frame has every instance identified, so a multi_class_* head has nothing to learn from. "
            "Re-run coco2sleap.py --require-identity, or annotate identities, then retry."
        )
    if usable < MIN_IDENTITY_FRAMES and not allow_thin:
        raise SystemExit(
            f"Only {usable} frame(s) have a complete set of identities, well under the {MIN_IDENTITY_FRAMES} this refuses below.\n"
            "Two problems, not one:\n"
            f"  1. {usable} frames is far too little to learn an identity classifier.\n"
            "  2. COCO identities are numbered per source clip, so 'object_id 1' in one clip and in another are\n"
            "     different animals. A single global identity head over several clips is therefore ill-posed.\n"
            "Unblock it by annotating identities across the dataset, or by converting one clip on its own so the\n"
            "identity space is consistent. To train anyway (the model will be near-useless), pass --allow-thin-identity."
        )
    if usable < MIN_IDENTITY_FRAMES:
        print(f"--allow-thin-identity: training on {usable} fully-identified frame(s) anyway. Do not read anything into the result.")


def build_head_configs(head_config: str, args) -> Dict:
    """Build the ``head_configs`` **dict** so the output strides can be set.

    Passing ``head_configs`` as a bare string makes sleap_nn use ``output_stride: 1``,
    which means ``generate_confmaps``/``generate_pafs`` build full-resolution target
    tensors -- 12x640x640 plus 22x640x640 floats -- on the CPU for every single sample.
    Measured, that pins the dataloader at 100% CPU with the GPU at 1% and costs ~21
    minutes an epoch. Coarser strides are the single biggest lever on training time.
    """
    confmaps: Dict = {"output_stride": args.confmap_stride}
    if args.sigma is not None:
        confmaps["sigma"] = args.sigma
    if args.confmap_loss_weight is not None:
        confmaps["loss_weight"] = args.confmap_loss_weight
    parts: Dict = {"confmaps": confmaps}

    secondary = SECONDARY_MAP_PART.get(head_config)
    if secondary:
        secondary_cfg: Dict = {"output_stride": args.paf_stride}
        if args.paf_sigma is not None:
            secondary_cfg["sigma"] = args.paf_sigma
        if args.paf_loss_weight is not None:
            secondary_cfg["loss_weight"] = args.paf_loss_weight
        parts[secondary] = secondary_cfg
    return {head_config: parts}


def build_backbone_config(backbone: str, output_stride: int) -> Dict:
    """Reproduce a named backbone variant as a dict, overriding only its output stride.

    sleap_nn resolves a backbone dict with ``UNetConfig(**d)``, which *replaces* rather
    than merges -- so a bare ``{"unet": {"output_stride": 2}}`` would silently drop
    ``unet_medium_rf``'s ``filters_rate=2`` and train a different architecture than asked
    for. Rebuilding from sleap_nn's own resolved variant keeps every other parameter and
    avoids duplicating its variant table here.

    ``in_channels`` is left as-is: ``model_trainer`` overwrites it from the real image
    channel count (model_trainer.py:898).
    """
    import attrs
    from sleap_nn.config.get_config import get_backbone_config

    resolved = get_backbone_config(backbone)
    for family in ("unet", "convnext", "swint", "pretrained"):
        sub = getattr(resolved, family, None)
        if sub is None:
            continue
        params = attrs.asdict(sub)
        if "output_stride" in params:
            params["output_stride"] = output_stride
        return {family: params}
    raise SystemExit(f"Could not resolve the backbone '{backbone}' into a family config.")


def check_batch_size(batch_size: int, n_train_frames: int) -> None:
    """Warn when the batch size makes an "epoch" stop meaning one pass over the data."""
    steps = n_train_frames // max(batch_size, 1)
    if steps < 200:
        print(
            f"Warning: batch size {batch_size} gives {steps} steps/epoch over {n_train_frames} frames, below "
            f"sleap_nn's min_train_steps_per_epoch of 200. It will pad the epoch by oversampling, so --max-epochs "
            f"no longer counts full passes. Use --batch-size {MAX_HONEST_BATCH_SIZE} or lower for a like-for-like "
            "epoch count."
        )


def train_one(head_config: str, run_name: str, train_slp: str, val_slp: str, args) -> str:
    """Run one ``sleap_nn.train`` and return the directory holding its checkpoint."""
    # ``train`` is not re-exported on the ``sleap_nn`` package (only ``predict``,
    # ``Predictor``, ``load_models`` and ``load_metrics`` are), so import the module.
    from sleap_nn.train import train as sleap_nn_train

    models_dir = osp.abspath(osp.expanduser(args.models_dir))
    os.makedirs(models_dir, exist_ok=True)

    backbone_stride = args.backbone_output_stride or args.confmap_stride
    kwargs = dict(
        train_labels_path=[train_slp],
        val_labels_path=[val_slp],
        head_configs=build_head_configs(head_config, args),
        backbone_config=build_backbone_config(args.backbone, backbone_stride),
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        scale=args.scale,
        seed=args.seed,
        data_pipeline_fw=args.data_pipeline_fw,
        # sleap_nn defaults save_ckpt to False, which trains a model and then throws it
        # away. Without this there is nothing for track.py to load.
        save_ckpt=True,
        ckpt_dir=models_dir,
        run_name=run_name,
        use_augmentations_train=not args.no_augmentation,
    )
    if args.cache_img_path:
        kwargs["cache_img_path"] = osp.abspath(osp.expanduser(args.cache_img_path))
    if args.crop_size is not None and head_config in CROPPING_HEADS:
        kwargs["crop_size"] = args.crop_size
    if args.device:
        kwargs["trainer_accelerator"] = args.device
    if not args.skip_evaluation:
        kwargs["test_file_path"] = val_slp

    print(f"\n=== Training '{head_config}' -> {osp.join(models_dir, run_name)} ===")
    sleap_nn_train(**kwargs)

    model_dir = osp.join(models_dir, run_name)
    if not osp.isdir(model_dir):
        raise SystemExit(f"Training finished but {model_dir} does not exist; sleap_nn may have chosen another run name.")
    return model_dir


def report_artifacts(model_dirs: Sequence[str]) -> List[str]:
    """Warn about any run directory missing the files inference needs."""
    incomplete = []
    for model_dir in model_dirs:
        missing = [name for name in ("best.ckpt", "training_config.yaml") if not osp.isfile(osp.join(model_dir, name))]
        if missing:
            incomplete.append(f"{model_dir} (missing {', '.join(missing)})")
    return incomplete


def main(args):
    head_configs = ARCHITECTURES[args.head_config]
    train_slp, val_slp = resolve_labels(args)
    base_run_name = args.run_name or args.head_config

    print(f"Architecture '{args.head_config}': {len(head_configs)} model(s) -> {list(head_configs)}")
    print(f"  train: {train_slp}\n  val:   {val_slp}")
    print(
        f"  backbone {args.backbone} @ stride {args.backbone_output_stride or args.confmap_stride}, "
        f"confmaps @ stride {args.confmap_stride}, secondary maps @ stride {args.paf_stride}"
    )
    print(f"  batch {args.batch_size}, {args.num_workers} workers, {args.data_pipeline_fw}")
    check_batch_size(args.batch_size, len(load_labels(train_slp).labeled_frames))
    check_identity_support(head_configs, train_slp, val_slp, args.allow_thin_identity)

    model_dirs = []
    for head_config in head_configs:
        # One model keeps the run name as-is; a two-model setup suffixes each stage so
        # both live under the same recognizable prefix.
        run_name = base_run_name if len(head_configs) == 1 else f"{base_run_name}_{head_config}"
        model_dirs.append(train_one(head_config, run_name, train_slp, val_slp, args))

    incomplete = report_artifacts(model_dirs)
    if incomplete:
        print("\nWarning: these run directories lack the files inference needs:")
        for entry in incomplete:
            print(f"  {entry}")

    if len(model_dirs) == 1:
        model_spec: Optional[str] = model_dirs[0]
    else:
        group_dir = osp.join(osp.abspath(osp.expanduser(args.models_dir)), base_run_name)
        os.makedirs(group_dir, exist_ok=True)
        model_spec = write_model_group(model_dirs, osp.join(group_dir, MODEL_GROUP_FILE))
        print(f"\nWrote {model_spec} recording the inference order: {[osp.basename(d) for d in model_dirs]}")

    print(f"\nDone. Track a video with:\n  python track.py <video>.mp4 --model {model_spec} --max-instances 5")


if __name__ == "__main__":
    main(parse_args())
