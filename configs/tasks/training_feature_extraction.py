_base_ = ["../models/rtmdet-pose.py", "../wandb/keys.py"]

# Settings
fe_batch_size = _base_.fe_batch_size
fe_base_lr = _base_.fe_base_lr
fe_weight_decay = _base_.fe_weight_decay
fe_num_epochs = _base_.fe_num_epochs
fe_val_interval = _base_.fe_val_interval

work_dir = _base_.training_work_dir

resume = False
load_from = _base_.fe_training_checkpoint


input_size = _base_.input_size
pad_value = _base_.pad_value

metainfo = _base_.metainfo
data_mode = _base_.data_mode
data_root = _base_.data_root
training_anns_path = _base_.training_anns_path
training_imgs_path = _base_.training_imgs_path
validation_anns_path = _base_.validation_anns_path
validation_imgs_path = _base_.validation_imgs_path
# /Settings


# /Settings

# Model
model = dict(
    head=dict(
        loss_cls=dict(loss_weight=0.0),
        loss_bbox=dict(loss_weight=0.0),
        loss_obj=dict(loss_weight=0.0),
        loss_oks=dict(loss_weight=0.0),
        loss_vis=dict(loss_weight=0.0),
        loss_bbox_aux=dict(loss_weight=0.0),
    ),
)
# /Model

# Loops
train_cfg = dict(
    _scope_=_base_.default_scope,
    type="EpochBasedTrainLoop",
    max_epochs=fe_num_epochs,
    val_interval=fe_val_interval,
)
val_cfg = dict(type="ValLoop")
# /Loops

# Optimization
optim_wrapper = dict(
    loss_scale="dynamic",
    type="AmpOptimWrapper",
    dtype="float16",
    optimizer=dict(type="AdamW", lr=fe_base_lr, weight_decay=fe_weight_decay),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
# /Optimization

# Dataloaders
codec = dict(type="YOLOXPoseAnnotationProcessor", input_size=input_size)

load_img = [dict(type="LoadImage")]
load_anns = [
    dict(type="FilterAnnotations", by_kpt=True, by_box=True, keep_empty=False),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
transforms_stage2 = [
    dict(
        type="BottomupRandomAffine",
        input_size=input_size,
        shift_prob=0,
        rotate_prob=0,
        scale_prob=0,
        scale_type="long",
        pad_val=(pad_value, pad_value, pad_value),
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomContrastAug"),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
]

train_pipeline_stage2 = load_img + transforms_stage2 + load_anns

param_scheduler = [
    dict(
        type="QuadraticWarmupLR",
        by_epoch=True,
        begin=0,
        end=_base_.warmup_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=fe_base_lr / 20,
        begin=_base_.warmup_epochs,
        T_max=int(0.8 * fe_num_epochs),
        end=int(0.8 * fe_num_epochs),
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(type="ConstantLR", by_epoch=True, factor=1, begin=int(0.8 * fe_num_epochs), end=fe_num_epochs),
]


train_dataloader = dict(
    batch_size=fe_batch_size,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="COCODataset",
        from_file=metainfo,
        data_mode=data_mode,
        filter_cfg=dict(filter_empty_gt=False, min_size=fe_batch_size),
        ann_file=training_anns_path,
        data_prefix=dict(img=training_imgs_path),
        pipeline=train_pipeline_stage2,
    ),
)
val_pipeline = load_img + [dict(type="Resize", input_size=input_size, pad_val=(pad_value, pad_value, pad_value))] + load_anns
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type="COCODataset",
        from_file=metainfo,
        ann_file=validation_anns_path,
        data_mode=data_mode,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        data_prefix=dict(img=validation_imgs_path),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
# /Dataloaders

# Evaluation
val_evaluator = [dict(type="SilhouetteScore")]
# /Evaluation


# Hooks
default_hooks = dict(
    checkpoint=dict(interval=-1, type="CheckpointHook", save_best="SilhouetteScore/avg", rule="greater"),
)
custom_hooks = [
    dict(type="SyncNormHook", priority=48),
    dict(type="ModuleFreezingHook", modules_to_freeze=["backbone", "neck", "head"], priority=30),
    dict(type="ValidateBeforeTrainingHook", priority=100),
]
# /Hooks

# Visualization
if _base_.wandb_logging:
    visualizer = dict(
        type="Visualizer",
        name="wandb_visualizer",
        vis_backends=dict(
            type="WandbVisBackend",
            init_kwargs=dict(
                project=_base_.project,
                entity=_base_.entity,
            ),
            save_dir=work_dir + "wandb/",
        ),
    )
else:
    visualizer = dict(
        _delete_=True,
    )
# /Visualization
