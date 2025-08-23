_base_ = ["../models/rtmdet-pose.py", "../wandb/keys.py"]

# Settings
batch_size = _base_.batch_size
base_lr = _base_.base_lr
weight_decay = _base_.weight_decay
num_epochs = _base_.num_epochs
num_epochs_pipeline1 = _base_.num_epochs_pipeline1

val_interval = _base_.val_interval
work_dir = _base_.training_work_dir

resume = _base_.resume
load_from = _base_.training_checkpoint


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

# Model
model = dict(
    feature_extraction_head=dict(
        loss_features=dict(loss_weight=0.0),
    ),
)
# /Model

# Loops
train_cfg = dict(
    _scope_=_base_.default_scope,
    type="EpochBasedTrainLoop",
    max_epochs=num_epochs,
    val_interval=val_interval,
    dynamic_intervals=[(num_epochs_pipeline1, 1)],
)
val_cfg = dict(
    type="ValidationLossLoop",
    val_cfg=_base_.model.test_cfg,
)
# /Loops

# Optimization
optim_wrapper = dict(
    loss_scale="dynamic",
    type="AmpOptimWrapper",
    dtype="float16",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=weight_decay),
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
    dict(type="FilterAnnotations", by_kpt=True, by_box=True, keep_empty=False, min_kpt_vis=3),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
transforms_stage1 = [
    dict(
        type="Mosaic",
        img_scale=input_size,
        pad_val=pad_value,
        pre_transform=[dict(type="LoadImage", backend_args=None)],
    ),
    dict(
        type="BottomupRandomAffine",
        input_size=input_size,
        shift_factor=0.1,
        rotate_factor=10,
        scale_factor=(0.75, 1.0),
        pad_val=pad_value,
        distribution="uniform",
        transform_mode="perspective",
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(
        type="YOLOXMixUp",
        img_scale=input_size,
        ratio_range=(0.8, 1.6),
        pad_val=pad_value,
        pre_transform=[dict(type="LoadImage", backend_args=None)],
    ),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomContrastAug"),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
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

train_pipeline_stage1 = load_img + transforms_stage1 + load_anns
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
        eta_min=base_lr / 20,
        begin=_base_.warmup_epochs,
        T_max=num_epochs_pipeline1,
        end=num_epochs_pipeline1,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(type="ConstantLR", by_epoch=True, factor=1, begin=num_epochs_pipeline1, end=num_epochs),
]


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="COCODataset",
        from_file=metainfo,
        data_mode=data_mode,
        filter_cfg=dict(filter_empty_gt=False, min_size=batch_size),
        ann_file=training_anns_path,
        data_prefix=dict(img=training_imgs_path),
        pipeline=train_pipeline_stage1,
    ),
)
val_pipeline = load_img + [dict(type="Resize", input_size=input_size, pad_val=(pad_value, pad_value, pad_value))] + load_anns
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type="COCODataset",
        from_file=metainfo,
        ann_file=validation_anns_path,
        data_mode=data_mode,
        data_prefix=dict(img=validation_imgs_path),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
# /Dataloaders

# Evaluation
val_evaluator = []
# /Evaluation


# Hooks
custom_hooks = [
    dict(
        type="YOLOXPoseModeSwitchHook",
        num_last_epochs=num_epochs - num_epochs_pipeline1,
        new_train_pipeline=train_pipeline_stage2,
        priority=48,
    ),
    dict(type="SyncNormHook", priority=48),
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=_base_.ema_momentum,
        update_buffers=True,
        strict_load=False,
        priority=49,
    ),
    dict(type="ModuleFreezingHook", modules_to_freeze=["feature_extraction_head"], priority=30),
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
