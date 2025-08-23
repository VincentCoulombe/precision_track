_base_ = ["../models/yolox-pose.py", "../wandb/keys.py"]

# Settings
auto_scale_lr = dict(base_batch_size=128, enable=False)

batch_size = _base_.action_recognition_batch_size
base_lr = _base_.action_recognition_base_lr
weight_decay = _base_.action_recognition_weight_decay
num_iter = _base_.action_recognition_num_iter
warmup_iter = _base_.action_recognition_warmup_iter
val_interval = _base_.action_recognition_val_interval

work_dir = _base_.training_work_dir

resume = _base_.resume

block_size = _base_.block_size
n_embd_features = _base_.n_embd_features

bboxes_gt_format = _base_.action_recognition_bboxes_gt_format
keypoints_gt_format = _base_.action_recognition_bboxes_gt_format
actions_gt_format = _base_.action_recognition_bboxes_gt_format

metainfo = _base_.metainfo
data_mode = _base_.data_mode
data_root = _base_.action_recognition_data_root

train_sequences = _base_.action_recognition_train_sequences
train_bboxes_gt_paths = _base_.action_recognition_train_bboxes_gt_paths
train_keypoints_gt_paths = _base_.action_recognition_train_keypoints_gt_paths
train_actions_gt_paths = _base_.action_recognition_train_actions_gt_paths

val_sequences = _base_.action_recognition_val_sequences
val_bboxes_gt_paths = _base_.action_recognition_val_bboxes_gt_paths
val_keypoints_gt_paths = _base_.action_recognition_val_keypoints_gt_paths
val_actions_gt_paths = _base_.action_recognition_val_actions_gt_paths
# /Settings

# Model
data_preprocessor = dict(
    type="InferencePreprocessor",
    mean=[0, 0, 0],
    std=[1, 1, 1],
    input_size=_base_.input_size,
    pad_val=(_base_.pad_value, _base_.pad_value, _base_.pad_value),
)
detector = dict(
    runtime=dict(
        model=_base_.model,
        checkpoint=_base_.tracking_checkpoint,
        half_precision=_base_.half_precision,
        input_shapes=[dict(type="ImageShape", n_channels=3, width=_base_.input_size[0], height=_base_.input_size[1])],
        output_names=[
            "cls_scores",
            "objectnesses",
            "bbox_preds",
            "kpt_preds",
            "kpt_vis",
            "features",
            "priors",
            "strides",
        ],
    ),
    data_preprocessor=data_preprocessor,
    data_postprocessor=_base_.model.data_postprocessor,
    temperature_file=_base_.hyperparams,
)

model = dict(
    _base_.analyzer.runtime.model,
    data_preprocessor=dict(
        type="ActionRecognitionPreprocessor",
        metainfo=metainfo,
        _delete_=True,
        block_size=block_size,
        mode="loss",
    ),
    _delete_=True,
)
# /Model

# Loops
train_cfg = dict(
    type="IterBasedTrainLoop",
    _scope_=_base_.default_scope,
    val_interval=val_interval,
    max_iters=num_iter,
)
val_cfg = dict(type="SequenceValidationLoop")
# /Loops

# Optimization
param_scheduler = [
    dict(
        type="QuadraticWarmupLR",
        by_epoch=False,
        begin=0,
        end=warmup_iter,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr / 100,
        begin=warmup_iter,
        T_max=9 * (num_iter // 10),
        end=9 * (num_iter // 10),
        by_epoch=False,
    ),
    dict(type="ConstantLR", by_epoch=False, factor=1, begin=9 * (num_iter // 10), end=num_iter),
]


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
codec = dict(type="SequenceAnnotationProcessor", input_size=_base_.input_size, convert_cats=True)
load_img = [dict(type="LoadImage"), dict(type="BottomupResize", input_size=_base_.input_size, pad_val=_base_.pad_value)]
load_anns = [
    dict(type="FilterAnnotations", by_kpt=True, by_box=True, keep_empty=False),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler"),
    dataset=dict(
        type="ActionRecognitionDataset",
        detector=detector,
        from_file=metainfo,
        n_feats=n_embd_features,
        n_velocities=2,
        keypoints_gt_format=keypoints_gt_format,
        bboxes_gt_format=bboxes_gt_format,
        actions_gt_format=actions_gt_format,
        data_root=data_root,
        data_prefix=dict(
            sequences=train_sequences,
            keypoints_gt_paths=train_keypoints_gt_paths,
            bboxes_gt_paths=train_bboxes_gt_paths,
            actions_gt_paths=train_actions_gt_paths,
        ),
        block_size=block_size,
        pipeline=load_img + load_anns,
        inference_resolution=_base_.inference_resolution,
        training=True,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type="ActionRecognitionDataset",
        detector=detector,
        from_file=metainfo,
        n_feats=n_embd_features,
        n_velocities=2,
        keypoints_gt_format=keypoints_gt_format,
        bboxes_gt_format=bboxes_gt_format,
        actions_gt_format=actions_gt_format,
        data_root=data_root,
        data_prefix=dict(
            sequences=val_sequences,
            keypoints_gt_paths=val_keypoints_gt_paths,
            bboxes_gt_paths=val_bboxes_gt_paths,
            actions_gt_paths=val_actions_gt_paths,
        ),
        block_size=block_size,
        pipeline=load_img + load_anns,
        inference_resolution=_base_.inference_resolution,
        training=False,
    ),
)
# /Dataloaders

# Evaluation
val_evaluator = [dict(type="MultiClassActionRecognitionMetrics", metainfo=metainfo, confusion_matrix_save_dir=work_dir, label_index_mode="last")]
# /Evaluation


# Hooks
default_hooks = dict(
    checkpoint=dict(interval=-1, type="CheckpointHook", save_best="ActionRecognition/Macro F1", rule="greater", by_epoch=False),
)
custom_hooks = [
    dict(type="SyncNormHook", priority=48),
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49,
    ),
    # dict(
    #     type="SequencePreprocessingHook",
    #     priority=50,
    # ),
]
# /Hooks

# Visualization
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=False, num_digits=6)
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
