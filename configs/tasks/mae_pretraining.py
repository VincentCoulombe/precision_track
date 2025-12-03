_base_ = "./training_action_recognition.py"


# Settings
base_lr = 3e-4
weight_decay = 0.1
warmup_iter = _base_.action_recognition_warmup_iter
# /Settings

# Model
model = dict(
    mode="pretrain",
    data_preprocessor=dict(
        mode="pretrain",
    ),
)
# /Model


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
train_dataloader = dict(
    dataset=dict(
        type="MAEDataset",
        detector=_base_.detector,
        from_file=_base_.metainfo,
        n_feats=_base_.n_embd_features,
        n_velocities=2,
        keypoints_gt_format=_base_.keypoints_gt_format,
        bboxes_gt_format=_base_.bboxes_gt_format,
        data_root=_base_.data_root,
        data_prefix=dict(
            sequences=_base_.train_sequences,
            keypoints_gt_paths=_base_.train_keypoints_gt_paths,
            bboxes_gt_paths=_base_.train_bboxes_gt_paths,
        ),
        block_size=_base_.block_size,
        pipeline=_base_.load_img + _base_.resize + _base_.transforms + _base_.load_anns,
        inference_resolution=_base_.inference_resolution,
        training=True,
        loading_ratio=0.25,
        _delete_=True,
    ),
)

val_dataloader = None
# /Dataloaders

# Loops
val_cfg = None
# /Loops

# Evaluation
val_evaluator = None
# /Evaluation

# Hooks
custom_hooks = [
    dict(
        type="SequencesSwitchHook",
        priority=51,
        generate_every=1000,
    ),
]
# /Hooks
