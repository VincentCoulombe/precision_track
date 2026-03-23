_base_ = "./training_action_recognition.py"

# Model
model = dict(
    type="GMART",
    mart_config=_base_.analyzer.runtime.model,
    mart_checkpoint=_base_.mart_testing_checkpoint,
    metainfo=_base_.metainfo,
    relationship_loss_weight=10.0,
    classification_loss_weight=0.0,
    classification_alpha=None,
    with_vel_coherence=False,
    with_vel_approach=False,
    with_orientation_priors=False,
    with_keypoint_priors=True,
    data_preprocessor=dict(
        type="GroupActionRecognitionPoseTrainingPreprocessor",
        velocity_encoder=_base_.velocity_encoder,
        metainfo=_base_.metainfo,
        _delete_=True,
        block_size=_base_.block_size,
    ),
    _delete_=True,
)
# /Model

# Loops
train_cfg = dict(
    max_iters=_base_.gar_num_iter,
)
# /Loops

# Optimization
param_scheduler = [
    dict(
        type="QuadraticWarmupLR",
        by_epoch=False,
        begin=0,
        end=_base_.gar_warmup_iter,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=_base_.action_recognition_base_lr / 100,
        begin=_base_.gar_warmup_iter,
        T_max=9 * (_base_.gar_num_iter // 10),
        end=9 * (_base_.gar_num_iter // 10),
        by_epoch=False,
    ),
    dict(type="ConstantLR", by_epoch=False, factor=1, begin=9 * (_base_.gar_num_iter // 10), end=_base_.gar_num_iter),
]


optim_wrapper = dict(
    loss_scale="dynamic",
    type="AmpOptimWrapper",
    dtype="float16",
    optimizer=dict(type="AdamW", lr=_base_.action_recognition_base_lr, weight_decay=_base_.gar_weight_decay),
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
    batch_size=_base_.gar_batch_size,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    dataset=dict(
        type="GroupActionRecognitionDataset",
        keep_bboxes=True,
        require_interaction=True,
    ),
)

val_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    dataset=dict(
        type="GroupActionRecognitionDataset",
        keep_bboxes=True,
    ),
)
# /Dataloaders

# Evaluation
val_evaluator = dict(
    type="GroupActionRecognitionMetrics",
    metainfo=_base_.metainfo,
    confusion_matrix_save_dir=_base_.work_dir,
    metric_save_dir=_base_.work_dir,
    label_index_mode="last",
)
# /Evaluation

# Hooks
default_hooks = dict(
    checkpoint=dict(interval=-1, type="CheckpointHook", save_best="GroupActionRecognition/Social mF1@0.5:0.95", rule="greater", by_epoch=False),
)
custom_hooks = [
    dict(type="ModuleFreezingHook", modules_to_freeze=["mart"], priority=30),
    dict(type="LossCurriculumSwitchHook", switch_iter=0.5 * _base_.gar_warmup_iter, classification_loss_weight=1.0, priority=50),
]
# /Hooks
