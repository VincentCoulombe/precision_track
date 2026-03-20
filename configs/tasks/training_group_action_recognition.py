_base_ = "./training_action_recognition.py"

# Model
model = dict(
    type="GMART",
    mart_config=_base_.analyzer.runtime.model,
    mart_checkpoint=_base_.mart_testing_checkpoint,
    metainfo=_base_.metainfo,
    relationship_loss_weight=100.0,
    classification_loss_weight=1.0,
    with_vel_coherence=False,
    with_vel_approach=False,
    with_orientation_priors=False,
    with_keypoint_priors=True,
    # data_preprocessor=dict(
    #     type="GroupActionRecognitionTrainingPreprocessor",
    #     with_distance_priors=True,
    #     with_vel_alignments=False,
    #     with_vel_cosine_similarities=False,
    #     velocity_encoder=_base_.velocity_encoder,
    #     metainfo=_base_.metainfo,
    #     _delete_=True,
    #     block_size=_base_.block_size,
    #     mode="loss",
    # ),
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

# Dataloaders
train_dataloader = dict(
    dataset=dict(
        type="GroupActionRecognitionDataset",
        keep_bboxes=True,
        require_interaction=True,
    ),
)

val_dataloader = dict(
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
    checkpoint=dict(interval=-1, type="CheckpointHook", save_best="GroupActionRecognition/Relationship F1", rule="greater", by_epoch=False),
)
dict(type="ModuleFreezingHook", modules_to_freeze=["MART"], priority=30),
# /Hooks
