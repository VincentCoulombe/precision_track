_base_ = "./testing_action_recognition.py"

# Loops
test_cfg = dict(type="SequenceTestingLoop", test_cfg=dict(checkpoint=_base_.gmart_testing_checkpoint))
# /Loops

# Model
model = dict(
    type="GMART",
    mart_config=_base_.analyzer.runtime.model,
    mart_checkpoint=None,
    metainfo=_base_.metainfo,
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
# model = dict(
#     type="RelationshipDetectionBaselineModel",
#     actions_of_interest=[2],
#     mart_config=_base_.analyzer.runtime.model.mart_config,
#     mart_checkpoint=_base_.mart_testing_checkpoint,
#     metainfo=_base_.metainfo,
#     data_preprocessor=dict(
#         type="RelationshipDetectionBaselinePreprocessor",
#         actions_of_interest=[2],
#         velocity_encoder=_base_.velocity_encoder,
#         metainfo=_base_.metainfo,
#         _delete_=True,
#         block_size=_base_.block_size,
#     ),
#     _delete_=True,
# )
# /Model

# Dataloaders
test_dataloader = dict(
    dataset=dict(
        type="GroupActionRecognitionDataset",
        keep_bboxes=True,
    ),
)
# /Dataloaders

# Evaluation
test_evaluator = dict(
    type="GroupActionRecognitionMetrics",
    metainfo=_base_.metainfo,
    confusion_matrix_save_dir=_base_.work_dir,
    metric_save_dir=_base_.work_dir,
    label_index_mode="last",
    _delete_=True,
)
# /Evaluation

# Hooks
dict(type="ModuleFreezingHook", modules_to_freeze=["MART"], priority=30),
# /Hooks
