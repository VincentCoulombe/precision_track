_base_ = "./testing_action_recognition.py"

# Model
# model = dict(
#     type="GMART",
#     mart_config=_base_.analyzer.runtime.model,
#     mart_checkpoint=_base_.mart_testing_checkpoint,
#     metainfo=_base_.metainfo,
#     relationship_loss_weight=10.0,
#     classification_loss_weight=1.0,
#     refine_nodes=True,
#     with_vel_coherence=False,
#     with_vel_approach=False,
#     with_orientation_priors=False,
#     data_preprocessor=dict(
#         type="GroupActionRecognitionTrainingPreprocessor",
#         with_distance_priors=True,
#         with_vel_alignments=True,
#         with_vel_cosine_similarities=True,
#         velocity_encoder=_base_.velocity_encoder,
#         metainfo=_base_.metainfo,
#         _delete_=True,
#         block_size=_base_.block_size,
#         mode="loss",
#     ),
#     _delete_=True,
# )
model = dict(
    type="RelationshipDetectionBaselineModel",
    actions_of_interest=[2],
    mart_config=_base_.analyzer.runtime.model.mart_config,
    mart_checkpoint=_base_.mart_testing_checkpoint,
    metainfo=_base_.metainfo,
    data_preprocessor=dict(
        type="RelationshipDetectionBaselinePreprocessor",
        actions_of_interest=[2],
        velocity_encoder=_base_.velocity_encoder,
        metainfo=_base_.metainfo,
        _delete_=True,
        block_size=_base_.block_size,
    ),
    _delete_=True,
)
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
    group_actions=["Interacting"],  # TODO doit être dans metadata....
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
