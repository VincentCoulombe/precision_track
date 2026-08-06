_base_ = "./action_recognition.py"

with_group_action_recognition = True

gar_input_names = ["distance_priors", "keypoint_priors"]

gar_output_names = []
if isinstance(_base_.gar_output_names, list):
    gar_output_names = _base_.gar_output_names

analyzer = dict(
    input_names=_base_.action_recognition_input_names + gar_input_names,
    output_names=_base_.action_recognition_output_names + gar_output_names,
    data_preprocessor=dict(
        with_distance_prior=True,
        with_keypoint_priors=True,
    ),
    runtime=dict(
        model=dict(
            type="RelationshipDetectionPoseBaselineModel",
            mart_config=_base_.analyzer.runtime.model,
            mart_checkpoint=_base_.mart_checkpoint,
            metainfo=_base_.metainfo,
            with_vel_coherence=False,
            with_vel_approach=False,
            with_orientation_priors=False,
            with_keypoint_priors=True,
            _delete_=True,
        ),
        checkpoint=None,
    ),
)
