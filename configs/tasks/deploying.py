_base_ = "../models/yolox-pose.py"


# Settings
device = _base_.deployment_device
calibration_output_dir = _base_.calibration_output_dir

input_size = _base_.input_size
pad_value = _base_.pad_value

metainfo = _base_.metainfo
tracking_cfg = "../configs/tasks/tracking.py"

img = _base_.sanity_check_img

data_preprocessor = dict(
    type="InferencePreprocessor",
    mean=[0, 0, 0],
    std=[1, 1, 1],
    input_size=input_size,
    pad_val=(pad_value, pad_value, pad_value),
)
calibration_checkpoint = _base_.testing_checkpoint
data_mode = _base_.data_mode
testing_anns_path = _base_.testing_anns_path
testing_imgs_path = _base_.testing_imgs_path
# /Settings


# Loop
calibration_cfg = dict(
    type="CalibrationLoop",
    calibration_cfg=_base_.model.test_cfg
    | dict(checkpoint=calibration_checkpoint, data_preprocessor=data_preprocessor, data_postprocessor=_base_.model.data_postprocessor),
)
test_cfg = dict()
# /Loop

# Dataloader
codec = dict(type="YOLOXPoseAnnotationProcessor", input_size=input_size)
test_pipeline = [
    dict(type="LoadImage"),
    dict(type="GenerateTarget", encoder=codec),
    dict(
        type="PackPoseInputs",
        meta_keys=(
            "id",
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "input_size",
            "input_center",
            "input_scale",
        ),
    ),
]
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type="COCODataset",
        from_file=metainfo,
        data_mode=data_mode,
        ann_file=testing_anns_path,
        data_prefix=dict(img=testing_imgs_path),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
# /Dataloader

# Evaluation
calibration_evaluator = dict(
    type="PoseEstimationECEMetric",
    output_dir=calibration_output_dir,
    iou_thr=0.65,
    bin_distribution="uniform",
    n_bins=20,
)
test_evaluator = []
# /Evaluation

# Runtime
#   Detection
_base_.model.data_preprocessor = data_preprocessor
output_names = [
    "cls_scores",
    "objectnesses",
    "bbox_preds",
    "kpt_preds",
    "kpt_vis",
    "features",
    "priors",
    "strides",
]
runtime_config = dict(
    type="tensorrt",
    paths=dict(
        directory=_base_.deploying_directory,
        deployed_name=_base_.deployed_name,
    ),
    common_config=dict(half_precision=_base_.half_precision, max_workspace_size=4 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, input_size[0], input_size[1]],
                    opt_shape=[_base_.tracking_batch_size, 3, input_size[0], input_size[1]],
                    max_shape=[_base_.tracking_batch_size, 3, input_size[0], input_size[1]],
                )
            )
        )
    ],
    output_names=output_names,
)
codebase_config = dict(
    type="precision_track",
    task="PrecisionTrackDetection",
    post_processing=dict(
        score_threshold=0.1,
        iou_threshold=0.5,
        max_output_boxes_per_class=100,
        pre_top_k=1000,
        keep_top_k=100,
        background_label_id=-1,
    ),
)
dynamic_axes = {o: {0: "batch"} for o in output_names}
dynamic_axes["input"] = {0: "batch"}
onnx_config = dict(
    type="onnx",
    verbose=False,
    opset_version=17,
    input_names=["input"],
    output_names=output_names,
    save_file=_base_.deployed_name[:-4] + ".onnx",
    optimize=True,
    keep_initializers_as_inputs=False,
    dynamic_axes=dynamic_axes,
)
#   Detection

#   MART
analyzer = _base_.analyzer
block_size = _base_.block_size
mart_runtime_config = dict(
    type="tensorrt",
    paths=dict(
        directory=_base_.mart_deploying_directory,
        deployed_name=_base_.mart_checkpoint_name,
    ),
    common_config=dict(half_precision=_base_.half_precision, max_workspace_size=4 << 30),
    output_names=_base_.action_recognition_output_names,
)
mart_dynamic_axes = dict(
    features={0: "nb_subjects"}, poses={0: "nb_subjects"}, dynamics={0: "nb_subjects"}, class_logits={0: "nb_subjects"}, action_embeddings={0: "nb_subjects"}
)
mart_onnx_config = dict(
    type="onnx",
    verbose=False,
    opset_version=17,
    input_names=_base_.action_recognition_input_names,
    output_names=_base_.action_recognition_output_names,
    save_file=_base_.mart_checkpoint_name[:-4] + ".onnx",
    optimize=True,
    keep_initializers_as_inputs=False,
    dynamic_axes=mart_dynamic_axes,
)
#  /MART
# GMART
gar_input_names = _base_.action_recognition_input_names + _base_.gar_input_names
gmart_runtime_config = dict(
    type="tensorrt",
    paths=dict(
        directory=_base_.mart_deploying_directory,
        deployed_name=_base_.gmart_checkpoint_name,
    ),
    common_config=dict(half_precision=_base_.half_precision, max_workspace_size=4 << 30),
    output_names=_base_.action_recognition_output_names + _base_.gar_output_names,
)
gmart_dynamic_axes = mart_dynamic_axes | dict(
    distance_priors={0: "nb_subjects", 1: "nb_subjects"},
    keypoint_priors={0: "nb_subjects", 1: "nb_subjects"},
    interaction_logits={0: "nb_subjects", 1: "nb_subjects"},
    social_logits={0: "nb_subjects"},
)
gmart_onnx_config = dict(
    type="onnx",
    verbose=False,
    opset_version=17,
    input_names=gar_input_names,
    output_names=_base_.action_recognition_output_names + _base_.gar_output_names,
    save_file=_base_.gmart_checkpoint_name[:-4] + ".onnx",
    optimize=True,
    keep_initializers_as_inputs=False,
    dynamic_axes=gmart_dynamic_axes,
)
if _base_.with_action_recognition and _base_.with_group_action_recognition:
    gmart_analyzer = dict(
        data_preprocessor=analyzer.data_preprocessor
        | dict(
            with_distance_prior=True,
            with_keypoint_priors=True,
        ),
        runtime=dict(
            model=dict(
                type="GMART",
                mart_config=analyzer.runtime.model,
                mart_checkpoint=_base_.mart_checkpoint,
                metainfo=_base_.metainfo,
                with_keypoint_priors=True,
            ),
            input_shapes=list(analyzer.runtime.input_shapes)
            + [
                dict(type="DistancePriorsShape"),
                dict(type="KeypointPriorsShape", metainfo=_base_.metainfo),
            ],
        ),
    )
# /GMART
# /Runtime

# Visualization
visualizer = dict(
    _delete_=True,
)
# /Visualization
