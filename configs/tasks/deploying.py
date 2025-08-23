_base_ = "../models/yolox-pose.py"


# Settings
work_dir = _base_.testing_work_dir
device = _base_.deployment_device
calibration_output_dir = _base_.calibration_output_dir

input_size = _base_.input_size
pad_value = _base_.pad_value

metainfo = _base_.metainfo
tracking_cfg = "../configs/tasks/tracking.py"
videos = _base_.testing_video_paths
gt_paths = _base_.testing_gt_paths

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
        directory=_base_.deployed_directory,
        deployed_name=_base_.deployed_name,
    ),
    common_config=dict(half_precision=_base_.half_precision, max_workspace_size=4 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 640, 640],
                    opt_shape=[30, 3, 640, 640],
                    max_shape=[30, 3, 640, 640],
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
        directory=_base_.mart_deployed_directory,
        deployed_name=_base_.mart_deployed_name,
    ),
    common_config=dict(half_precision=_base_.half_precision, max_workspace_size=4 << 30),
    model_inputs=dict(
        input_shapes=dict(
            features=dict(
                min_shape=[1, block_size, _base_.n_embd_features],
                opt_shape=[20, block_size, _base_.n_embd_features],
                max_shape=[100, block_size, _base_.n_embd_features],
            ),
            poses=dict(
                min_shape=[1, block_size, 18],  # TODO rendre dynamique
                opt_shape=[20, block_size, 18],
                max_shape=[100, block_size, 18],
            ),
            dynamics=dict(
                min_shape=[1, block_size, 2],
                opt_shape=[20, block_size, 2],
                max_shape=[100, block_size, 2],
            ),
        )
    ),
    output_names=_base_.action_recognition_output_names,
)
mart_dynamic_axes = dict(
    features={0: "batch_size", 1: "sequence_len"},
    poses={0: "batch_size", 1: "sequence_len"},
    dynamics={0: "batch_size", 1: "sequence_len"},
    class_logits={0: "batch_size", 1: "sequence_len"},
)
mart_onnx_config = dict(
    type="onnx",
    verbose=False,
    opset_version=17,
    input_names=_base_.action_recognition_input_names,
    output_names=_base_.action_recognition_output_names,
    save_file=_base_.mart_deployed_name[:-4] + ".onnx",
    optimize=True,
    keep_initializers_as_inputs=False,
    dynamic_axes=mart_dynamic_axes,
)
#   MART
# /Runtime

# Visualization
visualizer = dict(
    _delete_=True,
)
# /Visualization
