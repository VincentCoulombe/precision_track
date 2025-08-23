_base_ = "../models/rtmdet-pose.py"

# Settings
work_dir = _base_.testing_work_dir

batch_size = _base_.batch_size
half_precision = _base_.half_precision

load_from = _base_.testing_checkpoint

input_size = _base_.input_size
pad_value = _base_.pad_value

metainfo = _base_.metainfo
data_root = _base_.data_root
data_mode = _base_.data_mode
testing_anns_path = _base_.testing_anns_path
testing_imgs_path = _base_.testing_imgs_path
testing_output_file = _base_.testing_output_file
data_preprocessor = dict(
    type="InferencePreprocessor",
    mean=[0, 0, 0],
    std=[1, 1, 1],
    input_size=input_size,
    pad_val=(pad_value, pad_value, pad_value),
)
# /Settings

# Loop
test_cfg = dict(
    type="TestingLoop",
    test_cfg=_base_.model.test_cfg | dict(checkpoint=load_from, data_preprocessor=data_preprocessor, data_postprocessor=_base_.model.data_postprocessor),
)
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
test_evaluator = [
    dict(
        type="PoseTrackingMetric",
        ann_file=testing_anns_path,
        input_format="cxcywh",
        metafile=metainfo,
        output_file=testing_output_file,
        out_normalized_distances=_base_.with_pose_estimation,
        association_cfg=dict(
            metafile=metainfo,
            match_iou_thrs=dict(
                high=0.9,
                low=0.5,
                tentative=0.9,
            ),
            with_kpt_sigmas=False,
        ),
    ),
]
# /Evaluation

# Visualization
visualizer = dict(
    _delete_=True,
)
# /Visualization
