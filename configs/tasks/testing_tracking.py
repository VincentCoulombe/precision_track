_base_ = "../models/rtmdet-pose.py"

# Settings
testing_checkpoint = _base_.testing_checkpoint
half_precision = _base_.half_precision

batch_size = _base_.tracking_batch_size
hyperparams = _base_.hyperparams
work_dir = _base_.testing_work_dir

input_size = _base_.input_size
pad_value = _base_.pad_value

metainfo = _base_.metainfo

low_thr = _base_.low_thr
high_thr = _base_.high_thr
init_thr = _base_.init_thr
# /Settings

# Model
data_preprocessor = dict(
    type="InferencePreprocessor",
    mean=[0, 0, 0],
    std=[1, 1, 1],
    input_size=input_size,
    pad_val=(pad_value, pad_value, pad_value),
)
detector = dict(
    runtime=dict(
        model=_base_.model,
        checkpoint=testing_checkpoint,
        half_precision=half_precision,
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
    temperature_file=hyperparams,
)
assigner = dict(
    metafile=_base_.metainfo,
    nb_frames_retain=_base_.nb_frames_retain,
    num_tentatives=_base_.num_tentatives,
    thresholds_file=_base_.hyperparams,
    tracking_algorithm=dict(
        type="PrecisionTrack" if _base_.with_pose_estimation else "ByteTrack",
        obj_score_thrs=dict(high=high_thr, low=low_thr),
        weight_iou_with_det_scores=False,
        match_iou_thrs=dict(high=0.9, low=0.75, tentative=0.9),
        init_track_thr=init_thr,
        with_kpt_weights=True,
        with_kpt_sigmas=False,
        dynamic_temporal_scaling=False,
        alpha=0.5,
    ),
    motion_algorithm=dict(
        type="DynamicKalmanFilter",
    ),
    stitching_algorithm=_base_.stitching_algorithm,
)
# /Model

# Dataloader
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type="VideoDataset",
        video_paths=_base_.testing_video_paths,
        gt_paths=_base_.testing_gt_paths,
    ),
)
# /Dataloader

# Config
test_cfg = dict(
    type="TrackingTestingLoop",
    test_cfg=dict(
        detector=detector,
        assigner=assigner,
        validator=_base_.validator,
        analyzer=None,
        batch_size=batch_size,
        dataloader=test_dataloader,
    ),
)
# /Config

# Evaluation
test_evaluator = [dict(type="CLEARMetrics", metainfo=metainfo, output_file=_base_.testing_tracking_output_file)]
# /Evaluation

# Visualization
visualizer = dict(
    _delete_=True,
)
# /Visualization
