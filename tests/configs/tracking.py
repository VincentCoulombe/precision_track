_base_ = "../../configs/models/yolox-pose.py"

# Settings
half_precision = True

batch_size = 30
hyperparams = "./tests/configs/hyperparameters.json"

input_size = (640, 640)
pad_value = 114

metainfo = "./configs/metadata/mice.py"

low_thr = 0.1
high_thr = 0.5
init_thr = 0.75
# /Settings

# Model
_base_.model["metainfo"] = metainfo
_base_.model["head"]["assigner"]["oks_calculator"]["metainfo"] = metainfo
_base_.model["head"]["loss_oks"]["metainfo"] = metainfo
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
        checkpoint="./tests/configs/model_mice_clustering_DEPLOYED.pth",
        half_precision=half_precision,
        input_shapes=[dict(type="ImageShape", n_channels=3, width=input_size[0], height=input_size[1])],
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
    _delete_=True,
)
assigner = dict(
    metafile=metainfo,
    nb_frames_retain=10,
    num_tentatives=3,
    thresholds_file=hyperparams,
    tracking_algorithm=dict(
        type="PrecisionTrack",
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
    stitching_algorithm=dict(
        type="SearchBasedStitching",
        capped_classes={"mouse": 20},
        beta=0.5,
        match_thr=0.9,
    ),
    _delete_=True,
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
        video_paths="./assets/20mice_sanity_check.avi",
        gt_paths="./tests/work_dir/20mice_sanity_check.csv",
    ),
)
# /Dataloader

# Evaluation
test_evaluator = [dict(type="CLEARMetrics", metainfo=metainfo)]
# /Evaluation

# Config
test_cfg = dict(
    type="TrackingTestingLoop",
    test_cfg=dict(
        detector=detector,
        assigner=assigner,
        validator=None,
        analyzer=None,
        batch_size=batch_size,
        dataloader=test_dataloader,
    ),
)
# /Config

visualizer = dict(
    _delete_=True,
)
