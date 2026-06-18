_base_ = "../tasks/tracking.py"


# Settings
batch_size = _base_.tracking_batch_size
metainfo = _base_.metainfo

# /Settings

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
        video_paths=_base_.mot_data_root + "videos/val",
        gt_paths=_base_.mot_data_root + "bboxes/val",
    ),
)
# /Dataloader

# Config
test_cfg = dict(
    type="TrackingTestingLoop",
    test_cfg=dict(
        batch_size=batch_size,
        dataloader=test_dataloader,
        profile_output_file=_base_.testing_tracking_output_file_3,
    ),
)
# /Config

# Evaluation
test_evaluator = [
    dict(type="CLEARMetrics", metainfo=metainfo, output_file=_base_.testing_tracking_output_file, report_every_prcnt=0.05),
    dict(type="IdentityPurityMetrics", metainfo=metainfo, output_file=_base_.testing_tracking_output_file_2, report_every_prcnt=0.05),
]
# /Evaluation

# Visualization
visualizer = dict(
    _delete_=True,
)
# /Visualization
