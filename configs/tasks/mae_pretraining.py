_base_ = "./training_action_recognition.py"


# Settings
base_lr = 3e-4
weight_decay = 0.1
warmup_iter = _base_.action_recognition_warmup_iter

hyperparams = _base_.hyperparams

input_size = _base_.input_size
pad_value = _base_.pad_value

metainfo = _base_.metainfo

low_thr = _base_.low_thr
high_thr = _base_.high_thr
init_thr = _base_.init_thr

# data_root = "../../datasets/MICE/sequential_nano/"

# val_sequences = ["videos/14-20-02.avi"]
# val_bboxes_gt_paths = ["bboxes/14-20-02.csv"]
# val_keypoints_gt_paths = ["keypoints/14-20-02.csv"]
# val_actions_gt_paths = ["actions/14-20-02.csv"]

# train_sequences = val_sequences
# train_bboxes_gt_paths = val_bboxes_gt_paths
# train_keypoints_gt_paths = val_keypoints_gt_paths
# train_actions_gt_paths = val_actions_gt_paths


# /Settings

# Model
assigner = dict(
    metafile=_base_.metainfo,
    nb_frames_retain=_base_.nb_frames_retain,
    num_tentatives=_base_.num_tentatives,
    thresholds_file=hyperparams,
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

model = dict(
    mode="pretrain",
    data_preprocessor=dict(
        mode="pretrain",
    ),
)
# /Model

# Outputs
outputs = [
    dict(
        type="CsvBoundingBoxes",
        path=_base_.work_dir + "/bboxes.csv",
        instance_data="pred_track_instances",
        precision=64,
    ),
    dict(
        type="CsvVelocities",
        path=_base_.work_dir + "/velocities.csv",
        instance_data="pred_track_instances",
        precision=32,
    ),
    dict(type="OnlinePthEmbeddingOutput"),
]
if _base_.with_pose_estimation:
    outputs += [
        dict(
            type="CsvKeypoints",
            path=_base_.work_dir + "/kpts.csv",
            instance_data="pred_track_instances",
            precision=32,
        ),
    ]
# /Outputs

# Optimization
optim_wrapper = dict(
    optimizer=dict(lr=base_lr, weight_decay=weight_decay),
)
# /Optimization

# Dataloaders
train_dataloader = dict(
    dataset=dict(
        type="MAEDataset",
        detector=_base_.detector,
        assigner=assigner,
        outputs=outputs,
        tracking_batch_size=_base_.tracking_batch_size,
        from_file=_base_.metainfo,
        n_feats=_base_.n_embd_features,
        n_velocities=2,
        keypoints_gt_format=_base_.keypoints_gt_format,
        bboxes_gt_format=_base_.bboxes_gt_format,
        data_root=_base_.data_root,
        data_prefix=dict(
            sequences=_base_.train_sequences,
            # keypoints_gt_paths=_base_.train_keypoints_gt_paths,
            # bboxes_gt_paths=_base_.train_bboxes_gt_paths,
        ),
        block_size=_base_.block_size,
        pipeline=_base_.load_img + _base_.resize + _base_.transforms + _base_.load_anns,
        inference_resolution=_base_.inference_resolution,
        training=True,
        nb_simulteneous_seq=1,
        supervized=False,
        _delete_=True,
    ),
)

val_dataloader = None
# /Dataloaders

# Loops
val_cfg = None
# /Loops

# Evaluation
val_evaluator = None
# /Evaluation

# Hooks
custom_hooks = [
    dict(
        type="SequencesSwitchHook",
        priority=51,
        generate_every=1000,
    ),
]
# /Hooks
