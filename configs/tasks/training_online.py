_base_ = ["./training_detection.py"]

# Settings
# auto_scale_lr = dict(base_batch_size=128, enable=True)
auto_scale_lr = dict(base_batch_size=128, enable=False)


## TODO Settings to refactor... ##
resume = False
load_from = "../checkpoints/yolox-pose_s_old_mice/mart_fixed.pth"
# load_from = None

num_epochs = 1000
warmup_epochs = 5
batch_size = 8

#### TODO End here ###

# base_lr = _base_.action_recognition_base_lr
base_lr = 1e-5

weight_decay = _base_.action_recognition_weight_decay
# val_interval = _base_.action_recognition_val_interval
val_interval = 5


work_dir = _base_.training_work_dir

resume = _base_.resume

block_size = _base_.block_size
n_embd_features = _base_.n_embd_features

bboxes_gt_format = _base_.action_recognition_bboxes_gt_format
keypoints_gt_format = _base_.action_recognition_keypoints_gt_format
actions_gt_format = _base_.action_recognition_actions_gt_format

metainfo = _base_.metainfo
data_mode = _base_.data_mode
data_root = "../../datasets/MICE/sequential_images/"
# data_root = "../../datasets/MICE/sequential_images_single_block/"


train_sequences = [
    # "images/train/0/",
    "images/train/13-10-02/",
    "images/train/13-20-02/",
    "images/train/13-40-02/",
]
train_bboxes_gt_paths = [
    # "bboxes/dummy/0.csv"
    "bboxes/train/13-10-02.csv",
    "bboxes/train/13-20-02.csv",
    "bboxes/train/13-40-02.csv",
]
train_keypoints_gt_paths = [
    # "keypoints/dummy/0.csv"
    "keypoints/train/13-10-02.csv",
    "keypoints/train/13-20-02.csv",
    "keypoints/train/13-40-02.csv",
]
train_actions_gt_paths = [
    # "actions/dummy/0.csv"
    "actions/train/13-10-02.csv",
    "actions/train/13-20-02.csv",
    "actions/train/13-40-02.csv",
]


val_sequences = ["images/val/14-20-02/"]
val_bboxes_gt_paths = ["bboxes/val/14-20-02.csv"]
val_keypoints_gt_paths = ["keypoints/val/14-20-02.csv"]
val_actions_gt_paths = ["actions/val/14-20-02.csv"]

train_sequences = ["images/val/14-20-02/"]
train_bboxes_gt_paths = ["bboxes/val/14-20-02.csv"]
train_keypoints_gt_paths = ["keypoints/val/14-20-02.csv"]
train_actions_gt_paths = ["actions/val/14-20-02.csv"]

# val_sequences = train_sequences
# val_bboxes_gt_paths = train_bboxes_gt_paths
# val_keypoints_gt_paths = train_keypoints_gt_paths
# val_actions_gt_paths = train_actions_gt_paths
# /Settings

# Model
data_preprocessor = dict(
    type="InferencePreprocessor",
    mean=[0, 0, 0],
    std=[1, 1, 1],
    input_size=_base_.input_size,
    pad_val=(_base_.pad_value, _base_.pad_value, _base_.pad_value),
)
detector = dict(
    runtime=dict(
        model=_base_.model,
        checkpoint=_base_.tracking_checkpoint,
        half_precision=_base_.half_precision,
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
    temperature_file=_base_.hyperparams,
)

assigner = dict(
    tracking_algorithm=dict(
        type="OnlineGroundTruth",
    ),
    metafile=metainfo,
    motion_algorithm=dict(
        type="DynamicKalmanFilterPytorch",
    ),
    memory_length=2,
)

analyzer = dict(
    _base_.analyzer.runtime.model,
    data_preprocessor=dict(
        # type="FPVOnlinePreprocessor",
        type="TestPreprocessor",  # TODO changer le nom...
        embd_size=128,  # TODO à dynamiser, devrait être une cfg...
        metainfo=metainfo,
        _delete_=True,
        block_size=block_size,
        with_actions=True,
        with_kpts=True,
        with_vels=True,
    ),
    loss_actions=dict(
        type="LDAMWithDRW",  # TODO peut pas utiliser le DRW, pcq recoit pas le #batch à chaque epoch...
        metainfo=metainfo,
    ),
    _delete_=True,
)


model = dict(
    type="Tracker",
    detector=detector,
    assigner=assigner,
    analyzer=analyzer,
)

# /Model

# Loops
runner = "PrecisionTrackRunner"
train_cfg = dict(
    _scope_=_base_.default_scope,
    type="EpochBasedTrainLoop",
    max_epochs=num_epochs,
    val_interval=val_interval,
    # val_interval=num_epochs,
    _delete_=True,
)
# train_cfg = dict(
#     type="OnlineTrainLoop",
#     metafile=metainfo,
#     post_processor=_base_.data_postprocessor,
#     _scope_=_base_.default_scope,
#     val_interval=val_interval,
#     max_epochs=epochs,
#     _delete_=True,
# )
val_cfg = dict(
    type="SequenceValidationLoop",
    _delete_=True,
)
# /Loops

# Optimization
param_scheduler = [
    # dict(
    #     type="QuadraticWarmupLR",
    #     by_epoch=True,
    #     begin=0,
    #     end=warmup_epochs,
    # ),
    # dict(
    #     type="CosineAnnealingLR",
    #     eta_min=base_lr / 100,
    #     begin=warmup_epochs,
    #     T_max=9 * (num_epochs // 10),
    #     end=9 * (num_epochs // 10),
    #     by_epoch=True,
    # ),
    # dict(type="ConstantLR", by_epoch=True, factor=1, begin=9 * (num_epochs // 10), end=num_epochs),
]


optim_wrapper = dict(
    loss_scale="dynamic",
    type="AmpOptimWrapper",
    dtype="float16",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=weight_decay),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    accumulative_counts=1,
)
# /Optimization

# Dataloaders
codec = dict(type="YOLOXPoseAnnotationProcessor", input_size=_base_.input_size)

load_img = [dict(type="LoadImage")]
load_anns = [
    dict(type="FilterAnnotations", by_kpt=True, by_box=True, keep_empty=False, min_kpt_vis=2),
    dict(type="GenerateTarget", encoder=codec),
]
resize = [dict(type="BottomupResize", input_size=_base_.input_size, pad_val=(_base_.pad_value, _base_.pad_value, _base_.pad_value))]
crop = [dict(type="SequenceRandomCrop", crop_size=(0.85, 1.0))]
transforms = [
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomContrastAug"),
    dict(type="SequenceRandomFlip", direction="horizontal", prob=0.5),
    # dict(type="SequenceRandomOcclusion"),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=0,
    # num_workers=4,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="OnlineRandomSequenceDataset",
        from_file=metainfo,
        n_feats=n_embd_features,
        n_velocities=2,
        keypoints_gt_format=keypoints_gt_format,
        bboxes_gt_format=bboxes_gt_format,
        actions_gt_format=actions_gt_format,
        data_root=data_root,
        data_prefix=dict(
            sequences=train_sequences,
            keypoints_gt_paths=train_keypoints_gt_paths,
            bboxes_gt_paths=train_bboxes_gt_paths,
            actions_gt_paths=train_actions_gt_paths,
        ),
        block_size=block_size,
        # pipeline=load_img + resize + load_anns,
        # pipeline=load_img + crop + resize + transforms + load_anns,
        pipeline=load_img + resize + transforms + load_anns,
        coach=dict(
            type="ActionRecognitionCoach",
            metainfo=metainfo,
            block_size=block_size,
        ),
        inference_resolution=_base_.inference_resolution,
        test_mode=False,
    ),
    _delete_=True,
    collate_fn=dict(type="pseudo_collate_sequences"),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    # num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type="OnlineRandomSequenceDataset",
        from_file=metainfo,
        n_feats=n_embd_features,
        n_velocities=2,
        keypoints_gt_format=keypoints_gt_format,
        bboxes_gt_format=bboxes_gt_format,
        actions_gt_format=actions_gt_format,
        data_root=data_root,
        data_prefix=dict(
            sequences=val_sequences,
            keypoints_gt_paths=val_keypoints_gt_paths,
            bboxes_gt_paths=val_bboxes_gt_paths,
            actions_gt_paths=val_actions_gt_paths,
        ),
        block_size=block_size,
        pipeline=load_img + resize + load_anns,
        inference_resolution=_base_.inference_resolution,
        test_mode=True,
    ),
    _delete_=True,
    collate_fn=dict(type="pseudo_collate_sequences"),
)
# /Dataloaders

# Evaluation
val_evaluator = [dict(type="MultiClassActionRecognitionMetrics", metainfo=metainfo, confusion_matrix_save_dir=work_dir)]
# /Evaluation


# Hooks
default_hooks = dict(
    checkpoint=dict(interval=-1, type="CheckpointHook", save_best="ActionRecognition/Macro F1", rule="greater", by_epoch=True),
)
custom_hooks = [
    dict(type="SyncNormHook", priority=48),
    # dict(
    #     type="DetectorEMAHook",
    #     ema_type="ExpMomentumEMA",
    #     momentum=0.0002,
    #     update_buffers=True,
    #     strict_load=False,
    #     priority=49,
    # ),
    # dict( # TODO Work in progress. Doit marcher avec DDP et seulement updater MART...
    #     type="AnalyzerEMAHook",
    #     ema_type="ExpMomentumEMA",
    #     momentum=0.0002,
    #     update_buffers=True,
    #     strict_load=False,
    #     priority=49,
    # ),
    # dict(
    #     type="SequencePreprocessingHook",
    #     priority=50,
    # ),
]
# /Hooks

# Visualization
log_processor = dict(type="LogProcessor", window_size=1, by_epoch=True, num_digits=6)
if _base_.wandb_logging:
    visualizer = dict(
        type="Visualizer",
        name="wandb_visualizer",
        vis_backends=dict(
            type="WandbVisBackend",
            init_kwargs=dict(
                project=_base_.project,
                entity=_base_.entity,
            ),
            save_dir=work_dir + "wandb/",
        ),
    )
else:
    visualizer = dict(
        _delete_=True,
    )
# /Visualization
