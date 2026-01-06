_base_ = "../models/yolox-pose.py"

# Settings
balanced = True
work_dir = _base_.training_work_dir

block_size = _base_.block_size
n_embd_features = _base_.n_embd_features

bboxes_gt_format = _base_.action_recognition_bboxes_gt_format
keypoints_gt_format = _base_.action_recognition_bboxes_gt_format
actions_gt_format = _base_.action_recognition_bboxes_gt_format

metainfo = _base_.metainfo
data_mode = _base_.data_mode
data_root = _base_.action_recognition_data_root

test_sequences = _base_.action_recognition_test_sequences
test_bboxes_gt_paths = _base_.action_recognition_test_bboxes_gt_paths
test_keypoints_gt_paths = _base_.action_recognition_test_keypoints_gt_paths
test_actions_gt_paths = _base_.action_recognition_test_actions_gt_paths
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

model = dict(
    _base_.analyzer.runtime.model,
    data_preprocessor=dict(
        type="ActionRecognitionPreprocessor",
        metainfo=metainfo,
        _delete_=True,
        block_size=block_size,
        mode="loss" if balanced else "sequence",
    ),
    _delete_=True,
)
# /Model

# Loops
test_cfg = dict(type="SequenceTestingLoop", test_cfg=dict(checkpoint=_base_.mart_testing_checkpoint))
# /Loops

# Dataloaders
codec = dict(type="SequenceAnnotationProcessor", input_size=_base_.input_size, convert_cats=True)
load_img = [dict(type="LoadImage"), dict(type="BottomupResize", input_size=_base_.input_size, pad_val=_base_.pad_value)]
load_anns = [
    dict(type="FilterAnnotations", by_kpt=True, by_box=True, keep_empty=False),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type="ActionRecognitionDataset" if balanced else "ActionRecognitionPerFrameDataset",
        detector=detector,
        from_file=metainfo,
        n_feats=n_embd_features,
        n_velocities=2,
        keypoints_gt_format=keypoints_gt_format,
        bboxes_gt_format=bboxes_gt_format,
        actions_gt_format=actions_gt_format,
        data_root=data_root,
        data_prefix=dict(
            sequences=test_sequences,
            keypoints_gt_paths=test_keypoints_gt_paths,
            bboxes_gt_paths=test_bboxes_gt_paths,
            actions_gt_paths=test_actions_gt_paths,
        ),
        block_size=block_size,
        pipeline=load_img + load_anns,
        inference_resolution=_base_.inference_resolution,
        training=False,
    ),
)
# /Dataloaders

# Evaluation
test_evaluator = [
    dict(type="MultiClassActionRecognitionMetrics", metainfo=metainfo, confusion_matrix_save_dir=work_dir, label_index_mode="last" if balanced else "spacial"),
]
# /Evaluation

# Visualization
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=False, num_digits=6)
visualizer = dict(
    _delete_=True,
)
# /Visualization
