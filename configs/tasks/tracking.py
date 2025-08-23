_base_ = "../models/yolox-pose.py"

# Settings
half_precision = _base_.half_precision

batch_size = _base_.tracking_batch_size
hyperparams = _base_.hyperparams

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
        checkpoint=_base_.tracking_checkpoint,
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
    # dict(
    #     type="NpyEmbeddingOutput",
    #     path=_base_.work_dir + "/features.npy",
    # ),
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
if _base_.stitching_algorithm is not None:
    outputs += [
        dict(
            type="CsvSearchAreas",
            path=_base_.work_dir + "/search_areas.csv",
            instance_data="search_areas",
            precision=64,
        )
    ]
if _base_.validator is not None:
    outputs += [
        dict(
            type="CsvValidations",
            path=_base_.work_dir + "validations.csv",
            instance_data="validation_instances",
            precision=64,
        ),
        dict(
            type="CsvCorrections",
            path=_base_.work_dir + "corrections.csv",
            instance_data="correction_instances",
            precision=32,
        ),
    ]
if _base_.analyzer is not None:
    outputs += [
        dict(
            type="CsvActions",
            path=_base_.work_dir + "actions.csv",
            instance_data="pred_track_instances",
            metainfo=metainfo,
            precision=64,
        ),
        # dict(
        #     type="NpyEmbeddingOutput",
        #     path=_base_.work_dir + "/action_embeddings.npy",
        #     ids_field="instances_id",
        #     embs_field="action_embeddings",
        # ),
    ]
# /Outputs


# Visualization
painters = [
    dict(
        type="LabelPainter",
        info=["id"],
        metafile_path=metainfo,
        label_position="TOP_CENTER",
        text_color=[0, 0, 0],
        text_scale=2,
        text_thickness=2,
        text_padding=1,
        border_radius=1,
        format="cxcywh",
    ),
]
if _base_.with_pose_estimation:
    painters += [
        dict(
            type="KeypointsPainter",
            metafile_path=metainfo,
            joint_radius=8,
            link_thickness=4,
        ),
        dict(
            type="VelocityPainter",
            amplitude=4,
            anchor=0,
            thickness=8,
            color=[31, 31, 31],
        ),
    ]
else:
    painters += [
        dict(
            type="BoundingBoxPainter",
            annotations=[
                dict(
                    type="Box",
                    thickness=3,
                    format="cxcywh",
                )
            ],
        ),
    ]
if _base_.stitching_algorithm is not None:
    dict(
        type="SearchAreaPainter",
        annotations=[dict(type="Box")],
        color=[255, 0, 0],
    ),
if _base_.validator is not None:
    painters += [
        dict(
            type="ValidationPainter",
            annotations=[dict(type="Dot", radius=10)],
            palette=dict(nan_color=[255, 255, 255]),
        )
    ]

writers = [
    dict(
        type="FrameIdWriter",
        text_anchor=[100, 10],
        text_color=[255, 255, 255],
        text_scale=1,
        text_thickness=2,
        text_padding=10,
    ),
]
if _base_.validator is not None:
    writers += [
        dict(
            type="TagsDetectionWriter",
            text_color=[0, 0, 0],
            tag_ids=_base_.valid_tags,
        ),
    ]

visualizer = dict(
    _delete_=True,
    size=(1280, 1280),
    painters=painters,
    writers=writers,
    palette=dict(
        names=[
            "Spectral",
            "deep",
        ],
        size=20,
        nan_color=(255, 255, 255),
    ),
)
# /Visualization
