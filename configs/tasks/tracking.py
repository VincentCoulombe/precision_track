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
data_preprocessor = _base_.inference_data_preprocessor
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
        deploying_directory=_base_.deploying_directory,
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
        type="PrecisionTrack",
        obj_score_thrs=dict(high=high_thr, low=low_thr),
        weight_iou_with_det_scores=False,
        match_iou_thrs=dict(high=0.99, low=0.75, tentative=0.9),
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
if _base_.with_action_recognition and _base_.with_group_action_recognition:
    mart = _base_.analyzer.runtime.model
    analyzer = dict(
        input_names=_base_.action_recognition_input_names + _base_.gar_input_names,
        output_names=_base_.action_recognition_output_names + _base_.gar_output_names,
        data_preprocessor=dict(
            with_distance_prior=True,
            with_keypoint_priors=True,
        ),
        runtime=dict(
            model=dict(
                type="GMART",
                mart_config=mart,
                mart_checkpoint=None,
                metainfo=_base_.metainfo,
                with_vel_coherence=False,
                with_vel_approach=False,
                with_orientation_priors=False,
                with_keypoint_priors=True,
                _delete_=True,
            ),
        ),
    )
# /Model

# Outputs
outputs = [
    dict(
        type="CsvBoundingBoxes",
        path=_base_.saving_directory + "/detected_bboxes.csv",
        instance_data="pred_instances",
        subtype="detected_bboxes",
        precision=64,
    ),
    dict(
        type="CsvBoundingBoxes",
        path=_base_.saving_directory + "/tracked_bboxes.csv",
        instance_data="pred_track_instances",
        subtype="tracked_bboxes",
        precision=64,
    ),
    dict(
        type="CsvBoundingBoxes",
        path=_base_.saving_directory + "/predicted_bboxes.csv",
        instance_data="next_frame_pred_track_instances",
        subtype="predicted_bboxes",
        precision=64,
    ),
    dict(
        type="CsvVelocities",
        path=_base_.saving_directory + "/tracked_velocities.csv",
        instance_data="pred_track_instances",
        precision=32,
    ),
    dict(
        type="CsvTimestamps",
        path=_base_.saving_directory + "/timestamps.csv",
    ),
    # dict(
    #     type="NpyEmbeddingOutput",
    #     path=_base_.saving_directory + "/features.npy",
    #     ids_field="instances_id",
    # ),
    dict(
        type="CsvSearchAreas",
        path=_base_.saving_directory + "/stitching_search_areas.csv",
        instance_data="search_areas",
        precision=64,
    ),
    dict(
        type="CsvTailtagValidations",
        path=_base_.saving_directory + "/tracked_tailtag_validations.csv",
        precision=64,
    ),
    dict(
        type="CsvAppearanceValidations",
        path=_base_.saving_directory + "/tracked_appearance_validations.csv",
        precision=64,
    ),
    dict(
        type="CsvCorrections",
        path=_base_.saving_directory + "/tracked_corrections.csv",
        instance_data="correction_instances",
        precision=32,
    ),
    dict(
        type="CsvActions",
        path=_base_.saving_directory + "/tracked_actions.csv",
        instance_data="pred_track_instances",
        metainfo=metainfo,
        precision=64,
    ),
    # dict(
    #     type="NpyEmbeddingOutput",
    #     path=_base_.saving_directory + "/action_embeddings.npy",
    #     ids_field="instances_id",
    #     embs_field="action_embeddings",
    # ),
    # dict(
    #     type="PthAppearanceDatabaseOutput",
    #     path=_base_.saving_directory + "/appearance_database.pth",
    # ),
]

if _base_.with_pose_estimation:
    outputs += [
        dict(
            type="CsvKeypoints",
            path=_base_.saving_directory + "/tracked_kpts.csv",
            instance_data="pred_track_instances",
            precision=32,
        ),
    ]
# /Outputs


# Visualization
bbox_size = 4
text_size = 2
painters = []
if _base_.display_search_zones:
    painters += [
        dict(
            type="SearchAreaPainter",
            annotations=[dict(type="Box")],
            color=[255, 0, 0],
        ),
    ]
if _base_.display_untracked_detections:
    painters += [
        dict(
            type="BoundingBoxPainter",
            annotations=[
                dict(
                    type="Box",
                    thickness=bbox_size,
                    format="cxcywh",
                )
            ],
            subtype="detected_bboxes",
        ),
    ]
if _base_.display_bounding_boxes:
    painters += [
        dict(
            type="BoundingBoxPainter",
            annotations=[
                dict(
                    type="Box",
                    thickness=bbox_size,
                    format="cxcywh",
                )
            ],
            subtype="tracked_bboxes",
        ),
    ]
if _base_.display_velocities:
    painters += [
        dict(
            type="VelocityPainter",
            amplitude=4,
            anchor=0,
            thickness=8,
            color=[31, 31, 31],
        ),
    ]
if _base_.display_poses:
    painters += [
        dict(
            type="KeypointsPainter",
            metafile_path=metainfo,
            joint_radius=bbox_size + 6,
            link_thickness=bbox_size + 2,
        ),
    ]
if _base_.display_validations:
    painters += [
        dict(
            type="ValidationPainter",
            radius=20,
            palette=dict(nan_color=[255, 255, 255]),
        ),
        dict(
            type="AppearanceValidationPainter",
            thickness=bbox_size * 5,
            palette=dict(nan_color=[255, 255, 255]),
        ),
    ]
if _base_.display_predicted_bounding_boxes:
    painters += [
        dict(
            type="BoundingBoxPainter",
            annotations=[
                dict(
                    type="Corner",
                    thickness=bbox_size,
                    format="cxcywh",
                )
            ],
            subtype="predicted_bboxes",
        ),
    ]

info = []
if _base_.display_species:
    info += ["class"]
info += ["id"]
if _base_.display_confidence_scores:
    info += ["score"]

painters += [
    dict(
        type="LabelPainter",
        info=info,
        metafile_path=metainfo,
        label_position="TOP_CENTER",
        text_color=[0, 0, 0],
        text_scale=text_size,
        text_thickness=int(text_size),
        text_padding=1,
        border_radius=1,
        format="cxcywh",
        display_actions=_base_.display_actions,
    ),
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
if _base_.display_validations:
    writers += [
        dict(
            type="TagsDetectionWriter",
            text_color=[0, 0, 0],
        ),
        dict(
            type="AppearanceDetectionWriter",
            text_color=[0, 0, 0],
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
