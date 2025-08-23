_base_ = "./_base_.py"

# Common
metainfo = "../configs/metadata/mice.py"
wandb_logging = False
# /Common

# 1) Detection
with_pose_estimation = True

widen_factor = 0.5
deepen_factor = 0.33
#   1.1) Training
data_mode = _base_.data_mode
data_root = "../../datasets/MICE/pose-estimation/"
training_work_dir = _base_.work_dir + "training_runs/mice/"
resume = False
training_checkpoint = "../checkpoints/model_ap/model_ap.pth"

input_size = (640, 640)
pad_value = 114

base_lr = 0.004
batch_size = 38
weight_decay = 0.05

ema_momentum = 0.0003

num_epochs = 300
num_epochs_pipeline1 = 280
warmup_epochs = 5

val_interval = 100

training_anns_path = data_root + "annotations/train.json"
training_imgs_path = data_root + "images/"
validation_anns_path = data_root + "annotations/val.json"
validation_imgs_path = training_imgs_path

if with_pose_estimation:
    weight_loss_kpts = 30.0
    weight_loss_kpts_vis = 1.0
    assign_on = "iou"
else:
    weight_loss_kpts = 0.0
    weight_loss_kpts_vis = 0.0
    assign_on = "iou"
#   1.1) /Training

#   1.2) Testing
testing_work_dir = _base_.work_dir + "testing_runs/mice/"
testing_checkpoint = "../tests/configs/model_mice.pth"

half_precision = True

testing_anns_path = validation_anns_path
testing_imgs_path = validation_imgs_path
testing_output_file = testing_work_dir + "pose-detection_metrics.csv"
#   1.2) /Testing

#   1.3) Calibration
calibration_output_dir = _base_.work_dir + "calibration_runs/mice/"
#   1.3) /Calibration

#   1.4) Feature Extraction
fe_batch_size = 128
fe_base_lr = 0.01
fe_weight_decay = 0.0001
fe_num_epochs = 50
fe_val_interval = 10
fe_training_checkpoint = training_work_dir + f"epoch_{num_epochs}.pth"

#   1.4) /Feature Extraction

#   1.5) Deployment
sanity_check_img = data_root + "images/0000003435.jpg"
deployment_device = "auto"
deployed_directory = "../tests/configs/"
deployed_name = "model_mice_clustering_DEPLOYED.pth"
#   1.5) /Deployment
# 1) /Detection


# 2) Tracking
tracking_checkpoint = deployed_directory + "model_mice_clustering_DEPLOYED.onnx"


pipelined = True
tracking_batch_size = 30
num_tentatives = 3
nb_frames_retain = 10
with_validation = False
with_action_recognition = True


num_mice = 20
stitching_algorithm = dict(
    type="SearchBasedStitching",
    capped_classes={"mouse": num_mice},
    beta=0.5,
    match_thr=0.9,
)
if with_validation:
    valid_tags = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 22, 23, 25]
    assert len(valid_tags) == num_mice, f"To ensure a sucessful validation, please make sure that the number of mice match the number of valid tags."
    validator = dict(
        type="ArucoValidation",
        num_tags=32,
        tags_size=3,
        predefined_dict=None,
        parameters=dict(
            minMarkerPerimeterRate=0.1,
            maxMarkerPerimeterRate=0.9,
            adaptiveThreshWinSizeMin=7,
            adaptiveThreshWinSizeMax=23,
            adaptiveThreshWinSizeStep=10,
            polygonalApproxAccuracyRate=0.14,
            minOtsuStdDev=1,
            perspectiveRemovePixelPerCell=13,
            perspectiveRemoveIgnoredMarginPerCell=0.35,
        ),
        refinement="none",
        tag_kpt=7,
        kpt_conf_thr=0.5,
        estimation_range=120,
        timeout_after=0.02,
        min_sample_size=25,
        valid_tags=valid_tags,
    )
else:
    validator = None

#   2.1) Tuning
low_thr_range = [0.05, 0.1]
high_thr_range = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
init_thr_range = [0.65, 0.7, 0.75, 0.8]
beta_range = [0.25, 0.5, 1.0, 1.5]
match_thr_range = [0.8, 0.9]
eps_range = [1e-2, 1e-1]
#   2.1) /Tuning

#   2.2) Testing
hyperparams = deployed_directory + "hyperparameters.json"
low_thr = low_thr_range[1]
high_thr = high_thr_range[3]
init_thr = init_thr_range[1]
testing_video_paths = data_root + "/benchmark/data/"
testing_gt_paths = data_root + "benchmark/annotations/"
testing_tracking_output_file = testing_work_dir + "CLEAR_metrics.csv"
#   2.2) /Testing
# 2) /Tracking


# 3) Action Recognition
mart_checkpoint = deployed_directory + "mart_DEPLOYED.onnx"

inference_resolution = (2720, 2720)
block_size = 30

n_embd_dynamics = 32
n_embd_pose = 96
n_embd_features = 128

action_recognition_bboxes_gt_format = "CsvBoundingBoxes"
action_recognition_keypoints_gt_format = "CsvKeypoints"
action_recognition_actions_gt_format = "CsvActions"

assigner = dict(
    nb_frames_retain=block_size,
)

if with_action_recognition:
    action_recognition_input_names = ["features", "poses", "dynamics"]
    action_recognition_output_names = ["class_logits", "action_embeddings"]

    analyzer = dict(
        type="ActionRecognitionBackend",
        data_preprocessor=dict(
            type="ActionRecognitionPreprocessor",
            metainfo=metainfo,
            _delete_=True,
            block_size=block_size,
        ),
        metainfo=metainfo,
        input_names=action_recognition_input_names,
        data_postprocessor=dict(
            type="ActionPostProcessingSteps",
            postprocessing_steps=[
                dict(type="NearnessBasedActionFiltering", concerned_labels=["Interacting"], fallback_label="Other", metainfo=metainfo),
                dict(
                    type="KeypointBasedActionRefinement",
                    action_to_refine="Interacting",
                    source_keypoints=[0, 0],
                    sink_keypoints=[0, [6, 7]],
                    criterias=["nearest", "nearest"],
                    refined_actions=["Interacting: nose-to-nose", "Interacting: Anogenital"],
                    metainfo=metainfo,
                ),
            ],
        ),
        runtime=dict(
            model=dict(
                type="MART",
                config=dict(
                    n_embd=n_embd_features,
                    block_size=block_size,
                    n_embd_dynamics=n_embd_dynamics,
                    n_embd_pose=n_embd_pose,
                    n_block=4,
                    causal=True,
                    use_alibi=False,
                    n_head=4,
                    bias=False,
                    dropout=0.0,
                    n_output=5,
                ),
                metainfo=metainfo,
            ),
            checkpoint=mart_checkpoint,
            half_precision=True,
            freeze=True,
            output_names=action_recognition_output_names,
            input_shapes=[
                dict(type="FeaturesShape", block_size=block_size, n_embd=n_embd_features),
                dict(type="PosesShape", block_size=block_size, metainfo=metainfo),
                dict(type="VelocityShape", block_size=block_size),
            ],
        ),
    )
else:
    analyzer = None
    action_recognition_input_names = None
    action_recognition_output_names = None

#   3.1) Training
action_recognition_batch_size = 128
action_recognition_base_lr = 3e-5
action_recognition_weight_decay = 0.01
action_recognition_dropout = 0
action_recognition_num_iter = 100000
action_recognition_warmup_iter = int(0.1 * action_recognition_num_iter)
action_recognition_val_interval = action_recognition_num_iter // 100

action_recognition_data_root = "../../datasets/MICE/sequential/"

action_recognition_train_sequences = [
    "videos/train/13-10-02.avi",
    "videos/train/13-20-02.avi",
    "videos/train/13-40-02.avi",
]
action_recognition_train_bboxes_gt_paths = [
    "bboxes/train/13-10-02.csv",
    "bboxes/train/13-20-02.csv",
    "bboxes/train/13-40-02.csv",
]
action_recognition_train_keypoints_gt_paths = [
    "keypoints/train/13-10-02.csv",
    "keypoints/train/13-20-02.csv",
    "keypoints/train/13-40-02.csv",
]
action_recognition_train_actions_gt_paths = [
    "actions/train/13-10-02.csv",
    "actions/train/13-20-02.csv",
    "actions/train/13-40-02.csv",
]

action_recognition_val_sequences = ["videos/val/14-20-02.avi"]
action_recognition_val_bboxes_gt_paths = ["bboxes/val/14-20-02.csv"]
action_recognition_val_keypoints_gt_paths = ["keypoints/val/14-20-02.csv"]
action_recognition_val_actions_gt_paths = ["actions/val/14-20-02.csv"]
#   3.1) /Training

#   3.2) Testing
mart_testing_checkpoint = deployed_directory + "mart.pth"

action_recognition_test_sequences = action_recognition_val_sequences
action_recognition_test_bboxes_gt_paths = action_recognition_val_bboxes_gt_paths
action_recognition_test_keypoints_gt_paths = action_recognition_val_keypoints_gt_paths
action_recognition_test_actions_gt_paths = action_recognition_val_actions_gt_paths
#   3.2) /Testing

#   3.3) Deployment
mart_deployed_directory = deployed_directory
mart_deployed_name = "mart_DEPLOYED.pth"
#   3.3) /Deployment
# 3) /Action Recognition
