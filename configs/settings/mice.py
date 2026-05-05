_base_ = "./_base_.py"

# Common
metainfo = "../configs/metadata/mice.py"
wandb_logging = False
# /Common

# 1) Detection
with_pose_estimation = False
half_precision = True

widen_factor = 0.5
deepen_factor = 0.33
#   1.1) Training
data_mode = _base_.data_mode
data_root = "../../datasets/MICE/pose-estimation/"
dataset_name = "mice"
deploying_directory = "../checkpoints/stripedmice/"
deployed_name = "model_" + dataset_name + "_DEPLOYED.pth"
training_work_dir = _base_.work_dir + "training_runs/" + dataset_name + "/"
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

val_interval = 10

training_anns_path = data_root + "annotations/train.json"
training_imgs_path = data_root + "images/"
validation_anns_path = data_root + "annotations/val.json"
validation_imgs_path = training_imgs_path

assign_on = "iou"
if with_pose_estimation:
    weight_loss_kpts = 30.0
    weight_loss_kpts_vis = 1.0
else:
    weight_loss_kpts = 0.0
    weight_loss_kpts_vis = 0.0
#   1.1) /Training

#   1.2) Testing
testing_work_dir = _base_.work_dir + "testing_runs/" + dataset_name + "/"
testing_checkpoint = deploying_directory + deployed_name

testing_anns_path = validation_anns_path
testing_imgs_path = validation_imgs_path
testing_output_file = testing_work_dir + "pose-detection_metrics.csv"
#   1.2) /Testing

#   1.3) Calibration
calibration_output_dir = _base_.work_dir + "calibration_runs/" + dataset_name + "/"
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
deploying_sanity_check_img_path = "images/0000003435.jpg"
sanity_check_img = data_root + deploying_sanity_check_img_path
deployment_device = "auto"
#   1.5) /Deployment
# 1) /Detection


# 2) Tracking
tracking_checkpoint_name = "model_stripedmice_deployed.onnx"
tracking_checkpoint = deploying_directory + tracking_checkpoint_name

pipelined = True
saving_directory = "../work_dir/test_cam_reid"
tracking_batch_size = 30
num_tentatives = 3
nb_frames_retain = 10
with_validation = False
with_action_recognition = False
with_group_action_recognition = False

num_subjects = {"mouse": 20}
stitching_algorithm = dict(
    type="SearchBasedStitching",
    capped_classes=num_subjects,
    beta=0.5,
    match_thr=0.9,
)

#   2.1) Tuning
low_thr_range = [0.05, 0.1]
high_thr_range = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
init_thr_range = [0.65, 0.7, 0.75, 0.8]
beta_range = [0.25, 0.5, 1.0, 1.5]
match_thr_range = [0.8, 0.9]
eps_range = [1e-2, 1e-1]
#   2.1) /Tuning

#   2.2) Testing
hyperparameters_file_name = "hyperparameters.json"
hyperparams = deploying_directory + hyperparameters_file_name
low_thr = low_thr_range[1]
high_thr = high_thr_range[3]
init_thr = init_thr_range[1]
mot_data_root = "../../datasets/MICE/pose-estimation/benchmark/"
testing_tracking_output_file = testing_work_dir + "mean_CLEAR_metrics_over_all_videos.csv"
#   2.2) /Testing


#   2.3) Validation
validation_configuration_file = "../configs/settings/validation/appearance.yaml"
#   2.3) /Validation

# 2) /Tracking


# 3) Action Recognition

#   3.1) Checkpoints
mart_deploying_directory = deploying_directory
mart_checkpoint_name = "mart_DEPLOYED.pth"
gmart_checkpoint_name = "gmart_DEPLOYED.pth"

mart_checkpoint = deploying_directory + mart_checkpoint_name
gmart_checkpoint = deploying_directory + gmart_checkpoint_name
#   3.1) /Checkpoints

block_size = 30

n_encoded_dynamics = 2
n_embd_dynamics = 32
n_embd_poses = 96
n_embd_features = 128

action_recognition_bboxes_gt_format = "CsvBoundingBoxes"
action_recognition_keypoints_gt_format = "CsvKeypoints"
action_recognition_actions_gt_format = "CsvActions"

assigner = dict(
    nb_frames_retain=block_size,
)

if with_action_recognition:
    action_recognition_input_names = ["features", "poses", "dynamics"]
    gar_input_names = ["distance_priors", "keypoint_priors"]
    action_recognition_output_names = ["class_logits", "action_embeddings"]
    gar_output_names = ["interaction_logits", "social_logits"]

    velocity_encoder = dict(type="BaseVelocityEncoder")

    action_recognition_with_velocities = n_embd_dynamics > 0
    action_recognition_with_poses = n_embd_poses > 0 and with_pose_estimation
    action_recognition_with_features = n_embd_features > 0

    analyzer = dict(
        type="ActionRecognitionBackend",
        data_preprocessor=dict(
            type="ActionRecognitionPreprocessor",
            embd_size=n_embd_features,
            metainfo=metainfo,
            _delete_=True,
            block_size=block_size,
            with_actions=False,
            with_kpts=action_recognition_with_poses,
            with_vels=action_recognition_with_velocities,
            velocity_encoder=velocity_encoder,
        ),
        metainfo=metainfo,
        input_names=action_recognition_input_names,
        data_postprocessor=dict(
            type="ActionPostProcessingSteps",
            postprocessing_steps=[
                dict(type="NearnessBasedActionFiltering", fallback_label="Other", metainfo=metainfo),
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
                    with_features=action_recognition_with_features,
                    with_dynamics=action_recognition_with_velocities,
                    with_poses=action_recognition_with_poses,
                    n_embd_features=n_embd_features,
                    block_size=block_size,
                    n_encoded_dynamics=n_encoded_dynamics,
                    n_embd_dynamics=n_embd_dynamics,
                    n_embd_poses=n_embd_poses,
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
            half_precision=half_precision,
            freeze=True,
            output_names=action_recognition_output_names,
            input_shapes=[
                dict(type="FeaturesShape", block_size=block_size, n_embd=n_embd_features),
                dict(type="PosesShape", block_size=block_size, metainfo=metainfo),
                dict(type="VelocityShape", block_size=block_size, n_encoding=n_encoded_dynamics),
            ],
        ),
    )
else:
    analyzer = None
    action_recognition_input_names = []
    gar_input_names = []
    action_recognition_output_names = []
    gar_output_names = []

#   3.2) Training
action_recognition_batch_size = 128
gar_batch_size = 32
action_recognition_base_lr = 3e-5
gar_base_lr = 3e-5
action_recognition_weight_decay = 0.01
gar_weight_decay = 0.1
action_recognition_dropout = 0
action_recognition_num_iter = 100000
gar_num_iter = 50000
action_recognition_warmup_iter = 25000
gar_warmup_iter = 10000
action_recognition_val_interval = 1000


action_recognition_data_root = "../../datasets/MICE/sequential/"

action_recognition_train_sequences = ["videos/train/13-10-02.avi", "videos/train/13-20-02.avi", "videos/train/13-40-02.avi"]
action_recognition_train_bboxes_gt_paths = ["bboxes/train/13-10-02.csv", "bboxes/train/13-20-02.csv", "bboxes/train/13-40-02.csv"]
action_recognition_train_keypoints_gt_paths = ["keypoints/train/13-10-02.csv", "keypoints/train/13-20-02.csv", "keypoints/train/13-40-02.csv"]
action_recognition_train_actions_gt_paths = ["actions/train/13-10-02.csv", "actions/train/13-20-02.csv", "actions/train/13-40-02.csv"]

action_recognition_val_sequences = ["videos/val/14-20-02.avi"]
action_recognition_val_bboxes_gt_paths = ["bboxes/val/14-20-02.csv"]
action_recognition_val_keypoints_gt_paths = ["keypoints/val/14-20-02.csv"]
action_recognition_val_actions_gt_paths = ["actions/val/14-20-02.csv"]

#   3.2) /Training

#   3.3) Testing
mart_testing_checkpoint = deploying_directory + mart_checkpoint_name
gmart_testing_checkpoint = deploying_directory + gmart_checkpoint_name

action_recognition_test_sequences = action_recognition_val_sequences
action_recognition_test_bboxes_gt_paths = action_recognition_val_bboxes_gt_paths
action_recognition_test_keypoints_gt_paths = action_recognition_val_keypoints_gt_paths
action_recognition_test_actions_gt_paths = action_recognition_val_actions_gt_paths
#   3.3) /Testing
# 3) /Action Recognition


# 4) Visualization
display_bounding_boxes = True
display_poses = False
display_velocities = True
display_species = False
display_confidence_scores = False
display_actions = False
display_search_zones = False
display_validations = False
display_untracked_detections = False
display_predicted_bounding_boxes = False
# 4) /Visualization
output_clustered_features = False
output_action_recognition_embeddings = False
output_appearance_database = False
