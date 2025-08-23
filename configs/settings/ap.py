_base_ = "./_base_.py"

# Common
metainfo = "../configs/metadata/ap.py"
wandb_logging = False

data_mode = _base_.data_mode
data_root = "../../datasets/AP/"
# /Common

# 1) Detection
with_pose_estimation = True

widen_factor = 0.5
deepen_factor = 0.33
#   1.1) Training
data_mode = _base_.data_mode
data_root = "../../datasets/AP/"
training_work_dir = _base_.work_dir + "training_runs/ap/"
resume = False
training_checkpoint = "../checkpoints/model_coco/model_coco.pth"

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
training_imgs_path = data_root + "data/"
validation_anns_path = data_root + "annotations/val.json"
validation_imgs_path = training_imgs_path

if with_pose_estimation:
    weight_loss_kpts = 30.0
    weight_loss_kpts_vis = 1.0
    assign_on = "oks"
else:
    weight_loss_kpts = 0.0
    weight_loss_kpts_vis = 0.0
    assign_on = "iou"
#   1.1) /Training

#   1.2) Testing
testing_work_dir = _base_.work_dir + "testing_runs/ap/"
testing_checkpoint = "../checkpoints/model_ap/model_ap.pth"
half_precision = True

testing_anns_path = data_root + "annotations/test.json"
testing_imgs_path = training_imgs_path
testing_output_file = testing_work_dir + "pose-detection_metrics.csv"
#   1.2) /Testing

#   1.3) Calibration
calibration_output_dir = _base_.work_dir + "calibration_runs/ap/"
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
sanity_check_img = validation_imgs_path + "000000000001.jpg"
deployment_device = "auto"
deployed_directory = "../checkpoints/model_ap/"
deployed_name = "model_ap_DEPLOYED.pth"
#   1.5) /Deployment
# 1) /Detection


# 2) Tracking
tracking_checkpoint = deployed_directory + deployed_name
pipelined = True
tracking_batch_size = 30
num_tentatives = 1
nb_frames_retain = 5
stitching_algorithm = None
validator = None
analyzer = None
action_recognition_input_names = None
action_recognition_output_names = None

#   2.1) Tuning
low_thr_range = [0.05, 0.1]
high_thr_range = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
init_thr_range = [0.65, 0.7, 0.75, 0.8]
#   2.1) /Tuning

#   2.2) Testing
hyperparams = deployed_directory + "hyperparameters.json"
low_thr = low_thr_range[1]
high_thr = high_thr_range[3]
init_thr = init_thr_range[1]
testing_video_paths = "../../datasets/APT-test/data/"
testing_gt_paths = "../../datasets/APT-test/annotations/"
testing_tracking_output_file = testing_work_dir + "CLEAR_metrics.csv"
#   2.2) /Testing
# 2) /Tracking

# 3) Action Recognition
with_action_recognition = False
mart_checkpoint = None
block_size = None

n_embd_dynamics = None
n_embd_pose = None
n_embd_features = None

analyzer = None
action_recognition_input_names = None
action_recognition_output_names = None

#   3.1) Training
#   3.1) /Training

#   3.2) Testing
#   3.2) /Testing

#   3.3) Deployment
mart_deployed_directory = deployed_directory
mart_deployed_name = "mart_DEPLOYED.pth"
#   3.3) /Deployment
# 3) /Action Recognition
