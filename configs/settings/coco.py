_base_ = "./_base_.py"

# IMPORTANT!
# The Microsoft's COCO dataset is only used to pretraining the detector.
# Therefore, it is not configure to be deployed nor to track.
# /IMPORTANT!

# Common
metainfo = "../configs/metadata/coco.py"
wandb_logging = False

data_mode = _base_.data_mode
data_root = "../../datasets/coco/"
# /Common

# 1) Detection
with_pose_estimation = True

widen_factor = 0.5
deepen_factor = 0.33
#   1.1) Training
data_mode = _base_.data_mode
data_root = "../../datasets/coco/"
training_work_dir = _base_.work_dir + "training_runs/coco/"
resume = False
training_checkpoint = None

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

training_anns_path = data_root + "annotations/person_keypoints_train2017.json"
training_imgs_path = data_root + "train2017/"
validation_anns_path = data_root + "annotations/person_keypoints_val2017.json"
validation_imgs_path = data_root + "val2017/"

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
testing_work_dir = _base_.work_dir + "testing_runs/coco/"
testing_checkpoint = "../checkpoints/model_coco/model_coco.pth"
half_precision = True

testing_anns_path = validation_anns_path
testing_imgs_path = validation_imgs_path
testing_output_file = testing_work_dir + "metrics.csv"
#   1.2) /Testing

#   1.3) Feature Extraction
fe_batch_size = 128
fe_base_lr = 0.01
fe_weight_decay = 0.0001
fe_num_epochs = 50
fe_val_interval = 10

fe_training_checkpoint = training_work_dir + f"epoch_{num_epochs}.pth"
#   1.3) /Feature Extraction
# 1) /Detection
