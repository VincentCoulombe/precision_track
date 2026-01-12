_base_ = "../../configs/tasks/training_action_recognition.py"


train_sequences = _base_.val_sequences
train_bboxes_gt_paths = _base_.val_bboxes_gt_paths
train_keypoints_gt_paths = _base_.val_keypoints_gt_paths
train_actions_gt_paths = _base_.val_actions_gt_paths


num_iter = 10000
warmup_iter = 1000
val_interval = 10000
