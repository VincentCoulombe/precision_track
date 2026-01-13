_base_ = "../../configs/tasks/training_action_recognition.py"


train_sequences = _base_.train_sequences[0]
train_bboxes_gt_paths = _base_.train_bboxes_gt_paths[0]
train_keypoints_gt_paths = _base_.train_keypoints_gt_paths[0]
train_actions_gt_paths = _base_.train_actions_gt_paths[0]


num_iter = 10000
warmup_iter = 1000
val_interval = 10000
