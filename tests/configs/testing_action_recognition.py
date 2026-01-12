_base_ = "../../configs/tasks/testing_action_recognition.py"


test_cfg = dict(type="SequenceTestingLoop", test_cfg=dict(checkpoint=_base_.mart_testing_checkpoint))

test_sequences = "videos/val/"
test_bboxes_gt_paths = "bboxes/val/"
test_keypoints_gt_paths = "keypoints/val/"
test_actions_gt_paths = "actions/val/"
