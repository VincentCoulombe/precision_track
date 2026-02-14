_base_ = "../../configs/tasks/testing_action_recognition.py"


detector = dict(runtime=dict(checkpoint="../tests/configs/model_mice_DEPLOYED.pth"))

test_dataloader = dict(
    dataset=dict(
        detector=detector,
    )
)

test_cfg = dict(type="SequenceTestingLoop", test_cfg=dict(checkpoint="../tests/configs/mart.pth"))

test_sequences = "videos/val/"
test_bboxes_gt_paths = "bboxes/val/"
test_keypoints_gt_paths = "keypoints/val/"
test_actions_gt_paths = "actions/val/"
