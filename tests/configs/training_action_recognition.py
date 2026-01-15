_base_ = "../../configs/tasks/training_action_recognition.py"


train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            sequences=[_base_.train_sequences[0]],
            keypoints_gt_paths=[_base_.train_keypoints_gt_paths[0]],
            bboxes_gt_paths=[_base_.train_bboxes_gt_paths[0]],
            actions_gt_paths=[_base_.train_actions_gt_paths[0]],
        ),
    ),
)

train_cfg = dict(
    type="IterBasedTrainLoop",
    val_interval=10000,
    max_iters=10000,
)

param_scheduler = [
    dict(
        type="QuadraticWarmupLR",
        by_epoch=False,
        begin=0,
        end=1000,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=_base_.base_lr / 100,
        begin=1000,
        T_max=9 * (10000 // 10),
        end=9 * (10000 // 10),
        by_epoch=False,
    ),
    dict(type="ConstantLR", by_epoch=False, factor=1, begin=9 * (10000 // 10), end=10000),
]
