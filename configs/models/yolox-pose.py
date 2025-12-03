_base_ = "../settings/mice.py"

input_size = _base_.input_size
pad_value = _base_.pad_value

widen_factor = _base_.widen_factor
deepen_factor = _base_.deepen_factor
metainfo = _base_.metainfo

data_postprocessor = [
    dict(
        type="NMSPostProcessor",
        score_thr=0.01,
        nms_thr=0.65,
        pool_thr=0.9,
    ),
]

model = dict(
    type="DetectionModel",
    metainfo=metainfo,
    init_cfg=dict(
        type="Kaiming",
        layer="Conv2d",
        a=2.23606797749979,
        distribution="uniform",
        mode="fan_in",
        nonlinearity="leaky_relu",
    ),
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        pad_size_divisor=32,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        batch_augments=[
            dict(
                type="BatchSyncRandomResize",
                random_size_range=(480, 800),
                size_divisor=32,
                interval=1,
            ),
        ],
    ),
    backbone=dict(
        type="CSPDarknet",
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        out_indices=(2, 3, 4),
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="Swish"),
        # init_cfg=dict(
        #     type="Pretrained",
        #     checkpoint="https://download.openmmlab.com/mmdetection/v2.0/" "yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_" "20211121_095711-4592a793.pth",
        #     prefix="backbone.",
        # ),
    ),
    neck=dict(
        type="YOLOXPAFPN",
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode="nearest"),
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="Swish"),
    ),
    head=dict(
        type="DetectionHead",
        featmap_strides=(8, 16, 32),
        in_channels=256,
        feat_channels=256,
        widen_factor=widen_factor,
        stacked_convs=2,
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="Swish"),
        prior_generator=dict(type="MultiLevelPointGenerator", offset=0, strides=[8, 16, 32]),
        assigner=dict(
            type="SimOTAAssigner",
            dynamic_k_indicator=_base_.assign_on,
            oks_calculator=dict(type="PoseOKS", metainfo=metainfo),
        ),
        loss_cls=dict(type="BCELoss", reduction="sum", loss_weight=1.0),
        loss_bbox=dict(type="IoULoss", mode="square", eps=1e-16, reduction="sum", loss_weight=5.0),
        loss_obj=dict(type="BCELoss", use_target_weight=True, reduction="sum", loss_weight=1.0),
        loss_oks=dict(type="OKSLoss", reduction="none", metainfo=metainfo, norm_target_weight=True, loss_weight=_base_.weight_loss_kpts),
        loss_vis=dict(type="BCELoss", use_target_weight=True, reduction="mean", loss_weight=_base_.weight_loss_kpts_vis),
        overlaps_power=0.5,
    ),
    feature_extraction_head=dict(
        type="FeatureExtractionHead",
        featmap_strides=(8, 16, 32),
        in_channels=256,
        feat_channels=256,
        widen_factor=widen_factor,
        stacked_convs=4,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="SiLU"),
        loss_features=dict(type="TripletLoss", loss_weight=1.0, neg_strategy="semihard", pos_strategy="hard"),
    ),
    test_cfg=dict(data_postprocessor=data_postprocessor, input_size=input_size),
    data_postprocessor=data_postprocessor,
)
