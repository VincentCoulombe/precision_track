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
    backbone=dict(
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        channel_attention=True,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="SiLU"),
        # init_cfg=dict(
        #     type="Pretrained",
        #     prefix="backbone.",
        #     checkpoint="https://download.openmmlab.com/mmdetection/v3.0/" "rtmdet/cspnext_rsb_pretrain/" "cspnext-s_imagenet_600e-ea671761.pth",
        # ),
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
    neck=dict(
        type="CSPNeXtNeck",
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="SiLU", inplace=True),
        widen_factor=widen_factor,
    ),
    test_cfg=dict(data_postprocessor=data_postprocessor, input_size=input_size),
    head=dict(
        type="DetectionHead",
        featmap_strides=(8, 16, 32),
        in_channels=256,
        feat_channels=256,
        widen_factor=widen_factor,
        stacked_convs=2,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="SiLU"),
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
    data_postprocessor=data_postprocessor,
)
