_base_ = "./tracking.py"

assigner = dict(
    tracking_algorithm=dict(
        _delete_=True,
        type="StrongSORT",
        obj_score_thrs=dict(high=_base_.high_thr, low=_base_.low_thr),
        weight_iou_with_det_scores=False,
        match_iou_thrs=dict(high=0.99, low=0.75, tentative=0.9),
        init_track_thr=_base_.init_thr,
        appearance_weight=0.25,
        appearance_ema=0.1,
        crop_enlargement_factor=0.0,
        re_identificator=dict(
            metainfo="../tests/configs/re-identification_metadata.yaml",
            checkpoint="../tests/configs/precision_track_re-identificator.onnx",
        ),
        data_preprocessor=dict(type="WildLifeReIDPreprocessor"),
    ),
)
