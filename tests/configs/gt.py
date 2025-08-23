_base_ = "./tracking.py"

assigner = dict(
    nb_frames_retain=1,
    num_tentatives=3,
    tracking_algorithm=dict(
        type="GroundTruth",
        weight_iou_with_det_scores=False,
    ),
)

outputs = [
    dict(
        type="NpyEmbeddingOutput",
        path="",
    ),
]
