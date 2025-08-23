metainfo = "../configs/settings/mice/metadata.py"
visualizer = dict(
    painters=[
        dict(
            type="BoundingBoxPainter",
            annotations=[
                dict(
                    type="Box",
                    format="xywh",
                    thickness=3,
                )
            ],
        ),
    ],
    palette=dict(
        names=[
            "Spectral",
            "deep",
        ],
        size=20,
    ),
)

outputs = [
    dict(
        type="CsvBoundingBoxes",
        path="../precision_track/evaluation/comparisons/results/sleap/multi_class_bottomup.csv",
        instance_data="pred_track_instances",
        precision=64,
    ),
]
