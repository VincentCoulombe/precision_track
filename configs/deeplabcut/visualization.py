metainfo = "../configs/mice/metadata.py"
visualizer = dict(
    painters=[
        dict(
            type="BoundingBoxPainter",
            annotations=[dict(type="Corner")],
        ),
        dict(
            type="LabelPainter",
            info=["class", "id", "score"],
            class_id_to_class={0: "Mouse"},
            label_position="TOP_CENTER",
            text_color=[0, 0, 0],
            text_scale=0.5,
            text_thickness=1,
            text_padding=3,
            border_radius=3,
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
        path="../precision_track/evaluation/comparisons/results/deeplabcut/dlcrnet_ms5.csv",
        instance_data="pred_track_instances",
        precision=64,
    ),
]
