dataset_info = dict(
    dataset_name="mice",
    paper_info=dict(
        author="Coulombe. et al.",
        title="Scaling Up Social Behavior Studies: PrecisionTrack and MART for Real-Time, Large-Scale and Prolonged Social Behavior Analysis",
        homepage="https://www.biorxiv.org/content/10.1101/2024.12.26.630112v1",
    ),
    keypoint_info=[
        dict(name="Snout", swap=""),
        dict(name="Right Ear", swap="Left Ear"),
        dict(name="Right Leg", swap="Left Leg"),
        dict(name="Left Leg", swap="Right Leg"),
        dict(name="Left Ear", swap="Right Ear"),
        dict(name="Centroid", swap=""),
        dict(name="Base of Tail", swap=""),
        dict(name="Tailtag", swap=""),
    ],
    skeleton_info=[
        dict(link=("Snout", "Right Ear")),
        dict(link=("Snout", "Left Ear")),
        dict(link=("Right Ear", "Centroid")),
        dict(link=("Left Ear", "Centroid")),
        dict(link=("Centroid", "Right Leg")),
        dict(link=("Centroid", "Left Leg")),
        dict(link=("Right Leg", "Base of Tail")),
        dict(link=("Left Leg", "Base of Tail")),
        dict(link=("Base of Tail", "Tailtag")),
    ],
    joint_weights=[
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    sigmas=[
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ],
    classes=["mouse"],
    actions=[
        "Other",
        "Running",
        "Interacting",
        "Rearing",
        "Grooming",
    ],
    orientation_keypoint_pairs=[
        # (anterior, posterior) ordered best→worst axis reliability
        ("Snout", "Base of Tail"),  # full body axis — most reliable
        ("Snout", "Tailtag"),  # longest axis, but tailtag wobbles
        ("Snout", "Centroid"),  # front half axis
        ("Centroid", "Base of Tail"),  # rear half axis
        ("Centroid", "Tailtag"),  # rear half, longer
        ("Right Ear", "Base of Tail"),
        ("Left Ear", "Base of Tail"),
        ("Right Ear", "Centroid"),
        ("Left Ear", "Centroid"),
    ],
    distance_keypoint_pairs=[
        ("Snout", "Snout"),
        ("Snout", "Right Ear"),
        ("Snout", "Left Ear"),
        ("Snout", "Right Leg"),
        ("Snout", "Left Leg"),
        ("Snout", "Base of Tail"),
        ("Snout", "Tailtag"),
    ],
)
