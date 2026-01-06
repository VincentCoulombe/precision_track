_base_ = "./tracking.py"

inference_resolution = (1536, 1536)
block_size = 30

n_embd_dynamics = 32
n_embd_pose = 96
n_embd_features = 128

assigner = dict(
    nb_frames_retain=block_size,
)

metainfo = _base_.metainfo

action_recognition_input_names = ["features", "poses", "dynamics"]
action_recognition_output_names = ["class_logits", "action_embeddings"]
analyzer = dict(
    type="ActionRecognitionBackend",
    data_preprocessor=dict(
        type="ActionRecognitionPreprocessor",
        metainfo=metainfo,
        _delete_=True,
        block_size=block_size,
    ),
    metainfo=metainfo,
    input_names=action_recognition_input_names,
    data_postprocessor=dict(
        type="ActionPostProcessingSteps",
        postprocessing_steps=[
            dict(type="NearnessBasedActionFiltering", concerned_labels=["Interacting"], fallback_label="Other", metainfo=metainfo),
            dict(
                type="KeypointBasedActionRefinement",
                action_to_refine="Interacting",
                source_keypoints=[0, 0],
                sink_keypoints=[0, [6, 7]],
                criterias=["nearest", "nearest"],
                refined_actions=["Interacting: nose-to-nose", "Interacting: Anogenital"],
                metainfo=metainfo,
            ),
        ],
    ),
    runtime=dict(
        model=dict(
            type="MART",
            config=dict(
                n_embd=n_embd_features,
                block_size=block_size,
                n_embd_dynamics=n_embd_dynamics,
                n_embd_pose=n_embd_pose,
                n_block=4,
                causal=True,
                n_head=4,
                bias=False,
                dropout=0.0,
                n_output=5,
            ),
            metainfo=metainfo,
        ),
        checkpoint="./tests/configs/mart_DEPLOYED.pth",
        half_precision=True,
        freeze=True,
        output_names=action_recognition_output_names,
        input_shapes=[
            dict(type="FeaturesShape", block_size=block_size, n_embd=n_embd_features),
            dict(type="PosesShape", block_size=block_size, metainfo=metainfo),
            dict(type="VelocityShape", block_size=block_size),
        ],
    ),
    _delete_=True,
)

outputs = []
