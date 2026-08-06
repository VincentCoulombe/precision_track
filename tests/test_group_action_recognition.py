import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from mmengine import Config

from precision_track import PipelinedTracker, Tracker
from precision_track.models.group_mart import GMARTPredictions
from precision_track.utils import VideoReader, empty_fpv_action_recognition, load_user_configs, postprocess_fpv_action_recognition

ROOT = "./tests/"

TOOLS_DIR = "./tools"


@pytest.fixture
def config():
    return os.path.join(ROOT, "configs/group_action_recognition.py")


@pytest.fixture
def user_configs():
    return os.path.join(ROOT, "../configs/user_configs.yaml")


@pytest.fixture
def testing_config():
    return os.path.join(ROOT, "configs/testing_group_action_recognition.py")


@pytest.fixture
def deployed_checkpoints():
    ckpt_names = [
        "model_mart_DEPLOYED.pth",
        "model_mart_DEPLOYED_NVIDIAGeForceRTX3090_FP16.engine",
        "model_mart_DEPLOYED.onnx",
    ]
    return dict(
        checkpoint_names=ckpt_names,
        found=[False for _ in ckpt_names],
    )


def set_user_configs(with_action_recognition: bool, user_configs, system_config, deploying_directory="", metadata_file=""):
    with open(user_configs, "r") as f:
        user_configs = yaml.safe_load(f)
    user_configs["booleans"]["with_action_recognition"] = with_action_recognition
    if deploying_directory:
        user_configs["training"]["deploying_directory"] = deploying_directory + "/"
    if metadata_file:
        user_configs["training"]["metainfo"] = metadata_file
    load_user_configs(user_configs, system_config)


ACTIONS_MAP = np.array(["Other", "Rearing", "Interacting"], dtype="<U32")
GROUP_ACTIONS_MAP = np.array(["Interacting"], dtype="<U32")


def gmart_predictions(class_logits, social_logits, edge_probs):
    """Mimic the shapes GMART.predict returns: the batch axis is already stripped."""
    return GMARTPredictions(
        class_logits.squeeze(0),
        torch.randn(1, class_logits.shape[1], 8),
        edge_probs.squeeze(0),
        social_logits.squeeze(0),
    )


def data_sample_of(instances_id):
    return {
        "pred_track_instances": {
            "instances_id": np.asarray(instances_id),
            "valid_action_recognition_context": torch.ones(len(instances_id), dtype=torch.bool),
        }
    }


@pytest.mark.parametrize("nb_subjects", [0, 1, 2, 3])
def test_postprocessing_is_shape_stable(nb_subjects):
    """A single tracked subject must not collapse the instance axis (see the double-squeeze regression)."""
    preds = gmart_predictions(
        class_logits=torch.softmax(torch.randn(1, nb_subjects, len(ACTIONS_MAP)), dim=-1),
        social_logits=torch.softmax(torch.randn(1, nb_subjects, len(GROUP_ACTIONS_MAP) + 1), dim=-1),
        edge_probs=torch.rand(1, nb_subjects, nb_subjects),
    )
    data_sample = data_sample_of(range(nb_subjects))

    out = postprocess_fpv_action_recognition(
        preds,
        data_sample,
        ACTIONS_MAP,
        group_actions_map=GROUP_ACTIONS_MAP,
        null_action="Other",
    )["pred_track_instances"]

    assert len(out["actions"]) == nb_subjects
    assert len(out["action_scores"]) == nb_subjects
    assert len(out["target_ids"]) == nb_subjects
    assert out["action_embeddings"].shape[0] == nb_subjects


def test_empty_postprocessing_matches_the_regular_tail():
    """Frames without any subject bypass the model, but must expose the same fields."""
    empty = empty_fpv_action_recognition(data_sample_of([]), ACTIONS_MAP.dtype, with_target_ids=True)["pred_track_instances"]

    preds = gmart_predictions(
        class_logits=torch.empty(1, 0, len(ACTIONS_MAP)),
        social_logits=torch.empty(1, 0, len(GROUP_ACTIONS_MAP) + 1),
        edge_probs=torch.empty(1, 0, 0),
    )
    postprocessed = postprocess_fpv_action_recognition(
        preds,
        data_sample_of([]),
        ACTIONS_MAP,
        group_actions_map=GROUP_ACTIONS_MAP,
        null_action="Other",
    )["pred_track_instances"]

    for field in ("actions", "action_scores", "target_ids"):
        assert empty[field].size == 0
        assert postprocessed[field].size == 0


def test_postprocessing_pairs_social_subjects():
    """The subject above the social threshold gets the group label and points at its partner."""
    class_logits = torch.tensor([[[0.05, 0.05, 0.9], [0.9, 0.05, 0.05]]])
    social_logits = torch.tensor([[[0.1, 0.9], [0.9, 0.1]]])
    edge_probs = torch.tensor([[[0.0, 0.9], [0.9, 0.0]]])
    data_sample = data_sample_of([10, 20])

    out = postprocess_fpv_action_recognition(
        gmart_predictions(class_logits, social_logits, edge_probs),
        data_sample,
        ACTIONS_MAP,
        group_actions_map=GROUP_ACTIONS_MAP,
        null_action="Other",
    )["pred_track_instances"]

    assert out["actions"].tolist() == ["Interacting", "Other"]
    assert out["target_ids"].tolist() == ["20", "-1"]


@pytest.mark.timeout(10 * 60)
@pytest.mark.parametrize(
    "checkpoints",
    [
        (
            os.path.join(ROOT, "configs/mart.pth"),
            os.path.join(ROOT, "configs/mart_DEPLOYED.pth"),
        )
    ],
)
def test_inference(checkpoints, config):
    config = Config.fromfile(config)
    analyzer = config["analyzer"]

    for checkpoint in checkpoints:
        analyzer["runtime"]["model"]["mart_checkpoint"] = checkpoint
        for i, pipelined in enumerate([True, False]):
            video = VideoReader(os.path.join(ROOT, "../assets/20mice_sanity_check.avi"))
            config["outputs"] = [
                dict(
                    type="CsvActions",
                    path=os.path.join(ROOT, f"work_dir/actions{i}.csv"),
                    instance_data="pred_track_instances",
                    metainfo=config["metainfo"],
                    precision=64,
                )
            ]
            if pipelined:
                mp.set_start_method("spawn", force=True)
                tracker = PipelinedTracker(
                    detector=config.get("detector"),
                    assigner=config.get("assigner"),
                    validator=config.get("validator"),
                    analyzer=config.get("analyzer"),
                    outputs=config.get("outputs"),
                    expected_resolution=(video.resolution[1], video.resolution[0], 3),
                    batch_size=config.get("batch_size"),
                    verbose=True,
                )
                tracker(video=video)
            else:
                tracker = Tracker(
                    detector=config.get("detector"),
                    assigner=config.get("assigner"),
                    validator=config.get("validator"),
                    analyzer=config.get("analyzer"),
                    outputs=config.get("outputs"),
                    batch_size=config.get("batch_size"),
                    verbose=True,
                )
                tracker(video=video)

        df0 = pd.read_csv(os.path.join(ROOT, "work_dir/actions0.csv"))
        assert not df0.empty
        df1 = pd.read_csv(os.path.join(ROOT, "work_dir/actions1.csv"))
        assert df0.equals(df1)
