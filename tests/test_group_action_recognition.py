import multiprocessing as mp
import os

import pandas as pd
import pytest
import yaml
from mmengine import Config

from precision_track import PipelinedTracker, Tracker
from precision_track.utils import VideoReader, load_user_configs

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
