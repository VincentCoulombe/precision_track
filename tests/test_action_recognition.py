import multiprocessing as mp
import os
import subprocess
import numpy as np
import pandas as pd
import pytest
import torch
import time
import json
import yaml
from pathlib import Path
from mmengine import Config
from mmengine.structures import InstanceData
from utils import temp_csv_file

from precision_track import PipelinedTracker, Tracker
from precision_track.registry import MODELS
from precision_track.utils import VideoReader, cuda_available, load_user_configs

ROOT = "./tests/"
TOOLS_DIR = "./tools"


@pytest.fixture
def config():
    return os.path.join(ROOT, "configs/action_recognition.py")


@pytest.fixture
def user_configs():
    return os.path.join(ROOT, "../configs/user_configs.yaml")


@pytest.fixture
def training_config():
    return os.path.join(ROOT, "configs/training_action_recognition.py")


@pytest.fixture
def testing_config():
    return os.path.join(ROOT, "configs/testing_action_recognition.py")


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


@pytest.mark.timeout(10 * 60)
@pytest.mark.parametrize(
    "checkpoints",
    [
        (
            os.path.join(ROOT, "configs/mart_DEPLOYED.pth"),
            os.path.join(ROOT, "configs/mart_DEPLOYED.onnx"),
            os.path.join(ROOT, "configs/mart_DEPLOYED_NVIDIAGeForceRTX3090_FP16.engine"),
        )
    ],
)
def test_inference(checkpoints, config):
    config = Config.fromfile(config)
    analyzer = config["analyzer"]

    with (
        temp_csv_file(os.path.join(ROOT, "work_dir/actions0.csv")),
        temp_csv_file(os.path.join(ROOT, "work_dir/actions1.csv")),
        temp_csv_file(os.path.join(ROOT, "work_dir/reference.csv")),
    ):
        for checkpoint in checkpoints:
            if checkpoint.endswith(".engine") and not cuda_available():
                continue
            if checkpoint.endswith(".onnx") and not cuda_available():
                checkpoint = os.path.splitext(checkpoint)[0] + "_cpu.onnx"
            analyzer["runtime"]["checkpoint"] = checkpoint
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
                if checkpoint.endswith(".pth") and not pipelined:
                    config["outputs"].append(
                        dict(
                            type="CsvActions",
                            path=os.path.join(ROOT, "work_dir/reference.csv"),
                            instance_data="pred_track_instances",
                            metainfo=config["metainfo"],
                            precision=64,
                        )
                    )
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
            df1 = pd.read_csv(os.path.join(ROOT, "work_dir/actions1.csv"))
            assert df0.equals(df1)

            if os.path.exists(os.path.join(ROOT, "work_dir/reference.csv")):
                dv_ref = pd.read_csv(os.path.join(ROOT, "work_dir/reference.csv")).values

            dv0 = df0.values
            for i in range(len(df0)):
                # If the predicted action differs, the softmax scores need to at least be close.
                if not np.all(np.equal(dv0[i, :4], dv_ref[i, :4])):
                    assert np.isclose(dv0[i, 4], dv_ref[i, 4], atol=1e-2)


@pytest.mark.parametrize(
    "predict_inputs, loss_sequence_input",
    [
        (
            [
                os.path.join(ROOT, "work_dir/1079.pth"),
                os.path.join(ROOT, "work_dir/1080.pth"),
                os.path.join(ROOT, "work_dir/1081.pth"),
                os.path.join(ROOT, "work_dir/1082.pth"),
            ],
            os.path.join(ROOT, "work_dir/1079-1082_id17.pth"),
        ),
    ],
)
def test_training_preprocessing(predict_inputs, loss_sequence_input, config):
    ar_preprocessing = MODELS.build(Config.fromfile(config)["training_data_preprocessor"])
    ar_preprocessing.block_size = 4

    inf_ar_pre_cfg = Config.fromfile(config)["analyzer"]["data_preprocessor"]
    inf_ar_pre_cfg.block_size = 3
    inf_ar_preprocessing = MODELS.build(inf_ar_pre_cfg)

    map_location = "cuda" if cuda_available() else "cpu"

    for predict_input in predict_inputs:
        loaded_ds = torch.load(predict_input, weights_only=False, map_location=torch.device(map_location))
        predict_output = inf_ar_preprocessing(loaded_ds)

    loss_sequence_input = torch.load(loss_sequence_input, weights_only=False, map_location=torch.device(map_location))
    for ds in loss_sequence_input["data_samples"]:
        ds.pred_track_instances.velocities = ds.pred_track_instances.dynamics
    loss_output = ar_preprocessing.loss(loss_sequence_input)

    assert torch.allclose(loss_output["features"][-1][1:].to(torch.float16), predict_output["features"][-1].to(torch.float16))
    assert torch.allclose(loss_output["poses"][-1][1:].to(torch.float16), predict_output["poses"][-1].to(torch.float16))
    assert torch.allclose(loss_output["dynamics"][-1][1:].to(torch.float16), predict_output["dynamics"][-1].to(torch.float16))

    loss_sequence_input["inputs"][0] = loss_sequence_input["inputs"][0].view(1, 4, 128)

    pred_track_instances = InstanceData()

    pred_track_instances.velocities = loss_sequence_input["data_samples"][0].pred_track_instances.dynamics.view(1, 4, 2)
    pred_track_instances.kpts = loss_sequence_input["data_samples"][0].pred_track_instances.kpts.view(1, 4, 8, 2)
    pred_track_instances.kpt_vis = loss_sequence_input["data_samples"][0].pred_track_instances.kpt_vis.view(1, 4, 8)
    loss_sequence_input["data_samples"][0].pred_track_instances = pred_track_instances
    sequence_output = ar_preprocessing.sequence(loss_sequence_input)

    assert torch.allclose(sequence_output["features"][-1][1:].to(torch.float16), predict_output["features"][-1].to(torch.float16))
    assert torch.allclose(sequence_output["poses"][-1][1:].to(torch.float16), predict_output["poses"][-1].to(torch.float16))
    assert torch.allclose(sequence_output["dynamics"][-1][1:].to(torch.float16), predict_output["dynamics"][-1].to(torch.float16))


# def test_training(training_config, deployed_checkpoints):
#     start_time = time.perf_counter()

#     train_config = Config.fromfile(training_config)

#     train_sequences = train_config.train_sequences
#     train_bboxes_gt_paths = train_config.train_bboxes_gt_paths
#     train_keypoints_gt_paths = train_config.train_keypoints_gt_paths
#     train_actions_gt_paths = train_config.train_actions_gt_paths

#     if (
#         os.path.isdir(train_sequences)
#         and os.path.isdir(train_bboxes_gt_paths)
#         and os.path.isdir(train_keypoints_gt_paths)
#         and os.path.isdir(train_actions_gt_paths)
#         and torch.cuda.is_available()
#     ):

#         train_ar_tool_path = os.path.join(ROOT, "..", "tools", "train_action_recognition.py")

#         result = subprocess.run(
#             ["python", train_ar_tool_path, "--test=true", "--deploy=true", f"--config={os.path.join(training_config)}"],
#             capture_output=True,
#             text=True,
#         )
#         assert result.returncode == 0, f"Training failed with: {result.stderr}"

#         with open(os.path.join(train_config.work_dir, "best_ActionRecognition_f1.json"), "r") as f:
#             obtained = json.load(f)

#         with open(os.path.join(ROOT, "work_dir", "dummy_bare_minimum_ActionRecognition_f1.json"), "r") as f:
#             bare_minimum = json.load(f)

#         for metric in obtained:
#             obtained_metric = obtained[metric]
#             bare_min_metric = bare_minimum[metric]
#             for obt_cls_metric, bare_nim_cls_metric in zip(obtained_metric, bare_min_metric):
#                 assert obt_cls_metric >= bare_nim_cls_metric, f"Metric: {metric}, Obtained: {obt_cls_metric:.4f}, Bare minimum: {bare_nim_cls_metric:.4f}."

#         deployed_dir = train_config.mart_deploying_directory

#         ckpt_names = deployed_checkpoints["checkpoint_names"]
#         ckpt_found = deployed_checkpoints["found"]
#         for i, ckpt_name in enumerate(ckpt_names):
#             file_path = os.path.join(deployed_dir, ckpt_name)
#             if os.path.exists(file_path):
#                 file_mtime = os.path.getmtime(file_path)
#                 if file_mtime >= start_time:
#                     ckpt_found[i] = True

#         assert all(ckpt_found), f"Not all checkpoints were created or updated. Found: {dict(zip(ckpt_names, ckpt_found))}"


def test_testing(testing_config, user_configs):
    with open(user_configs, "r") as f:
        user_configs = yaml.safe_load(f)
    user_configs["booleans"]["with_action_recognition"] = True
    load_user_configs(user_configs, testing_config)
    test_config = Config.fromfile(testing_config)

    ar_data_root = Path(test_config.action_recognition_data_root)
    ar_data_root = str(Path(*ar_data_root.parts[1:]))

    test_sequences = os.path.join(ar_data_root, test_config.test_sequences)
    test_bboxes_gt_paths = os.path.join(ar_data_root, test_config.test_bboxes_gt_paths)
    test_keypoints_gt_paths = os.path.join(ar_data_root, test_config.test_keypoints_gt_paths)
    test_actions_gt_paths = os.path.join(ar_data_root, test_config.test_actions_gt_paths)

    if (
        os.path.isdir(test_sequences)
        and os.path.isdir(test_bboxes_gt_paths)
        and os.path.isdir(test_keypoints_gt_paths)
        and os.path.isdir(test_actions_gt_paths)
        and torch.cuda.is_available()
    ):

        result = subprocess.run(
            ["python", "./test_action_recognition.py", f"--config={os.path.abspath(testing_config)}"],
            capture_output=True,
            text=True,
            cwd=TOOLS_DIR,
        )
        assert result.returncode == 0, f"Training failed with: {result.stderr}"

        with open(os.path.abspath(os.path.join(ROOT, test_config.work_dir, "best_ActionRecognition_f1.json")), "r") as f:
            obtained = json.load(f)

        with open(os.path.abspath(os.path.join(ROOT, "work_dir", "bare_minimum_ActionRecognition_f1.json")), "r") as f:
            bare_minimum = json.load(f)

        for metric in obtained:
            obtained_metric = obtained[metric]
            bare_minimum_metric = bare_minimum[metric]
            assert obtained_metric >= bare_minimum_metric, f"Metric: {metric}, Obtained: {obtained_metric:.4f}, Bare minimum: {bare_minimum_metric:.4f}."


if __name__ == "__main__":
    pytest.main(["-x", os.path.realpath(__file__), "-s"])
