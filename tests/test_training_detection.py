import json
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

ROOT = "./tests/"
TOOLS_DIR = "./tools"


@pytest.fixture
def user_configs():
    with open(os.path.join(ROOT, "../configs/user_configs.yaml"), "r") as f:
        user_configs = yaml.safe_load(f)
    return user_configs


@pytest.fixture
def metrics():
    return [
        "Metric",
        "Recall",
        "Precision",
        "F1 Score",
        "Assignation's OKS",
        "Assignation's PCK:0.2",
        "Assignation's PCK:0.05",
        "Assignation's Keypoints AUC",
    ]


@pytest.fixture
def checkpoints():
    ckpt_names = [
        "model_mice_DEPLOYED.pth",
        "model_mice_DEPLOYED_NVIDIAGeForceRTX3090_FP16.engine",
        "model_mice_DEPLOYED.onnx",
    ]
    return dict(
        checkpoint_names=ckpt_names,
        found=[False for _ in ckpt_names],
    )


@pytest.fixture
def hyperparameters():
    return "hyperparameters.json"


def test_training_detection(user_configs, metrics, checkpoints, hyperparameters):
    start_time = time.perf_counter()
    data_root = user_configs["training"]["data_root"]
    data_root = Path(data_root)
    data_root = str(Path(*data_root.parts[1:]))

    if os.path.isdir(data_root) and torch.cuda.is_available():
        train_det_tool_path = os.path.abspath(os.path.join(ROOT, "..", "tools", "train_detection.py"))
        result = subprocess.run(
            ["python", train_det_tool_path, "--test=true", "--format_dataset=true", "--calibrate=true", "--deploy=true", "--optimize_hyperparams=true"],
            capture_output=True,
            text=True,
            cwd=TOOLS_DIR,
        )
        assert result.returncode == 0, f"Training failed with: {result.stderr}"

        obtained_pt_metrics = pd.read_csv(os.path.join(ROOT, "..", "work_dir", "testing_runs", "mice", "pose-detection_metrics.csv"))

        bare_minimum_pt_metrics = pd.read_csv(os.path.join(ROOT, "work_dir", "pose-detection_metrics.csv"))
        for metric in metrics:
            obtained_metric = obtained_pt_metrics[metric]
            bare_min_metric = bare_minimum_pt_metrics[metric]
            for obt_cls_metric, bare_nim_cls_metric in zip(obtained_metric, bare_min_metric):
                assert obt_cls_metric >= bare_nim_cls_metric, f"Metric: {metric}, Obtained: {obt_cls_metric:.4f}, Bare minimum: {bare_nim_cls_metric:.4f}."

        deployed_dir = user_configs["training"]["deploying_directory"]
        deployed_dir = Path(deployed_dir)
        deployed_dir = str(Path(*deployed_dir.parts[1:]))

        obtained_hyperparameters_path = os.path.abspath(os.path.join(deployed_dir, hyperparameters))
        expected_hyperparameters_path = os.path.abspath(os.path.join(ROOT, "configs", "hyperparameters.json"))

        with open(expected_hyperparameters_path, "r") as f:
            expected_hyperparams = json.load(f)
        with open(obtained_hyperparameters_path, "r") as f:
            obtained_hyperparams = json.load(f)

        for key in expected_hyperparams:
            assert key in obtained_hyperparams

        ckpt_names = checkpoints["checkpoint_names"]
        ckpt_found = checkpoints["found"]
        for i, ckpt_name in enumerate(ckpt_names):
            file_path = os.path.join(deployed_dir, ckpt_name)
            if os.path.exists(file_path):
                file_mtime = os.path.getmtime(file_path)
                if file_mtime >= start_time:
                    ckpt_found[i] = True

        assert all(ckpt_found), f"Not all checkpoints were created or updated. Found: {dict(zip(ckpt_names, ckpt_found))}"


if __name__ == "__main__":
    pytest.main(["-x", os.path.realpath(__file__), "-v", "-s"])
