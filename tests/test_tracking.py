import json
import os

import pytest
import torch
from mmengine import Config

from precision_track import PipelinedTracker, Runner, Tracker
from precision_track.utils import cuda_available

ROOT = "./tests/"


@pytest.fixture
def config():
    return os.path.join(ROOT, "configs/tracking.py")


@pytest.fixture
def expected_results_path():
    return os.path.join(ROOT, "work_dir/expected_results.json")


@pytest.mark.parametrize(
    "checkpoints",
    [
        (
            os.path.join(ROOT, "configs/yolox-pose_s_clustering_DEPLOYED.pth"),
            os.path.join(ROOT, "configs/yolox-pose_s_clustering_DEPLOYED.onnx"),
            os.path.join(ROOT, "configs/yolox-pose_s_clustering_DEPLOYED_NVIDIAGeForceRTX3090_FP16.engine"),
        )
    ],
)
def test_tracking(checkpoints, expected_results_path, config):
    with open(expected_results_path, "r") as f:
        expected_results = json.load(f)

    cfg = Config.fromfile(config)
    detector = cfg["detector"]

    for checkpoint in checkpoints:
        if checkpoint.endswith(".engine") and not cuda_available():
            continue
        if checkpoint.endswith(".onnx") and not cuda_available():
            checkpoint = os.path.splitext(checkpoint)[0] + "_cpu.onnx"
        detector["runtime"]["checkpoint"] = checkpoint
        for pipelined in [True, False]:
            runner = Runner(config, "none", mode="test")
            if pipelined:
                runner.test_loop.tracker = PipelinedTracker
            else:
                runner.test_loop.tracker = Tracker
            obtained_metrics = runner.test_loop.run()

            for k, v in expected_results["mouse"].items():
                t_v = obtained_metrics[f"CLEAR/mouse/{k}"]
                max_diff = 1 if t_v > 1 else 1e-2
                assert v == pytest.approx(t_v, max_diff)


if __name__ == "__main__":
    pytest.main()
