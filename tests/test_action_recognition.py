import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import pytest
import torch
from mmengine import Config
from mmengine.structures import InstanceData
from utils import temp_csv_file

from precision_track import PipelinedTracker, Tracker
from precision_track.registry import MODELS
from precision_track.utils import VideoReader, cuda_available

ROOT = "./tests/"


@pytest.fixture
def config():
    return os.path.join(ROOT, "configs/action_recognition.py")


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
                # If the predicted action differs, at least the softmax scores are close, meaning the error is due to runtime conversion.
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
def test_preprocessing(predict_inputs, loss_sequence_input, config):
    ar_preprocessing = MODELS.build(Config.fromfile(config)["analyzer"]["data_preprocessor"])
    ar_preprocessing.block_size = 4

    map_location = "cuda" if cuda_available() else "cpu"

    for predict_input in predict_inputs:
        predict_output = ar_preprocessing.predict(torch.load(predict_input, weights_only=False, map_location=torch.device(map_location)))

    loss_sequence_input = torch.load(loss_sequence_input, weights_only=False, map_location=torch.device(map_location))
    loss_output = ar_preprocessing.loss(loss_sequence_input)

    assert torch.allclose(loss_output["features"][-1].to(torch.float16), predict_output["features"][-1].to(torch.float16))
    assert torch.allclose(loss_output["poses"][-1].to(torch.float16), predict_output["poses"][-1].to(torch.float16))
    assert torch.allclose(loss_output["dynamics"][-1].to(torch.float16), predict_output["dynamics"][-1].to(torch.float16))

    loss_sequence_input["inputs"][0] = loss_sequence_input["inputs"][0].view(1, 4, 128)

    pred_track_instances = InstanceData()

    pred_track_instances.dynamics = loss_sequence_input["data_samples"][0].pred_track_instances.dynamics.view(1, 4, 2)
    pred_track_instances.kpts = loss_sequence_input["data_samples"][0].pred_track_instances.kpts.view(1, 4, 8, 2)
    pred_track_instances.kpt_vis = loss_sequence_input["data_samples"][0].pred_track_instances.kpt_vis.view(1, 4, 8)
    loss_sequence_input["data_samples"][0].pred_track_instances = pred_track_instances
    sequence_output = ar_preprocessing.sequence(loss_sequence_input)

    assert torch.allclose(sequence_output["features"][-1].to(torch.float16), predict_output["features"][-1].to(torch.float16))
    assert torch.allclose(sequence_output["poses"][-1].to(torch.float16), predict_output["poses"][-1].to(torch.float16))
    assert torch.allclose(sequence_output["dynamics"][-1].to(torch.float16), predict_output["dynamics"][-1].to(torch.float16))


if __name__ == "__main__":
    pytest.main()
