import cv2
import os
import torch
import pytest
from mmengine import Config

from precision_track import Tracker
from precision_track.utils import reformat

ROOT = os.path.join(os.getcwd(), "tests")


@pytest.fixture
def config():
    return os.path.join(ROOT, "configs/tracking.py")


@pytest.fixture
def input_():
    return cv2.imread(os.path.join(ROOT, "..", "assets/0000004668.jpg"))


@pytest.mark.parametrize(
    "checkpoints",
    [
        (
            os.path.join(ROOT, "configs/model_mice_clustering_DEPLOYED.pth"),
            os.path.join(ROOT, "configs/model_mice_clustering_DEPLOYED.onnx"),
        )
    ],
)
def test_feature_map_coherance(checkpoints, config, input_):
    config = Config.fromfile(config)
    for checkpoint in checkpoints:
        if checkpoint.endswith(".pth"):
            config["detector"]["runtime"]["checkpoint"] = checkpoint
            tracker = Tracker(
                detector=config.get("detector"),
                assigner=config.get("assigner"),
                validator=None,
                analyzer=None,
                outputs=config.get("outputs"),
                batch_size=config.get("batch_size"),
                verbose=True,
            )

            detector = tracker.detector
            detector.runtime.model.feature_extraction_head = None

            head_output = detector.predict([input_], [0])
            feature_maps = head_output[0]["pred_instances"]["feature_maps"]

            if checkpoint.endswith(".pth"):
                detector.runtime.model.head = None
                reference_feature_map = detector.tensor([input_], [0])["features"]
                reference_feature_map = [x.permute(0, 2, 3, 1).flatten(1, 2) for x in reference_feature_map]
                reference_feature_map = torch.cat(reference_feature_map, dim=1)[0, ...]

            assert torch.allclose(feature_maps, reference_feature_map)


if __name__ == "__main__":
    pytest.main(["-x", os.path.realpath(__file__)])
