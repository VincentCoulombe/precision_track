import os

import numpy as np
import pytest
import torch
from mmengine import Config

from precision_track.apis import Tracker
from precision_track.outputs import NpyEmbeddingOutput
from precision_track.utils import VideoReader, cuda_available

SLIDING_WINDOW_SIZE = 30
ROOT = "./tests/"


def all_equal(arr):
    return np.all(arr == arr.flat[0])


@pytest.fixture
def tracking_cfg():
    return Config.fromfile(os.path.join(ROOT, "configs/gt.py"))


@pytest.mark.parametrize(
    "video, bboxes_path",
    [
        (os.path.join(ROOT, "../assets/20mice_sanity_check.avi"), os.path.join(ROOT, "work_dir/20mice_sanity_check.csv")),
    ],
)
def test_call(video, bboxes_path, tracking_cfg):
    if not not cuda_available():
        return
    tracking_cfg["assigner"]["tracking_algorithm"]["gt_bbox_path"] = bboxes_path
    tracking_cfg["analyzer"] = None
    tracker = Tracker(**tracking_cfg)

    assert len(tracker.result.outputs) == 1 and isinstance(tracker.result.outputs[0], NpyEmbeddingOutput)
    assert len(tracker.result.outputs[0]) == 0

    unique_ids_encountered = list()
    reader = VideoReader(video)
    features_size = None
    past_features = []
    for i, frame in enumerate(reader):
        output = tracker.detector(inputs=[frame], data_samples=[i])
        output = tracker.association_step(output[0], None)
        assert "features" in output["pred_track_instances"]
        curr_f_size = output["pred_track_instances"]["features"].shape[1]
        if features_size is None and curr_f_size > 1:
            features_size = curr_f_size
        tracker.result(output)

        ids = output["pred_track_instances"]["ids"]
        new_ids_encountered = set(ids.tolist()) - set(unique_ids_encountered)
        unique_ids_encountered.extend(new_ids_encountered)

        ids_slice = tracker.result.outputs[0][-1]
        assert ids_slice.shape == (len(unique_ids_encountered), features_size)
        for id_row in ids_slice:
            assert all_equal(id_row)
            assert id_row[0] in unique_ids_encountered

        curr_slice = tracker.result.outputs[0][i]
        features = output["pred_track_instances"]["features"].cpu().numpy()
        for id_ in unique_ids_encountered:
            if id_ in ids.tolist():
                id_features = features[ids == id_]
            else:
                id_features = np.zeros((1, features_size))
            id_idx = (ids_slice == id_)[:, 0]
            assert np.allclose(curr_slice[id_idx, :], id_features)

        for j in range(i):
            assert np.allclose(tracker.result.outputs[0][j].astype(np.float16), past_features[j])

        past_features.append(curr_slice.astype(np.float16))


if __name__ == "__main__":
    pytest.main()
