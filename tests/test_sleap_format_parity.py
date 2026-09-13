"""Guards ``sleap/pt_format.py`` against drifting from ``precision_track/outputs/csv.py``.

The SLEAP pipeline lives in its own environment and cannot import ``precision_track``,
so it ships vendored copies of ``CsvKeypoints`` and ``CsvBoundingBoxes``. This test
feeds one identical data sample to both implementations and asserts the resulting
dataframes are indistinguishable.

The twin of ``tests/test_dlc_format_parity.py``: each benchmark directory keeps its own
copy of the writers, so each needs its own guard.
"""

import importlib.util
import os.path as osp
import sys

import numpy as np
import pytest

from precision_track.outputs.csv import CsvBoundingBoxes, CsvKeypoints

SLEAP_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "sleap")


def _load_vendored():
    spec = importlib.util.spec_from_file_location("sleap_pt_format", osp.join(SLEAP_DIR, "pt_format.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def vendored():
    return _load_vendored()


@pytest.fixture
def data_samples():
    """Two frames of three tracked instances with five keypoints each."""
    rng = np.random.default_rng(0)
    samples = []
    for frame_id in range(2):
        keypoints = rng.uniform(0, 640, size=(3, 5, 2))
        keypoint_scores = rng.uniform(0, 1, size=(3, 5))
        bboxes = np.stack(
            [
                np.array(
                    [
                        (kpt[:, 0].min() + kpt[:, 0].max()) / 2,
                        (kpt[:, 1].min() + kpt[:, 1].max()) / 2,
                        kpt[:, 0].max() - kpt[:, 0].min(),
                        kpt[:, 1].max() - kpt[:, 1].min(),
                    ]
                )
                for kpt in keypoints
            ]
        )
        samples.append(
            {
                "img_id": frame_id,
                "pred_track_instances": dict(
                    labels=np.zeros(3, dtype=np.int64),
                    instances_id=np.arange(3, dtype=np.int64),
                    bboxes=bboxes,
                    scores=rng.uniform(0.5, 1.0, size=3),
                    keypoints=keypoints,
                    keypoint_scores=keypoint_scores,
                ),
            }
        )
    return samples


def test_keypoints_parity(tmp_path, vendored, data_samples):
    reference = CsvKeypoints(path=str(tmp_path / "ref_kpts.csv"), instance_data="pred_track_instances", precision=32)
    candidate = vendored.CsvKeypoints(path=str(tmp_path / "sleap_kpts.csv"), instance_data="pred_track_instances", precision=32)

    for sample in data_samples:
        reference(sample)
        candidate(sample)

    expected, actual = reference.to_dataframe(), candidate.to_dataframe()
    assert list(actual.columns) == list(expected.columns)
    assert actual.dtypes.tolist() == expected.dtypes.tolist()
    assert actual.equals(expected)
    assert list(expected.columns[:6]) == ["frame_id", "class_id", "instance_id", "x0", "y0", "score0"]


@pytest.mark.parametrize("save_bbox_format", [None, ["x", "y", "w", "h"]])
def test_bounding_boxes_parity(tmp_path, vendored, data_samples, save_bbox_format):
    kwargs = dict(instance_data="pred_track_instances", precision=64, save_bbox_format=save_bbox_format)
    reference = CsvBoundingBoxes(path=str(tmp_path / "ref_bboxes.csv"), subtype="tracked_bboxes", **kwargs)
    candidate = vendored.CsvBoundingBoxes(path=str(tmp_path / "sleap_bboxes.csv"), subtype="tracked_bboxes", **kwargs)

    for sample in data_samples:
        reference(sample)
        candidate(sample)

    expected, actual = reference.to_dataframe(), candidate.to_dataframe()
    assert list(actual.columns) == list(expected.columns)
    assert actual.dtypes.tolist() == expected.dtypes.tolist()
    assert actual.equals(expected)


def test_scale_parity(tmp_path, vendored, data_samples):
    reference = CsvKeypoints(path=str(tmp_path / "ref_scaled.csv"), instance_data="pred_track_instances", precision=32)
    candidate = vendored.CsvKeypoints(path=str(tmp_path / "sleap_scaled.csv"), instance_data="pred_track_instances", precision=32)

    for sample in data_samples:
        reference(sample)
        candidate(sample)

    reference.scale((640, 640), (1536, 1536))
    candidate.scale((640, 640), (1536, 1536))
    assert candidate.to_dataframe().equals(reference.to_dataframe())


def test_vendored_bbox_derivation_matches_precision_track(vendored):
    from precision_track.utils.formatting import keypoints_cxcywh

    keypoints = np.array([[10.0, 20.0], [30.0, 60.0], [np.nan, np.nan]])
    assert np.allclose(vendored.keypoints_cxcywh(keypoints), keypoints_cxcywh(keypoints))


def test_vendored_copies_agree_with_each_other():
    """The dlc/ and sleap/ copies must stay identical to each other, not just to the source."""
    dlc_path = osp.join(osp.dirname(SLEAP_DIR), "dlc", "pt_format.py")
    if not osp.isfile(dlc_path):
        pytest.skip("dlc/pt_format.py is not present")

    def code_only(path):
        with open(path, "r") as f:
            source = f.read()
        # Drop the module docstring, which names its own directory on purpose.
        return source.split('"""', 2)[-1]

    assert code_only(dlc_path) == code_only(osp.join(SLEAP_DIR, "pt_format.py"))
