import numpy as np
import pytest
import torch

from precision_track.utils import clip, reformat


@pytest.fixture
def dummy_cxcywh():
    return np.array(
        [
            [5, 5, 5, 10],
            [25, 15, 5, 5],
            [25, 25, 100, 100],
            [0, 0, 10, 10],
            [50, 50, 10, 10],
            [25, 0, 50, 50],
            [0, 25, 50, 50],
            [7, 12.5, 16, 25],
            [40, 37.5, 30, 25],
            [12.5, 5, 15, 20],
            [50, 30, 40, 20],
            [2, 2, 2, 2],
            [3, 3, 2, 2],
            [1, 1, 2, 2],
        ],
        np.float32,
    )


@pytest.fixture
def dummy_clipped_cxcywh():
    return np.array(
        [
            [5, 5, 5, 10],
            [25, 15, 5, 5],
            [25, 25, 50, 50],
            [0, 0, 0, 0],
            [50, 50, 0, 0],
            [25, 0, 50, 0],
            [0, 25, 0, 50],
            [6.5, 12.5, 13, 25],
            [40, 37.5, 20, 25],
            [12.5, 5, 15, 10],
            [50, 30, 0, 20],
            [2, 2, 2, 2],
            [3, 3, 2, 2],
            [1, 1, 2, 2],
        ],
        np.float32,
    )


@pytest.fixture
def dummy_xyxy():
    return torch.tensor(
        [
            [2.5, 0, 7.5, 10],
            [22.5, 12.5, 27.5, 17.5],
            [-25, -25, 75, 75],
            [-5, -5, 5, 5],
            [45, 45, 55, 55],
            [0, -25, 50, 25],
            [-25, 0, 25, 50],
            [-1, 0, 15, 25],
            [25, 25, 55, 50],
            [5, -5, 20, 15],
            [30, 20, 70, 40],
            [1, 1, 3, 3],
            [2, 2, 4, 4],
            [0, 0, 2, 2],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def dummy_cxcyah():
    return np.array(
        [
            [5, 5, 0.5, 10],
            [25, 15, 1, 5],
            [25, 25, 1, 100],
            [0, 0, 1, 10],
            [50, 50, 1, 10],
            [25, 0, 1, 50],
            [0, 25, 1, 50],
            [6.5, 12.5, 0.6, 25],
            [40, 37.5, 1.2, 25],
            [12.5, 5, 0.75, 20],
            [50, 30, 2, 20],
            [2, 2, 1, 2],
            [3, 3, 1, 2],
            [1, 1, 1, 2],
        ],
        np.float32,
    )


@pytest.fixture
def dummy_xywh_1d():
    return np.array([22.5, 12.5, 5, 5], np.float32)


@pytest.fixture
def dummy_cxcywh_1d():
    return np.array([25, 15, 5, 5], np.float32)


@pytest.fixture
def dummy_keypoints():
    return np.array(
        [22.5, 12.5, 27.5, 17.5, 25, 15],
        np.float32,
    ).reshape(3, 2)


@pytest.mark.parametrize(
    "old_format, new_format, instance_fixture, expected_fixture",
    [
        ("cxcywh", "xyxy", "dummy_cxcywh", "dummy_xyxy"),
        ("xyxy", "cxcywh", "dummy_xyxy", "dummy_cxcywh"),
        ("cxcywh", "cxcyah", "dummy_cxcywh", "dummy_cxcyah"),
        ("cxcyah", "cxcywh", "dummy_cxcyah", "dummy_cxcywh"),
        ("cxcywh", "xywh", "dummy_cxcywh_1d", "dummy_xywh_1d"),
        ("keypoints", "cxcywh", "dummy_keypoints", "dummy_cxcywh_1d"),
    ],
)
def test_reformat(old_format, new_format, instance_fixture, expected_fixture, request):
    instance = request.getfixturevalue(instance_fixture)
    expected = request.getfixturevalue(expected_fixture)
    reformatted = reformat(instance, old_format, new_format)
    assert isinstance(reformatted, (np.ndarray, torch.Tensor))
    if isinstance(reformatted, torch.Tensor):
        reformatted = reformatted.numpy()
    np.testing.assert_array_almost_equal(reformatted, expected, 1e-2)
    if (old_format == "cxcywh" and new_format == "xyxy") or (old_format == "cxcywh" and new_format == "cxcyah"):
        back_instances = reformat(reformatted, new_format, old_format)
        np.testing.assert_array_almost_equal(back_instances, instance, 1e-1)


@pytest.mark.parametrize(
    "format, max_width, max_height, instance_fixture, expected_fixture",
    [
        ("cxcywh", "50", "50", "dummy_cxcywh", "dummy_clipped_cxcywh"),
    ],
)
def test_clip(format, max_width, max_height, instance_fixture, expected_fixture, request):
    expected = request.getfixturevalue(expected_fixture)
    instance = request.getfixturevalue(instance_fixture)
    clipped = np.empty_like(expected)
    for i, inst in enumerate(instance):
        clipped[i] = clip(inst, format, int(max_width), int(max_height))
    assert isinstance(clipped, np.ndarray)
    assert clipped.shape == instance.shape
    np.testing.assert_array_almost_equal(clipped, expected, 1e-2)

    torch_instance = torch.from_numpy(instance)
    clipped = torch.empty_like(torch.from_numpy(expected))
    for i, inst in enumerate(torch_instance):
        clipped[i] = clip(inst, format, int(max_width), int(max_height))
    assert isinstance(clipped, torch.Tensor)
    assert clipped.shape == torch_instance.shape
    np.testing.assert_array_almost_equal(clipped.numpy(), expected, 1e-2)


if __name__ == "__main__":
    pytest.main()
