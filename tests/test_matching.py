import numpy as np
import pytest

from precision_track.utils import batch_bbox_areas, iou_batch, reformat


@pytest.fixture
def expected_iou_batch_xyxy():

    def _expected_iou_batch_xyxy(a, b, general=False, eps=1e-6):
        a_exp = np.expand_dims(a, 1)
        b_exp = np.expand_dims(b, 0)

        xx1 = np.maximum(a_exp[..., 0], b_exp[..., 0])
        yy1 = np.maximum(a_exp[..., 1], b_exp[..., 1])
        xx2 = np.minimum(a_exp[..., 2], b_exp[..., 2])
        yy2 = np.minimum(a_exp[..., 3], b_exp[..., 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        b_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = a_area[:, np.newaxis] + b_area - intersection

        iou = intersection / (union + eps)

        if general:
            enclose_x1 = np.minimum(a_exp[..., 0], b_exp[..., 0])
            enclose_y1 = np.minimum(a_exp[..., 1], b_exp[..., 1])
            enclose_x2 = np.maximum(a_exp[..., 2], b_exp[..., 2])
            enclose_y2 = np.maximum(a_exp[..., 3], b_exp[..., 3])
            enclose_w = np.maximum(0.0, enclose_x2 - enclose_x1)
            enclose_h = np.maximum(0.0, enclose_y2 - enclose_y1)
            enclose_area = enclose_w * enclose_h
            giou = iou - (enclose_area - union) / (enclose_area + eps)
            normalize_giou = (giou + 1) / 2
            return normalize_giou
        else:
            return iou

    return _expected_iou_batch_xyxy


def test_iou_batch(expected_iou_batch_xyxy):
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    b = np.array([[0, 0, 10, 10], [10, 10, 20, 20]])
    expected_iou = expected_iou_batch_xyxy(a, b)
    result_iou = iou_batch(reformat(a, "xyxy", "cxcywh"), reformat(b, "xyxy", "cxcywh"))
    np.testing.assert_allclose(result_iou, expected_iou, rtol=1e-5)

    a = np.array([[0, 0, 1, 1], [1, 1, 2, 2]])
    b = np.array([[2, 2, 3, 3], [3, 3, 4, 4]])
    expected_iou = expected_iou_batch_xyxy(a, b)
    result_iou = iou_batch(reformat(a, "xyxy", "cxcywh"), reformat(b, "xyxy", "cxcywh"))
    np.testing.assert_allclose(result_iou, expected_iou, rtol=1e-5)

    a = np.array([[0, 0, 2, 2], [1, 1, 3, 3]])
    b = np.array([[1, 1, 3, 3], [2, 2, 4, 4]])
    expected_iou = expected_iou_batch_xyxy(a, b)
    result_iou = iou_batch(reformat(a, "xyxy", "cxcywh"), reformat(b, "xyxy", "cxcywh"))
    np.testing.assert_allclose(result_iou, expected_iou, rtol=1e-5)
    a = np.array([[0, 0, 2, 2], [1, 1, 3, 3]])
    b = np.array([[1, 1, 3, 3], [2, 2, 4, 4]])

    expected_iou = expected_iou_batch_xyxy(a, b, general=True)
    result_iou = iou_batch(
        reformat(a, "xyxy", "cxcywh"),
        reformat(b, "xyxy", "cxcywh"),
        general=True,
    )
    np.testing.assert_allclose(result_iou, expected_iou, rtol=1e-5)


def test_batch_bbox_areas():
    bboxes = np.array([[0, 0, 10, 10], [20, 20, 5, 5]])
    expected_areas = np.array([100, 25])
    result_areas = batch_bbox_areas(bboxes)
    np.testing.assert_array_equal(result_areas, expected_areas)

    bboxes = np.array([[0, 0, 0, 10], [20, 20, 5, 0]])
    expected_areas = np.array([0, 0])
    result_areas = batch_bbox_areas(bboxes)
    np.testing.assert_array_equal(result_areas, expected_areas)

    bboxes = np.array([[0, 0, -10, 10], [20, 20, 5, -5]])
    result_areas = batch_bbox_areas(bboxes)
    assert result_areas is None

    bboxes = np.array([[0, 0, 10, 10], [20, 20, -5, 5]])
    result_areas = batch_bbox_areas(bboxes)
    assert result_areas is None

    bboxes = np.array([[0, 0, 10000, 10000], [20, 20, 5000, 5000]])
    expected_areas = np.array([100000000, 25000000])
    result_areas = batch_bbox_areas(bboxes)
    np.testing.assert_array_equal(result_areas, expected_areas)


if __name__ == "__main__":
    pytest.main()
