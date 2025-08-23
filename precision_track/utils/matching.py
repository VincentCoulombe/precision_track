from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit
from scipy.optimize import linear_sum_assignment

from .formatting import xyxy_cxcywh
from .structures import PoseDataSample


def _assert_kpts_input(a_keypoints, b_keypoints, symmetric=False):
    if len(a_keypoints.shape) != 3:
        raise ValueError(f"Input arrays must have three dimensions, not {len(a_keypoints.shape)}.")
    if symmetric and a_keypoints.shape != b_keypoints.shape:
        raise ValueError(f"Both input arrays must have the same shape, {a_keypoints.shape} != {b_keypoints.shape}.")


def _assert_bboxes_input(bboxes):
    if len(bboxes.shape) != 2:
        raise ValueError(f"Input array must have 2 dimensions, not {len(bboxes.shape)}")
    if bboxes.shape[1] != 4:
        raise ValueError(f"Input array must contains either xywh or cxcywh bounding boxes coordinates, meaning 4 parameters, not {bboxes.shape[1]}.")


def biou_batch(tracks, detections, scales, general=False):
    """scales should be values between 0 and 1"""
    if isinstance(scales, np.ndarray) and scales.ndim == 1:
        scales = np.expand_dims(scales, axis=1)
    detections[:, 2:] += 2 * scales * detections[:, 2:]
    return iou_batch(tracks, detections, general)


@njit()
def iou_batch(a, b, general=False, eps=1e-6):
    n = a.shape[0]
    m = b.shape[0]
    result = np.empty((n, m), dtype=np.float32)

    for i in range(n):
        for j in range(m):
            acx, acy, aw, ah = a[i, 0], a[i, 1], a[i, 2], a[i, 3]
            bcx, bcy, bw, bh = b[j, 0], b[j, 1], b[j, 2], b[j, 3]
            ax1 = acx - aw / 2
            ay1 = acy - ah / 2
            ax2 = acx + aw / 2
            ay2 = acy + ah / 2
            bx1 = bcx - bw / 2
            by1 = bcy - bh / 2
            bx2 = bcx + bw / 2
            by2 = bcy + bh / 2
            xx1 = max(ax1, bx1)
            yy1 = max(ay1, by1)
            xx2 = min(ax2, bx2)
            yy2 = min(ay2, by2)
            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            intersection = w * h
            union = aw * ah + bw * bh - intersection
            iou = intersection / (union + eps)
            if general:
                enclose_x1 = min(ax1, bx1)
                enclose_y1 = min(ay1, by1)
                enclose_x2 = max(ax2, bx2)
                enclose_y2 = max(ay2, by2)
                enclose_w = max(0.0, enclose_x2 - enclose_x1)
                enclose_h = max(0.0, enclose_y2 - enclose_y1)
                enclose_area = enclose_w * enclose_h
                giou = iou - (enclose_area - union) / (enclose_area + eps)
                normalize_giou = (giou + 1) / 2  # min/max normalization
                result[i, j] = normalize_giou
            else:
                result[i, j] = iou
    return result


@njit
def filter_matches(x, y, cost_matrix, thresh):
    num_matches = 0
    for i in range(len(x)):
        if cost_matrix[x[i], y[i]] <= thresh:
            num_matches += 1

    if num_matches == 0:
        return np.empty(0), np.empty(0)

    thr_x = np.empty(num_matches, dtype=np.float64)
    thr_y = np.empty(num_matches, dtype=np.float64)
    index = 0
    for i in range(len(x)):
        if cost_matrix[x[i], y[i]] <= thresh:
            thr_x[index] = x[i]
            thr_y[index] = y[i]
            index += 1

    return thr_x, thr_y


def batch_bbox_areas(bboxes):
    if bboxes.shape[0] == 0:
        return np.zeros(0, dtype=bboxes.dtype)
    _assert_bboxes_input(bboxes)
    return _batch_bbox_areas_numba(bboxes)


@njit
def _batch_bbox_areas_numba(bboxes):
    """Calculate the areas of a batch of bounding boxes given in xyah format.

    Args:
    bboxes (np.ndarray): Array of bounding boxes, shape (B, 4) where B is the batch size and
                         each bounding box can defined as [x, y, w, h] or [cx, cy, w, h].

    Returns:
    np.ndarray: Array of areas for each bounding box in the batch.
    """
    B = bboxes.shape[0]
    areas = np.empty(B, dtype=bboxes.dtype)

    for i in range(B):
        if bboxes[i, 2] < 0 or bboxes[i, 3] < 0:
            return
        areas[i] = bboxes[i, 2] * bboxes[i, 3]

    return areas


def oks_batch(
    true_keypoints,
    pred_keypoints,
    areas,
    keypoint_std=None,
    keypoint_weights=None,
    eps=1e-3,
):
    if true_keypoints.shape[0] == 0 or pred_keypoints.shape[0] == 0:
        return np.zeros(0)
    _assert_kpts_input(true_keypoints, pred_keypoints)
    return _oks_batch_numba(
        true_keypoints,
        pred_keypoints,
        areas,
        keypoint_std,
        keypoint_weights,
        eps,
    )


@njit
def _oks_batch_numba(
    true_keypoints,
    pred_keypoints,
    areas,
    keypoint_std=None,
    keypoint_weights=None,
    eps=1e-3,
):
    """Calculate the Object Keypoint Similarity (OKS) for each combination of
    predicted and true keypoints across different batches, optimized with
    Numba.

    Args:
    true_keypoints (np.ndarray): True keypoints, shape (B1, N, 2).
    pred_keypoints (np.ndarray): Predicted keypoints, shape (B2, N, 2).
    areas (np.ndarray): Areas of the true bounding boxes, shape (B1).
    keypoint_std (np.ndarray): Standard deviations for each keypoint, shape (N,).
    keypoint_weights (np.ndarray): Weights for each predicted keypoints, shape (B2, N).

    Returns:
    np.ndarray: Array of normalized OKS scores for each combination of true and predicted instances, shape (B1, B2).
    """
    B1, N, _ = true_keypoints.shape
    B2, _, _ = pred_keypoints.shape
    scores = np.zeros((B1, B2), dtype=np.float32)
    if keypoint_std is None:
        keypoint_std = np.ones(N, dtype=np.float32)
    if keypoint_weights is None:
        keypoint_weights = np.ones((B2, N), dtype=np.float32)

    for i in range(B1):
        area_scale = np.sqrt(areas[i]) if areas is not None else 1.0
        for j in range(B2):
            nb_kpts_counted = 0
            for n in range(N):
                if np.all(pred_keypoints[j][n] > 0.0) and np.all(true_keypoints[i][n] > 0.0):
                    scores[i, j] = (
                        scores[i, j]
                        + np.exp(
                            -(
                                (
                                    np.sqrt(
                                        np.sum(
                                            (pred_keypoints[j][n] - true_keypoints[i][n]) ** 2,
                                            axis=0,
                                        )
                                    )
                                    / (area_scale + eps)
                                )
                                / (keypoint_std[n] + eps)
                            )
                        )
                        * keypoint_weights[j][n]
                    )

                    nb_kpts_counted = nb_kpts_counted + 1
            if nb_kpts_counted > 0:
                scores[i, j] = scores[i, j] / nb_kpts_counted
    return scores


@njit()
def euc_dist_batch(keypoints, mask=None):
    if mask is not None:
        keypoints = keypoints[mask]
    return np.sqrt(np.sum((keypoints[:, np.newaxis] - keypoints) ** 2, axis=2))


def mae_batch(a_keypoints, b_keypoints):
    """Calculate Mean Average Euclidean error (MAE) between two batches of keypoints of shape [N, K, 2],
    where N is the number of entities, K is the number of keypoints, and 2 represents each keypoint's coordinates.

    Args:
        a_keypoints (np.ndarray): Keypoints in the [N, K, 2] format.
        b_keypoints (np.ndarray): Keypoints in the [N, K, 2] format.

    Returns:
        float: The MAE
    """
    if a_keypoints.shape[0] == 0 or b_keypoints.shape[0] == 0:
        return 0.0
    _assert_kpts_input(a_keypoints, b_keypoints, symmetric=True)
    return _mae_batch_numba(a_keypoints, b_keypoints)


@njit
def _mae_batch_numba(a_keypoints, b_keypoints):
    N, K, _ = a_keypoints.shape
    total_error = 0.0
    count = 0

    for i in range(N):
        for j in range(K):
            if np.all(a_keypoints[i, j] > 0.0) and np.all(b_keypoints[i, j] > 0.0):
                distance = np.sqrt(np.sum((a_keypoints[i, j] - b_keypoints[i, j]) ** 2))
                total_error += distance
                count += 1

    return total_error / count if count > 0 else 0.0


def rmse_batch(a_keypoints, b_keypoints, mask=None):
    """Calculate the RMSE between two batches of keypoints of shape [N, K, 2],
    where: N is the number of entities, K is the number of keypoints and 2 is each keypoint coordinates.

    Args:
        a_keypoints (np.ndarray): keypoints in the [N, K, 2] format.
        b_keypoints (np.ndarray): keypoints in the [N, K, 2] format.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.

    Returns:
        np.ndarray: The RMSE
    """
    if a_keypoints.shape[0] == 0 or b_keypoints.shape[0] == 0:
        return 0.0
    _assert_kpts_input(a_keypoints, b_keypoints)
    if mask is not None:
        assert isinstance(mask, np.ndarray)
        assert mask.shape == a_keypoints.shape[:2]
        a_keypoints = a_keypoints[mask]
        b_keypoints = b_keypoints[mask]
    return _rmse_batch_numba(a_keypoints, b_keypoints)


@njit()
def _rmse_batch_numba(a_keypoints, b_keypoints):
    """Calculate the RMSE between two batches of keypoints of shape [N, K, 2],
    where: N is the number of entities, K is the number of keypoints and 2 is each keypoint coordinates.

    Args:
        a_keypoints (np.ndarray): keypoints in the [N, K, 2] format.
        b_keypoints (np.ndarray): keypoints in the [N, K, 2] format.

    Returns:
        np.ndarray: The RMSE
    """
    N, K, _ = a_keypoints.shape
    batch_rmse = np.zeros(N, dtype=np.float32)
    for i in range(N):
        cross_entities_dist = np.zeros(K, dtype=np.float32)
        for j in range(K):
            if np.all(a_keypoints[i][j] > 0.0) and np.all(b_keypoints[i][j] > 0.0):
                cross_entities_dist[j] = np.sum((a_keypoints[i][j] - b_keypoints[i][j]) ** 2, axis=0)
        batch_rmse[i] = np.sum(cross_entities_dist, axis=0)
    return np.sqrt(np.mean(batch_rmse))


@njit
def filter_matches_dynamic(x, y, cost_matrix, thresh):
    num_matches = 0
    for i in range(len(x)):
        if cost_matrix[x[i], y[i]] <= thresh[i]:
            num_matches += 1

    thr_x = np.empty(num_matches, dtype=np.float32)
    thr_y = np.empty(num_matches, dtype=np.float32)

    if num_matches > 0:
        index = 0
        for i in range(len(x)):
            if cost_matrix[x[i], y[i]] <= thresh[i]:
                thr_x[index] = x[i]
                thr_y[index] = y[i]
                index += 1

    return thr_x, thr_y


def linear_assignment(cost_matrix, thresh=None):
    if cost_matrix.size == 0:
        return (
            np.empty(0),
            np.empty(0),
        )
    x, y = linear_sum_assignment(cost_matrix)
    if thresh is None:
        return x, y
    elif isinstance(thresh, float):
        return filter_matches(x, y, cost_matrix, thresh)
    elif isinstance(thresh, np.ndarray):
        thresh = thresh[y].astype(np.float32)
        return filter_matches_dynamic(x, y, cost_matrix, thresh)
    else:
        TypeError(f"thresh must be one of [NoneType, float, np.ndarray]. Not, {type(thresh)}.")


def fp16_clamp(x, min_val=None, max_val=None):
    if not x.is_cuda and x.dtype == torch.float16:
        return x.float().clamp(min_val, max_val).half()
    return x.clamp(min_val, max_val)


def bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6) -> torch.Tensor:
    """Calculate overlap between two sets of bounding boxes.

    Args:
        bboxes1 (torch.Tensor): Bounding boxes of shape (..., m, 4) or empty.
        bboxes2 (torch.Tensor): Bounding boxes of shape (..., n, 4) or empty.
        mode (str): "iou" (intersection over union),
                    "iof" (intersection over foreground),
                    or "giou" (generalized intersection over union).
                    Defaults to "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A small constant added to the denominator for
            numerical stability. Default 1e-6.

    Returns:
        torch.Tensor: Overlap values of shape (..., m, n) if is_aligned is
            False, else shape (..., m).

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )
    """
    assert mode in ["iou", "iof", "giou"], f"Unsupported mode {mode}"
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0

    if bboxes1.ndim == 1:
        bboxes1 = bboxes1.unsqueeze(0)
    if bboxes2.ndim == 1:
        bboxes2 = bboxes2.unsqueeze(0)

    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])
        wh = fp16_clamp(rb - lt, min_val=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        wh = fp16_clamp(rb - lt, min_val=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])

    eps_tensor = union.new_tensor([eps])
    union = torch.max(union, eps_tensor)
    ious = overlap / union
    if mode in ["iou", "iof"]:
        return ious
    elif mode == "giou":
        enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min_val=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        enclose_area = torch.max(enclose_area, eps_tensor)
        gious = ious - (enclose_area - union) / enclose_area
        return gious


def map_sequence(sequence: List[PoseDataSample]):
    for data_sample in sequence:
        bboxes = data_sample.pred_instances["bboxes"].detach().cpu().numpy()
        labels = data_sample.pred_instances["labels"].detach().cpu().numpy()
        features = data_sample.pred_instances["features"]

        # Remove duplicate GT (solves labelling errors)
        gt_labels_all = data_sample.gt_instance_labels
        uniques, counts = torch.unique(gt_labels_all.ids, return_counts=True)
        duplicate_unique_labels_mask = counts > 1
        duplicate_uniques = uniques[duplicate_unique_labels_mask]
        non_duplicate_labels = ~torch.isin(gt_labels_all.ids, duplicate_uniques)
        gt_labels_cleaned = gt_labels_all[non_duplicate_labels]
        data_sample.gt_instance_labels = gt_labels_cleaned

        # Match GT to Detections
        gt_bboxes = xyxy_cxcywh(data_sample.gt_instance_labels["bboxes"]).detach().cpu().numpy()
        gt_labels = data_sample.gt_instance_labels["labels"].detach().cpu().numpy()

        dists = iou_batch(gt_bboxes, bboxes)
        if gt_bboxes.size > 0:
            same_labels = labels[None, :] == gt_labels[:, None]
            dists = (dists * same_labels).astype(np.float32)
        matched_gt, matched_dets = linear_assignment(1 - dists, 0.9)

        matched_gt = torch.from_numpy(matched_gt).to(torch.long).to(features.device)
        matched_dets = torch.from_numpy(matched_dets).to(torch.long).to(features.device)
        matches = torch.concat((matched_gt.view(-1, 1), matched_dets.view(-1, 1)), dim=1)

        data_sample.gt_instance_labels = data_sample.gt_instance_labels.to(matches.device)
        data_sample.gt_instance_labels = data_sample.gt_instance_labels[matches[:, 0]]

        data_sample.pred_instances = data_sample.pred_instances[matches[:, 1]]

        assert data_sample.pred_instances["features"].size(0) == data_sample.gt_instance_labels.ids.size(0)
    return sequence


def cosine_similarity(a: torch.Tensor, b: torch.Tensor):
    return F.normalize(a) @ F.normalize(b).T


# @njit()
def match_nearest_keypoint(queries, gallery, idx_map, mask=None):
    N, _ = queries.shape
    matched_gallery_idx = np.ones(N, np.int64) * -1
    for i in range(N):
        if mask is not None:
            masked_gallery = gallery[mask[i]]
        else:
            masked_gallery = gallery
        matched_gallery_idx[i] = np.argmin(np.linalg.norm(queries[i] - masked_gallery, axis=-1))
    return idx_map[matched_gallery_idx]
