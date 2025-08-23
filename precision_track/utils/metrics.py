# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np


def calc_distances(preds: np.ndarray, gts: np.ndarray, mask: np.ndarray, norm_factor: np.ndarray) -> np.ndarray:
    """Calculate the normalized distances between preds and target.

    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        gts (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        norm_factor (np.ndarray[N, D]): Normalization factor.
            Typical value is heatmap_size.

    Returns:
        np.ndarray[K, N]: The normalized distances. \
            If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    # set mask=0 when norm_factor==0
    _mask = mask.copy()
    _mask[np.where((norm_factor == 0).sum(1))[0], :] = False

    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    norm_factor[np.where(norm_factor <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(((preds - gts) / norm_factor[:, None, :])[_mask], axis=-1)
    return distances.T


def keypoint_auc(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, norm_factor: np.ndarray, num_thrs: int = 20) -> float:
    """Calculate the Area under curve (AUC) of keypoint PCK accuracy.

    Note:
        - instance number: N
        - keypoint number: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        norm_factor (float): Normalization factor.
        num_thrs (int): number of thresholds to calculate auc.

    Returns:
        float: Area under curve (AUC) of keypoint PCK accuracy.
    """
    nor = np.tile(np.array([[norm_factor, norm_factor]]), (pred.shape[0], 1))
    thrs = [1.0 * i / num_thrs for i in range(num_thrs)]
    avg_accs = []
    for thr in thrs:
        _, avg_acc, _ = keypoint_pck_accuracy(pred, gt, mask, thr, nor)
        avg_accs.append(avg_acc)

    auc = 0
    for i in range(num_thrs):
        auc += 1.0 / num_thrs * avg_accs[i]
    return auc


def keypoint_pck_accuracy(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, thr: np.ndarray, norm_factor: np.ndarray) -> tuple:
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - instance number: N
        - keypoint number: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        norm_factor (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = calc_distances(pred, gt, mask, norm_factor)
    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0.0
    return acc, avg_acc, cnt


def _distance_acc(distances: np.ndarray, thr: float = 0.5) -> float:
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.

    Note:
        - instance number: N

    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        float: Percentage of distances below the threshold. \
            If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1


def compute_similarity_transform(source_points, target_points):
    """Computes a similarity transform (sR, t) that takes a set of 3D points
    source_points (N x 3) closest to a set of 3D points target_points, where R
    is an 3x3 rotation matrix, t 3x1 translation, s scale. And return the
    transformed 3D points source_points_hat (N x 3). i.e. solves the orthogonal
    Procrutes problem.

    Note:
        Points number: N

    Args:
        source_points (np.ndarray): Source point set with shape [N, 3].
        target_points (np.ndarray): Target point set with shape [N, 3].

    Returns:
        np.ndarray: Transformed source point set with shape [N, 3].
    """

    assert target_points.shape[0] == source_points.shape[0]
    assert target_points.shape[1] == 3 and source_points.shape[1] == 3

    source_points = source_points.T
    target_points = target_points.T

    # 1. Remove mean.
    mu1 = source_points.mean(axis=1, keepdims=True)
    mu2 = target_points.mean(axis=1, keepdims=True)
    X1 = source_points - mu1
    X2 = target_points - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, _, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Transform the source points:
    source_points_hat = scale * R.dot(source_points) + t

    source_points_hat = source_points_hat.T

    return source_points_hat
