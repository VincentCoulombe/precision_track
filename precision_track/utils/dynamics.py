import time

import numpy as np
import torch
from numba import njit


def calculate_bbox_velocities(curr, prev, dt, conf_thr: float = 0.0):
    curr_boxes = curr.bboxes
    curr_ids = curr.instances_id

    prev_boxes = prev.bboxes
    prev_ids = prev.instances_id
    still_tracked = torch.isin(prev_ids, curr_ids, assume_unique=True)
    prev_ids = prev_ids[still_tracked]
    prev_boxes = prev_boxes[still_tracked]

    device = curr_boxes.device
    dtype = curr_boxes.dtype
    Nc = curr_boxes.shape[0]

    if prev_ids.numel() == 0:
        return torch.zeros((Nc, 2), device=device, dtype=dtype)

    sort_idx = torch.argsort(prev_ids)
    prev_ids_sorted = prev_ids[sort_idx]
    pos = torch.searchsorted(prev_ids_sorted, curr_ids)

    n = prev_ids_sorted.numel()
    in_range = pos < n
    eq = torch.zeros(Nc, dtype=torch.bool, device=device)
    if in_range.any():
        eq[in_range] = prev_ids_sorted[pos[in_range]] == curr_ids[in_range]
    valid = in_range & eq

    prev_idx_for_curr = torch.full((Nc,), -1, dtype=torch.long, device=device)
    prev_idx_for_curr[valid] = sort_idx[pos[valid]]
    has_prev = prev_idx_for_curr >= 0

    aligned_prev_boxes = torch.zeros_like(curr_boxes)
    if has_prev.any():
        aligned_prev_boxes[has_prev] = prev_boxes[prev_idx_for_curr[has_prev]]

    conf_mask = curr.scores > conf_thr

    use_mask = has_prev & conf_mask

    velocities = torch.zeros((Nc, 2), device=device, dtype=dtype)
    if use_mask.any():
        delta_c = curr_boxes[use_mask, :2] - aligned_prev_boxes[use_mask, :2]
        velocities[use_mask] = delta_c / dt

    return velocities


def calculate_pose_velocities(curr, prev, dt, vis_thr: float = 0.5):
    curr_kpts = curr.keypoints
    curr_vis = curr.keypoint_scores
    curr_ids = curr.instances_id

    prev_kpts = prev.keypoints
    prev_ids = prev.instances_id
    still_tracked = torch.isin(prev_ids, curr_ids, assume_unique=True)
    prev_ids = prev_ids[still_tracked]
    prev_kpts = prev_kpts[still_tracked]

    Nc, K, _ = curr_kpts.shape
    device = curr_kpts.device
    dtype = curr_kpts.dtype

    if prev_ids.numel() == 0:
        return torch.zeros((Nc, K, 2), device=device, dtype=dtype)

    sort_idx = torch.argsort(prev_ids)
    prev_ids_sorted = prev_ids[sort_idx]

    pos = torch.searchsorted(prev_ids_sorted, curr_ids)
    n = prev_ids_sorted.numel()
    in_range = pos < n
    eq = torch.zeros(Nc, dtype=torch.bool, device=device)
    if in_range.any():
        eq[in_range] = prev_ids_sorted[pos[in_range]] == curr_ids[in_range]
    valid = in_range & eq

    prev_idx_for_curr = torch.full((Nc,), -1, device=device, dtype=torch.long)
    prev_idx_for_curr[valid] = sort_idx[pos[valid]]

    aligned_prev_kpts = torch.zeros((Nc, K, 2), device=device, dtype=dtype)
    has_prev = prev_idx_for_curr >= 0
    if has_prev.any():
        aligned_prev_kpts[has_prev] = prev_kpts[prev_idx_for_curr[has_prev]]

    vis_mask = (curr_vis > vis_thr).unsqueeze(-1)
    returning_mask = has_prev.view(Nc, 1, 1)
    use_mask = returning_mask & vis_mask

    velocities = torch.zeros((Nc, K, 2), device=device, dtype=dtype)
    delta = curr_kpts - aligned_prev_kpts
    velocities[use_mask.expand_as(delta)] = (delta / dt)[use_mask.expand_as(delta)]

    return velocities


@njit
def update_dynamics_2d(
    dynamics: np.ndarray,
    location: np.ndarray,
    previous_location: np.ndarray,
    alpha: float,
    dt: int,
) -> None:
    dynamics[0], dynamics[1] = location
    dx, dy = location - previous_location
    vx_new = alpha * (dx / dt) + (1 - alpha) * dynamics[2]
    vy_new = alpha * (dy / dt) + (1 - alpha) * dynamics[3]
    dvx = vx_new - dynamics[2]
    dvy = vy_new - dynamics[3]
    dynamics[4] = (np.sign(vx_new) == np.sign(dynamics[2])) * dvx / dt
    dynamics[5] = (np.sign(vy_new) == np.sign(dynamics[3])) * dvy / dt
    dynamics[2], dynamics[3] = vx_new, vy_new
    return dynamics


@njit
def sequential_ema_smoothing(t0_ids: np.ndarray, t0_probs: np.ndarray, t1_ids: np.ndarray, t1_probs: np.ndarray, smoothing_factor: float = 0.1):
    n = t1_ids.shape[0]
    m = t0_ids.shape[0]

    for i in range(n):
        idx = -1
        for j in range(m):
            if t0_ids[j] == t1_ids[i]:
                idx = j
                break
        if idx >= 0:
            t1_probs[i] = t0_probs[idx] * smoothing_factor + t1_probs[i] * (1 - smoothing_factor)
    return t1_probs


def wait_until_clear(event, timeout: float, poll: float = 0.05) -> bool:
    deadline = None if timeout is None else (time.monotonic() + timeout)
    while event.is_set():
        if deadline and time.monotonic() >= deadline:
            return False
        time.sleep(poll)
    return True
