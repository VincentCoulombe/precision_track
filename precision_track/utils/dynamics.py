import time

import numpy as np
from numba import njit


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
