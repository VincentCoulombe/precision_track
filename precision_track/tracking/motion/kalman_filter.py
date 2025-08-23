from typing import Optional, Tuple

import numpy as np
import scipy.linalg

from precision_track.registry import TRACKING
from precision_track.utils import clip, reformat

from .base_kalman_filter import BaseKalmanFilter


@TRACKING.register_module()
class KalmanFilter(BaseKalmanFilter):
    """A simple Kalman filter for tracking bounding boxes in image space.

    Taken From: https://github.com/open-mmlab/mmdetection/tree/cfd5d3a985b0249de009b67d04f37263e11cdf3d/mmdet/models/task_modules/tracking

    The 8-dimensional state space

    x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).


    Args:
        use_nsa (bool): Whether to use the NSA (Noise Scale Adaptive) Kalman
            Filter, which adaptively modulates the noise scale according to
            the quality of detections. More details in
            https://arxiv.org/abs/2202.11983. Defaults to False.
    """

    def __init__(self, use_nsa: Optional[bool] = False, **kwargs) -> None:

        super().__init__(**kwargs)
        assert isinstance(use_nsa, bool)
        self.use_nsa = use_nsa
        self.ndim, self.dt = 4, 1.0

        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, track: dict) -> None:
        """Initiate the mean (the 8-dimensional state space) and covariance,
        (the 8x8-dimensional noise space) of a new track.

        Args:
            track (dict): The track to be initiated.

        Returns:
             None: The track is updated in place.
        """
        measurement = self.get_measurement(track)

        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        track["mean"] = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        track["covariance"] = np.diag(np.square(std))
        track["pred_bboxe"] = track["bboxes"][-1]
        track["pred_keypoints"] = track["keypoints"][-1]
        track["velocity"] = np.zeros(2)

    def predict(
        self,
        track: dict,
    ) -> None:
        """Run Kalman filter prediction step.

        Args:
            track (dict): The track to predict its next location.

        Returns:
            None: The track is updated in place.
        """
        mean, covariance = track["mean"], track["covariance"]
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        new_mean = np.dot(self._motion_mat, mean)
        track["velocity"] = new_mean[:2] - mean[:2]
        track["mean"] = new_mean
        track["pred_bboxe"] = clip(
            reformat(new_mean[:4], "cxcyah", "cxcywh"),
            "cxcywh",
            self.frame_size[0],
            self.frame_size[1],
        )
        track["pred_keypoints"] = track["keypoints"][-1] + track["velocity"]
        track["covariance"] = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

    def project(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        bbox_score: Optional[float] = 0.0,
    ) -> Tuple[np.array, np.array]:
        """Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8
                dimensional).
            bbox_score (float): The confidence score of the bbox.
                Defaults to 0.

        Returns:
            (ndarray, ndarray):  Returns the projected mean and covariance
            matrix of the given state estimate.
        """
        # The standard deviations (uncertainty of measurement) for each dimension: [x, y, aspect ratio, height]
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]

        if self.use_nsa:
            # reduce uncertainty for high confidence detections
            std = [(1 - bbox_score) * x for x in std]

        # A diagonal matrix where the diagonal elements are the squared standard deviations
        # Captures the uncertainty of each measurement in its space
        innovation_cov = np.diag(np.square(std))

        # Project the mean by dot product of the measurement matrix and the mean
        # The measurement matrix is a identity matrix that filter out velocities
        mean = np.dot(self._update_mat, mean)

        # Project the covariance by dot product of the measurement matrix and the covariance
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))

        # Introduce the uncertainty of the measurement to the projected covariance
        return mean, covariance + innovation_cov

    def update(
        self,
        track: dict,
    ) -> None:
        """Run Kalman filter correction step.

        Args:
            track (dict): The track to be updated.

        Returns:
             None: The track is updated in place.
        """
        measurement = self.get_measurement(track)

        mean, covariance, score = (
            track["mean"],
            track["covariance"],
            track["scores"][-1],
        )

        projected_mean, projected_cov = self.project(mean, covariance, score)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        track["mean"] = mean + np.dot(innovation, kalman_gain.T)
        track["covariance"] = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

    def get_measurement(self, track: dict, idx: Optional[int] = -1) -> np.ndarray:
        return reformat(super().get_measurement(track, idx), "cxcywh", "cxcyah")
