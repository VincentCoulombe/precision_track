from typing import Optional

import numpy as np
import torch

from precision_track.registry import TRACKING
from precision_track.utils import clip, update_dynamics_2d

from .base_kalman_filter import BaseKalmanFilter


@TRACKING.register_module()
class DynamicKalmanFilter(BaseKalmanFilter):

    def __init__(self, alpha: Optional[float] = 0.5, **kwargs):
        """A kalman filter tailored to handle cases where objects exhibit
        varying velocities, and accelerations along both the x-axis and the
        y-axis of the frame.

        Args:
            alpha (Optional[float], optional): The weight attributed the the previous velocity in the EMA velocity calculation. Defaults to 0.15.
        """
        super().__init__(**kwargs)
        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha
        self.dt = 1
        self.IDENTITY = np.eye(6)

        self._motion_mat = np.array(
            [
                [1, 0, self.dt, 0, 0.5 * self.dt**2, 0],  # x
                [0, 1, 0, self.dt, 0, 0.5 * self.dt**2],  # y
                [0, 0, 1, 0, self.dt, 0],  # v_x
                [0, 0, 0, 1, 0, self.dt],  # v_y
                [0, 0, 0, 0, 1, 0],  # a_x
                [0, 0, 0, 0, 0, 1],  # a_y
            ]
        )

        self._update_mat = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        self._measurement_noise = np.array([[1.0, 0], [0, 1.0]])

    def _get_process_noise(self):
        """
        Compute the process noise covariance matrix.

        Returns:
            np.ndarray: Process noise covariance.
        """
        q = 0.1
        motion_cov = np.zeros((6, 6))
        motion_cov[:2, :2] = np.eye(2) * 0.5 * self.dt**2 * q  # Position noise
        motion_cov[2:4, 2:4] = np.eye(2) * self.dt * q  # Velocity noise
        motion_cov[4:6, 4:6] = np.eye(2) * q  # Acceleration noise
        return motion_cov

    def initiate(self, track: dict) -> None:
        """Initiate the mean (the 6-dimensional state space) and covariance,
        (the 6x6-dimensional noise space) of a new track.

        Args:
            track (dict): The track to be initiated.

        Returns:
             None: The track is updated in place.
        """
        centroid = self.get_measurement(track)
        track["mean"] = np.array([centroid[0], centroid[1], 0, 0, 0, 0])
        track["covariance"] = self.IDENTITY * 1000
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
        self.dt = self.frame_id - track["frame_ids"][-1]

        mean = track["mean"]
        pred_centroid = self._motion_mat @ mean
        velocity = pred_centroid[:2] - mean[:2]
        track["velocity"] = velocity

        bbox = track["bboxes"][-1]
        cxcywh = np.array([bbox[0] + velocity[0], bbox[1] + velocity[1], *bbox[2:]])
        track["pred_bboxe"] = clip(cxcywh, "cxcywh", self.frame_size[0], self.frame_size[1])

        track["pred_keypoints"] = track["keypoints"][-1] + velocity

        # process_noise = self._get_process_noise()
        # track["covariance"] = np.linalg.multi_dot((self._motion_mat, track["covariance"], self._motion_mat.T)) + process_noise

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
        track_frames = track["frame_ids"]
        self.dt = track_frames[-1] - track_frames[-2] if len(track_frames) > 1 else 1

        centroid = self.get_measurement(track)
        previous_centroid = self.get_measurement(track, -2) if self.dt == 1 else centroid

        track["mean"] = update_dynamics_2d(track["mean"], centroid, previous_centroid, self.alpha, self.dt)

        # track["covariance"] = np.linalg.multi_dot((self._motion_mat, track["covariance"], self._motion_mat.T))
        # chol_factor, lower = scipy.linalg.cho_factor(track["covariance"], lower=True, check_finite=False)
        # kalman_gain = scipy.linalg.cho_solve(
        #     (chol_factor, lower),
        #     np.dot(track["covariance"], self._update_mat.T),
        #     check_finite=False,
        # ).T
        # track["covariance"] = track["covariance"] - np.linalg.multi_dot((kalman_gain, track["covariance"], kalman_gain.T))

    def get_measurement(self, track: dict, idx: Optional[int] = -1) -> np.ndarray:
        return super().get_measurement(track, idx)[:2]


@TRACKING.register_module()
class DynamicKalmanFilterPytorch(BaseKalmanFilter):

    def __init__(self, alpha: Optional[float] = 0.15, **kwargs):
        """A kalman filter tailored to handle cases where objects exhibit
        varying velocities, and accelerations along both the x-axis and the
        y-axis of the frame.

        Args:
            alpha (Optional[float], optional): The weight attributed the the previous velocity in the EMA velocity calculation. Defaults to 0.15.
        """
        super().__init__(**kwargs)
        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha
        self.dt = 1
        self.IDENTITY = torch.eye(6)

        self._motion_mat = torch.tensor(
            [
                [1, 0, self.dt, 0, 0.5 * self.dt**2, 0],  # x
                [0, 1, 0, self.dt, 0, 0.5 * self.dt**2],  # y
                [0, 0, 1, 0, self.dt, 0],  # v_x
                [0, 0, 0, 1, 0, self.dt],  # v_y
                [0, 0, 0, 0, 1, 0],  # a_x
                [0, 0, 0, 0, 0, 1],  # a_y
            ]
        )

        self._update_mat = torch.tensor([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        self._measurement_noise = torch.tensor([[1.0, 0], [0, 1.0]])

    def _get_process_noise(self):
        """
        Compute the process noise covariance matrix.

        Returns:
            torch.Tensor: Process noise covariance.
        """
        q = 0.1
        motion_cov = torch.zeros((6, 6))
        motion_cov[:2, :2] = torch.eye(2) * 0.5 * self.dt**2 * q  # Position noise
        motion_cov[2:4, 2:4] = torch.eye(2) * self.dt * q  # Velocity noise
        motion_cov[4:6, 4:6] = torch.eye(2) * q  # Acceleration noise
        return motion_cov

    def initiate(self, track: dict) -> None:
        """Initiate the mean (the 6-dimensional state space) and covariance,
        (the 6x6-dimensional noise space) of a new track.

        Args:
            track (dict): The track to be initiated.

        Returns:
             None: The track is updated in place.
        """
        centroid = self.get_measurement(track)
        track["mean"] = torch.tensor([centroid[0], centroid[1], 0, 0, 0, 0], device=centroid.device)
        track["covariance"] = self.IDENTITY * 1000
        track["pred_bboxe"] = track["bboxes"][-1]
        track["pred_keypoints"] = track["keypoints"][-1]
        track["velocity"] = torch.zeros(2, device=centroid.device)

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
        self.dt = self.frame_id - track["frame_ids"][-1]

        if self._motion_mat.device != track["mean"].device:
            self._motion_mat = self._motion_mat.to(track["mean"].device)
        pred_centroid = self._motion_mat @ track["mean"]
        dx, dy = pred_centroid[:2] - track["mean"][:2]
        track["velocity"] = torch.cat((dx.unsqueeze(0), dy.unsqueeze(0)))

        cxcywh = track["bboxes"][-1]
        cxcywh[0] += dx
        cxcywh[1] += dy
        track["pred_bboxe"] = clip(cxcywh, "cxcywh", self.frame_size[0], self.frame_size[1])
        track["pred_keypoints"] = track["keypoints"][-1] + track["velocity"]

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
        track_frames = track["frame_ids"]
        self.dt = track_frames[-1] - track_frames[-2] if len(track_frames) > 1 else 1

        centroid = self.get_measurement(track)
        previous_centroid = self.get_measurement(track, -2) if self.dt == 1 else centroid

        track["mean"][0], track["mean"][1] = centroid
        dx, dy = centroid - previous_centroid
        vx_new = self.alpha * (dx / self.dt) + (1 - self.alpha) * track["mean"][2]
        vy_new = self.alpha * (dy / self.dt) + (1 - self.alpha) * track["mean"][3]
        dvx = vx_new - track["mean"][2]
        dvy = vy_new - track["mean"][3]
        track["mean"][4] = (torch.sign(vx_new) == torch.sign(track["mean"][2])) * dvx / self.dt
        track["mean"][5] = (torch.sign(vy_new) == torch.sign(track["mean"][3])) * dvy / self.dt
        track["mean"][2], track["mean"][3] = vx_new, vy_new

    def get_measurement(self, track: dict, idx: Optional[int] = -1) -> torch.Tensor:
        return super().get_measurement(track, idx)[:2]
