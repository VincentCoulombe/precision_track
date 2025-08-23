from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from precision_track.utils import PoseDataSample

from .motion.base_kalman_filter import BaseKalmanFilter


class BaseAlgorithm(metaclass=ABCMeta):

    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        if hasattr(self, "return_masks") and self.return_masks:
            self.fields_to_remove = ["_data_fields", "_metainfo_fields"]
        else:
            self.fields_to_remove = ["_data_fields", "_metainfo_fields", "masks"]

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass


class BaseAssignationAlgorithm(BaseAlgorithm):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._kf = None
        self.prior_feature_size = 0
        self.feature_device = "cpu"

    @property
    def kf(self):
        assert self._kf is not None, "The association algorithm do not have any Kalman Filter yet."
        return self._kf

    @kf.setter
    def kf(self, new_kf: BaseKalmanFilter):
        assert isinstance(new_kf, BaseKalmanFilter), "The Kalman Filter must extend the BaseKalmanFilter"
        self._kf = new_kf

    @abstractmethod
    def update_thresholds(self, tracking_thresholds: dict) -> None:
        pass

    def loss(self, data_samples: List[List[PoseDataSample]]) -> dict:
        return dict()

    def val_step(self, data_samples: list, *args, **kwargs) -> list:
        return list()

    def test_step(self, data_samples: list, *args, **kwargs) -> list:
        return list()

    @abstractmethod
    def __call__(
        self,
        frame_id: int,
        data_sample: dict,
        tracks: dict,
        confirmed_ids: List[int],
        unconfirmed_ids: List[int],
    ) -> Dict[str, np.ndarray]:
        """Assign the new detections to the existing tracks. Update the given
        data_sample's pred_track_instances.

        Args:
            frame_id (int): The currnt frame number.
            data_sample (dict): The new detections. Minimaly containing bboxes, labels and scores.
            tracks (dict): A dictionary containing every tracks currently active.
            confirmed_ids (List[int]): The ids of the track that have been alive since >= num_tentatives.
            unconfirmed_ids (List[int]): The ids of the track that have been alive since < num_tentatives.
        """

    def save_to_ds(
        self,
        data_samples: dict,
        detections: dict,
        track_ids: np.ndarray,
        pred_idx: np.ndarray,
        predicted_bboxes: Optional[np.ndarray] = None,
        features: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        data_samples["pred_track_instances"] = {"ids": track_ids}
        data_samples["pred_track_instances"].update({k: v[pred_idx] for k, v in detections.items() if k not in self.fields_to_remove})

        if predicted_bboxes is not None:
            data_samples["pred_track_instances"].update(
                {
                    "next_frame_bboxes": predicted_bboxes,
                }
            )
        if features is not None:
            data_samples["pred_track_instances"].update({"features": features})

    def init_call(self, data_sample: dict, tracks: dict):
        for k, v in data_sample["pred_instances"].items():
            if k != "features":
                data_sample["pred_instances"][k] = v.cpu().numpy()
        self.kf.frame_id = data_sample["img_id"]
        num_tracks = len(tracks)
        num_dets = data_sample["pred_instances"]["scores"].shape[0]
        max_size = num_tracks + num_dets
        prior_features = data_sample["pred_instances"].get("features", torch.empty(0))
        if not self.prior_feature_size:
            self.prior_feature_size = prior_features.size(1)
        self.feature_device = prior_features.device
        return (
            data_sample["pred_instances"],
            data_sample["img_id"],
            data_sample["pred_instances"]["scores"],
            num_tracks,
            num_dets,
            np.zeros(max_size, dtype=int),
            np.zeros((max_size, 4), dtype=np.float32),
            torch.zeros((max_size, self.prior_feature_size), dtype=torch.float32, device=prior_features.device),
            np.zeros(max_size, dtype=int),
            0,
        )


class BaseStitchingAlgorithm(BaseAlgorithm):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @abstractmethod
    def __call__(
        self,
        tracks: dict,
        track_instances: dict,
    ) -> Dict[str, np.ndarray]:
        self.tracks = tracks
        self.track_instances = track_instances

    def __masktrack__(self, track_id: int) -> None:
        idx = self._assert_id(track_id)
        self.track_instances["instances_id"][idx] = -1

    def __setinstid__(self, track_id: int, inst_id: int) -> None:
        idx = self._assert_id(track_id)
        self.tracks[track_id].instances_id = inst_id
        self.track_instances["instances_id"][idx] = inst_id

    def _assert_id(self, track_id: int) -> None:
        assert track_id in self.tracks, f"Track {track_id} does not exist."
        idx = self.track_instances["ids"] == track_id
        assert np.sum(idx) <= 1, f"Track {track_id} should have no more than 1 instance, but have {np.sum(idx)}."
        return idx
