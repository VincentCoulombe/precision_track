from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class BaseKalmanFilter(metaclass=ABCMeta):

    def __init__(self, **kwargs):
        self._frame_size = None
        self.frame_id = 0

    @property
    def frame_size(self):
        if self._frame_size is None:
            raise ValueError("Frame size not set for the Kalman filter.")
        return self._frame_size

    @frame_size.setter
    def frame_size(self, frame_size: Tuple[int, int]):
        assert len(frame_size) == 2
        for f_s in frame_size:
            assert 0 < f_s
        self._frame_size = frame_size

    @abstractmethod
    def initiate(self, track: dict, bboxe: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, track: dict) -> None:
        pass

    @abstractmethod
    def update(self, track: dict) -> None:
        pass

    def multi_predict(self, tracks: Dict[int, dict], ids: List[int]) -> None:
        for id in ids:
            self.predict(tracks[id])

    def get_measurement(self, track: dict, idx: Optional[int] = -1) -> np.ndarray:
        return track.bboxes[idx]
