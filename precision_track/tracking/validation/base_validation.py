from abc import ABCMeta, abstractmethod
from typing import List, Tuple


class BaseValidation(metaclass=ABCMeta):

    def __init__(self, *args, **kwargs) -> None:
        self._frame_size = None

    @abstractmethod
    def __call__(self, *args, **kwargs) -> List[Tuple]:
        pass

    @property
    def frame_size(self):
        if self._frame_size is None:
            raise ValueError("Frame size not set for the validation.")
        return self._frame_size

    @frame_size.setter
    def frame_size(self, frame_size: Tuple[int, int]):
        assert len(frame_size) == 2
        for f_s in frame_size:
            assert 0 < f_s
        self._frame_size = frame_size
