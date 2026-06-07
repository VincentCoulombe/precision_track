from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Optional, Any


class BaseValidation(metaclass=ABCMeta):

    def __init__(
        self,
        validated_classes: List[str],
        identities: List[Any],
        disabled_identities: Optional[List[Any]] = None,
        *args,
        **kwargs,
    ) -> None:
        self._frame_size = None
        assert isinstance(validated_classes, list)
        for cls in validated_classes:
            assert isinstance(cls, str)
        self.validated_classes = validated_classes
        if not isinstance(identities, list):
            identities = []
        self.identities = identities
        if not isinstance(disabled_identities, list):
            disabled_identities = []
        self.disabled_identities = disabled_identities

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
