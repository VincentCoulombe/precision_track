from .common import RandomFlip, RandomCrop
from .base import BaseTransform

from precision_track.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SequenceRandomFlip(RandomFlip):
    pass


@TRANSFORMS.register_module()
class SequenceRandomOcclusion(BaseTransform):
    pass

    def transform(self, results: dict) -> dict:
        pass


@TRANSFORMS.register_module()
class SequenceRandomCrop(RandomCrop):
    pass
