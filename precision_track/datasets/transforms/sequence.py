from abc import ABCMeta, abstractmethod
from typing import Union, List, Tuple
import numpy as np

from .common import RandomFlip, RandomCrop
from .base import BaseTransform
from .utils import cache_randomness

from precision_track.registry import TRANSFORMS


class BaseSequenceTransform(metaclass=ABCMeta):

    @abstractmethod
    def set_stochastic_params(self) -> None:
        pass


@TRANSFORMS.register_module()
class SequenceRandomFlip(RandomFlip, BaseSequenceTransform):
    def __init__(
        self,
        prob: Union[float, List[float]] = 0.5,
        direction: Union[str, List[str]] = "horizontal",
    ) -> None:
        super().__init__(prob, direction)
        self.cur_dir = None

    def _choose_direction(self) -> str:
        direction_list = [self.direction, None]

        if isinstance(self.prob, list):
            non_prob: float = 1 - sum(self.prob)
            prob_list = self.prob + [non_prob]
        elif isinstance(self.prob, float):
            non_prob = 1.0 - self.prob
            single_ratio = self.prob / (len(direction_list) - 1)
            prob_list = [single_ratio] * (len(direction_list) - 1) + [non_prob]

        cur_dir = np.random.choice(direction_list, p=prob_list)

        return cur_dir

    def set_stochastic_params(self) -> None:
        self.cur_dir = self._choose_direction()

    def transform(self, results: dict) -> dict:
        results["flip_direction"] = self.cur_dir
        results["flip"] = False if self.cur_dir is None else True

        return super().transform(results)


@TRANSFORMS.register_module()
class SequenceRandomOcclusion(BaseTransform, BaseSequenceTransform):
    pass

    def set_stochastic_params(self) -> None:
        pass

    def transform(self, results: dict) -> dict:
        pass


@TRANSFORMS.register_module()
class SequenceRandomCrop(RandomCrop, BaseSequenceTransform):
    def __init__(self, crop_size: tuple) -> None:
        assert isinstance(crop_size, tuple)
        assert len(crop_size) == 2
        for cs in crop_size:
            assert 0 < cs <= 1, f"The crop sizes must are relative to the image size. {cs} not in range."
        assert crop_size[0] < crop_size[1]
        super().__init__(
            crop_size=crop_size,
            crop_type="relative_range",
            allow_negative_crop=True,
        )
        self.crop_range = np.random.default_rng()
        self.seq_crop_size = 1
        self.offset_h = -1
        self.offset_w = -1

    def _get_crop_size(self) -> int:
        return self.crop_range.integers(self.crop_size[0] * 100, self.crop_size[1] * 100 + 1) / 100

    def _get_reltiave_offsets(self, margin: Tuple[int, int]) -> Tuple[int, int]:
        margin_h, margin_w = margin
        self.offset_h = np.random.randint(0, margin_h + 1)
        self.offset_w = np.random.randint(0, margin_w + 1)

    def _rand_offset(self, margin: Tuple[int, int]) -> Tuple[int, int]:
        margin_h, margin_w = margin
        return margin_h + self.offset_h, margin_w + self.offset_w

    def set_stochastic_params(self) -> None:
        self.seq_crop_size = self._get_crop_size()
        self.offset_h = -1
        self.offset_w = -1

    def transform(self, results: dict) -> Union[dict, None]:
        img = results["img"]
        crop_size = tuple(int(img_s * self.seq_crop_size) for img_s in img.shape[:2])
        if self.offset_h < 0 and self.offset_w < 0:
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            self._get_reltiave_offsets((margin_h, margin_w))
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results
