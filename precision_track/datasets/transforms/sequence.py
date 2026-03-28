from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Union

import cv2
import numpy as np

from precision_track.registry import TRANSFORMS

from .base import BaseTransform
from .common import RandomCrop, RandomFlip


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


@TRANSFORMS.register_module()
class SequenceRandomContrastAug(BaseTransform):
    def __init__(self, contrast_range=(0.75, 1.25)) -> None:
        assert isinstance(contrast_range, tuple) and len(contrast_range) == 2, "contrast_range must be a tuple of two floats"
        self.contrast_range = contrast_range
        self.contrast_factor = 1.0

    def set_stochastic_params(self) -> None:
        self.contrast_factor = np.random.uniform(*self.contrast_range)

    def transform(self, results: dict) -> dict:
        img = results["img"]

        img = img.astype(np.float32)
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * self.contrast_factor + mean
        img = np.clip(img, 0, 255).astype(np.uint8)

        results["img"] = img
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(contrast_range={self.contrast_range})"


@TRANSFORMS.register_module()
class SequenceYOLOXHSVRandomAug(BaseTransform):
    def __init__(self, hue_delta: int = 5, saturation_delta: int = 30, value_delta: int = 30) -> None:
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta
        self.hsv_gains = None

    def _get_hsv_gains(self):
        hsv_gains = np.random.uniform(-1, 1, 3) * [
            self.hue_delta,
            self.saturation_delta,
            self.value_delta,
        ]
        # random selection of h, s, v
        hsv_gains *= np.random.randint(0, 2, 3)
        # prevent overflow
        hsv_gains = hsv_gains.astype(np.int16)
        return hsv_gains

    def set_stochastic_params(self) -> None:
        self.hsv_gains = self._get_hsv_gains()

    def transform(self, results: dict) -> dict:
        if self.hsv_gains is None:
            return results

        img = results["img"]

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + self.hsv_gains[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + self.hsv_gains[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + self.hsv_gains[2], 0, 255)
        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)

        results["img"] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(hue_delta={self.hue_delta}, "
        repr_str += f"saturation_delta={self.saturation_delta}, "
        repr_str += f"value_delta={self.value_delta})"
        return repr_str
