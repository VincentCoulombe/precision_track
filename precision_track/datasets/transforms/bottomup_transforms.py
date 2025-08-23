# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.stats import truncnorm

from precision_track.registry import TRANSFORMS
from precision_track.utils import clip, reformat

from .base import BaseTransform
from .formatting import get_pers_warp_matrix, get_udp_warp_matrix, get_warp_matrix
from .utils import cache_randomness


@TRANSFORMS.register_module()
class BottomupRandomAffine(BaseTransform):
    r"""Randomly shift, resize and rotate the image.

    Required Keys:

        - img
        - img_shape
        - keypoints (optional)

    Modified Keys:

        - img
        - keypoints (optional)

    Added Keys:

        - input_size
        - warp_mat

    Args:
        input_size (Tuple[int, int]): The input image size of the model in
            [w, h]
        shift_factor (float): Randomly shift the image in range
            :math:`[-dx, dx]` and :math:`[-dy, dy]` in X and Y directions,
            where :math:`dx(y) = img_w(h) \cdot shift_factor` in pixels.
            Defaults to 0.2
        shift_prob (float): Probability of applying random shift. Defaults to
            1.0
        scale_factor (Tuple[float, float]): Randomly resize the image in range
            :math:`[scale_factor[0], scale_factor[1]]`. Defaults to
            (0.75, 1.5)
        scale_prob (float): Probability of applying random resizing. Defaults
            to 1.0
        scale_type (str): wrt ``long`` or ``short`` length of the image.
            Defaults to ``short``
        rotate_factor (float): Randomly rotate the bbox in
            :math:`[-rotate_factor, rotate_factor]` in degrees. Defaults
            to 40.0
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(
        self,
        input_size: Optional[Tuple[int, int]] = None,
        shift_factor: float = 0.2,
        shift_prob: float = 1.0,
        scale_factor: Tuple[float, float] = (0.75, 1.5),
        scale_prob: float = 1.0,
        scale_type: str = "short",
        rotate_factor: float = 30.0,
        rotate_prob: float = 1,
        shear_factor: float = 2.0,
        shear_prob: float = 1.0,
        use_udp: bool = False,
        pad_val: Union[float, Tuple[float]] = 0,
        border: Tuple[int, int] = (0, 0),
        distribution="trunc_norm",
        transform_mode="affine",
        bbox_keep_corner: bool = True,
        clip_border: bool = False,
    ) -> None:
        super().__init__()

        assert transform_mode in ("affine", "affine_udp", "perspective"), (
            f"the argument transform_mode should be either 'affine', " f"'affine_udp' or 'perspective', but got '{transform_mode}'"
        )

        self.input_size = input_size
        self.shift_factor = shift_factor
        self.shift_prob = shift_prob
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob
        self.scale_type = scale_type
        self.rotate_factor = rotate_factor
        self.rotate_prob = rotate_prob
        self.shear_factor = shear_factor
        self.shear_prob = shear_prob

        self.use_udp = use_udp
        self.distribution = distribution
        self.clip_border = clip_border
        self.bbox_keep_corner = bbox_keep_corner

        self.transform_mode = transform_mode

        if isinstance(pad_val, (int, float)):
            pad_val = (pad_val, pad_val, pad_val)

        if "affine" in transform_mode:
            self._transform = partial(cv2.warpAffine, flags=cv2.INTER_LINEAR, borderValue=pad_val)
        else:
            self._transform = partial(cv2.warpPerspective, borderValue=pad_val)

    def _random(self, low: float = -1.0, high: float = 1.0, size: tuple = ()) -> np.ndarray:
        if self.distribution == "trunc_norm":
            """Sample from a truncated normal distribution."""
            return truncnorm.rvs(low, high, size=size).astype(np.float32)
        elif self.distribution == "uniform":
            x = np.random.rand(*size)
            return x * (high - low) + low
        else:
            raise ValueError(f"the argument `distribution` should be either" f"'trunc_norn' or 'uniform', but got " f"{self.distribution}.")

    def _fix_aspect_ratio(self, scale: np.ndarray, aspect_ratio: float):
        """Extend the scale to match the given aspect ratio.

        Args:
            scale (np.ndarray): The image scale (w, h) in shape (2, )
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.ndarray: The reshaped image scale in (2, )
        """
        w, h = scale
        if w > h * aspect_ratio:
            if self.scale_type == "long":
                _w, _h = w, w / aspect_ratio
            elif self.scale_type == "short":
                _w, _h = h * aspect_ratio, h
            else:
                raise ValueError(f"Unknown scale type: {self.scale_type}")
        else:
            if self.scale_type == "short":
                _w, _h = w, w / aspect_ratio
            elif self.scale_type == "long":
                _w, _h = h * aspect_ratio, h
            else:
                raise ValueError(f"Unknown scale type: {self.scale_type}")
        return np.array([_w, _h], dtype=scale.dtype)

    @cache_randomness
    def _get_transform_params(self) -> Tuple:
        """Get random transform parameters.

        Returns:
            tuple:
            - offset (np.ndarray): Image offset rate in shape (2, )
            - scale (np.ndarray): Image scaling rate factor in shape (1, )
            - rotate (np.ndarray): Image rotation degree in shape (1, )
        """
        # get offset
        if np.random.rand() < self.shift_prob:
            offset = self._random(size=(2,)) * self.shift_factor
        else:
            offset = np.zeros((2,), dtype=np.float32)

        # get scale
        if np.random.rand() < self.scale_prob:
            scale_min, scale_max = self.scale_factor
            scale = scale_min + (scale_max - scale_min) * (self._random(size=(1,)) + 1) / 2
        else:
            scale = np.ones(1, dtype=np.float32)

        # get rotation
        if np.random.rand() < self.rotate_prob:
            rotate = self._random() * self.rotate_factor
        else:
            rotate = 0

        # get shear
        if "perspective" in self.transform_mode and np.random.rand() < self.shear_prob:
            shear = self._random(size=(2,)) * self.shear_factor
        else:
            shear = np.zeros((2,), dtype=np.float32)

        return offset, scale, rotate, shear

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`BottomupRandomAffine` to perform
        photometric distortion on images.

        See ``transform()`` method of :class:`BaseTransform` for details.


        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img_h, img_w = results["img_shape"][:2]
        w, h = self.input_size

        offset_rate, scale_rate, rotate, shear = self._get_transform_params()

        if "affine" in self.transform_mode:
            offset = offset_rate * [img_w, img_h]
            scale = scale_rate * [img_w, img_h]
            # adjust the scale to match the target aspect ratio
            scale = self._fix_aspect_ratio(scale, aspect_ratio=w / h)

            if self.transform_mode == "affine_udp":
                center = np.array([(img_w - 1.0) / 2, (img_h - 1.0) / 2], dtype=np.float32)
                warp_mat = get_udp_warp_matrix(center=center + offset, scale=scale, rot=rotate, output_size=(w, h))
            else:
                center = np.array([img_w / 2, img_h / 2], dtype=np.float32)
                warp_mat = get_warp_matrix(center=center + offset, scale=scale, rot=rotate, output_size=(w, h))

        else:
            offset = offset_rate * [w, h]
            center = np.array([w / 2, h / 2], dtype=np.float32)
            warp_mat = get_pers_warp_matrix(center=center, translate=offset, scale=scale_rate[0], rot=rotate, shear=shear)

        if "warp_mat" in results:
            warp_mat = results["warp_mat"]

        # warp image and keypoints
        results["img"] = self._transform(results["img"], warp_mat, (int(w), int(h)))

        if "keypoints" in results:
            # Only transform (x, y) coordinates
            kpts = cv2.transform(results["keypoints"], warp_mat)
            if kpts.shape[-1] == 3:
                kpts = kpts[..., :2] / kpts[..., 2:3]
            results["keypoints"] = kpts

            if self.clip_border:
                results["keypoints"], results["keypoints_visible"] = clip(
                    results["keypoints"], "keypoints_vis", max_width=w, max_height=h, keypoints_visible=results["keypoints_visible"]
                )

        if "bbox" in results:
            bbox = reformat(results["bbox"], "xyxy", "corner")
            bbox = cv2.transform(bbox, warp_mat)
            if bbox.shape[-1] == 3:
                bbox = bbox[..., :2] / bbox[..., 2:3]
            if not self.bbox_keep_corner:
                bbox = reformat(bbox, "corner", "xyxy")
            if self.clip_border:
                bbox = clip(bbox, "xyxy", w, h)
            results["bbox"] = bbox

        if "area" in results:
            warp_mat_for_area = warp_mat
            if warp_mat.shape[0] == 2:
                aux_row = np.array([[0.0, 0.0, 1.0]], dtype=warp_mat.dtype)
                warp_mat_for_area = np.concatenate((warp_mat, aux_row))
            results["area"] *= np.linalg.det(warp_mat_for_area)

        results["input_size"] = self.input_size
        results["warp_mat"] = warp_mat

        return results


@TRANSFORMS.register_module()
class BottomupResize(BaseTransform):
    """Resize the image to the input size of the model. Optionally, the image
    can be resized to multiple sizes to build a image pyramid for multi-scale
    inference.

    Required Keys:

        - img
        - ori_shape

    Modified Keys:

        - img
        - img_shape

    Added Keys:

        - input_size
        - warp_mat
        - aug_scale

    Args:
        input_size (Tuple[int, int]): The input size of the model in [w, h].
            Note that the actually size of the resized image will be affected
            by ``resize_mode`` and ``size_factor``, thus may not exactly equals
            to the ``input_size``
        aug_scales (List[float], optional): The extra input scales for
            multi-scale testing. If given, the input image will be resized
            to different scales to build a image pyramid. And heatmaps from
            all scales will be aggregated to make final prediction. Defaults
            to ``None``
        size_factor (int): The actual input size will be ceiled to
                a multiple of the `size_factor` value at both sides.
                Defaults to 16
        resize_mode (str): The method to resize the image to the input size.
            Options are:

                - ``'fit'``: The image will be resized according to the
                    relatively longer side with the aspect ratio kept. The
                    resized image will entirely fits into the range of the
                    input size
                - ``'expand'``: The image will be resized according to the
                    relatively shorter side with the aspect ratio kept. The
                    resized image will exceed the given input size at the
                    longer side
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        aug_scales: Optional[List[float]] = None,
        size_factor: int = 32,
        resize_mode: str = "fit",
        pad_val: tuple = (0, 0, 0),
        use_udp: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.aug_scales = aug_scales
        self.resize_mode = resize_mode
        self.size_factor = size_factor
        self.use_udp = use_udp
        self.pad_val = pad_val

    @staticmethod
    def _ceil_to_multiple(size: Tuple[int, int], base: int):
        """Ceil the given size (tuple of [w, h]) to a multiple of the base."""
        return tuple(int(np.ceil(s / base) * base) for s in size)

    def _get_input_size(self, img_size: Tuple[int, int], input_size: Tuple[int, int]) -> Tuple:
        """Calculate the actual input size (which the original image will be
        resized to) and the padded input size (which the resized image will be
        padded to, or which is the size of the model input).

        Args:
            img_size (Tuple[int, int]): The original image size in [w, h]
            input_size (Tuple[int, int]): The expected input size in [w, h]

        Returns:
            tuple:
            - actual_input_size (Tuple[int, int]): The target size to resize
                the image
            - padded_input_size (Tuple[int, int]): The target size to generate
                the model input which will contain the resized image
        """
        img_w, img_h = img_size
        ratio = img_w / img_h

        if self.resize_mode == "fit":
            padded_input_size = self._ceil_to_multiple(input_size, self.size_factor)
            if padded_input_size != input_size:
                raise ValueError(
                    "When ``resize_mode=='fit', the input size (height and"
                    " width) should be mulitples of the size_factor("
                    f"{self.size_factor}) at all scales. Got invalid input "
                    f"size {input_size}."
                )

            pad_w, pad_h = padded_input_size
            rsz_w = min(pad_w, pad_h * ratio)
            rsz_h = min(pad_h, pad_w / ratio)
            actual_input_size = (rsz_w, rsz_h)

        elif self.resize_mode == "expand":
            _padded_input_size = self._ceil_to_multiple(input_size, self.size_factor)
            pad_w, pad_h = _padded_input_size
            rsz_w = max(pad_w, pad_h * ratio)
            rsz_h = max(pad_h, pad_w / ratio)

            actual_input_size = (rsz_w, rsz_h)
            padded_input_size = self._ceil_to_multiple(actual_input_size, self.size_factor)

        else:
            raise ValueError(f"Invalid resize mode {self.resize_mode}")

        return actual_input_size, padded_input_size

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`BottomupResize` to perform
        photometric distortion on images.

        See ``transform()`` method of :class:`BaseTransform` for details.


        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results["img"]
        img_h, img_w = results["ori_shape"]
        w, h = self.input_size

        input_sizes = [(w, h)]
        if self.aug_scales:
            input_sizes += [(int(w * s), int(h * s)) for s in self.aug_scales]

        imgs = []
        for i, (_w, _h) in enumerate(input_sizes):

            actual_input_size, padded_input_size = self._get_input_size(img_size=(img_w, img_h), input_size=(_w, _h))

            if self.use_udp:
                center = np.array([(img_w - 1.0) / 2, (img_h - 1.0) / 2], dtype=np.float32)
                scale = np.array([img_w, img_h], dtype=np.float32)
                warp_mat = get_udp_warp_matrix(center=center, scale=scale, rot=0, output_size=actual_input_size)
            else:
                center = np.array([img_w / 2, img_h / 2], dtype=np.float32)
                scale = np.array([img_w * padded_input_size[0] / actual_input_size[0], img_h * padded_input_size[1] / actual_input_size[1]], dtype=np.float32)
                warp_mat = get_warp_matrix(center=center, scale=scale, rot=0, output_size=padded_input_size)

            _img = cv2.warpAffine(img, warp_mat, padded_input_size, flags=cv2.INTER_LINEAR, borderValue=self.pad_val)

            imgs.append(_img)

            # Store the transform information w.r.t. the main input size
            if i == 0:
                results["img_shape"] = padded_input_size[::-1]
                results["input_center"] = center
                results["input_scale"] = scale
                results["input_size"] = padded_input_size

        if self.aug_scales:
            results["img"] = imgs
            results["aug_scales"] = self.aug_scales
        else:
            results["img"] = imgs[0]
            results["aug_scale"] = None

        return results
