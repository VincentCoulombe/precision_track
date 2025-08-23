# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from mmengine import is_list_of
from mmengine.config import ConfigDict
from mmengine.dist import get_dist_info

from precision_track.datasets.codecs import *  # noqa
from precision_track.registry import KEYPOINT_CODECS, TRANSFORMS
from precision_track.utils import clip, reformat

from .base import BaseTransform
from .formatting import flip_bbox, flip_keypoints, get_udp_warp_matrix, get_warp_matrix
from .utils import cache_randomness


@TRANSFORMS.register_module()
class Resize(BaseTransform):
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
                scale = np.array(
                    [
                        img_w * padded_input_size[0] / actual_input_size[0],
                        img_h * padded_input_size[1] / actual_input_size[1],
                    ],
                    dtype=np.float32,
                )
                warp_mat = get_warp_matrix(center=center, scale=scale, rot=0, output_size=padded_input_size)

            _img = cv2.warpAffine(
                img,
                warp_mat,
                padded_input_size,
                flags=cv2.INTER_LINEAR,
                borderValue=self.pad_val,
            )

            imgs.append(_img)

            # Store the transform information w.r.t. the main input size
            if i == 0:
                results["img_shape"] = padded_input_size[::-1]
                results["input_center"] = center
                results["input_scale"] = scale
                results["input_size"] = padded_input_size

            if "keypoints" in results:
                # Only transform (x, y) coordinates
                kpts = cv2.transform(results["keypoints"], warp_mat)
                if kpts.shape[-1] == 3:
                    kpts = kpts[..., :2] / kpts[..., 2:3]
                results["keypoints"] = kpts
                results["keypoints"], results["keypoints_visible"] = clip(
                    results["keypoints"], "keypoints_vis", max_width=w, max_height=h, keypoints_visible=results["keypoints_visible"]
                )

            if "bbox" in results:
                bbox = reformat(results["bbox"], "xyxy", "corner")
                bbox = cv2.transform(bbox, warp_mat)
                if bbox.shape[-1] == 3:
                    bbox = bbox[..., :2] / bbox[..., 2:3]
                bbox = reformat(bbox, "corner", "xyxy")
                bbox = clip(bbox, "xyxy", w, h)
                results["bbox"] = bbox

            if "area" in results:
                warp_mat_for_area = warp_mat
                if warp_mat.shape[0] == 2:
                    aux_row = np.array([[0.0, 0.0, 1.0]], dtype=warp_mat.dtype)
                    warp_mat_for_area = np.concatenate((warp_mat, aux_row))
                results["area"] *= np.linalg.det(warp_mat_for_area)

        if self.aug_scales:
            results["img"] = imgs
            results["aug_scales"] = self.aug_scales
        else:
            results["img"] = imgs[0]
            results["aug_scale"] = None

        return results


def imflip(img: np.ndarray, direction: str = "horizontal") -> np.ndarray:
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image.
    """
    assert direction in ["horizontal", "vertical", "diagonal"]
    if direction == "horizontal":
        return np.flip(img, axis=1)
    elif direction == "vertical":
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))


@TRANSFORMS.register_module()
class RandomFlip(BaseTransform):
    """Randomly flip the image, bbox and keypoints.

    Required Keys:

        - img
        - img_shape
        - flip_indices
        - input_size (optional)
        - bbox (optional)
        - bbox_center (optional)
        - keypoints (optional)
        - keypoints_visible (optional)
        - img_mask (optional)

    Modified Keys:

        - img
        - bbox (optional)
        - bbox_center (optional)
        - keypoints (optional)
        - keypoints_visible (optional)
        - img_mask (optional)

    Added Keys:

        - flip
        - flip_direction

    Args:
        prob (float | list[float]): The flipping probability. If a list is
            given, the argument `direction` should be a list with the same
            length. And each element in `prob` indicates the flipping
            probability of the corresponding one in ``direction``. Defaults
            to 0.5
        direction (str | list[str]): The flipping direction. Options are
            ``'horizontal'``, ``'vertical'`` and ``'diagonal'``. If a list is
            is given, each data sample's flipping direction will be sampled
            from a distribution determined by the argument ``prob``. Defaults
            to ``'horizontal'``.
    """

    def __init__(
        self,
        prob: Union[float, List[float]] = 0.5,
        direction: Union[str, List[str]] = "horizontal",
    ) -> None:
        if isinstance(prob, list):
            assert is_list_of(prob, float)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        else:
            raise ValueError(
                f"probs must be float or list of float, but \
                              got `{type(prob)}`."
            )
        self.prob = prob

        valid_directions = ["horizontal", "vertical", "diagonal"]
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError(
                f"direction must be either str or list of str, \
                               but got `{type(direction)}`."
            )
        self.direction = direction

        if isinstance(prob, list):
            assert len(prob) == len(self.direction)

    @cache_randomness
    def _choose_direction(self) -> str:
        """Choose the flip direction according to `prob` and `direction`"""
        if isinstance(self.direction, List) and not isinstance(self.direction, str):
            # None means non-flip
            direction_list: list = list(self.direction) + [None]
        elif isinstance(self.direction, str):
            # None means non-flip
            direction_list = [self.direction, None]

        if isinstance(self.prob, list):
            non_prob: float = 1 - sum(self.prob)
            prob_list = self.prob + [non_prob]
        elif isinstance(self.prob, float):
            non_prob = 1.0 - self.prob
            # exclude non-flip
            single_ratio = self.prob / (len(direction_list) - 1)
            prob_list = [single_ratio] * (len(direction_list) - 1) + [non_prob]

        cur_dir = np.random.choice(direction_list, p=prob_list)

        return cur_dir

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`RandomFlip`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        if "flip" in results and "flip_direction" in results:
            flip_dir = results["flip_direction"]
        else:
            flip_dir = self._choose_direction()

        if flip_dir is None:
            results["flip"] = False
            results["flip_direction"] = None
        else:
            results["flip"] = True
            results["flip_direction"] = flip_dir

            h, w = results.get("input_size", results["img_shape"])
            # flip image and mask
            if isinstance(results["img"], list):
                results["img"] = [imflip(img, direction=flip_dir) for img in results["img"]]
            else:
                results["img"] = imflip(results["img"], direction=flip_dir)

            if "img_mask" in results:
                results["img_mask"] = imflip(results["img_mask"], direction=flip_dir)

            # flip bboxes
            if results.get("bbox", None) is not None:
                results["bbox"] = flip_bbox(
                    results["bbox"],
                    image_size=(w, h),
                    bbox_format="xyxy",
                    direction=flip_dir,
                )

            if results.get("bbox_center", None) is not None:
                results["bbox_center"] = flip_bbox(
                    results["bbox_center"],
                    image_size=(w, h),
                    bbox_format="center",
                    direction=flip_dir,
                )

            # flip keypoints
            if results.get("keypoints", None) is not None:
                keypoints, keypoints_visible = flip_keypoints(
                    results["keypoints"],
                    results.get("keypoints_visible", None),
                    image_size=(w, h),
                    flip_indices=results["flip_indices"],
                    direction=flip_dir,
                )

                results["keypoints"] = keypoints
                results["keypoints_visible"] = keypoints_visible

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f"(prob={self.prob}, "
        repr_str += f"direction={self.direction})"
        return repr_str


@TRANSFORMS.register_module()
class GenerateTarget(BaseTransform):
    """Encode keypoints into Target.

    The generated target is usually the supervision signal of the model
    learning, e.g. heatmaps or regression labels.

    Required Keys:

        - keypoints
        - keypoints_visible
        - dataset_keypoint_weights

    Added Keys:

        - The keys of the encoded items from the codec will be updated into
            the results, e.g. ``'heatmaps'`` or ``'keypoint_weights'``. See
            the specific codec for more details.

    Args:
        encoder (dict | list[dict]): The codec config for keypoint encoding.
            Both single encoder and multiple encoders (given as a list) are
            supported
        multilevel (bool): Determine the method to handle multiple encoders.
            If ``multilevel==True``, generate multilevel targets from a group
            of encoders of the same type (e.g. multiple :class:`MSRAHeatmap`
            encoders with different sigma values); If ``multilevel==False``,
            generate combined targets from a group of different encoders. This
            argument will have no effect in case of single encoder. Defaults
            to ``False``
        use_dataset_keypoint_weights (bool): Whether use the keypoint weights
            from the dataset meta information. Defaults to ``False``
        target_type (str, deprecated): This argument is deprecated and has no
            effect. Defaults to ``None``
    """

    def __init__(
        self,
        encoder: ConfigDict,
        target_type: Optional[str] = None,
        multilevel: bool = False,
        use_dataset_keypoint_weights: bool = False,
    ) -> None:
        super().__init__()

        if target_type is not None:
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn(
                    "The argument `target_type` is deprecated in" " GenerateTarget. The target type and encoded " "keys will be determined by encoder(s).",
                    DeprecationWarning,
                )

        self.encoder_cfg = deepcopy(encoder)
        self.multilevel = multilevel
        self.use_dataset_keypoint_weights = use_dataset_keypoint_weights

        if isinstance(self.encoder_cfg, list):
            self.encoder = [KEYPOINT_CODECS.build(cfg) for cfg in self.encoder_cfg]
        else:
            assert not self.multilevel, "Need multiple encoder configs if ``multilevel==True``"
            self.encoder = KEYPOINT_CODECS.build(self.encoder_cfg)

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`GenerateTarget`.

        See ``transform()`` method of :class:`BaseTransform` for details.
        """

        if results.get("transformed_keypoints", None) is not None:
            # use keypoints transformed by TopdownAffine
            keypoints = results["transformed_keypoints"]
        else:
            if results.get("keypoints", None) is None:
                results["bbox"] = np.array([])
                results["bbox_score"] = np.array([])
                results["num_keypoints"] = np.array([])
                results["keypoints"] = np.array([])
                results["keypoints_visible"] = np.array([])
                results["area"] = np.array([])
            keypoints = results["keypoints"]

        keypoints_visible = results["keypoints_visible"]
        if keypoints_visible.ndim == 3 and keypoints_visible.shape[2] == 2:
            keypoints_visible, keypoints_visible_weights = (
                keypoints_visible[..., 0],
                keypoints_visible[..., 1],
            )
            results["keypoints_visible"] = keypoints_visible
            results["keypoints_visible_weights"] = keypoints_visible_weights

        # Encoded items from the encoder(s) will be updated into the results.
        # Please refer to the document of the specific codec for details about
        # encoded items.
        if not isinstance(self.encoder, list):
            # For single encoding, the encoded items will be directly added
            # into results.
            auxiliary_encode_kwargs = {key: results.get(key, None) for key in self.encoder.auxiliary_encode_keys}
            encoded = self.encoder.encode(
                keypoints=keypoints,
                keypoints_visible=keypoints_visible,
                **auxiliary_encode_kwargs,
            )

            if self.encoder.field_mapping_table:
                encoded["field_mapping_table"] = self.encoder.field_mapping_table
            if self.encoder.instance_mapping_table:
                encoded["instance_mapping_table"] = self.encoder.instance_mapping_table
            if self.encoder.label_mapping_table:
                encoded["label_mapping_table"] = self.encoder.label_mapping_table

        else:
            encoded_list = []
            _field_mapping_table = dict()
            _instance_mapping_table = dict()
            _label_mapping_table = dict()
            for _encoder in self.encoder:
                auxiliary_encode_kwargs = {key: results[key] for key in _encoder.auxiliary_encode_keys}
                encoded_list.append(
                    _encoder.encode(
                        keypoints=keypoints,
                        keypoints_visible=keypoints_visible,
                        **auxiliary_encode_kwargs,
                    )
                )

                _field_mapping_table.update(_encoder.field_mapping_table)
                _instance_mapping_table.update(_encoder.instance_mapping_table)
                _label_mapping_table.update(_encoder.label_mapping_table)

            if self.multilevel:
                # For multilevel encoding, the encoded items from each encoder
                # should have the same keys.

                keys = encoded_list[0].keys()
                if not all(_encoded.keys() == keys for _encoded in encoded_list):
                    raise ValueError("Encoded items from all encoders must have the same " "keys if ``multilevel==True``.")

                encoded = {k: [_encoded[k] for _encoded in encoded_list] for k in keys}

            else:
                # For combined encoding, the encoded items from different
                # encoders should have no overlapping items, except for
                # `keypoint_weights`. If multiple `keypoint_weights` are given,
                # they will be multiplied as the final `keypoint_weights`.

                encoded = dict()
                keypoint_weights = []

                for _encoded in encoded_list:
                    for key, value in _encoded.items():
                        if key == "keypoint_weights":
                            keypoint_weights.append(value)
                        elif key not in encoded:
                            encoded[key] = value
                        else:
                            raise ValueError(f'Overlapping item "{key}" from multiple ' "encoders, which is not supported when " "``multilevel==False``")

                if keypoint_weights:
                    encoded["keypoint_weights"] = keypoint_weights

            if _field_mapping_table:
                encoded["field_mapping_table"] = _field_mapping_table
            if _instance_mapping_table:
                encoded["instance_mapping_table"] = _instance_mapping_table
            if _label_mapping_table:
                encoded["label_mapping_table"] = _label_mapping_table

        if self.use_dataset_keypoint_weights and "keypoint_weights" in encoded:
            if isinstance(encoded["keypoint_weights"], list):
                for w in encoded["keypoint_weights"]:
                    w = w * results["dataset_keypoint_weights"]
            else:
                encoded["keypoint_weights"] = encoded["keypoint_weights"] * results["dataset_keypoint_weights"]

        results.update(encoded)

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f"(encoder={str(self.encoder_cfg)}, "
        repr_str += "use_dataset_keypoint_weights=" f"{self.use_dataset_keypoint_weights})"
        return repr_str


@TRANSFORMS.register_module()
class GenerateActionRecognitionTargets(BaseTransform):
    pass


@TRANSFORMS.register_module()
class GenerateSequenceTargets(BaseTransform):
    pass


@TRANSFORMS.register_module()
class YOLOXHSVRandomAug(BaseTransform):
    """Apply Hue Saturation Value augmentation to image sequentially. It is referenced from
    https://github.com/Megvii-
    BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py#L21.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        hue_delta (int): delta of hue. Defaults to 5.
        saturation_delta (int): delta of saturation. Defaults to 30.
        value_delta (int): delat of value. Defaults to 30.
    """

    def __init__(self, hue_delta: int = 5, saturation_delta: int = 30, value_delta: int = 30) -> None:
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    @cache_randomness
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

    def transform(self, results: dict) -> dict:
        img = results["img"]
        hsv_gains = self._get_hsv_gains()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)

        results["img"] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(hue_delta={self.hue_delta}, "
        repr_str += f"saturation_delta={self.saturation_delta}, "
        repr_str += f"value_delta={self.value_delta})"
        return repr_str


@TRANSFORMS.register_module()
class RandomContrastAug(BaseTransform):
    """Randomly adjust the contrast of an image within a specified range.

    Required Keys:
    - img

    Modified Keys:
    - img

    Args:
        contrast_range (tuple): A tuple of two floats indicating the min and max
            contrast factor. 1.0 means no change.
    """

    def __init__(self, contrast_range=(0.75, 1.25)) -> None:
        assert isinstance(contrast_range, tuple) and len(contrast_range) == 2, "contrast_range must be a tuple of two floats"
        self.contrast_range = contrast_range

    @cache_randomness
    def _get_contrast_factor(self):
        return np.random.uniform(*self.contrast_range)

    def transform(self, results: dict) -> dict:
        img = results["img"]
        factor = self._get_contrast_factor()

        img = img.astype(np.float32)
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * factor + mean
        img = np.clip(img, 0, 255).astype(np.uint8)

        results["img"] = img
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(contrast_range={self.contrast_range})"


@TRANSFORMS.register_module()
class FilterAnnotations(BaseTransform):
    """Eliminate undesirable annotations based on specific conditions.

    This class is designed to sift through annotations by examining multiple
    factors such as the size of the bounding box, the visibility of keypoints,
    and the overall area. Users can fine-tune the criteria to filter out
    instances that have excessively small bounding boxes, insufficient area,
    or an inadequate number of visible keypoints.

    Required Keys:

    - bbox (np.ndarray) (optional)
    - area (np.int64) (optional)
    - keypoints_visible (np.ndarray) (optional)

    Modified Keys:

    - bbox (optional)
    - bbox_score (optional)
    - category_id (optional)
    - keypoints (optional)
    - keypoints_visible (optional)
    - area (optional)

    Args:
        min_gt_bbox_wh (tuple[float]): Minimum width and height of ground
            truth boxes. Default: (1., 1.)
        min_gt_area (int): Minimum foreground area of instances.
            Default: 1
        min_kpt_vis (int): Minimum number of visible keypoints. Default: 1
        by_box (bool): Filter instances with bounding boxes not meeting the
            min_gt_bbox_wh threshold. Default: False
        by_area (bool): Filter instances with area less than min_gt_area
            threshold. Default: False
        by_kpt (bool): Filter instances with keypoints_visible not meeting the
            min_kpt_vis threshold. Default: True
        keep_empty (bool): Whether to return None when it
            becomes an empty bbox after filtering. Defaults to True.
    """

    def __init__(
        self,
        min_gt_bbox_wh: Tuple[int, int] = (1, 1),
        min_gt_area: int = 1,
        min_kpt_vis: int = 1,
        by_box: bool = False,
        by_area: bool = False,
        by_kpt: bool = True,
        keep_empty: bool = True,
    ) -> None:

        assert by_box or by_kpt or by_area
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.min_gt_area = min_gt_area
        self.min_kpt_vis = min_kpt_vis
        self.by_box = by_box
        self.by_area = by_area
        self.by_kpt = by_kpt
        self.keep_empty = keep_empty

    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to filter annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        assert "keypoints" in results
        kpts = results["keypoints"]
        if kpts.shape[0] == 0:
            return results

        tests = []
        if self.by_box and "bbox" in results:
            bbox = results["bbox"]
            tests.append(((bbox[..., 2] - bbox[..., 0] > self.min_gt_bbox_wh[0]) & (bbox[..., 3] - bbox[..., 1] > self.min_gt_bbox_wh[1])))
        if self.by_area and "area" in results:
            area = results["area"]
            tests.append(area >= self.min_gt_area)
        if self.by_kpt and results["num_keypoints"][0] > 0:
            kpts_vis = results["keypoints_visible"]
            if kpts_vis.ndim == 3:
                kpts_vis = kpts_vis[..., 0]
            tests.append(kpts_vis.sum(axis=1) >= self.min_kpt_vis)

        keep = tests[0]
        for t in tests[1:]:
            keep = keep & t

        if not keep.any():
            if self.keep_empty:
                return None

        keys = (
            "bbox",
            "bbox_score",
            "category_id",
            "keypoints",
            "keypoints_visible",
            "area",
            "id",
            "action",
            "action_label",
        )
        for key in keys:
            if key in results:
                results[key] = np.atleast_1d(np.array(results[key]))[keep]

        return results


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    """Random crop the image & bboxes & masks.

    The absolute ``crop_size`` is sampled based on ``crop_type`` and
    ``image_size``, then the cropped results are generated.

    Required Keys:

        - img
        - keypoints
        - bbox (optional)
        - masks (BitmapMasks | PolygonMasks) (optional)

    Modified Keys:

        - img
        - img_shape
        - keypoints
        - keypoints_visible
        - num_keypoints
        - bbox (optional)
        - bbox_score (optional)
        - id (optional)
        - category_id (optional)
        - raw_ann_info (optional)
        - iscrowd (optional)
        - segmentation (optional)
        - masks (optional)

    Added Keys:

        - warp_mat

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            (width, height).
        crop_type (str, optional): One of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])].
            Defaults to "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Defaults to False.
        recompute_bbox (bool, optional): Whether to re-compute the boxes based
            on cropped instance masks. Defaults to False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - If the crop does not contain any gt-bbox region and
          ``allow_negative_crop`` is set to False, skip this image.
    """

    def __init__(
        self, crop_size: tuple, crop_type: str = "absolute", allow_negative_crop: bool = False, recompute_bbox: bool = False, bbox_clip_border: bool = True
    ) -> None:
        if crop_type not in ["relative_range", "relative", "absolute", "absolute_range"]:
            raise ValueError(f"Invalid crop_type {crop_type}.")
        if crop_type in ["absolute", "absolute_range"]:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(crop_size[1], int)
            if crop_type == "absolute_range":
                assert crop_size[0] <= crop_size[1]
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        self.recompute_bbox = recompute_bbox

    def _crop_data(self, results: dict, crop_size: Tuple[int, int], allow_negative_crop: bool) -> Union[dict, None]:
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (Tuple[int, int]): Expected absolute size after
                cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        img = results["img"]
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h, offset_w = self._rand_offset((margin_h, margin_w))
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        # Record the warp matrix for the RandomCrop
        warp_mat = np.array([[1, 0, -offset_w], [0, 1, -offset_h], [0, 0, 1]], dtype=np.float32)
        if results.get("warp_mat", None) is None:
            results["warp_mat"] = warp_mat
        else:
            results["warp_mat"] = warp_mat @ results["warp_mat"]

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results["img"] = img
        results["img_shape"] = img_shape[:2]

        # crop bboxes accordingly and clip to the image boundary
        if results.get("bbox", None) is not None:
            distances = (-offset_w, -offset_h)
            bboxes = results["bbox"]
            bboxes = bboxes + np.tile(np.asarray(distances), 2)

            if self.bbox_clip_border:
                bboxes[..., 0::2] = bboxes[..., 0::2].clip(0, img_shape[1])
                bboxes[..., 1::2] = bboxes[..., 1::2].clip(0, img_shape[0])

            valid_inds = (bboxes[..., 0] < img_shape[1]) & (bboxes[..., 1] < img_shape[0]) & (bboxes[..., 2] > 0) & (bboxes[..., 3] > 0)

            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if not valid_inds.any() and not allow_negative_crop:
                return None

            results["bbox"] = bboxes[valid_inds]
            meta_keys = ["bbox_score", "id", "category_id", "raw_ann_info", "iscrowd"]
            for key in meta_keys:
                if results.get(key):
                    if isinstance(results[key], list):
                        results[key] = np.asarray(results[key])[valid_inds].tolist()
                    else:
                        results[key] = results[key][valid_inds]

            if results.get("keypoints", None) is not None:
                keypoints = results["keypoints"]
                distances = np.asarray(distances).reshape(1, 1, 2)
                keypoints = keypoints + distances
                if self.bbox_clip_border:
                    keypoints_outside_x = keypoints[:, :, 0] < 0
                    keypoints_outside_y = keypoints[:, :, 1] < 0
                    keypoints_outside_width = keypoints[:, :, 0] > img_shape[1]
                    keypoints_outside_height = keypoints[:, :, 1] > img_shape[0]

                    kpt_outside = np.logical_or.reduce((keypoints_outside_x, keypoints_outside_y, keypoints_outside_width, keypoints_outside_height))

                    results["keypoints_visible"][kpt_outside] *= 0
                keypoints[:, :, 0] = keypoints[:, :, 0].clip(0, img_shape[1])
                keypoints[:, :, 1] = keypoints[:, :, 1].clip(0, img_shape[0])
                results["keypoints"] = keypoints[valid_inds]
                results["keypoints_visible"] = results["keypoints_visible"][valid_inds]

            if results.get("segmentation", None) is not None:
                results["segmentation"] = results["segmentation"][crop_y1:crop_y2, crop_x1:crop_x2]

            if results.get("masks", None) is not None:
                results["masks"] = results["masks"][valid_inds.nonzero()[0]].crop(np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results["bbox"] = results["masks"].get_bboxes(type(results["bbox"]))

        return results

    @cache_randomness
    def _rand_offset(self, margin: Tuple[int, int]) -> Tuple[int, int]:
        """Randomly generate crop offset.

        Args:
            margin (Tuple[int, int]): The upper bound for the offset generated
                randomly.

        Returns:
            Tuple[int, int]: The random offset for the crop.
        """
        margin_h, margin_w = margin
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        return offset_h, offset_w

    @cache_randomness
    def _get_crop_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.

        Args:
            image_size (Tuple[int, int]): (h, w).

        Returns:
            crop_size (Tuple[int, int]): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == "absolute":
            return min(self.crop_size[1], h), min(self.crop_size[0], w)
        elif self.crop_type == "absolute_range":
            crop_h = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w
        elif self.crop_type == "relative":
            crop_w, crop_h = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        else:
            # 'relative_range'
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        image_size = results["img"].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results
