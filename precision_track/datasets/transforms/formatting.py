# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from mmengine.structures import InstanceData, PixelData
from mmengine.utils import is_seq_of

from precision_track.registry import TRANSFORMS
from precision_track.utils import PoseDataSample

from .base import BaseTransform

try:
    from PIL import Image
except ImportError:
    Image = None

imread_backend = "cv2"

if Image is not None:
    if hasattr(Image, "Resampling"):
        pillow_interp_codes = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "box": Image.Resampling.BOX,
            "lanczos": Image.Resampling.LANCZOS,
            "hamming": Image.Resampling.HAMMING,
        }
    else:
        pillow_interp_codes = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "box": Image.BOX,
            "lanczos": Image.LANCZOS,
            "hamming": Image.HAMMING,
        }

cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def imresize(
    img: np.ndarray,
    size: Tuple[int, int],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    out: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
) -> Union[Tuple[np.ndarray, float, float], np.ndarray]:
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    h, w = img.shape[:2]
    if backend is None:
        backend = imread_backend
    if backend not in ["cv2", "pillow"]:
        raise ValueError(f"backend: {backend} is not supported for resize." f"Supported backends are 'cv2', 'pillow'")

    if backend == "pillow":
        assert img.dtype == np.uint8, "Pillow backend only support uint8 type"
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def image_to_tensor(img: Union[np.ndarray, Sequence[np.ndarray]]) -> torch.torch.Tensor:
    """Translate image or sequence of images to tensor. Multiple image tensors
    will be stacked.

    Args:
        value (np.ndarray | Sequence[np.ndarray]): The original image or
            image sequence

    Returns:
        torch.Tensor: The output tensor.
    """

    if isinstance(img, np.ndarray):
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)

        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    else:
        assert is_seq_of(img, np.ndarray)
        tensor = torch.stack([image_to_tensor(_img) for _img in img])

    return tensor


def keypoints_to_tensor(keypoints: Union[np.ndarray, Sequence[np.ndarray]]) -> torch.torch.Tensor:
    """Translate keypoints or sequence of keypoints to tensor. Multiple
    keypoints tensors will be stacked.

    Args:
        keypoints (np.ndarray | Sequence[np.ndarray]): The keypoints or
            keypoints sequence.

    Returns:
        torch.Tensor: The output tensor.
    """
    if isinstance(keypoints, np.ndarray):
        keypoints = np.ascontiguousarray(keypoints)
        tensor = torch.from_numpy(keypoints).contiguous()
    else:
        assert is_seq_of(keypoints, np.ndarray)
        tensor = torch.stack([keypoints_to_tensor(_keypoints) for _keypoints in keypoints])

    return tensor


@TRANSFORMS.register_module()
class PackPoseInputs(BaseTransform):
    """Pack the inputs data for pose estimation.

    The ``img_meta`` item is always populated. The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default it includes:

        - ``id``: id of the data sample

        - ``img_id``: id of the image

        - ``'category_id'``: the id of the instance category

        - ``img_path``: path to the image file

        - ``crowd_index`` (optional): measure the crowding level of an image,
            defined in CrowdPose dataset

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``input_size``: the input size to the network

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

        - ``flip_indices``: the indices of each keypoint's symmetric keypoint

        - ``raw_ann_info`` (optional): raw annotation of the instance(s)

    Args:
        meta_keys (Sequence[str], optional): Meta keys which will be stored in
            :obj: `PoseDataSample` as meta info. Defaults to ``('id',
            'img_id', 'img_path', 'category_id', 'crowd_index, 'ori_shape',
            'img_shape', 'input_size', 'input_center', 'input_scale', 'flip',
            'flip_direction', 'flip_indices', 'raw_ann_info')``
    """

    # items in `instance_mapping_table` will be directly packed into
    # PoseDataSample.gt_instances without converting to Tensor
    instance_mapping_table = dict(
        bbox="bboxes",
        bbox_score="bbox_scores",
        keypoints="keypoints",
        keypoints_cam="keypoints_cam",
        keypoints_visible="keypoints_visible",
        # In CocoMetric, the area of predicted instances will be calculated
        # using gt_instances.bbox_scales. To unsure correspondence with
        # previous version, this key is preserved here.
        bbox_scale="bbox_scales",
        # `head_size` is used for computing MpiiPCKAccuracy metric,
        # namely, PCKh
        head_size="head_size",
    )

    # items in `field_mapping_table` will be packed into
    # PoseDataSample.gt_fields and converted to Tensor. These items will be
    # used for computing losses
    field_mapping_table = dict(
        heatmaps="heatmaps",
        instance_heatmaps="instance_heatmaps",
        heatmap_mask="heatmap_mask",
        heatmap_weights="heatmap_weights",
        displacements="displacements",
        displacement_weights="displacement_weights",
    )

    # items in `label_mapping_table` will be packed into
    # PoseDataSample.gt_instance_labels and converted to Tensor. These items
    # will be used for computing losses
    label_mapping_table = dict(
        keypoint_labels="keypoint_labels",
        keypoint_weights="keypoint_weights",
        keypoints_visible_weights="keypoints_visible_weights",
    )

    def __init__(
        self,
        meta_keys=(
            "id",
            "img_id",
            "img_path",
            "category_id",
            "crowd_index",
            "ori_shape",
            "img_shape",
            "input_size",
            "input_center",
            "input_scale",
            "flip",
            "flip_direction",
            "flip_indices",
            "warp_mat",
            "raw_ann_info",
            "dataset_name",
        ),
        pack_transformed=False,
    ):
        self.meta_keys = meta_keys
        self.pack_transformed = pack_transformed

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_samples' (obj:`PoseDataSample`): The annotation info of the
                sample.
        """
        # Pack image(s) for 2d pose estimation
        if "img" in results:
            img = results["img"]
            inputs_tensor = image_to_tensor(img)
        # Pack keypoints for 3d pose-lifting
        elif "lifting_target" in results and "keypoints" in results:
            if "keypoint_labels" in results:
                keypoints = results["keypoint_labels"]
            else:
                keypoints = results["keypoints"]
            inputs_tensor = keypoints_to_tensor(keypoints)

        data_sample = PoseDataSample()

        # pack instance data
        gt_instances = InstanceData()
        _instance_mapping_table = results.get("instance_mapping_table", self.instance_mapping_table)
        for key, packed_key in _instance_mapping_table.items():
            if key in results:
                gt_instances.set_field(results[key], packed_key)

        # pack `transformed_keypoints` for visualizing data transform
        # and augmentation results
        if self.pack_transformed and "transformed_keypoints" in results:
            gt_instances.set_field(results["transformed_keypoints"], "transformed_keypoints")

        data_sample.gt_instances = gt_instances

        # pack instance labels
        gt_instance_labels = InstanceData()
        _label_mapping_table = results.get("label_mapping_table", self.label_mapping_table)
        for key, packed_key in _label_mapping_table.items():
            if key in results:
                if isinstance(results[key], list):
                    # A list of labels is usually generated by combined
                    # multiple encoders (See ``GenerateTarget`` in
                    # mmpose/datasets/transforms/common_transforms.py)
                    # In this case, labels in list should have the same
                    # shape and will be stacked.
                    _labels = np.stack(results[key])
                    gt_instance_labels.set_field(_labels, packed_key)
                else:
                    gt_instance_labels.set_field(results[key], packed_key)
        data_sample.gt_instance_labels = gt_instance_labels.to_tensor()

        # pack fields
        gt_fields = None
        _field_mapping_table = results.get("field_mapping_table", self.field_mapping_table)
        for key, packed_key in _field_mapping_table.items():
            if key in results:
                if isinstance(results[key], list):
                    if gt_fields is None:
                        gt_fields = PixelData()
                    else:
                        assert isinstance(gt_fields, PixelData), "Got mixed single-level and multi-level pixel data."
                else:
                    if gt_fields is None:
                        gt_fields = PixelData()
                    else:
                        assert isinstance(gt_fields, PixelData), "Got mixed single-level and multi-level pixel data."

                gt_fields.set_field(results[key], packed_key)

        if gt_fields:
            data_sample.gt_fields = gt_fields.to_tensor()

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)

        packed_results = dict()
        packed_results["inputs"] = inputs_tensor
        packed_results["data_samples"] = data_sample

        return packed_results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f"(meta_keys={self.meta_keys}, "
        repr_str += f"pack_transformed={self.pack_transformed})"
        return repr_str


def flip_bbox(bbox: np.ndarray, image_size: Tuple[int, int], bbox_format: str = "xywh", direction: str = "horizontal") -> np.ndarray:
    """Flip the bbox in the given direction.

    Args:
        bbox (np.ndarray): The bounding boxes. The shape should be (..., 4)
            if ``bbox_format`` is ``'xyxy'`` or ``'xywh'``, and (..., 2) if
            ``bbox_format`` is ``'center'``
        image_size (tuple): The image shape in [w, h]
        bbox_format (str): The bbox format. Options are ``'xywh'``, ``'xyxy'``
            and ``'center'``.
        direction (str): The flip direction. Options are ``'horizontal'``,
            ``'vertical'`` and ``'diagonal'``. Defaults to ``'horizontal'``

    Returns:
        np.ndarray: The flipped bounding boxes.
    """
    direction_options = {"horizontal", "vertical", "diagonal"}
    assert direction in direction_options, f'Invalid flipping direction "{direction}". ' f"Options are {direction_options}"

    format_options = {"xywh", "xyxy", "center"}
    assert bbox_format in format_options, f'Invalid bbox format "{bbox_format}". ' f"Options are {format_options}"

    bbox_flipped = bbox.copy()
    w, h = image_size

    # TODO: consider using "integer corner" coordinate system
    if direction == "horizontal":
        if bbox_format == "xywh" or bbox_format == "center":
            bbox_flipped[..., 0] = w - bbox[..., 0] - 1
        elif bbox_format == "xyxy":
            bbox_flipped[..., ::2] = w - bbox[..., -2::-2] - 1
    elif direction == "vertical":
        if bbox_format == "xywh" or bbox_format == "center":
            bbox_flipped[..., 1] = h - bbox[..., 1] - 1
        elif bbox_format == "xyxy":
            bbox_flipped[..., 1::2] = h - bbox[..., ::-2] - 1
    elif direction == "diagonal":
        if bbox_format == "xywh" or bbox_format == "center":
            bbox_flipped[..., :2] = [w, h] - bbox[..., :2] - 1
        elif bbox_format == "xyxy":
            bbox_flipped[...] = [w, h, w, h] - bbox - 1
            bbox_flipped = np.concatenate((bbox_flipped[..., 2:], bbox_flipped[..., :2]), axis=-1)

    return bbox_flipped


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """

    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def get_udp_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """Calculate the affine transformation matrix under the unbiased
    constraint. See `UDP (CVPR 2020)`_ for details.

    Note:

        - The bbox number: N

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (tuple): Size ([w, h]) of the output image

    Returns:
        np.ndarray: A 2x3 transformation matrix

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    input_size = center * 2
    rot_rad = np.deg2rad(rot)
    warp_mat = np.zeros((2, 3), dtype=np.float32)
    scale_x = (output_size[0] - 1) / scale[0]
    scale_y = (output_size[1] - 1) / scale[1]
    warp_mat[0, 0] = math.cos(rot_rad) * scale_x
    warp_mat[0, 1] = -math.sin(rot_rad) * scale_x
    warp_mat[0, 2] = scale_x * (-0.5 * input_size[0] * math.cos(rot_rad) + 0.5 * input_size[1] * math.sin(rot_rad) + 0.5 * scale[0])
    warp_mat[1, 0] = math.sin(rot_rad) * scale_y
    warp_mat[1, 1] = math.cos(rot_rad) * scale_y
    warp_mat[1, 2] = scale_y * (-0.5 * input_size[0] * math.sin(rot_rad) - 0.5 * input_size[1] * math.cos(rot_rad) + 0.5 * scale[1])
    return warp_mat


def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: Tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
    fix_aspect_ratio: bool = True,
) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)
        fix_aspect_ratio (bool): Whether to fix aspect ratio during transform.
            Defaults to True.

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    shift = np.array(shift)
    src_w, src_h = scale[:2]
    dst_w, dst_h = output_size[:2]

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([src_w * -0.5, 0.0]), rot_rad)
    dst_dir = np.array([dst_w * -0.5, 0.0])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    if fix_aspect_ratio:
        src[2, :] = _get_3rd_point(src[0, :], src[1, :])
        dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    else:
        src_dir_2 = _rotate_point(np.array([0.0, src_h * -0.5]), rot_rad)
        dst_dir_2 = np.array([0.0, dst_h * -0.5])
        src[2, :] = center + src_dir_2 + scale * shift
        dst[2, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir_2

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return warp_mat


def get_pers_warp_matrix(center: np.ndarray, translate: np.ndarray, scale: float, rot: float, shear: np.ndarray) -> np.ndarray:
    """Compute a perspective warp matrix based on specified transformations.

    Args:
        center (np.ndarray): Center of the transformation.
        translate (np.ndarray): Translation vector.
        scale (float): Scaling factor.
        rot (float): Rotation angle in degrees.
        shear (np.ndarray): Shearing angles in degrees along x and y axes.

    Returns:
        np.ndarray: Perspective warp matrix.

    Example:
        >>> center = np.array([0, 0])
        >>> translate = np.array([10, 20])
        >>> scale = 1.2
        >>> rot = 30.0
        >>> shear = np.array([15.0, 0.0])
        >>> warp_matrix = get_pers_warp_matrix(center, translate,
                                               scale, rot, shear)
    """
    translate_mat = np.array([[1, 0, translate[0] + center[0]], [0, 1, translate[1] + center[1]], [0, 0, 1]], dtype=np.float32)

    shear_x = math.radians(shear[0])
    shear_y = math.radians(shear[1])
    shear_mat = np.array([[1, np.tan(shear_x), 0], [np.tan(shear_y), 1, 0], [0, 0, 1]], dtype=np.float32)

    rotate_angle = math.radians(rot)
    rotate_mat = np.array([[np.cos(rotate_angle), -np.sin(rotate_angle), 0], [np.sin(rotate_angle), np.cos(rotate_angle), 0], [0, 0, 1]], dtype=np.float32)

    scale_mat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)

    recover_center_mat = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]], dtype=np.float32)

    warp_matrix = np.dot(np.dot(np.dot(np.dot(translate_mat, shear_mat), rotate_mat), scale_mat), recover_center_mat)

    return warp_matrix


def _get_3rd_point(a: np.ndarray, b: np.ndarray):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c


def flip_keypoints(
    keypoints: np.ndarray, keypoints_visible: Optional[np.ndarray], image_size: Tuple[int, int], flip_indices: List[int], direction: str = "horizontal"
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Flip keypoints in the given direction.

    Note:

        - keypoint number: K
        - keypoint dimension: D

    Args:
        keypoints (np.ndarray): Keypoints in shape (..., K, D)
        keypoints_visible (np.ndarray, optional): The visibility of keypoints
            in shape (..., K, 1) or (..., K, 2). Set ``None`` if the keypoint
            visibility is unavailable
        image_size (tuple): The image shape in [w, h]
        flip_indices (List[int]): The indices of each keypoint's symmetric
            keypoint
        direction (str): The flip direction. Options are ``'horizontal'``,
            ``'vertical'`` and ``'diagonal'``. Defaults to ``'horizontal'``

    Returns:
        tuple:
        - keypoints_flipped (np.ndarray): Flipped keypoints in shape
            (..., K, D)
        - keypoints_visible_flipped (np.ndarray, optional): Flipped keypoints'
            visibility in shape (..., K, 1) or (..., K, 2). Return ``None`` if
            the input ``keypoints_visible`` is ``None``
    """

    ndim = keypoints.ndim
    assert keypoints.shape[:-1] == keypoints_visible.shape[: ndim - 1], (
        f"Mismatched shapes of keypoints {keypoints.shape} and " f"keypoints_visible {keypoints_visible.shape}"
    )
    assert keypoints.shape[1] == len(flip_indices), f"Mismatched shapes of keypoints {keypoints.shape} and " f"flip_indices {keypoints_visible.shape}"

    direction_options = {"horizontal", "vertical", "diagonal"}
    assert direction in direction_options, f'Invalid flipping direction "{direction}". ' f"Options are {direction_options}"

    # swap the symmetric keypoint pairs
    if direction == "horizontal" or direction == "vertical":
        keypoints = keypoints.take(flip_indices, axis=ndim - 2)
        if keypoints_visible is not None:
            keypoints_visible = keypoints_visible.take(flip_indices, axis=ndim - 2)

    # flip the keypoints
    w, h = image_size
    if direction == "horizontal":
        keypoints[..., 0] = w - 1 - keypoints[..., 0]
    elif direction == "vertical":
        keypoints[..., 1] = h - 1 - keypoints[..., 1]
    else:
        keypoints = [w, h] - keypoints - 1

    return keypoints, keypoints_visible


def rescale_bboxes(bboxes, original_size, new_size, format="xywh"):
    """
    Rescale bounding boxes to a new image size.

    Args:
        bboxes (ndarray): Array of shape (N, 4) representing bounding boxes.
        original_size (tuple): (original_width, original_height)
        new_size (tuple): (new_width, new_height)
        format (str): One of {"xywh", "xyxy", "center"}

    Returns:
        ndarray: Rescaled bounding boxes in the same format.
    """
    assert format in {"xywh", "xyxy", "center"}, f"Invalid format: {format}"

    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]
    bboxes = np.array(bboxes, dtype=np.float32)

    if format in ["xywh", "center"]:
        bboxes[:, 0] *= scale_x
        bboxes[:, 1] *= scale_y
        bboxes[:, 2] *= scale_x
        bboxes[:, 3] *= scale_y

    elif format == "xyxy":
        bboxes[:, [0, 2]] *= scale_x
        bboxes[:, [1, 3]] *= scale_y

    return bboxes


def rescale_keypoints(keypoints, original_size, new_size):
    """
    Rescale keypoints to a new image size.

    Args:
        keypoints (ndarray): Array of shape (N, K, 2) representing keypoints (x, y).
        original_size (tuple): (original_width, original_height)
        new_size (tuple): (new_width, new_height)

    Returns:
        ndarray: Rescaled keypoints of shape (N, K, 2).
    """
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]

    keypoints = np.array(keypoints, dtype=np.float32)
    keypoints[..., 0] *= scale_x
    keypoints[..., 1] *= scale_y

    return keypoints
