# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

import numpy as np
import torch
from numba import njit

from precision_track.registry import TASK_UTILS

from .datasets import parse_pose_metainfo


def to_numpy(x):
    if isinstance(x, list):
        return np.array(x)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (np.ndarray, int, float, str, np.generic)):
        return x
    else:
        raise TypeError(f"{type(x)} not yet supported.")


def keypoints_cxcywh(keypoints: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(keypoints).any(1)
    if not mask.any():
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    keypoints = keypoints[mask]
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    w, h = xmax - xmin, ymax - ymin
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    return np.array([cx, cy, w, h], dtype=np.float32)


@njit
def cxcywh_xywh_1d(cxcywh: np.ndarray) -> np.ndarray:
    if np.isnan(cxcywh).any():
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    cx, cy, w, h = cxcywh
    return np.array([cx - w / 2, cy - h / 2, w, h], dtype=np.float32)


def cxcywh_xywh(bboxes: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
    return bboxes


def xywh_cxcywh(bboxes: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2
    return bboxes


@njit
def cxcywh_cxcyah_1d(cxcywh: np.ndarray) -> np.ndarray:
    if np.isnan(cxcywh).any():
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    cx, cy, w, h = cxcywh
    a = w / h
    return np.array([cx, cy, a, h], dtype=np.float32)


@njit
def cxcyah_cxcywh_1d(cxcyah: np.ndarray) -> np.ndarray:
    if np.isnan(cxcyah).any():
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    cx, cy, a, h = cxcyah
    w = a * h
    return np.array([cx, cy, w, h], dtype=np.float32)


@njit
def polygon_xywh(x1y1x2y2x3y3x4y4: np.ndarray) -> np.ndarray:
    min_x = np.min(x1y1x2y2x3y3x4y4[::2])
    max_x = np.max(x1y1x2y2x3y3x4y4[::2])
    min_y = np.min(x1y1x2y2x3y3x4y4[1::2])
    max_y = np.max(x1y1x2y2x3y3x4y4[1::2])
    return np.array([min_x, min_y, max_x - min_x, max_y - min_y], dtype=np.float32)


@njit
def polygon_cxcywh(x1y1x2y2x3y3x4y4: np.ndarray) -> np.ndarray:
    x = np.min(x1y1x2y2x3y3x4y4[:, ::2])
    w = np.max(x1y1x2y2x3y3x4y4[:, ::2]) - x
    y = np.min(x1y1x2y2x3y3x4y4[:, 1::2])
    h = np.max(x1y1x2y2x3y3x4y4[:, 1::2]) - y
    return np.array([x + w / 2, y + h / 2, w, h], dtype=np.float32)


def xyxy_cxcywh(bboxes: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2
    return bboxes


@njit
def xyxy_cxcywh_1d(bboxe: np.ndarray) -> np.ndarray:
    bboxe[2] = bboxe[2] - bboxe[0]
    bboxe[3] = bboxe[3] - bboxe[1]
    bboxe[0] = bboxe[0] + bboxe[2] / 2
    bboxe[1] = bboxe[1] + bboxe[3] / 2
    return bboxe


@njit
def cxcywh_cxcyah(bboxes: np.ndarray) -> np.ndarray:
    bboxes_out = bboxes.copy()
    bboxes_out[:, 2] = bboxes_out[:, 2] / bboxes_out[:, 3]
    return bboxes_out


@njit
def cxcyah_cxcywh(bboxes: np.ndarray) -> np.ndarray:
    bboxes_out = bboxes.copy()
    bboxes_out[:, 2] = bboxes_out[:, 2] * bboxes_out[:, 3]
    return bboxes_out


@njit
def cxcywh_xyxy_1d(bboxes: np.ndarray) -> np.ndarray:
    bboxes_out = bboxes.copy()
    bboxes_out[0] = bboxes_out[0] - bboxes_out[2] / 2
    bboxes_out[1] = bboxes_out[1] - bboxes_out[3] / 2
    bboxes_out[2] = bboxes_out[0] + bboxes_out[2]
    bboxes_out[3] = bboxes_out[1] + bboxes_out[3]
    return bboxes_out


@njit
def cxcywh_xyxy(bboxes: np.ndarray) -> np.ndarray:
    bboxes_out = bboxes.copy()
    bboxes_out[:, 0] = bboxes_out[:, 0] - bboxes_out[:, 2] / 2
    bboxes_out[:, 1] = bboxes_out[:, 1] - bboxes_out[:, 3] / 2
    bboxes_out[:, 2] = bboxes_out[:, 0] + bboxes_out[:, 2]
    bboxes_out[:, 3] = bboxes_out[:, 1] + bboxes_out[:, 3]
    return bboxes_out


def cxcywh_xyxy_torch(bboxes: torch.Tensor) -> torch.Tensor:
    bboxes_out = bboxes.clone()
    bboxes_out[:, 0] = bboxes_out[:, 0] - bboxes_out[:, 2] / 2
    bboxes_out[:, 1] = bboxes_out[:, 1] - bboxes_out[:, 3] / 2
    bboxes_out[:, 2] = bboxes_out[:, 0] + bboxes_out[:, 2]
    bboxes_out[:, 3] = bboxes_out[:, 1] + bboxes_out[:, 3]
    return bboxes_out


@njit
def xyxy_xywh(bboxes: np.ndarray) -> np.ndarray:
    bboxes_out = bboxes.copy()
    bboxes_out[:, 2] = bboxes_out[:, 2] - bboxes_out[:, 0]
    bboxes_out[:, 3] = bboxes_out[:, 3] - bboxes_out[:, 1]
    return bboxes_out


@njit
def xywh_xyxy(bboxes: np.ndarray) -> np.ndarray:
    bboxes_out = bboxes.copy()
    bboxes_out[:, 2] = bboxes_out[:, 2] + bboxes_out[:, 0]
    bboxes_out[:, 3] = bboxes_out[:, 3] + bboxes_out[:, 1]
    return bboxes_out


def xywh_xyxy_torch(bboxes: torch.Tensor) -> torch.Tensor:
    bboxes_out = bboxes.clone()
    bboxes_out[:, 2] = bboxes_out[:, 2] + bboxes_out[:, 0]
    bboxes_out[:, 3] = bboxes_out[:, 3] + bboxes_out[:, 1]
    return bboxes_out


def xyxy_corner(bbox: np.ndarray):
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None]

    bbox = np.tile(bbox, 2).reshape(-1, 4, 2)
    bbox[:, 1:3, 0] = bbox[:, 0:2, 0]

    if dim == 1:
        bbox = bbox[0]

    return bbox


def corner_xyxy(bbox: np.ndarray):
    if bbox.shape[-1] == 8:
        bbox = bbox.reshape(*bbox.shape[:-1], 4, 2)

    dim = bbox.ndim
    if dim == 2:
        bbox = bbox[None]

    bbox = np.concatenate((bbox.min(axis=1), bbox.max(axis=1)), axis=1)

    if dim == 2:
        bbox = bbox[0]
    return bbox


transformation_functions = {
    "xyxy_cxcywh_torch": xyxy_cxcywh,
    "xyxy_cxcywh": xyxy_cxcywh,
    "xyxy_cxcywh_1d": xyxy_cxcywh_1d,
    "xyxy_xywh": xyxy_xywh,
    "cxcywh_xyxy": cxcywh_xyxy,
    "cxcywh_xyxy_torch": cxcywh_xyxy_torch,
    "cxcywh_xyxy_1d": cxcywh_xyxy_1d,
    "cxcywh_cxcyah_1d": cxcywh_cxcyah_1d,
    "cxcywh_cxcyah": cxcywh_cxcyah,
    "keypoints_cxcywh": keypoints_cxcywh,
    "cxcywh_xywh_1d": cxcywh_xywh_1d,
    "cxcywh_xywh": cxcywh_xywh,
    "polygon_xywh_1d": polygon_xywh,
    "polygon_cxcywh": polygon_cxcywh,
    "cxcyah_cxcywh": cxcyah_cxcywh,
    "cxcyah_cxcywh_1d": cxcyah_cxcywh_1d,
    "xywh_cxcywh": xywh_cxcywh,
    "xywh_xyxy": xywh_xyxy,
    "xywh_xyxy_torch": xywh_xyxy_torch,
    "corner_xyxy_1d": corner_xyxy,
    "corner_xyxy": corner_xyxy,
    "xyxy_corner": xyxy_corner,
    "xyxy_corner_1d": xyxy_corner,
}


def reformat(instance: Union[np.ndarray, torch.Tensor], old: str, new: str) -> Union[np.ndarray, torch.Tensor]:
    """Reformat one or multiple bounding boxe(s) from 'old' format to 'new'
    format. If the reformat is not supported, raises an error.

    Args:
        instance (Union[np.ndarray, torch.Tensor]): The bounding boxe(s)
        old (str): The old format.
        new (str): The new format.

    Returns:
        Union[np.ndarray, torch.Tensor]: The bounding boxe(s) in the 'new' format
    """
    if instance.size == 0:
        return instance
    if not isinstance(instance, (np.ndarray, torch.Tensor)):
        raise TypeError("The instance argument must be a Numpy array or a Pytorch tensor.")
    ndim = instance.ndim

    is_torch = isinstance(instance, torch.Tensor)

    function_key = f"{old}_{new}"
    if is_torch:
        function_key += "_torch"
    if ndim == 1:
        function_key += "_1d"

    transformation_function = transformation_functions.get(function_key, None)
    if transformation_function is None:
        raise ValueError(f"Transformation: {function_key} is not defined.")

    return transformation_function(instance)


@njit
def clip_cxcywh(cxcywh: np.ndarray, max_width: int, max_height: int):
    cxcywh[0] = min(max(cxcywh[0], 0), max_width)
    cxcywh[1] = min(max(cxcywh[1], 0), max_height)
    cxcywh[2] = max(min(cxcywh[2], 2 * min(cxcywh[0], max_width - cxcywh[0])), 0)
    cxcywh[3] = max(min(cxcywh[3], 2 * min(cxcywh[1], max_height - cxcywh[1])), 0)
    return cxcywh


def clip_torch_cxcywh(cxcywh: torch.Tensor, max_width: int, max_height: int):
    cxcywh[0] = torch.clamp(cxcywh[0], 0, max_width)
    cxcywh[1] = torch.clamp(cxcywh[1], 0, max_height)
    cxcywh[2] = torch.clamp(cxcywh[2], 0, 2 * torch.min(cxcywh[0], max_width - cxcywh[0]))
    cxcywh[3] = torch.clamp(cxcywh[3], 0, 2 * torch.min(cxcywh[1], max_height - cxcywh[1]))
    return cxcywh


def clip_xyxy(bbox: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    if bbox.shape[-1] == 2:
        bbox[..., 0] = np.clip(bbox[..., 0], a_min=0, a_max=max_width)
        bbox[..., 1] = np.clip(bbox[..., 1], a_min=0, a_max=max_height)
    else:
        bbox[..., ::2] = np.clip(bbox[..., ::2], a_min=0, a_max=max_width)
        bbox[..., 1::2] = np.clip(bbox[..., 1::2], a_min=0, a_max=max_height)

    return bbox


@njit
def clip_keypoints(keypoints, max_width: int, max_height: int):
    keypoints[..., 0] = np.clip(keypoints[..., 0], 0, max_width)
    keypoints[..., 1] = np.clip(keypoints[..., 1], 0, max_height)
    return keypoints


def clip_keypoints_vis(keypoints: np.ndarray, keypoints_visible: np.ndarray, max_width: int, max_height: int) -> Tuple[np.ndarray, np.ndarray]:
    # Create a mask for keypoints outside the frame
    outside_mask = (keypoints[..., 0] > max_width) | (keypoints[..., 0] < 0) | (keypoints[..., 1] > max_height) | (keypoints[..., 1] < 0)

    # Update visibility values for keypoints outside the frame
    if keypoints_visible.ndim == 2:
        keypoints_visible[outside_mask] = 0.0
    elif keypoints_visible.ndim == 3:
        keypoints_visible[outside_mask, 0] = 0.0

    return keypoints, keypoints_visible


clip_functions = {
    "cxcywh_torch_1d": clip_torch_cxcywh,
    "cxcywh_1d": clip_cxcywh,
    "xyxy_1d": clip_xyxy,
    "xyxy": clip_xyxy,
    "keypoints": clip_keypoints,
    "keypoints_vis": clip_keypoints_vis,
    "keypoints_vis_1d": clip_keypoints_vis,
}


def clip(
    instance: Union[np.ndarray, torch.Tensor],
    format: str,
    max_width: int,
    max_height: int,
    *args,
    **kwargs,
) -> Union[np.ndarray, torch.Tensor]:
    """Clip the bounding boxe into the limits (0, 0, max_width, max_height) of
    a frame.

    Args:
        instance (Union[np.ndarray, torch.Tensor]): The bounding box to clip
        format (str): The format of the bounding box
        max_width (int): The max width of the frame
        max_height (int): The max height of the frame

    Returns:
        Union[np.ndarray, torch.Tensor]: The clipped bounding box.
    """
    if not isinstance(instance, (np.ndarray, torch.Tensor)):
        raise TypeError("The instance argument must be a Numpy array or a Pytorch tensor.")
    ndim = instance.ndim

    is_torch = isinstance(instance, torch.Tensor)

    function_key = f"{format}"
    if is_torch:
        function_key += "_torch"
    if ndim == 1:
        function_key += "_1d"

    clip_function = clip_functions.get(function_key, None)
    if clip_function is None:
        raise ValueError(f"Clipping: {function_key} is not defined.")

    return clip_function(instance, max_width=max_width, max_height=max_height, *args, **kwargs)


def data_sample_to_dict(data_sample) -> dict:
    return dict(
        ori_shape=getattr(data_sample, "ori_shape"),
        img_id=getattr(
            data_sample,
            "img_id",
        ),
    )


def data_sample_to_tracker_input(data_sample) -> dict:
    out = data_sample_to_dict(data_sample)
    out["pred_instances"] = data_sample.pred_instances.to_dict()
    return out


def data_sample_to_gt(data_sample) -> dict:
    out = data_sample_to_dict(data_sample)
    out["gt_instances"] = data_sample.gt_instances.to_dict()
    return out


def normalize_kpts_by_bboxes(keypoints: torch.Tensor, bboxes: torch.Tensor):
    N, K, _ = keypoints.shape
    bbox_width = (bboxes[:, 2] - bboxes[:, 0]).view(N, 1)
    bbox_height = (bboxes[:, 3] - bboxes[:, 1]).view(N, 1)
    bbox_x = bboxes[:, 0].view(N, 1) - bbox_width / 2
    bbox_y = bboxes[:, 1].view(N, 1) - bbox_height / 2

    keypoints = keypoints.view(N, K * 2)
    keypoints[:, ::2] = (keypoints[:, ::2] - bbox_x) / bbox_width
    keypoints[:, 1::2] = (keypoints[:, 1::2] - bbox_y) / bbox_height
    return keypoints.view(N, K, 2)


def kpts_to_poses(
    kpts: torch.Tensor,
    kpt_vis: torch.Tensor,
    skeleton_sources: torch.Tensor,
    skeleton_targets: torch.Tensor,
    conf_thr: float = 0.2,
    normalize: bool = True,
    eps: float = 1e-6,
):
    bone_vec = kpts[:, skeleton_sources] - kpts[:, skeleton_targets]
    vis_mask = (kpt_vis[:, skeleton_sources] >= conf_thr) & (kpt_vis[:, skeleton_targets] >= conf_thr)

    scale = 1
    if normalize:
        lens = torch.norm(bone_vec, dim=-1)
        lens[~vis_mask] = torch.nan
        scale, _ = torch.nanmedian(lens, dim=1, keepdim=True)
        scale = torch.where(torch.isnan(scale), torch.ones_like(scale), scale)

        bone_vec = bone_vec / (scale[..., None] + eps)

    bone_vec[~vis_mask[..., None].expand_as(bone_vec)] = 0.0

    return bone_vec, scale

    # ---------- cosine-angle stream ----------------------------------------
    # TODO à optimiser.....
    # T, V, C = kpts.shape
    # angles_cos = torch.zeros((T, V), dtype=kpts.dtype, device=kpts.device)

    # # pre-gather bone indices touching each joint
    # src_map = {j: torch.where(skeleton_sources == j)[0] for j in range(V)}
    # tgt_map = {j: torch.where(skeleton_targets == j)[0] for j in range(V)}

    # for j in range(V):
    #     if len(src_map[j]) == 0 or len(tgt_map[j]) == 0:
    #         continue  # is an extremity

    #     child_idx = tgt_map[j][0]
    #     parent_idx = src_map[j][0]

    #     v_child = bone_vec[:, child_idx]
    #     v_parent = -bone_vec[:, parent_idx]

    #     valid = vis_mask[:, child_idx] & vis_mask[:, parent_idx]

    #     dot_product = (v_child * v_parent).sum(-1)
    #     cosθ = dot_product / (v_child.norm(dim=-1) * v_parent.norm(dim=-1)) + eps

    #     angles_cos[valid, j] = cosθ[valid]

    # return bone_vec, angles_cos
    # return torch.hstack((bone_vec.view(-1, 18), angles_cos))  # TODO rendrep lus efficace, shotgun mémoire au début.

    return bone_vec


def velocity_to_dir_speed(velocity, eps=1e-6):
    speed = velocity.norm(p=2, dim=-1).clamp(min=eps)
    direction = velocity / speed.unsqueeze(-1)
    log_speed = torch.log1p(speed)
    return torch.cat([direction, log_speed.unsqueeze(-1)], dim=-1)


class InputShape(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass


@TASK_UTILS.register_module()
class PosesShape(InputShape):
    def __init__(self, block_size: int, metainfo: str):
        assert 0 < block_size
        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        assert "skeleton_links" in metainfo
        skeleton = metainfo["skeleton_links"]
        assert isinstance(skeleton, list)
        self.shape = (block_size, len(skeleton) * 2)


@TASK_UTILS.register_module()
class VelocityShape(InputShape):

    def __init__(self, block_size: int):
        assert 0 < block_size
        self.shape = (block_size, 2)


@TASK_UTILS.register_module()
class FeaturesShape(InputShape):

    def __init__(self, block_size: int, n_embd: int):
        for arg_ in [block_size, n_embd]:
            assert 0 < arg_
        self.shape = (block_size, n_embd)


@TASK_UTILS.register_module()
class ImageShape(InputShape):
    def __init__(self, n_channels: int, width: int, height: int):
        for arg_ in [n_channels, width, height]:
            assert 0 < arg_
        self.shape = (n_channels, width, height)
