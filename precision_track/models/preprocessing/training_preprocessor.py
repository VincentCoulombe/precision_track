# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import random
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import MessageHub
from mmengine.dist import barrier, broadcast, get_dist_info
from mmengine.model import ImgDataPreprocessor
from mmengine.structures import PixelData
from mmengine.utils import is_seq_of
from torch import Tensor

from precision_track.registry import MODELS
from precision_track.utils import PoseDataSample


@MODELS.register_module()
class PoseDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for pose estimation tasks.

    Comparing with the :class:`ImgDataPreprocessor`,

    1. It will additionally append batch_input_shape
    to data_samples considering the DETR-based pose estimation tasks.

    2. Support image augmentation transforms on batched data.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Apply batch augmentation transforms.

    Args:
        mean (sequence of float, optional): The pixel mean of R, G, B
            channels. Defaults to None.
        std (sequence of float, optional): The pixel standard deviation
            of R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to BGR.
            Defaults to False.
        non_blocking (bool): Whether block current process
            when transferring data to device. Defaults to False.
        batch_augments: (list of dict, optional): Configs of augmentation
            transforms on batched data. Defaults to None.
    """

    def __init__(
        self,
        mean: Sequence[float] = None,
        std: Sequence[float] = None,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        non_blocking: Optional[bool] = False,
        batch_augments: Optional[List[dict]] = None,
    ):
        super().__init__(
            mean=mean, std=std, pad_size_divisor=pad_size_divisor, pad_value=pad_value, bgr_to_rgb=bgr_to_rgb, rgb_to_bgr=rgb_to_bgr, non_blocking=non_blocking
        )

        if batch_augments is not None:
            self.batch_augments = nn.ModuleList([MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        batch_pad_shape = self._get_pad_shape(data)
        data = super().forward(data=data, training=training)
        inputs, data_samples = data["inputs"], data["data_samples"]

        # update metainfo since the image shape might change
        batch_input_shape = tuple(inputs[0].size()[-2:])
        for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
            data_sample.set_metainfo({"batch_input_shape": batch_input_shape, "pad_shape": pad_shape})

        # apply batch augmentations
        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {"inputs": inputs, "data_samples": data_samples}

    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        _batch_inputs = data["inputs"]
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                pad_h = int(np.ceil(ori_input.shape[1] / self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(np.ceil(ori_input.shape[2] / self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                "The input of `ImgDataPreprocessor` should be a NCHW tensor " "or a list of tensor, but got a tensor with shape: " f"{_batch_inputs.shape}"
            )
            pad_h = int(np.ceil(_batch_inputs.shape[1] / self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(np.ceil(_batch_inputs.shape[2] / self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError("Output of `cast_data` should be a dict " "or a tuple with inputs and data_samples, but got" f"{type(data)}: {data}")
        return batch_pad_shape


@MODELS.register_module()
class BatchSyncRandomResize(nn.Module):
    """Batch random resize which synchronizes the random size across ranks.

    Args:
        random_size_range (tuple): The multi-scale random range during
            multi-scale training.
        interval (int): The iter interval of change
            image size. Defaults to 10.
        size_divisor (int): Image size divisible factor.
            Defaults to 32.
    """

    def __init__(self, random_size_range: Tuple[int, int], interval: int = 10, size_divisor: int = 32) -> None:
        super().__init__()
        self.rank, self.world_size = get_dist_info()
        self._input_size = None
        self._random_size_range = (round(random_size_range[0] / size_divisor), round(random_size_range[1] / size_divisor))
        self._interval = interval
        self._size_divisor = size_divisor

    def forward(self, inputs: Tensor, data_samples: List[PoseDataSample]) -> Tuple[Tensor, List[PoseDataSample]]:
        """resize a batch of images and bboxes to shape ``self._input_size``"""
        h, w = inputs.shape[-2:]
        if self._input_size is None:
            self._input_size = (h, w)
        scale_y = self._input_size[0] / h
        scale_x = self._input_size[1] / w
        if scale_x != 1 or scale_y != 1:
            inputs = F.interpolate(inputs, size=self._input_size, mode="bilinear", align_corners=False)
            for data_sample in data_samples:
                img_shape = (int(data_sample.img_shape[0] * scale_y), int(data_sample.img_shape[1] * scale_x))
                pad_shape = (int(data_sample.pad_shape[0] * scale_y), int(data_sample.pad_shape[1] * scale_x))
                data_sample.set_metainfo({"img_shape": img_shape, "pad_shape": pad_shape, "batch_input_shape": self._input_size})

                if "gt_instance_labels" not in data_sample:
                    continue

                if "bboxes" in data_sample.gt_instance_labels:
                    data_sample.gt_instance_labels.bboxes[..., 0::2] *= scale_x
                    data_sample.gt_instance_labels.bboxes[..., 1::2] *= scale_y

                if "keypoints" in data_sample.gt_instance_labels:
                    data_sample.gt_instance_labels.keypoints[..., 0] *= scale_x
                    data_sample.gt_instance_labels.keypoints[..., 1] *= scale_y

                if "areas" in data_sample.gt_instance_labels:
                    data_sample.gt_instance_labels.areas *= scale_x * scale_y

                if "gt_fields" in data_sample and "heatmap_mask" in data_sample.gt_fields:

                    mask = data_sample.gt_fields.heatmap_mask.unsqueeze(0)
                    gt_fields = PixelData()
                    gt_fields.set_field(F.interpolate(mask.float(), size=self._input_size, mode="bilinear", align_corners=False).squeeze(0), "heatmap_mask")

                    data_sample.gt_fields = gt_fields

        message_hub = MessageHub.get_current_instance()
        if (message_hub.get_info("iter") + 1) % self._interval == 0:
            self._input_size = self._get_random_size(aspect_ratio=float(w / h), device=inputs.device)
        return inputs, data_samples

    def _get_random_size(self, aspect_ratio: float, device: torch.device) -> Tuple[int, int]:
        """Randomly generate a shape in ``_random_size_range`` and broadcast to
        all ranks."""
        tensor = torch.empty(2, dtype=torch.long, device=device)
        if self.rank == 0:
            size = random.randint(*self._random_size_range)
            size = (self._size_divisor * size, self._size_divisor * int(aspect_ratio * size))
            tensor[0] = size[0]
            tensor[1] = size[1]
        barrier()
        broadcast(tensor, 0)
        input_size = (tensor[0].item(), tensor[1].item())
        return input_size
