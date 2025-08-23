from typing import Tuple

import torch.nn as nn
from mmengine.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class BasePostProcessor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        scores: Tensor,
        objectness: Tensor,
        bboxes: Tensor,
        kpts: Tensor,
        kpt_vis: Tensor,
        features: Tensor,
        kept_idxs: Tensor,
        *args,
        **kwargs,
    ) -> Tuple[Tensor]:
        return scores, objectness, bboxes, kpts, kpt_vis, features, kept_idxs


class BaseActionPostProcessor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data_sample: dict) -> dict:
        return data_sample
