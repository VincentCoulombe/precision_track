from typing import Any, List, Union

import numpy as np
import torch
from mmengine import Config

from precision_track.registry import MODELS
from precision_track.utils import PoseDataSample

from .base import BaseBackend


@MODELS.register_module()
class ReIDBackend(BaseBackend):

    def __init__(
        self,
        config: Config,
    ) -> None:
        super(ReIDBackend, self).__init__(config.runtime)

    def preprocess(self, inputs: Any, data_samples: List[Union[int, PoseDataSample]]):
        return dict(inputs=inputs, data_samples=data_samples)

    def postprocess(self, features: torch.Tensor, data_samples: List[PoseDataSample]) -> List[dict]:
        return features

    def loss(
        self,
        inputs: List[Union[np.ndarray, str]],
        data_samples: List[Union[int, PoseDataSample]],
        *args,
        **kwargs,
    ) -> dict:
        data = self.preprocess(inputs, data_samples)
        losses = self._runtime.loss(**data)
        return {"losses": losses, "data_samples": data_samples}

    def val_step(self, inputs: List[torch.Tensor], data_samples: List[dict], *args, **kwargs) -> list:
        return self._runtime.val_step(inputs, data_samples, *args, **kwargs)

    def test_step(self, inputs: List[torch.Tensor], data_samples: List[dict], *args, **kwargs) -> list:
        return self._runtime.test_step(inputs, data_samples, *args, **kwargs)
