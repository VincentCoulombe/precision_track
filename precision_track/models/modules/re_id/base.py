from abc import abstractmethod
from typing import List, Optional, Union

import torch
from mmengine.model import BaseModel

from precision_track.tracking.utils.structures import PoseDataSample


class BaseReIDModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super(BaseReIDModel, self).__init__()

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: List[PoseDataSample],
        mode: Optional[str] = "predict",
        *args,
        **kwargs,
    ) -> Union[List[dict], dict]:
        if mode == "predict":
            return self.predict(inputs=inputs, data_samples=data_samples)
        elif mode == "loss":
            return self.loss(inputs=inputs, data_samples=data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". ' "Only supports loss and predict mode.")

    @abstractmethod
    def predict(self, inputs: torch.Tensor, data_samples: dict) -> torch.Tensor:
        pass

    @abstractmethod
    def loss(self, inputs: torch.Tensor, data_samples: List[PoseDataSample]) -> dict:
        pass
