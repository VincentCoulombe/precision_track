import logging
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.logging import print_log

from precision_track.utils import PoseDataSample, get_device


class BaseRuntime(nn.Module, metaclass=ABCMeta):

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        checkpoint: Optional[str] = None,
        half_precision: Optional[bool] = False,
        device: Optional[str] = None,
        verbose: Optional[bool] = True,
        *args,
        **kwargs,
    ):
        if device is None:
            device == "auto"
        super(BaseRuntime, self).__init__()
        self.device = get_device() if device == "auto" else device
        if half_precision and device == "cpu":
            print_log(f"FP16 is not supported for: {device}", logger="current", level=logging.WARNING)
        self.half_precision = half_precision and self.device != "cpu"
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.model = model

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
        elif mode == "tensor":
            return self.tensor(inputs=inputs, data_samples=data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". ' "Only supports loss, tensor and predict mode.")

    @abstractmethod
    def predict(self, inputs: torch.Tensor, data_samples: List[PoseDataSample]) -> Tuple[torch.Tensor]:
        pass

    @abstractmethod
    def loss(self, inputs: torch.Tensor, data_samples: List[PoseDataSample]) -> dict:
        pass

    @abstractmethod
    def tensor(self, inputs: torch.Tensor, data_samples: List[PoseDataSample]) -> dict:
        pass

    @abstractmethod
    def _assert_runtime(self):
        pass

    def log_runtime(self, backend_str: str):
        if self.verbose:
            print_log(
                f"{backend_str} with FP{16 if self.half_precision else 32} precision on device: {self.device}.",
                logger="current",
                level=logging.INFO,
            )

    def train(self, mode: bool = True):
        if isinstance(self.model, nn.Module):
            self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def val_step(self, *args, **kwargs) -> list:
        return self.model.val_step(*args, **kwargs)

    def test_step(self, *args, **kwargs) -> list:
        return self.model.test_step(*args, **kwargs)


class InferenceOnlyRuntime(BaseRuntime):
    def loss(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__} does not support training.")

    def tensor(self, *args, **kwargs):
        return self.predict(**args, **kwargs)
