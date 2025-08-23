from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from addict import Dict
from mmengine import Config
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from mmengine.runner.amp import autocast

from precision_track.registry import RUNTIMES
from precision_track.utils import PoseDataSample, get_device, get_runtime_type


class BaseBackend(BaseModel, metaclass=ABCMeta):
    PYTHON_CKPT_EXTS = [".pt", ".pth"]
    ONNX_CKPT_EXTS = [".onnx"]
    TRT_CKPT_EXTS = [".engine"]

    def __init__(self, runtime: Config, *args, **kwargs):
        super(BaseBackend, self).__init__()
        checkpoint = runtime.get("checkpoint")
        runtime["type"] = get_runtime_type(checkpoint)
        self._runtime_type = runtime["type"]
        self._runtime = RUNTIMES.build(runtime)
        self.device = getattr(runtime, "device", get_device())
        self.half_precision = getattr(runtime, "half_precision", False) and self.device != "cpu"
        self.verbose = getattr(runtime, "verbose", False)

    @property
    def runtime(self):
        return self._runtime

    @abstractmethod
    def preprocess(self, inputs: Any, data_samples: List[Union[int, PoseDataSample]]):
        pass

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

    def train_step(self, data: Union[dict, tuple, list], optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        assert self._runtime_type == "PytorchRuntime", "Training is only supported by the PytorchRuntime."
        return self._run_forward(data=data, optim_wrapper=optim_wrapper, mode="loss")

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        return self.test_step(data)

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        return self._run_forward(data, mode="predict")

    @torch.inference_mode()
    def predict(
        self,
        inputs: Union[List[Union[np.ndarray, str, torch.Tensor]], torch.Tensor],
        data_samples: Union[List[Union[int, PoseDataSample, dict]], dict],
    ) -> Union[list[dict], dict]:
        with autocast(enabled=self.half_precision):
            data = self.preprocess(inputs, data_samples)
            feats = self._runtime.predict(**data)
            outputs = self.postprocess(*feats, data["data_samples"])
        return outputs

    @abstractmethod
    def loss(
        self,
        inputs: Union[List[Union[np.ndarray, str]], torch.Tensor],
        data_samples: Union[List[Union[int, PoseDataSample, dict]], dict],
        optim_wrapper: OptimWrapper,
        *args,
        **kwargs,
    ) -> dict:
        pass

    @torch.inference_mode()
    def tensor(
        self,
        inputs: Union[List[Union[np.ndarray, str, torch.Tensor]], torch.Tensor],
        data_samples: Union[List[Union[int, PoseDataSample, dict]], dict],
    ) -> Union[list[dict], dict]:
        with autocast(enabled=self.half_precision):
            data = self.preprocess(inputs, data_samples)
            feats = self._runtime.predict(**data)
        return dict(features=feats)

    @abstractmethod
    def postprocess(features: Union[Tuple[torch.Tensor], torch.Tensor], data_samples: List[PoseDataSample]) -> List[dict]:
        pass

    def train(self, mode: bool = True):
        self.runtime.train(mode)
        return self

    def eval(self):
        return self.train(False)
