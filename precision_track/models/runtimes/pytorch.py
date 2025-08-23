from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine import Config
from mmengine.runner.amp import autocast

from precision_track.registry import MODELS, RUNTIMES
from precision_track.utils import PoseDataSample, load_checkpoint

from .base import BaseRuntime


@RUNTIMES.register_module()
class PytorchRuntime(BaseRuntime):

    def __init__(
        self,
        model: Union[Config, nn.Module],
        checkpoint: Optional[str] = None,
        device: Optional[str] = "auto",
        half_precision: Optional[bool] = False,
        verbose: Optional[bool] = True,
        freeze: Optional[bool] = False,
        **kwargs,
    ) -> None:
        if not isinstance(model, nn.Module):
            model = MODELS.build(model)

        super(PytorchRuntime, self).__init__(
            device=device,
            half_precision=half_precision,
            verbose=verbose,
            checkpoint=checkpoint,
            model=model,
            **kwargs,
        )
        self._assert_runtime()

        if self.checkpoint is not None:
            self.load_checkpoint(map_location="cpu")

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.to(self.device)
        self.model.eval()

    def loss(self, inputs: torch.Tensor, data_samples: List[PoseDataSample], *args, **kwargs) -> dict:
        return self.model.loss(inputs=inputs, data_samples=data_samples, *args, **kwargs)

    def tensor(self, inputs: torch.Tensor, data_samples: List[PoseDataSample], *args, **kwargs) -> dict:
        return self.model(inputs=inputs, data_samples=data_samples, mode="tensor", *args, **kwargs)

    @torch.no_grad()
    def predict(self, *args, **kwargs) -> Tuple[torch.Tensor]:
        with autocast(enabled=self.half_precision):
            feats = self.model.predict(*args, **kwargs)
        return feats

    def _assert_runtime(self):
        if "cuda" in self.device:
            assert torch.cuda.is_available()
        self.log_runtime(f"Inference backend set to: torch: {torch.__version__}, cuda: {torch.version.cuda}, cudnn: {torch.backends.cudnn.version()}")

    def load_checkpoint(self, map_location=None, strict=False, logger=None, revise_keys=[(r"^module\.", "")]):
        load_checkpoint(self.model, self.checkpoint, map_location=map_location, strict=strict, logger=logger, revise_keys=revise_keys)
