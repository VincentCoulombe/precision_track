from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine import Config
from mmengine.evaluator import Evaluator
from mmengine.registry import LOOPS
from mmengine.runner.amp import autocast
from mmengine.runner.loops import ValLoop
from torch.utils.data import DataLoader

from precision_track.models.backends import DetectionBackend
from precision_track.utils import get_device


@LOOPS.register_module()
class ValidationLossLoop(ValLoop):

    def __init__(
        self,
        runner,
        val_cfg: Config,
        dataloader: Union[DataLoader, Dict],
        evaluator: Union[Evaluator, Dict, List],
        fp16: bool = True,
    ):
        super().__init__(
            runner=runner,
            dataloader=dataloader,
            evaluator=evaluator,
            fp16=fp16,
        )
        assert isinstance(val_cfg, (Config, dict)), "If the validation loop is not sequential, a validation config must be provided."
        self.backend = DetectionBackend(
            temperature_file=val_cfg.get("hyperparameters", ""),
            data_preprocessor=self.runner.model.data_preprocessor,
            data_postprocessor=val_cfg.get("data_postprocessor"),
            kpt_score_thr=val_cfg.get("kpt_score_thr", 0.0),
            runtime=dict(
                type="PytorchRuntime",
                checkpoint=val_cfg.get("checkpoint"),
                model=self.runner.model,
                device=get_device(),
                half_precision=self.fp16,
            ),
        )
        val_data_preprocessor = val_cfg.get("data_preprocessor")
        if val_data_preprocessor is not None:
            self.backend.data_preprocessor = val_data_preprocessor
        self.reset_losses()

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], *args, **kwargs) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        with autocast(enabled=self.fp16):
            inputs = self.backend.data_preprocessor(data_batch, False)
            val_losses = self.backend.runtime.loss(**inputs)
        for val_loss in val_losses:
            if "loss" in val_loss:
                self.val_losses[f"val/{val_loss}"].append(val_losses[val_loss].flatten().mean(0).item())
            if "overlaps" in val_loss:
                self.val_losses[f"val/{val_loss}"].append(val_losses[val_loss].flatten().max(0)[0].item())

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook("before_val")
        self.runner.call_hook("before_val_epoch")
        self.runner.model.eval()
        self.reset_losses()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)
        total_loss = 0
        for val_loss in self.val_losses:
            mean_loss = np.mean(self.val_losses[val_loss])
            self.val_losses[val_loss] = mean_loss
            if "loss" in val_loss:
                total_loss += mean_loss
        self.val_losses["val/loss"] = total_loss
        self.runner.call_hook("after_val_epoch", metrics=self.val_losses)
        self.runner.call_hook("after_val")
        return self.val_losses

    def reset_losses(self):
        self.val_losses = defaultdict(list)
        tags_to_del = []
        for tag in self.runner.message_hub._log_scalars:
            if "val" in tag:
                tags_to_del.append(tag)
        for tag in tags_to_del:
            del self.runner.message_hub._log_scalars[tag]


@LOOPS.register_module()
class ValidationLoop(ValLoop):

    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        evaluator: Union[Evaluator, Dict, List],
        val_cfg: Optional[Config] = None,
        fp16: bool = True,
        is_sequence: bool = False,
    ):
        super().__init__(
            runner=runner,
            dataloader=dataloader,
            evaluator=evaluator,
            fp16=fp16,
        )
        if is_sequence:
            self.backend = None
        else:
            assert isinstance(val_cfg, (Config, dict)), "If the validation loop is not sequential, a validation config must be provided."
            self.backend = DetectionBackend(
                temperature_file=val_cfg.get("hyperparameters", ""),
                data_preprocessor=self.runner.model.data_preprocessor,
                data_postprocessor=val_cfg.get("data_postprocessor"),
                kpt_score_thr=val_cfg.get("kpt_score_thr", 0.0),
                runtime=dict(
                    type="PytorchRuntime",
                    checkpoint=val_cfg.get("checkpoint"),
                    model=self.runner.model,
                    device=get_device(),
                    half_precision=self.fp16,
                ),
            )

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], *args, **kwargs) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook("before_val_iter", batch_idx=idx, data_batch=data_batch)
        with autocast(enabled=self.fp16):
            outputs = self.backend.val_step(data_batch, *args, **kwargs)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            "after_val_iter",
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs,
        )

    def run(self, *args, **kwargs) -> dict:
        """Launch validation."""
        self.runner.call_hook("before_val")
        self.runner.call_hook("before_val_epoch")
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch, *args, **kwargs)

        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook("after_val_epoch", metrics=metrics)
        self.runner.call_hook("after_val")
        return metrics


@LOOPS.register_module()
class SequenceValidationLoop(ValidationLoop):

    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        evaluator: Union[Evaluator, Dict, List],
        fp16: bool = False,
    ):
        super().__init__(
            runner=runner,
            val_cfg=None,
            dataloader=dataloader,
            evaluator=evaluator,
            fp16=fp16,
            is_sequence=True,
        )
        self.backend = self.runner.model
