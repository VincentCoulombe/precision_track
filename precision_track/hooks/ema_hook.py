# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional

import torch
import torch.nn as nn
from mmengine.model import ExponentialMovingAverage, is_model_wrapper
from mmengine.registry import MODELS
from torch import Tensor
from mmengine.hooks import Hook

from precision_track.registry import HOOKS


@MODELS.register_module()
class ExpMomentumEMA(ExponentialMovingAverage):
    """Exponential moving average (EMA) with exponential momentum strategy,
    which is used in YOLOX.

    Ported from ` the implementation of MMDetection
    <https://github.com/open-mmlab/mmdetection/blob/3.x/mmdet/models/layers/ema.py>`_.

    Args:
        model (nn.Module): The model to be averaged.
        momentum (float): The momentum used for updating ema parameter.
            Ema's parameter are updated with the formula:
           `averaged_param = (1-momentum) * averaged_param + momentum *
           source_param`. Defaults to 0.0002.
        gamma (int): Use a larger momentum early in training and gradually
            annealing to a smaller value to update the ema model smoothly. The
            momentum is calculated as
            `(1 - momentum) * exp(-(1 + steps) / gamma) + momentum`.
            Defaults to 2000.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """

    def __init__(
        self, model: nn.Module, momentum: float = 0.0002, gamma: int = 2000, interval=1, device: Optional[torch.device] = None, update_buffers: bool = False
    ) -> None:
        super().__init__(model=model, momentum=momentum, interval=interval, device=device, update_buffers=update_buffers)
        assert gamma > 0, f"gamma must be greater than 0, but got {gamma}"
        self.gamma = gamma

    def avg_func(self, averaged_param: Tensor, source_param: Tensor, steps: int) -> None:
        """Compute the moving average of the parameters using the exponential
        momentum strategy.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        momentum = (1 - self.momentum) * math.exp(-float(1 + steps) / self.gamma) + self.momentum
        averaged_param.mul_(1 - momentum).add_(source_param, alpha=momentum)


@HOOKS.register_module()
class AnalyzerEMAHook(ExpMomentumEMA):
    def __init__(self, ema_type: str = "ExponentialMovingAverage", strict_load: bool = False, begin_iter: int = 0, begin_epoch: int = 0, **kwargs):
        self.strict_load = strict_load
        self.ema_cfg = dict(type=ema_type, **kwargs)
        assert not (begin_iter != 0 and begin_epoch != 0), "`begin_iter` and `begin_epoch` should not be both set."
        assert begin_iter >= 0, "`begin_iter` must larger than or equal to 0, " f"but got begin_iter: {begin_iter}"
        assert begin_epoch >= 0, "`begin_epoch` must larger than or equal to 0, " f"but got begin_epoch: {begin_epoch}"
        self.begin_iter = begin_iter
        self.begin_epoch = begin_epoch
        self.enabled_by_epoch = self.begin_epoch > 0

    def before_run(self, runner) -> None:
        """Create an ema copy of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.analyzer.model.module
        else:
            model = model.analyzer.model
        self.src_model = model
        self.ema_model = MODELS.build(self.ema_cfg, default_args=dict(model=self.src_model))


@HOOKS.register_module()
class DetectorEMAHook(Hook):
    priority = "NORMAL"

    def __init__(self, ema_type: str = "ExponentialMovingAverage", strict_load: bool = False, begin_iter: int = 0, begin_epoch: int = 0, **kwargs):
        self.strict_load = strict_load
        self.ema_cfg = dict(type=ema_type, **kwargs)
        assert not (begin_iter != 0 and begin_epoch != 0), "`begin_iter` and `begin_epoch` should not be both set."
        assert begin_iter >= 0, "`begin_iter` must larger than or equal to 0, " f"but got begin_iter: {begin_iter}"
        assert begin_epoch >= 0, "`begin_epoch` must larger than or equal to 0, " f"but got begin_epoch: {begin_epoch}"
        self.begin_iter = begin_iter
        self.begin_epoch = begin_epoch
        # If `begin_epoch` and `begin_iter` are not set, `EMAHook` will be
        # enabled at 0 iteration.
        self.enabled_by_epoch = self.begin_epoch > 0

    def before_run(self, runner) -> None:
        """Create an ema copy of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        if not hasattr(runner, "detector"):
            raise ValueError("The provided Runner does not have a detector.")
        model = runner.detector
        if is_model_wrapper(model):
            model = model.module
        self.src_model = model
        self.ema_model = MODELS.build(self.ema_cfg, default_args=dict(model=self.src_model))

    def after_load_detector_checkpoint(self, runner, checkpoint: dict) -> None:
        super().after_load_checkpoint(runner, checkpoint)
