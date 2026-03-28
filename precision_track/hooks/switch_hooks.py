# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
import functools
from typing import Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from precision_track.registry import HOOKS


def rsetattr(obj, attr, val):
    """Set the value of a nested attribute of an object.

    This function splits the attribute path and sets the value of the
    nested attribute. If the attribute path is nested (e.g., 'x.y.z'), it
    traverses through each attribute until it reaches the last one and sets
    its value.

    Args:
        obj (object): The object whose attribute needs to be set.
        attr (str): The attribute path in dot notation (e.g., 'x.y.z').
        val (any): The value to set at the specified attribute path.
    """
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """Recursively get a nested attribute of an object.

    This function splits the attribute path and retrieves the value of the
    nested attribute. If the attribute path is nested (e.g., 'x.y.z'), it
    traverses through each attribute. If an attribute in the path does not
    exist, it returns the value specified as the third argument.

    Args:
        obj (object): The object whose attribute needs to be retrieved.
        attr (str): The attribute path in dot notation (e.g., 'x.y.z').
        *args (any): Optional default value to return if the attribute
            does not exist.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


@HOOKS.register_module()
class YOLOXPoseModeSwitchHook(Hook):
    """Switch the mode of YOLOX-Pose during training.

    This hook:
    1) Turns off mosaic and mixup data augmentation.
    2) Uses instance mask to assist positive anchor selection.
    3) Uses auxiliary L1 loss in the head.

    Args:
        num_last_epochs (int): The number of last epochs at the end of
            training to close the data augmentation and switch to L1 loss.
            Defaults to 20.
        new_train_dataset (dict): New training dataset configuration that
            will be used in place of the original training dataset. Defaults
            to None.
        new_train_pipeline (Sequence[dict]): New data augmentation pipeline
            configuration that will be used in place of the original pipeline
            during training. Defaults to None.
    """

    def __init__(self, num_last_epochs: int = 20, new_train_dataset: dict = None, new_train_pipeline: Sequence[dict] = None):
        self.num_last_epochs = num_last_epochs
        self.new_train_dataset = new_train_dataset
        self.new_train_pipeline = new_train_pipeline

    def _modify_dataloader(self, runner: Runner):
        """Modify dataloader with new dataset and pipeline configurations."""
        runner.logger.info(f"New Pipeline: {self.new_train_pipeline}")

        train_dataloader_cfg = copy.deepcopy(runner.cfg.train_dataloader)
        if self.new_train_dataset:
            train_dataloader_cfg.dataset = self.new_train_dataset
        if self.new_train_pipeline:
            train_dataloader_cfg.dataset.pipeline = self.new_train_pipeline

        new_train_dataloader = Runner.build_dataloader(train_dataloader_cfg)
        runner.train_loop.dataloader = new_train_dataloader
        runner.logger.info("Recreated the dataloader!")

    def before_train_epoch(self, runner: Runner):
        """Close mosaic and mixup augmentation, switch to use L1 loss."""
        epoch = runner.epoch
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        if epoch + 1 == runner.max_epochs - self.num_last_epochs:
            self._modify_dataloader(runner)
            runner.logger.info("Added additional reg loss now!")
            model.head.use_aux_loss = True


@HOOKS.register_module()
class SequencesSwitchHook(Hook):
    def __init__(self, generate_every: int):
        self.generate_every = int(generate_every)
        assert self.generate_every > 0

    def _modify_dataloader(self, runner: Runner):
        runner.logger.info("Generating new training sequences.")
        runner.train_dataloader.dataset.reset()

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        if batch_idx > 0 and batch_idx % self.generate_every == 0:
            self._modify_dataloader(runner)

    def before_train_epoch(self, runner: Runner):
        epoch = runner.epoch
        if epoch > 0 and epoch % self.generate_every == 0:
            self._modify_dataloader(runner)


@HOOKS.register_module()
class LossCurriculumSwitchHook(Hook):
    """Two-phase curriculum: train relationship_loss only, then enable classification_loss.

    Phase 1 (iters 0 → switch_iter-1): relationship_loss only.
      model.classification_loss_weight must start at 0.0 in config.
    Phase 2 (iter >= switch_iter): both losses active.
      Hook sets model.classification_loss_weight = classification_loss_weight.

    Args:
        switch_iter (int): Global iteration at which to enable classification_loss.
        classification_loss_weight (float): Weight to apply in phase 2. Defaults to 1.0.
    """

    def __init__(self, switch_iter: int, classification_loss_weight: float = 1.0):
        self.switch_iter = switch_iter
        self.classification_loss_weight = classification_loss_weight
        self._switched = False

    def before_train_iter(self, runner: Runner, batch_idx: int, data_batch=None):
        if not self._switched and runner.iter >= self.switch_iter:
            model = runner.model
            if is_model_wrapper(model):
                model = model.module
            model.classification_loss_weight = self.classification_loss_weight
            self._switched = True
            runner.logger.info(f"[LossCurriculumSwitchHook] Iter {runner.iter}: " f"enabling classification_loss (weight={self.classification_loss_weight})")
