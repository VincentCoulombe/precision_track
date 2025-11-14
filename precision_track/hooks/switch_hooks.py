# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
import functools
from typing import Dict, Sequence

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
class ActionRecognitionRegenerationSwitchHook(Hook):
    def __init__(self, generate_every: int):
        self.generate_every = int(generate_every)
        assert self.generate_every > 0

    def _modify_dataloader(self, runner: Runner):
        runner.logger.info("Generating new training sequences.")
        runner.train_dataloader.dataset.load_data_list()

    def before_train_epoch(self, runner: Runner):
        epoch = runner.epoch
        if epoch > 0 and epoch % self.generate_every == 0:
            self._modify_dataloader(runner)
