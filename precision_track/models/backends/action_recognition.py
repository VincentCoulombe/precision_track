import logging
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine import Config
from mmengine.logging import print_log
from mmengine.runner.amp import autocast

from precision_track.registry import MODELS
from precision_track.utils import PoseDataSample, kwargs_to_args, parse_pose_metainfo, postprocess_fpv_action_recognition

from .base import BaseBackend


@MODELS.register_module()
class ActionRecognitionBackend(BaseBackend):
    DEFAULT_KPT_SCORE_THR = 0.0

    def __init__(
        self,
        runtime: Config,
        metainfo: str,
        data_preprocessor: Union[nn.Module, Config],
        data_postprocessor: Optional[Config] = None,
        input_names: Optional[list] = ["features", "poses", "dynamics"],
        smooth_factor: Optional[float] = 0.1,
        **kwargs,
    ) -> None:
        """
        Initialize the ActionRecognitionBackend.

        Args:
            data_preprocessor (Union[nn.Module, Config]): Preprocessor configuration or module.
            data_postprocessor (Config, optional): Post-processor configuration.
            smooth_factor (Optional[float], optional): The EMA strength (default: 0.1).
        """
        print_log(
            "Setting up the action recognition backend...",
            logger="current",
            level=logging.INFO,
        )
        super(ActionRecognitionBackend, self).__init__(runtime)
        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        self.actions = np.array(metainfo.get("actions", []), dtype="<U32")
        self.group_actions = np.array(metainfo.get("social_actions", []), dtype="<U32")
        self._set_preprocessor(data_preprocessor)
        if data_postprocessor is not None:
            self.data_postprocessor = MODELS.build(data_postprocessor)
        assert 0 <= smooth_factor <= 1
        assert isinstance(input_names, list)
        self.input_names = input_names
        self.smooth_factor = smooth_factor
        self.last_timestep_probs = None
        self.last_timestep_ids = None

    @property
    def runtime(self) -> nn.Module:
        return self._runtime

    @property
    def data_preprocessor(self) -> nn.Module:
        return self._data_preprocessor

    @data_preprocessor.setter
    def data_preprocessor(self, value: Union[nn.Module, Config]):
        self._set_preprocessor(value)

    def _set_preprocessor(self, preprocessor_config: Union[nn.Module, Config]) -> None:
        """
        Sets the data preprocessor.

        Args:
            preprocessor_config (Union[nn.Module, Config]): Preprocessor configuration or module.
        """
        if isinstance(preprocessor_config, nn.Module):
            self._data_preprocessor = preprocessor_config
        elif isinstance(preprocessor_config, dict):
            self._data_preprocessor = MODELS.build(preprocessor_config)
        else:
            raise TypeError("data_preprocessor should be a `dict` or " f"`nn.Module` instance, but got " f"{type(preprocessor_config)}")
        self._data_preprocessor.to(self.device)

    def val_step(self, data: Union[dict, tuple, list], *args, **kwargs) -> list:
        return self.test_step(data=data)

    def test_step(self, data: Union[dict, tuple, list], *args, **kwargs) -> list:
        return self(**data, mode="predict")

    @torch.inference_mode()
    def predict(
        self,
        data_samples: Union[List[Union[int, PoseDataSample, dict]], dict],
    ) -> Union[list[dict], dict]:
        with autocast(enabled=self.half_precision):
            data = self.preprocess(data_samples)
            preds = self._runtime.predict(kwargs_to_args(data, self.input_names), data["data_samples"])
            outputs = self.postprocess(preds, data["data_samples"])
        return outputs

    def loss(
        self,
        inputs: List[Union[np.ndarray, str]],
        data_samples: List[Union[int, PoseDataSample]],
        *args,
        **kwargs,
    ) -> dict:
        data = self.preprocess(inputs, data_samples, *args, **kwargs)
        return self._runtime.loss(**data)

    def preprocess(
        self,
        data_samples: Union[List[Union[int, PoseDataSample, dict]], dict],
    ):
        return self._data_preprocessor(dict(data_samples=[data_samples] if not isinstance(data_samples, list) else data_samples))

    def postprocess(
        self,
        preds: Union[torch.Tensor, tuple],
        data_samples: List[PoseDataSample],
    ):
        return postprocess_fpv_action_recognition(
            preds,
            data_samples,
            self.actions,
            self.data_postprocessor,
            group_actions_map=self.group_actions,
        )
