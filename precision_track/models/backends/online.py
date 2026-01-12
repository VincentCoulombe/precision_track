import logging
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine import Config
from mmengine.logging import print_log
from mmengine.runner.amp import autocast

from precision_track.registry import MODELS
from precision_track.utils import PoseDataSample, kwargs_to_args, parse_pose_metainfo, sequential_ema_smoothing

from .base import BaseBackend


@MODELS.register_module()
class OnlineBackend(BaseBackend):
    DEFAULT_KPT_SCORE_THR = 0.0

    def __init__(
        self,
        runtime: Config,
        metainfo: str,
        data_preprocessor: Optional[Union[nn.Module, Config]] = None,
        data_postprocessor: Optional[Config] = None,
        input_names: Optional[list] = ["features", "poses", "dynamics"],
        smooth_factor: Optional[float] = 0.1,
        **kwargs,
    ) -> None:
        """
        Initialize the OnlineBackend.

        Args:
            data_preprocessor (Union[nn.Module, Config]): Preprocessor configuration or module.
            data_postprocessor (Config, optional): Post-processor configuration.
            smooth_factor (Optional[float], optional): The EMA strength (default: 0.1).
        """
        print_log(
            "Setting up the backend...",
            logger="current",
            level=logging.INFO,
        )
        super(OnlineBackend, self).__init__(runtime)
        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        self.actions = np.array(metainfo.get("actions", []), dtype="<U32")

        self._set_preprocessor(data_preprocessor)

        self.data_postprocessor = data_postprocessor
        if data_postprocessor is not None:
            self.data_postprocessor = MODELS.build(data_postprocessor)

        assert 0 <= smooth_factor <= 1
        assert isinstance(input_names, list)
        self.input_names = input_names
        self.smooth_factor = smooth_factor
        self.last_timestep_probs = None
        self.last_timestep_ids = None

        self.tubes = dict()

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
        if isinstance(preprocessor_config, nn.Module) or preprocessor_config is None:
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
        inputs: torch.Tensor,
        data_samples: Union[List[Union[int, PoseDataSample, dict]], dict],
    ) -> Union[list[dict], dict]:
        with autocast(enabled=self.half_precision):
            data = self.preprocess(inputs, data_samples, "predict")
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
        data = self.preprocess(inputs, data_samples, "loss")
        return self._runtime.loss(**data)

    def preprocess(
        self,
        inputs: torch.Tensor,
        data_samples: Union[List[Union[int, PoseDataSample, dict]], dict],
        mode: Optional[str] = "loss",
    ):
        if self.data_preprocessor is None:
            return dict(inputs=inputs, data_samples=data_samples)

        if mode == "loss":
            return self._data_preprocessor.loss(dict(inputs=inputs, data_samples=data_samples, tubes=self.tubes))
        elif mode == "predict":
            return self._data_preprocessor.predict(dict(inputs=inputs, data_samples=data_samples, tubes=self.tubes))
        else:
            raise ValueError(f"OnlineBackend only support loss and predict mode, not: {mode}.")

    def postprocess(
        self,
        preds: Union[torch.Tensor, tuple],
        data_samples: List[PoseDataSample],
    ):
        assert len(data_samples) == 1, "Action Recognition does not support batches."
        data_sample = data_samples[0]
        if isinstance(preds, tuple):
            data_sample["pred_track_instances"]["action_embeddings"] = preds[1]
            preds = preds[0]

        current_timestep_ids = data_sample["pred_track_instances"]["instances_id"]
        current_timestep_probs = preds.detach().cpu().numpy().astype(np.float32)
        if isinstance(self.last_timestep_ids, np.ndarray) and isinstance(self.last_timestep_probs, np.ndarray):
            assert current_timestep_probs.shape[1] == self.last_timestep_probs.shape[1]
            current_timestep_probs = sequential_ema_smoothing(
                self.last_timestep_ids,
                self.last_timestep_probs,
                current_timestep_ids,
                current_timestep_probs,
                self.smooth_factor,
            )
        actions = self.actions[np.argmax(current_timestep_probs, axis=1).reshape(-1)]
        action_scores = np.max(current_timestep_probs, axis=1).reshape(-1)

        valid_context = data_sample["pred_track_instances"]["valid_action_recognition_context"]
        actions[~valid_context] = "Analyzing..."
        action_scores[~valid_context] = 1

        data_sample["pred_track_instances"]["actions"] = actions
        data_sample["pred_track_instances"]["action_scores"] = action_scores

        self.last_timestep_ids = current_timestep_ids
        self.last_timestep_probs = current_timestep_probs

        if self.data_postprocessor is not None:
            data_sample = self.data_postprocessor(data_sample, tubes=self.tubes)  # TODO roll les action tubes

        return data_sample
