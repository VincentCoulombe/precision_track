import json
import logging
import os.path as osp
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from addict import Dict
from mmengine import Config
from mmengine.logging import print_log

from precision_track.models import DetectionHead
from precision_track.models.postprocessing.steps import PostProcessingSteps
from precision_track.registry import MODELS
from precision_track.utils import PoseDataSample, xyxy_cxcywh

from .base import BaseBackend


@MODELS.register_module()
class DetectionBackend(BaseBackend):
    DEFAULT_KPT_SCORE_THR = 0.0

    def __init__(
        self,
        runtime: Config,
        temperature_file: Optional[str] = "",
        data_preprocessor: Optional[Union[nn.Module, Config]] = None,
        data_postprocessor: Optional[Union[Config, list, dict]] = None,
        kpt_score_thr: Optional[float] = DEFAULT_KPT_SCORE_THR,
        freeze: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """
        Initialize the DetectionBackend.

        Args:
            data_preprocessor (Union[nn.Module, Config]): Preprocessor configuration or module.
            data_postprocessor (Config, optional): Post-processor configuration.
            kpt_score_thr (Optional[float], optional): Threshold for keypoint scores (default: 0.0).
        """
        print_log(
            "Setting up the detection backend...",
            logger="current",
            level=logging.INFO,
        )
        super(DetectionBackend, self).__init__(runtime)
        self.post_processor = data_postprocessor
        if data_postprocessor is not None:
            self.post_processor = PostProcessingSteps(data_postprocessor)
        else:
            self.post_processor = []
        self.kpt_score_thr = kpt_score_thr
        self._set_preprocessor(data_preprocessor)

        temperature = 1
        if isinstance(temperature_file, str) and osp.exists(temperature_file):
            with open(temperature_file, "r") as f:
                try:
                    hyperparams = json.load(f)
                except json.JSONDecodeError:
                    hyperparams = {}
            temperature = hyperparams.get("calibrated_temperature", 1)
        if hasattr(self._runtime, "model") and isinstance(self._runtime.model.head, DetectionHead):
            self._runtime.model.head.temperature = temperature
            if self.verbose:
                print_log(
                    f"Dynamically updating detection's head temperature to: {temperature}.",
                    logger="current",
                    level=logging.WARNING,
                )

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
        if preprocessor_config is None:
            preprocessor_config = dict(type="BaseDataPreprocessor")
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
        assert isinstance(data, dict), f"Test's inputs should be dict, but got {type(data)}"
        inputs = [img.numpy() for img in data["inputs"]]
        return self(inputs, data["data_samples"], mode="predict")

    def loss(
        self,
        inputs: List[Union[np.ndarray, str]],
        data_samples: List[Union[int, PoseDataSample]],
        *args,
        **kwargs,
    ) -> dict:
        data = self.preprocess(inputs, data_samples, *args, **kwargs)
        data["return_features"] = True
        losses, feats = self._runtime.loss(**data)
        return {"losses": losses, "outputs": self.postprocess(*feats, data["data_samples"])}

    def preprocess(
        self,
        images: List[Union[np.ndarray, str]],
        ids: List[Union[int, PoseDataSample]],
    ):
        return self._data_preprocessor(dict(inputs=images, data_samples=ids))

    def postprocess(
        self,
        scores: torch.Tensor,
        objectness: torch.Tensor,
        bboxes: torch.Tensor,
        kpts: torch.Tensor,
        kpt_vis: torch.Tensor,
        features: torch.Tensor,
        priors: torch.Tensor,
        strides: torch.Tensor,
        data_samples: List[PoseDataSample],
    ):
        assert bboxes.shape[0] == len(data_samples)

        scores = scores.sigmoid()
        objectness = objectness.sigmoid()
        scores *= objectness
        scores, labels = scores.max(2, keepdim=True)

        formatted_outputs = []
        for i, data_sample in enumerate(data_samples):
            input_size = data_sample.metainfo["input_size"]
            input_center = data_sample.metainfo["input_center"]
            input_scale = data_sample.metainfo["input_scale"]

            i_bboxes = bboxes[i]
            i_scores = scores[i]
            i_kpts = kpts[i]
            i_kpt_vis = kpt_vis[i]
            i_labels = labels[i]
            i_features = features[i]

            i_bboxes, i_scores, i_kpts, i_kpt_vis, i_labels, i_features, i_kept_idxs = self.post_processor(
                i_bboxes, i_scores, i_kpts, i_kpt_vis, i_labels, i_features, torch.tensor(0)
            )

            i_scores = i_scores.flatten()
            i_kpt_vis = i_kpt_vis.sigmoid()
            i_labels = i_labels.flatten()

            scale = torch.tensor(input_scale, dtype=torch.float32, device=i_bboxes.device)
            rescale = scale / torch.tensor(input_size, dtype=torch.float32, device=i_bboxes.device)
            translation = torch.tensor(input_center, dtype=torch.float32, device=i_bboxes.device) - 0.5 * scale

            i_kpts = i_kpts * rescale.view(1, 1, 2) + translation.view(1, 1, 2)
            i_kpts[i_kpt_vis < self.kpt_score_thr] = 0.0

            i_bboxes = i_bboxes * torch.tile(rescale, (i_bboxes.shape[0], 2)) + torch.tile(translation, (i_bboxes.shape[0], 2))
            i_bboxes = xyxy_cxcywh(i_bboxes)

            pred_instances = Dict()
            pred_instances.bboxes = i_bboxes
            pred_instances.scores = i_scores
            pred_instances.keypoints = i_kpts
            pred_instances.keypoint_scores = i_kpt_vis
            pred_instances.labels = i_labels
            pred_instances.features = i_features
            pred_instances.kept_idxs = i_kept_idxs

            formatted_pred_instances = {
                "ori_shape": data_sample.ori_shape,
                "img_id": data_sample.img_id,
                "img_path": getattr(data_sample, "img_path", None),
                "id": getattr(data_sample, "id", None),
                "category_id": getattr(data_sample, "category_id", 1),
                "gt_instances": getattr(data_sample, "gt_instance_labels", None),
            }

            formatted_pred_instances["pred_instances"] = pred_instances
            formatted_outputs.append(formatted_pred_instances)
        return formatted_outputs
