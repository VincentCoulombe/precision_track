from typing import Optional, Union

from mmengine import Config
from mmengine.registry import MODELS

from .base import BaseActionPostProcessor, BasePostProcessor


@MODELS.register_module()
class PostProcessingSteps(BasePostProcessor):

    def __init__(
        self,
        postprocessing_steps: Optional[Union[Config, list, dict]] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        if postprocessing_steps is None:
            self.postprocessing_steps = []
        else:
            if isinstance(postprocessing_steps, (dict, Config)):
                postprocessing_steps = [postprocessing_steps]
            assert isinstance(postprocessing_steps, list)
            self.postprocessing_steps = [MODELS.build(p) for p in postprocessing_steps]

    def forward(self, bboxes, scores, keypoints, kpt_vis, labels, features, kept_idx):
        for p in self.postprocessing_steps:
            (
                bboxes,
                scores,
                keypoints,
                kpt_vis,
                labels,
                features,
                kept_idx,
            ) = p(bboxes, scores, keypoints, kpt_vis, labels, features, kept_idx)
        return bboxes, scores, keypoints, kpt_vis, labels, features, kept_idx


@MODELS.register_module()
class ActionPostProcessingSteps(BaseActionPostProcessor):

    def __init__(
        self,
        postprocessing_steps: Optional[Union[Config, list, dict]] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        if postprocessing_steps is None:
            self.postprocessing_steps = []
        else:
            if isinstance(postprocessing_steps, (dict, Config)):
                postprocessing_steps = [postprocessing_steps]
            assert isinstance(postprocessing_steps, list)
            self.postprocessing_steps = [MODELS.build(p) for p in postprocessing_steps]

    def forward(self, data_sample: dict):
        for p in self.postprocessing_steps:
            data_sample = p(data_sample)
        return data_sample
