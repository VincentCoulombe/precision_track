import operator
import os
from collections import defaultdict
from typing import Any, List, Optional
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from sklearn.metrics import classification_report, confusion_matrix

from precision_track.outputs import CsvBoundingBoxes, CsvSearchAreas
from precision_track.registry import METRICS
from precision_track.utils import PoseDataSample, batch_bbox_areas, iou_batch, linear_assignment, oks_batch, parse_pose_metainfo, reformat, wilson_bounds


@METRICS.register_module()
class MSEMetric(BaseMetric):
    default_prefix = "MSEMetric"

    def __init__(
        self,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: Any, data_samples: Any) -> None:
        for data_sample in data_samples:
            self.results.append(data_sample["mse_loss"].item())

    def compute_metrics(self, results: list) -> dict:
        return dict(mean=np.mean(results), median=np.median(results), min=np.min(results), max=np.max(results))
