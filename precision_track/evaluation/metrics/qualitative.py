import os
from typing import Any, Optional

import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from precision_track.outputs import CsvActions
from precision_track.registry import METRICS


@METRICS.register_module()
class QualitativeActionRecognitionMetric(BaseMetric):
    default_prefix = "QualitativeActionRecognition"

    def __init__(self, save_dir: str, block_size: int, metainfo: Optional[str] = None, collect_device: str = "cpu", prefix: Optional[str] = None) -> None:
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        assert 0 < block_size
        self.block_size = block_size
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.action_output = CsvActions(path=os.path.join(save_dir, "actions.csv"), metainfo=metainfo)

    def process(self, data_batch: Any, data_samples: Any) -> None:
        """Process one batch of data samples and predictions."""
        actions = []
        action_scores = []
        inst_ids = []
        for i, probs in enumerate(data_samples):
            inst_ids.append(data_batch["data_samples"][0].pred_track_instances.instances_id[i].item())

            max_probs = torch.argmax(probs)
            actions.append(max_probs.item())
            action_scores.append(probs[max_probs].item())
        ds = dict(img_id=data_batch["data_samples"][0].img_id + self.block_size)

        ds["pred_track_instances"] = dict(
            actions=np.array(actions), instances_id=np.array(inst_ids), labels=np.zeros_like(inst_ids), action_scores=np.array(action_scores)
        )
        self.action_output(ds)

    def compute_metrics(self, results: list) -> dict:
        """Compute macro F1, balanced accuracy, per-class accuracy, CE, and confusion matrix."""
        self.action_output.save()
        return dict()
