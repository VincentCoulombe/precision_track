from typing import Any, Optional

import numpy as np
from mmengine.evaluator import BaseMetric

from precision_track.registry import METRICS


@METRICS.register_module()
class FeaturesReconstructionMetric(BaseMetric):
    default_prefix = "FeaturesReconstructionMetric"

    def __init__(
        self,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: Any, data_samples: Any) -> None:
        for ds in data_samples:
            self.results.append(ds["loss_features"].item())

    def compute_metrics(self, results: list) -> dict:
        return dict(avg=np.mean(results), median=np.median(results), min=np.min(results), max=np.max(results))
