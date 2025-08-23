from typing import Any, Optional

import numpy as np
from mmengine.evaluator import BaseMetric
from sklearn.metrics import silhouette_score

from precision_track.registry import METRICS


@METRICS.register_module()
class SilhouetteScore(BaseMetric):
    default_prefix = "SilhouetteScore"

    def __init__(
        self,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: Any, data_samples: Any) -> None:
        """Process one batch of data samples and predictions.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Any): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            batch_features = data_sample.get("extracted_features")
            if batch_features is not None:
                for features, labels in zip(*batch_features):
                    labels = labels.cpu().numpy()
                    features = features.cpu().numpy()
                    if not np.all(labels == labels[0]):  # More than one individual in the frame.
                        score = silhouette_score(features, labels, metric="cosine")
                        self.results.append(score)

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        if not results:
            return dict(avg=0.0)
        return dict(avg=np.mean(results))
