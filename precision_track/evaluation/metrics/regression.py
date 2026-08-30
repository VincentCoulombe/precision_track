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


@METRICS.register_module()
class MSEMetric(BaseMetric):
    """Reconstruction error of the masked-autoencoder pretraining.

    Consumes the ``dict(mse_loss=...)`` returned by :meth:`precision_track.models.MART.pretrain`,
    which :class:`~precision_track.loops.SequenceValidationLoop` forwards verbatim when it runs
    in ``mode="pretrain"``. Lower is better, hence the ``rule="less"`` of the checkpoint hook.
    """

    default_prefix = "MSEMetric"

    def __init__(
        self,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
        key: str = "mse_loss",
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.key = key

    def process(self, data_batch: Any, data_samples: Any) -> None:
        """Process one batch of data samples and predictions.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Any): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            loss = data_sample.get(self.key)
            if loss is None:
                continue
            self.results.append(float(loss.item() if hasattr(loss, "item") else loss))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        if not results:
            return dict(mean=0.0, median=0.0, min=0.0, max=0.0)
        return dict(mean=np.mean(results), median=np.median(results), min=np.min(results), max=np.max(results))
