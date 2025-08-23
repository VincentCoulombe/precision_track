from typing import Dict, List, Sequence, Union

from mmengine import Config
from mmengine.registry import LOOPS
from mmengine.runner.amp import autocast
from torch.utils.data.dataloader import DataLoader

from .testing_loop import TestingLoop


@LOOPS.register_module()
class CalibrationLoop(TestingLoop):

    def __init__(
        self,
        calibration_cfg: Config,
        runner,
        dataloader: Union[DataLoader, Dict],
        evaluator: Union[Dict, List],
    ):
        super().__init__(
            runner=runner,
            test_cfg=calibration_cfg,
            dataloader=dataloader,
            evaluator=evaluator,
        )
        self.backend.runtime.model.head.temperature = 1
        self.device = self.backend.runtime.device

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook("before_calibration")
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        self.runner.call_hook("after_calibration")
        return metrics

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook("before_calibration_iter", batch_idx=idx, data_batch=data_batch)
        with autocast(enabled=self.fp16):
            images = [img.numpy() for img in data_batch["inputs"]]
            data = self.backend.preprocess(images, data_batch["data_samples"])
            logits = self.backend.runtime.forward(**data)
            outputs = self.backend.postprocess(*logits, data_samples=data["data_samples"])
        for i, output in enumerate(outputs):
            pt_idx = output["pred_instances"]["kept_idxs"].to(self.device)
            output["logits"] = [logit[i, pt_idx, ...] for logit in logits]

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            "after_calibration_iter",
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs,
        )
