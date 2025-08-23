import multiprocessing as mp
import os
from typing import Dict, List, Sequence, Union

import torch
from mmengine import Config
from mmengine.evaluator import Evaluator
from mmengine.registry import LOOPS
from mmengine.runner.amp import autocast
from mmengine.runner.loops import BaseLoop, TestLoop
from torch.utils.data import DataLoader

from precision_track import PipelinedTracker, Tracker
from precision_track.models.backends import DetectionBackend
from precision_track.models.runtimes import PytorchRuntime
from precision_track.utils import VideoReader, get_device


@LOOPS.register_module(force=True)
class SequenceTestingLoop(TestLoop):

    def __init__(
        self,
        runner,
        test_cfg: Config,
        dataloader: Union[DataLoader, Dict],
        evaluator: Union[Evaluator, Dict, List],
        fp16: bool = False,
    ):
        super().__init__(runner, dataloader, evaluator, fp16)
        self.backend = PytorchRuntime(
            checkpoint=test_cfg.get("checkpoint"),
            model=self.runner.model,
            device=get_device(),
            half_precision=self.fp16,
        )

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], *args, **kwargs) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook("before_test_iter", batch_idx=idx, data_batch=data_batch)
        with autocast(enabled=self.fp16):
            outputs = self.backend.test_step(data_batch, *args, **kwargs)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            "after_test_iter",
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs,
        )

    def run(self, *args, **kwargs) -> dict:
        """Launch validation."""
        self.runner.call_hook("before_test")
        self.runner.call_hook("before_test_epoch")
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch, *args, **kwargs)

        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook("after_test_epoch", metrics=metrics)
        self.runner.call_hook("after_test")
        return metrics


@LOOPS.register_module(force=True)
class TestingLoop(TestLoop):
    def __init__(
        self,
        runner,
        test_cfg: Config,
        dataloader: Union[DataLoader, Dict],
        evaluator: Union[Evaluator, Dict, List],
        fp16: bool = False,
    ):
        super().__init__(runner, dataloader, evaluator, fp16)
        self.backend = DetectionBackend(
            temperature_file=test_cfg.get("hyperparameters", ""),
            data_preprocessor=test_cfg.get("data_preprocessor"),
            data_postprocessor=test_cfg.get("data_postprocessor"),
            kpt_score_thr=test_cfg.get("kpt_score_thr", 0.0),
            runtime=dict(
                type="PytorchRuntime",
                checkpoint=test_cfg.get("checkpoint"),
                model=self.runner.model,
                device=get_device(),
                half_precision=self.fp16,
            ),
        )

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook("before_test_iter", batch_idx=idx, data_batch=data_batch)
        with autocast(enabled=self.fp16):
            inputs = [img.detach().cpu().numpy() for img in data_batch["inputs"]]
            outputs = self.backend(inputs, data_batch["data_samples"])
        self.runner.call_hook(
            "after_test_step",
            batch_idx=idx,
            data_batch=data_batch,
            data_samples=outputs,
        )

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            "after_test_iter",
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs,
        )


@LOOPS.register_module(force=True)
class TrackingTestingLoop(BaseLoop):

    def __init__(
        self,
        runner,
        test_cfg: Config,
        evaluator: Union[Evaluator, Dict, List],
        verbose: bool = True,
        pipelined: bool = False,
        *args,
        **kwargs,
    ):
        dataloader = test_cfg.get("dataloader")
        super().__init__(runner, dataloader)
        work_dir = test_cfg.get("work_dir", "./")
        self.output_path = os.path.join(work_dir, "tracking_predictions.csv")
        self.outputs = [
            dict(
                type="CsvBoundingBoxes",
                path=self.output_path,
                instance_data="pred_track_instances",
                precision=64,
            ),
        ]
        self.test_cfg = test_cfg
        if isinstance(evaluator, dict) or isinstance(evaluator, list):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator
        self.verbose = verbose

        self._tracker = None
        if pipelined:
            self.tracker = PipelinedTracker
        else:
            self.tracker = Tracker

    @property
    def tracker(self):
        return self._tracker

    @tracker.setter
    def tracker(self, tracker):
        if tracker is Tracker:
            self._tracker = tracker
        elif tracker is PipelinedTracker:
            mp.set_start_method("spawn", force=True)
            self._tracker = PipelinedTracker
        else:
            raise ValueError

    def run(self, *args, **kwargs) -> dict:
        """Launch validation."""

        self.runner.call_hook("before_test")
        self.runner.call_hook("before_test_epoch")
        for data_batch in self.dataloader:
            for video_path, gt_path in zip(data_batch["inputs"], data_batch["data_samples"]):
                video = VideoReader(video_path)
                tracker = self._tracker(
                    detector=self.test_cfg.get("detector"),
                    assigner=self.test_cfg.get("assigner"),
                    validator=self.test_cfg.get("validator"),
                    analyzer=self.test_cfg.get("analyzer"),
                    outputs=self.outputs,
                    batch_size=self.test_cfg.get("batch_size"),
                    verbose=self.verbose,
                    expected_resolution=(video.resolution[1], video.resolution[0], 3),
                )
                tracker(video=video)
                self.evaluator.process(data_batch=[self.output_path], data_samples=[gt_path])

        metrics = self.evaluator.evaluate(len(self.dataloader))
        self.runner.call_hook("after_test")
        return metrics
