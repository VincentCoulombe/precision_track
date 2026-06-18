import json
import multiprocessing as mp
import os
from typing import Dict, List, Sequence, Union

import pandas as pd
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
from precision_track.registry import TRACKING
from precision_track.utils import VideoReader, get_device, refine_corrections_offline


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
        if not isinstance(outputs, list):
            outputs = [outputs]
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
                subtype="tracked_bboxes",
            ),
        ]
        self.test_cfg = test_cfg
        self.validator_cfg = runner.cfg.validator
        self.assigner_cfg = runner.cfg.assigner
        self.detector_cfg = runner.cfg.detector

        # When offline correction refinement is enabled, also write the corrections and
        # appearance validations so the refinement pass has its inputs on disk. It then
        # refines tracking_predictions.csv in place, which is the file the metrics read.
        self.with_offline_correction_refinement = bool(runner.cfg.get("with_offline_correction_refinement")) and self.validator_cfg is not None
        self._refine_validator = None
        if self.with_offline_correction_refinement:
            self.outputs += [
                dict(
                    type="CsvCorrections",
                    path=os.path.join(work_dir, "tracking_corrections.csv"),
                    precision=32,
                ),
                dict(
                    type="CsvAppearanceValidations",
                    path=os.path.join(work_dir, "tracking_appearance_validations.csv"),
                    precision=64,
                ),
            ]

        # Per-substep throughput profiling. When ``profile_output_file`` is set, each video is
        # tracked with the non-pipelined Tracker's profiling enabled, dumping per-frame substep
        # latencies to a temporary JSON. The latencies are pooled over all videos and a single
        # throughput CSV is written next to the other testing metric CSVs.
        self.profile_output_file = test_cfg.get("profile_output_file")
        self._profile_tmp_dir = None
        self._profile_records = {}
        if self.profile_output_file:
            self._profile_tmp_dir = os.path.join(work_dir, ".substep_profiles")
            os.makedirs(self._profile_tmp_dir, exist_ok=True)

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
        # Profiling is only supported by the non-pipelined Tracker; the PipelinedTracker runs
        # substeps across processes and would silently swallow the ``profile`` argument.
        profiling = bool(self.profile_output_file) and self._tracker is Tracker
        for data_batch in self.dataloader:
            for video_path, gt_path in zip(data_batch["inputs"], data_batch["data_samples"]):
                video = VideoReader(video_path)
                profile = ""
                if profiling:
                    video_stem = os.path.splitext(os.path.basename(video_path))[0]
                    profile = os.path.join(self._profile_tmp_dir, f"{video_stem}.json")
                tracker = self._tracker(
                    detector=self.detector_cfg,
                    assigner=self.assigner_cfg,
                    validator=self.validator_cfg,
                    analyzer=None,
                    outputs=self.outputs,
                    batch_size=self.test_cfg.get("batch_size"),
                    verbose=self.verbose,
                    expected_resolution=(video.resolution[1], video.resolution[0], 3),
                    profile=profile,
                )
                tracker(video=video)
                if profiling:
                    self._accumulate_profile(profile)
                if self.with_offline_correction_refinement:
                    validator = getattr(tracker, "validator", None)
                    if validator is None or not hasattr(validator, "identities"):
                        if self._refine_validator is None:
                            self._refine_validator = TRACKING.build(self.validator_cfg)
                        validator = self._refine_validator
                    refine_corrections_offline(self.outputs, validator)
                self.evaluator.process(data_batch=[self.output_path], data_samples=[gt_path])

        if profiling:
            self._save_throughput_csv()

        metrics = self.evaluator.evaluate(len(self.dataloader))
        self.runner.call_hook("after_test")
        return metrics

    # Order substeps from earliest to latest in the tracking pipeline; absent keys are skipped.
    _PROFILE_SUBSTEP_ORDER = [
        "detection",
        "init_frame",
        "tracking",
        "stitching",
        "association_housekeeping",
        "validation",
        "analysis",
        "saving_results",
    ]

    def _accumulate_profile(self, profile_path: str) -> None:
        """Pool a video's per-frame substep latencies into ``self._profile_records``."""
        if not os.path.isfile(profile_path):
            return
        with open(profile_path, "r") as f:
            profile = json.load(f)
        for substep, latencies in profile.items():
            self._profile_records.setdefault(substep, []).extend(latencies)

    def _save_throughput_csv(self) -> None:
        """Write the mean per-substep throughput (frames/sec) pooled over all videos."""
        rows = []
        keys = [k for k in self._PROFILE_SUBSTEP_ORDER if k in self._profile_records]
        keys += [k for k in self._profile_records if k not in self._PROFILE_SUBSTEP_ORDER]
        total_mean_latency = 0.0
        for substep in keys:
            latencies = self._profile_records[substep]
            if not latencies:
                continue
            mean_latency = sum(latencies) / len(latencies)
            total_mean_latency += mean_latency
            rows.append(dict(substep=substep, throughput_fps=1.0 / mean_latency if mean_latency > 0 else float("inf")))
        if total_mean_latency > 0:
            rows.append(dict(substep="end_to_end", throughput_fps=1.0 / total_mean_latency))
        os.makedirs(os.path.dirname(self.profile_output_file), exist_ok=True)
        pd.DataFrame(rows).to_csv(self.profile_output_file, index=False)
