import multiprocessing as mp
import traceback
from collections import deque
from logging import WARNING
from multiprocessing import shared_memory
from time import perf_counter
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.logging import print_log
from mmengine.model import BaseModel
from tqdm import tqdm

from precision_track.models.backends import DetectionBackend
from precision_track.outputs.display import display_latency
from precision_track.registry import MODELS, TRACKING, OUTPUTS
from precision_track.utils import PoseDataSample, VideoReader, wait_until_clear, batch_tracking

from .association_step import AssociationStep
from .result import Result


@MODELS.register_module()
class Tracker(BaseModel):

    def __init__(
        self,
        detector: Config,
        assigner: Config,
        validator: Optional[Config] = None,
        analyzer: Optional[Config] = None,
        outputs: Optional[List[dict]] = None,
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 1,
        profile: Optional[str] = "",
        *args,
        **kwargs,
    ):
        super().__init__()

        self.verbose = verbose

        detector["verbose"] = self.verbose
        self.detector = DetectionBackend(**detector)
        is_frozen = detector.runtime.get("freeze", False) or detector.runtime.get("type") != "PytorchRuntime"
        self._detection_mode = "predict" if is_frozen else "loss"

        assigner["verbose"] = self.verbose
        self._assigner = assigner
        self._init_association_step()

        if validator is not None:
            validator = TRACKING.build(validator)
        self.validator = validator

        if isinstance(batch_size, int) and batch_size > 0:
            self.batch_size = batch_size
        else:
            self.batch_size = 1
        self.result = Result(outputs=outputs)

        self.analyzer = analyzer
        self._analyzing = False
        if self.analyzer is not None:
            self.analyzer = MODELS.build(analyzer)
            self._analyzing = True

        assert isinstance(profile, str)
        self.profile = profile

    def _init_association_step(self):
        self.association_step = AssociationStep(**self._assigner)

    def forward(self, mode: Optional[str] = "predict", *args, **kwargs) -> Any:
        if mode == "predict":
            return self.predict(*args, **kwargs)
        elif mode == "loss":
            return self.loss(*args, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". ' "Only supports loss and predict mode.")

    def train(self, mode: bool = True):
        train_detector = self._detection_mode == "loss" and mode
        self.detector.train(train_detector)
        if isinstance(self.association_step.tracking_algorithm, nn.Module):
            self.association_step.tracking_algorithm.train(mode)
        if isinstance(self.analyzer, nn.Module) and self._analyzing:
            self.analyzer.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def loss(self, inputs: List[torch.Tensor], data_samples: List[PoseDataSample]) -> dict:
        losses = dict()
        batched_outputs = []
        for seq_inputs, seq_data_samples in zip(inputs, data_samples):
            seq_outputs = self.detector(inputs=seq_inputs, data_samples=seq_data_samples, mode=self._detection_mode)
            self._maybe_update_losses(losses, seq_outputs)
            output = self._process_sequence(
                seq_outputs,
                seq_data_samples,
                losses=losses,
                remove_gt_instances=True,
                remove_pred_instances=True,
            )
            if self._analyzing:
                batched_outputs.append(output)
        if self._analyzing:
            outputs = self.analyzer.loss(inputs=batched_outputs, data_samples=data_samples)
            self._maybe_update_losses(losses, outputs)
        return losses

    def val_step(self, data_samples: Union[dict, tuple, list], *args, **kwargs) -> list:
        return self.test_step(data_samples=data_samples, *args, **kwargs)

    def test_step(self, data_samples: Union[dict, tuple, list], *args, **kwargs) -> list:
        batched_outputs = []
        inputs = data_samples["inputs"]
        data_samples = data_samples["data_samples"]
        for seq_inputs, seq_data_samples in zip(inputs, data_samples):
            outputs = self.detector(inputs=seq_inputs, data_samples=seq_data_samples, mode="predict")
            output = self._process_sequence(
                outputs,
                seq_data_samples,
                losses=None,
                remove_gt_instances=True,
            )
            if self._analyzing:
                batched_outputs.append(output)
        if self._analyzing:
            action_preds, action_embds = self.analyzer.test_step(batched_outputs)
            output.pred_track_instances.update(dict(action_preds=action_preds, action_embds=action_embds))
            return [output]
        return batched_outputs

    def predict(self, video: VideoReader, save: bool = True) -> Result:
        assert isinstance(video, VideoReader)

        total_frames = len(video)
        t0 = perf_counter()
        batch_tracking(
            video=video,
            detector=self.detector,
            batch_size=self.batch_size,
            result=self.result,
            association_step=self.association_step,
            validator=self.validator,
            analyzer=self.analyzer,
            verbose=self.verbose,
            profile=self.profile,
        )
        if self.verbose:
            display_latency(
                np.array([perf_counter() - t0]) / total_frames,
                "Tracking Latency",
                buffer_size=0,
                precision=4,
            )
        if save:
            self.result.save()
        return self.result

    def _process_sequence(
        self,
        detections,
        data_samples,
        losses=None,
        remove_gt_instances=False,
        remove_pred_instances=False,
        remove_pred_track_instances=False,
    ):
        self._init_association_step()
        self._load_predictions(detections, data_samples)
        for seq_data_sample in data_samples:
            output = self.association_step.associate(data_sample=seq_data_sample)
            if isinstance(losses, dict):
                self._maybe_update_losses(losses, output)
            if self._analyzing:
                output = self.analyzer.data_preprocessor(dict(data_samples=output))
            if remove_gt_instances and hasattr(output, "gt_instances"):
                del output.gt_instances
                del output.gt_instance_labels
            if remove_pred_instances:
                del output.pred_instances
            if remove_pred_track_instances:
                del output.pred_track_instances
        return output

    @staticmethod
    def _slim_down_output(output: dict):
        pass  # TODO enlàve TOUTE ce qui ne sert pas dans le analyzer!

    @staticmethod
    def _maybe_update_losses(losses, outputs):
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                if "loss" in k and isinstance(v, torch.Tensor) and v.requires_grad == True:
                    if k in losses:
                        losses[k] = losses[k] + v
                    else:
                        losses[k] = v

    @staticmethod
    def _load_predictions(outputs: List[dict], data_samples: List[PoseDataSample]) -> Tuple:
        assert len(outputs) == len(data_samples)
        for output, data_sample in zip(outputs, data_samples):
            assert output["img_id"] == data_sample.img_id and output["seq_id"] == data_sample.seq_id
            data_sample.pred_instances = output["pred_instances"]


class SharedFrameBatch:
    def __init__(self, shape: Tuple, input_is_loaded):
        self.shape = shape
        self.B, self.H, self.W, self.C = shape
        self.nbytes = 2 * np.prod(shape)

        self.shm = shared_memory.SharedMemory(create=True, size=self.nbytes)
        self.frames_np = np.ndarray((2, self.B, self.H, self.W, self.C), dtype=np.uint8, buffer=self.shm.buf)

        self.shm_indices = shared_memory.SharedMemory(create=True, size=2 * self.B * np.dtype(np.uint64).itemsize)
        self.indices_np = np.ndarray((2, self.B), dtype=np.uint64, buffer=self.shm_indices.buf)

        self.input_is_loaded = input_is_loaded

        self.fill_status = [0, 0]
        self.running_batch = 0

    def is_full(self):
        return self.fill_status[self.running_batch] / self.B == 1

    def update(self, frame_idx: int, frame: np.ndarray, send_pipe):
        rel_idx = frame_idx % self.B
        if self.is_full():
            assert rel_idx == 0
            send_pipe.send((self.running_batch, self.B))
            self.input_is_loaded.wait()
            self.input_is_loaded.clear()
            self.fill_status[self.running_batch] = 0
            self.running_batch = 1 if self.running_batch == 0 else 0
        self.frames_np[self.running_batch, rel_idx, ...] = frame
        self.indices_np[self.running_batch, rel_idx, ...] = frame_idx
        self.fill_status[self.running_batch] += 1

    def send_remaining(self, send_pipe):
        send_pipe.send((self.running_batch, self.fill_status[self.running_batch]))
        self.input_is_loaded.wait()
        self.input_is_loaded.clear()

    def close(self):
        for shm in [self.shm, self.shm_indices]:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


def tracking_process(
    detector_cfg,
    assigner_cfg,
    shape,
    shm_name,
    shm_indices_name,
    input_pipe,
    input_is_loaded,
    tracking_ready,
    stop_tracking,
    ann_input_is_loaded,
    trk_output_connexion,
    validator_cfg=None,
    outout_cfg=None,
    verbose=False,
):
    detector_cfg["verbose"] = verbose
    detector = DetectionBackend(**detector_cfg)

    assigner_cfg["verbose"] = verbose
    association_step = AssociationStep(**assigner_cfg)
    switches = None
    validator = None
    if validator_cfg is not None:
        validator = TRACKING.build(validator_cfg)

    result = Result(outputs=outout_cfg)  # TODO refactor the result saving pipeline # noqa

    B, H, W, C = shape

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    frames_np = np.ndarray((2, B, H, W, C), dtype=np.uint8, buffer=existing_shm.buf)

    shm_indices = shared_memory.SharedMemory(name=shm_indices_name)
    indices_np = np.ndarray((2, B), dtype=np.uint64, buffer=shm_indices.buf)

    tracking_ready.set()
    while not stop_tracking.is_set():
        if input_pipe.poll():
            batch_idx, batch_size = input_pipe.recv()
            if batch_idx >= 0:
                frames = [frames_np[batch_idx, i, ...] for i in range(batch_size)]
                indices = indices_np[batch_idx, :batch_size].tolist()
                input_is_loaded.set()
                outputs = detector(inputs=frames, data_samples=indices)
                for i, output in enumerate(outputs):
                    output = association_step(output, switches)
                    if validator is not None:
                        frame = frames_np[batch_idx, i, ...]
                        if validator._frame_size is None:
                            validator.frame_size = frame.shape[:2]
                        output, switches = validator(frame, output)
                    trk_output_connexion.send(output)
                    ann_input_is_loaded.wait()
                    ann_input_is_loaded.clear()
                    # result(output)
            elif batch_idx == -1:
                input_is_loaded.set()
                trk_output_connexion.send(None)

    # result.save()
    for shm in [existing_shm, shm_indices]:
        shm.close()
    tracking_ready.clear()


def analyzing_process(
    ann_input_connexion,
    ann_input_is_loaded,
    analyzer_ready,
    analyzer_cfg=None,
    outout_cfg=None,
    save=True,
):
    analyzer = analyzer_cfg
    if analyzer is not None:
        analyzer = MODELS.build(analyzer_cfg)
    result = Result(outputs=outout_cfg)

    analyzer_ready.set()

    while True:
        if ann_input_connexion.poll():
            output = ann_input_connexion.recv()
            ann_input_is_loaded.set()
            if output is not None:
                if analyzer is not None:
                    output = analyzer.predict(output)
                result(output)
            else:
                break
    if save:
        result.save()  # TODO refactor the result saving pipeline
    analyzer_ready.clear()


class PipelinedTracker:
    def __init__(
        self,
        detector: Config,
        assigner: Config,
        expected_resolution: Tuple,
        validator: Optional[Config] = None,
        analyzer: Optional[Config] = None,
        outputs: Optional[List[dict]] = None,
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 1,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.expected_resolution = expected_resolution
        shape = (batch_size,) + self.expected_resolution

        self.stop_tracking = mp.Event()
        self.main_connexion, tracking_input_connexion = mp.Pipe()
        self.input_is_loaded = mp.Event()
        self.tracking_ready = mp.Event()

        trk_output_connexion, ann_input_connexion = mp.Pipe()
        ann_input_is_loaded = mp.Event()
        self.analyzer_ready = mp.Event()

        self.shared_batch = SharedFrameBatch(shape, self.input_is_loaded)

        timestamps_output = None
        if outputs is not None:
            filtered_outputs = []
            for output_cfg in outputs:
                if output_cfg.get("type") == "CsvTimestamps":
                    timestamps_output = output_cfg
                else:
                    filtered_outputs.append(output_cfg)
            outputs = filtered_outputs if filtered_outputs else None

        self.timestamps_output = OUTPUTS.build(timestamps_output) if timestamps_output else None

        self.tracking = mp.Process(
            target=tracking_process,
            args=(
                detector,
                assigner,
                shape,
                self.shared_batch.shm.name,
                self.shared_batch.shm_indices.name,
                tracking_input_connexion,
                self.input_is_loaded,
                self.tracking_ready,
                self.stop_tracking,
                ann_input_is_loaded,
                trk_output_connexion,
                validator,
                outputs,
                verbose,
            ),
        )
        self.tracking.start()
        self.tracking_ready.wait()

        self.analyzing = mp.Process(
            target=analyzing_process,
            args=(
                ann_input_connexion,
                ann_input_is_loaded,
                self.analyzer_ready,
                analyzer,
                outputs,
            ),
        )
        self.analyzing.start()
        self.analyzer_ready.wait()

    def __call__(self, video: VideoReader) -> None:
        fps = video.fps
        try:
            for i, frame in tqdm(enumerate(video)):
                assert frame.shape == self.expected_resolution
                if self.timestamps_output is not None:
                    self.timestamps_output(dict(img_id=i, fps=fps))
                self.shared_batch.update(i, frame, self.main_connexion)
            self.shared_batch.send_remaining(self.main_connexion)
            if self.timestamps_output is not None:
                self.timestamps_output.save()
        except Exception:
            error_trace = traceback.format_exc()  # TODO log
            print(error_trace)
            self.shared_batch.close()
        finally:
            self.main_connexion.send((-1, -1))
            self.input_is_loaded.wait()
            self.input_is_loaded.clear()
            self.stop_tracking.set()
            trk_cleared = wait_until_clear(self.tracking_ready, timeout=60)
            an_cleared = wait_until_clear(self.analyzer_ready, timeout=60)
            for lbl, cleared in zip(["Tracking", "Analyzer"], [trk_cleared, an_cleared]):
                if not cleared:
                    print_log(f"{lbl} process was not closed properly.", level=WARNING)

            for step in [self.tracking, self.analyzing]:
                step.join()

            self.shared_batch.close()
