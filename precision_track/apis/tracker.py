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
from precision_track.outputs.display import display_latency, display_progress_bar
from precision_track.registry import MODELS, TRACKING
from precision_track.utils import PoseDataSample, VideoReader, wait_until_clear

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
        *args,
        **kwargs,
    ):
        super().__init__()

        self.verbose = verbose

        detector["verbose"] = self.verbose
        self.detector = DetectionBackend(**detector)
        self._detection_mode = "predict" if detector.runtime.get("freeze", False) else "loss"

        assigner["verbose"] = self.verbose
        self.association_step = AssociationStep(**assigner)

        if validator is not None:
            validator = TRACKING.build(validator)
        self.validator = validator

        if isinstance(batch_size, int) and batch_size > 0:
            self.batch_size = batch_size
        else:
            self.batch_size = 1
        self.result = Result(outputs=outputs)

        self.analyzer = analyzer
        if self.analyzer is not None:
            self.analyzer = MODELS.build(analyzer)

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
        return self

    def eval(self):
        return self.train(False)

    def loss(self, inputs: List[torch.Tensor], data_samples: List[PoseDataSample]) -> dict:
        inputs, data_samples = self._flatten_sequences(inputs, data_samples)
        losses = dict()
        outputs = self.detector(inputs=inputs, data_samples=data_samples, mode="tensor" if self._detection_mode != "loss" else self._detection_mode)
        if isinstance(outputs, dict):
            losses.update(outputs.get("losses", dict()))
            outputs = outputs["outputs"]
        self._load_predictions(outputs, data_samples)
        outputs = self.association_step.tracking_algorithm.loss(data_samples=data_samples)
        losses.update(outputs.get("losses", dict()))
        outputs = self.analyzer.loss(data_samples=data_samples)
        losses.update(outputs.get("losses", dict()))
        return losses

    def val_step(self, data_samples: Union[dict, tuple, list], *args, **kwargs) -> list:
        return self.test_step(data_samples=data_samples, *args, **kwargs)

    def test_step(self, data_samples: Union[dict, tuple, list], *args, **kwargs) -> list:
        inputs, data_samples = self._flatten_sequences(**data_samples)
        outputs = self.detector.test_step(dict(inputs=inputs, data_samples=data_samples, *args, **kwargs))
        self._load_predictions(outputs, data_samples)
        return self.association_step.tracking_algorithm.test_step(data_samples=data_samples, *args, **kwargs)

    def predict(self, video: VideoReader) -> Result:
        assert isinstance(video, VideoReader)

        b_frames = []
        b_idx = []
        outputs = deque()
        frames = deque()
        frame_id = 0
        empty = False
        switches = None
        total_frames = len(video)
        t0 = perf_counter()
        while True:
            frame = video.read()
            if len(b_frames) == self.batch_size or (empty and b_frames):
                for output in self.detector(inputs=b_frames, data_samples=b_idx):
                    outputs.appendleft(output)
                b_frames, b_idx = [], []
            if frame is not None:
                b_frames.append(frame)
                b_idx.append(frame_id)
                frames.appendleft(frame)
                frame_id += 1
                if self.verbose:
                    display_progress_bar(frame_id, total_frames)
            else:
                empty = True
            if outputs:
                output = outputs.pop()
                output = self.association_step(output, switches)
                frame = frames.pop()
                if self.validator is not None:
                    if self.validator._frame_size is None:
                        self.validator.frame_size = frame.shape[:2]
                    output, switches = self.validator(frame, output)
                if self.analyzer is not None:
                    output = self.analyzer.predict(output)

                self.result(output)
            elif empty and not b_frames:
                break

        if self.verbose:
            display_latency(
                np.array([perf_counter() - t0]) / total_frames,
                "Tracking Latency",
                buffer_size=0,
                precision=4,
            )
        self.result.save()
        return self.result

    @staticmethod
    def _flatten_sequences(inputs: List[torch.Tensor], data_samples: List[PoseDataSample]) -> Tuple:
        sequence_length = inputs[0].shape[0]
        assert len(data_samples) == sequence_length
        flatten_inputs = []
        flatten_ds = []
        for batch_idx in range(len(inputs)):
            assert sequence_length == inputs[batch_idx].shape[0]
            for sequence_idx in range(sequence_length):
                flatten_inputs.append(inputs[batch_idx][sequence_idx])
                flatten_ds.append(data_samples[sequence_idx][batch_idx])
        return flatten_inputs, flatten_ds

    @staticmethod
    def _load_predictions(outputs: List[dict], data_samples: List[PoseDataSample]) -> Tuple:
        assert len(outputs) == len(data_samples)
        for output, data_sample in zip(outputs, data_samples):
            assert output["img_id"] == data_sample.img_id
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
        try:
            for i, frame in tqdm(enumerate(video)):
                assert frame.shape == self.expected_resolution
                self.shared_batch.update(i, frame, self.main_connexion)
            self.shared_batch.send_remaining(self.main_connexion)
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
