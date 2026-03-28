from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from addict import Dict as ADict
from mmengine import Config
from mmengine.evaluator import Evaluator
from mmengine.registry import LOOPS
from mmengine.runner.amp import autocast
from mmengine.runner.loops import ValLoop
from torch.utils.data import DataLoader

from precision_track.models.backends import DetectionBackend
from precision_track.models.postprocessing.steps import PostProcessingSteps
from precision_track.registry import MODELS
from precision_track.tracking import OnlineGroundTruth
from precision_track.utils import PoseDataSample, get_device, parse_pose_metainfo, postprocess_fpv_action_recognition, postprocess_one_stage_detections


@LOOPS.register_module()
class ValidationLossLoop(ValLoop):

    def __init__(
        self,
        runner,
        val_cfg: Config,
        dataloader: Union[DataLoader, Dict],
        evaluator: Union[Evaluator, Dict, List],
        fp16: bool = True,
    ):
        super().__init__(
            runner=runner,
            dataloader=dataloader,
            evaluator=evaluator,
            fp16=fp16,
        )
        assert isinstance(val_cfg, (Config, dict)), "If the validation loop is not sequential, a validation config must be provided."
        self.backend = DetectionBackend(
            temperature_file=val_cfg.get("hyperparameters", ""),
            data_preprocessor=self.runner.model.data_preprocessor,
            data_postprocessor=val_cfg.get("data_postprocessor"),
            kpt_score_thr=val_cfg.get("kpt_score_thr", 0.0),
            runtime=dict(
                type="PytorchRuntime",
                checkpoint=val_cfg.get("checkpoint"),
                model=self.runner.model,
                device=get_device(),
                half_precision=self.fp16,
            ),
        )
        val_data_preprocessor = val_cfg.get("data_preprocessor")
        if val_data_preprocessor is not None:
            val_data_preprocessor = MODELS.build(val_data_preprocessor)
            self.backend.data_preprocessor = val_data_preprocessor
        self.reset_losses()

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], *args, **kwargs) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        with autocast(enabled=self.fp16):
            inputs = self.backend.data_preprocessor(data_batch, False)
            val_losses = self.backend.runtime.loss(**inputs)
        for val_loss in val_losses:
            if "loss" in val_loss:
                self.val_losses[f"val/{val_loss}"].append(val_losses[val_loss].flatten().mean(0).item())
            if "overlaps" in val_loss:
                self.val_losses[f"val/{val_loss}"].append(val_losses[val_loss].flatten().max(0)[0].item())

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook("before_val")
        self.runner.call_hook("before_val_epoch")
        self.runner.model.eval()
        self.reset_losses()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)
        total_loss = 0
        for val_loss in self.val_losses:
            mean_loss = np.mean(self.val_losses[val_loss])
            self.val_losses[val_loss] = mean_loss
            if "loss" in val_loss:
                total_loss += mean_loss
        self.val_losses["val/loss"] = total_loss
        self.runner.call_hook("after_val_epoch", metrics=self.val_losses)
        self.runner.call_hook("after_val")
        return self.val_losses

    def reset_losses(self):
        self.val_losses = defaultdict(list)
        tags_to_del = []
        for tag in self.runner.message_hub._log_scalars:
            if "val" in tag:
                tags_to_del.append(tag)
        for tag in tags_to_del:
            del self.runner.message_hub._log_scalars[tag]


@LOOPS.register_module()
class ValidationLoop(ValLoop):

    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        evaluator: Union[Evaluator, Dict, List],
        val_cfg: Optional[Config] = None,
        fp16: bool = True,
        is_sequence: bool = False,
    ):
        super().__init__(
            runner=runner,
            dataloader=dataloader,
            evaluator=evaluator,
            fp16=fp16,
        )
        if is_sequence:
            self.backend = None
        else:
            assert isinstance(val_cfg, (Config, dict)), "If the validation loop is not sequential, a validation config must be provided."
            self.backend = DetectionBackend(
                temperature_file=val_cfg.get("hyperparameters", ""),
                data_preprocessor=self.runner.model.data_preprocessor,
                data_postprocessor=val_cfg.get("data_postprocessor"),
                kpt_score_thr=val_cfg.get("kpt_score_thr", 0.0),
                runtime=dict(
                    type="PytorchRuntime",
                    checkpoint=val_cfg.get("checkpoint"),
                    model=self.runner.model,
                    device=get_device(),
                    half_precision=self.fp16,
                ),
            )

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], *args, **kwargs) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook("before_val_iter", batch_idx=idx, data_batch=data_batch)
        with autocast(enabled=self.fp16):
            outputs = self.backend.val_step(data_batch, *args, **kwargs)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            "after_val_iter",
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs,
        )

    def run(self, *args, **kwargs) -> dict:
        """Launch validation."""
        self.runner.call_hook("before_val")
        self.runner.call_hook("before_val_epoch")
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch, *args, **kwargs)

        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook("after_val_epoch", metrics=metrics)
        self.runner.call_hook("after_val")
        return metrics


@LOOPS.register_module()
class SequenceValidationLoop(ValidationLoop):
    VALID_MODES = ["pretrain", "predict"]

    def __init__(self, runner, dataloader: Union[DataLoader, Dict], evaluator: Union[Evaluator, Dict, List], fp16: bool = False, mode: bool = "predict"):
        super().__init__(
            runner=runner,
            val_cfg=None,
            dataloader=dataloader,
            evaluator=evaluator,
            fp16=fp16,
            is_sequence=True,
        )
        self.backend = self.runner.model
        self.mode = mode

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], *args, **kwargs) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook("before_val_iter", batch_idx=idx, data_batch=data_batch)
        with autocast(enabled=self.fp16):
            if self.mode == "predict":
                outputs = self.backend.val_step(data_batch, *args, **kwargs)
            else:
                data = self.backend.data_preprocessor(data_batch)
                outputs = self.backend.pretrain(**data)
        self.evaluator.process(data_samples=[outputs], data_batch=data_batch)
        self.runner.call_hook(
            "after_val_iter",
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs,
        )


@LOOPS.register_module()
class OnlineValLoop(ValLoop):
    def __init__(
        self,
        runner,
        metainfo: str,
        dataloader: Union[DataLoader, Dict],
        post_processor: Config,
        evaluator: Union[Evaluator, Dict, List],
        fp16: bool = False,
    ) -> None:

        assert hasattr(runner, "detector")

        self.post_processor = PostProcessingSteps(post_processor)
        self.tracker = OnlineGroundTruth()
        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        self.actions = np.array(metainfo.get("actions", []), dtype="<U32")

        super().__init__(
            runner,
            dataloader,
            evaluator,
            fp16,
        )

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], *args, **kwargs) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook("before_val_iter", batch_idx=idx, data_batch=data_batch)

        sequences_inputs = data_batch["inputs"]
        sequences_data_samples = data_batch["data_samples"]

        B = len(sequences_inputs)
        T, C, H, W = sequences_inputs[0].shape

        no_grad_frames = []
        no_grad_data_samples = []
        splitted_seq_data_samples = [OrderedDict() for _ in sequences_data_samples]

        for i, (sequence_inputs, sequence_data_samples) in enumerate(zip(sequences_inputs, sequences_data_samples)):
            seq_no_grad_frames = []
            seq_no_grad_ds = []
            for no_grad_idx in range(T):
                seq_no_grad_frames.append(sequence_inputs[no_grad_idx])
                ds = self._load_data_sample(sequence_data_samples, no_grad_idx)
                seq_no_grad_ds.append(ds)
                splitted_seq_data_samples[i][no_grad_idx] = ds
            no_grad_frames.append(torch.cat(seq_no_grad_frames).view(T, C, H, W))
            no_grad_data_samples.append(seq_no_grad_ds)

        no_grad_frames = torch.cat(no_grad_frames).view(T * B, C, H, W)

        with autocast(enabled=self.fp16):
            data = self.runner.detector.data_preprocessor(
                dict(inputs=no_grad_frames, data_samples=[PoseDataSample() for _ in range(no_grad_frames.shape[0])]), False
            )
            (
                scores,
                objectness,
                bboxes,
                kpts,
                kpt_vis,
                features,
                priors,
                _,
            ) = self.runner.detector.test_step(data)

        P = priors.shape[1]
        scores = scores.view(B, T, P, -1)
        objectness = objectness.view(B, T, P, -1)
        bboxes = bboxes.view(B, T, P, 4)
        kpts = kpts.view(B, T, P, -1, 2)
        kpt_vis = kpt_vis.view(B, T, P, -1)
        features = features.view(B, T, P, -1)

        inputs = defaultdict(list)
        input_dims = defaultdict(tuple)
        for i, ds in enumerate(splitted_seq_data_samples):
            splitted_seq_data_samples[i] = [t[1] for t in sorted(ds.items())]
            with torch.inference_mode():
                post_processed_seq = postprocess_one_stage_detections(
                    self.post_processor,
                    scores[i],
                    objectness[i],
                    bboxes[i],
                    kpts[i],
                    kpt_vis[i],
                    features[i],
                    priors,
                    splitted_seq_data_samples[i],
                )
            seq_data_samples = []
            prev = None
            for post_processed_frame in post_processed_seq:
                data_samples = self.tracker(post_processed_frame, prev)
                seq_inputs = self.runner.model.data_preprocessor(dict(inputs=[], data_samples=data_samples))
                data_samples = seq_inputs.pop("data_samples")
                prev = data_samples
                seq_inputs.pop("block_ids", None)
                seq_data_samples.append(ADict(data_samples))
            for seq_input in seq_inputs:
                inputs[seq_input].append(seq_inputs[seq_input])
                input_dims[seq_input] = tuple(seq_inputs[seq_input].shape[1:])
        for in_ in inputs:
            in_dims = input_dims[in_]
            inputs[in_] = torch.cat(inputs[in_]).view(-1, *in_dims)

        with autocast(enabled=self.fp16):
            outputs = self.runner.model.predict(tuple([inputs["features"], inputs["poses"], inputs["dynamics"]]), data_samples=seq_data_samples)

        seq_data_samples[-1] = postprocess_fpv_action_recognition(
            outputs[0],
            seq_data_samples[-1],
            self.actions,
            None,
        )

        # TODO passer tout les ds pour evaluer tracking et passer juste le -1 pour évaluer AR...
        self.evaluator.process(data_samples=outputs, data_batch=dict(data_samples=seq_data_samples[-1]))
        self.runner.call_hook(
            "after_val_iter",
            batch_idx=idx,
            data_batch=seq_data_samples,
            outputs=outputs,
        )

    @staticmethod
    def _load_data_sample(data_samples, idx):
        out = PoseDataSample(metainfo=data_samples.metainfo)
        mask = data_samples.gt_instances.frame_id == idx
        out.gt_instance_labels = data_samples.gt_instances[mask]
        out.ori_shape = None
        out.img_id = None
        out.img_path = None
        out.id = None
        # out.img_shape = data_samples.img_shape
        return out
