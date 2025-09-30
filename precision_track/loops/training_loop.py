from typing import Sequence, Union, Dict, Optional, List, Tuple
import torch
from torch.utils.data import DataLoader
from mmengine.runner import EpochBasedTrainLoop
from mmengine.structures import InstanceData
from mmengine.config import Config
from precision_track.registry import LOOPS
from precision_track.utils import PoseDataSample, postprocess_one_stage_detections
from precision_track.models.postprocessing.steps import PostProcessingSteps


@LOOPS.register_module()
class TrackingEpochBasedTrainLoop(EpochBasedTrainLoop):
    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        post_processor: Config,
        max_epochs: int,
        val_begin: int = 1,
        val_interval: int = 1,
        dynamic_intervals: Optional[List[Tuple[int, int]]] = None,
    ) -> None:

        assert hasattr(runner, "one_stage")
        assert hasattr(runner, "detector")
        assert hasattr(runner, "det_optim_wrapper")

        self.post_processor = PostProcessingSteps(post_processor)

        super().__init__(
            runner,
            dataloader,
            max_epochs,
            val_begin,
            val_interval,
            dynamic_intervals,
        )

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook("before_train_iter", batch_idx=idx, data_batch=data_batch)

        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.

        sequences_inputs = data_batch["inputs"]
        sequences_data_samples = data_batch["data_samples"]

        B = len(sequences_inputs)
        T, C, W, H = sequences_inputs[0].shape

        if self.runner.one_stage:
            num_train_frames = min(self.runner.train_frames, T)
            if num_train_frames == 0:
                self.runner.logger.warn("One-Stage training is activated, but either train_frames is not set or your are training on empty sequences.")

            # Select the training and no_grad frames:
            random_frame_idxs = torch.randperm(T)  # use these random indices to select the frames.

            go_back_frame_idxs = torch.argsort(random_frame_idxs)  # use these indices to go back to the original order.

            train_frame_idxs = random_frame_idxs[:num_train_frames]
            train_frames = []
            det_train_data_samples = []
            for sequence_inputs, sequence_data_samples in zip(sequences_inputs, sequences_data_samples):
                for train_idx in train_frame_idxs:
                    train_frames.append(sequence_inputs[train_idx])
                    det_train_data_sample = PoseDataSample()
                    train_mask = sequence_data_samples.gt_instances.frame_id == train_idx
                    det_train_data_sample.gt_instance_labels = sequence_data_samples.gt_instances[train_mask]
                    det_train_data_samples.append(det_train_data_sample)

            train_frames = torch.cat(train_frames).view(num_train_frames * B, C, W, H)

            no_grad_frame_idxs = random_frame_idxs[num_train_frames:]
            no_grad_frames = torch.cat([i[no_grad_frame_idxs] for i in sequences_inputs]).view((T - num_train_frames) * B, C, W, H)

            log_vars, det_train_outputs = self.runner.detector.train_step(
                dict(inputs=train_frames, data_samples=det_train_data_samples),
                optim_wrapper=self.runner.det_optim_wrapper,
                return_preds=True,
            )
            det_train_outputs = det_train_outputs["detections"]
        else:
            det_train_outputs = torch.tensor([])
            go_back_frame_idxs = torch.arange(T)
            no_grad_frames = torch.cat([i[no_grad_frame_idxs] for i in inputs]).view(T * B, C, W, H)

        det_pred_outputs = self.runner.detector.test_step(dict(inputs=no_grad_frames, data_samples=[PoseDataSample() for _ in range(no_grad_frames.shape[0])]))

        scores = []
        objectness = []
        bboxes = []
        kpts = []
        kpt_vis = []
        features = []
        priors = []
        data_samples = []
        for i, pred in enumerate([scores, objectness, bboxes, kpts, kpt_vis, features, priors]):
            pred.append(det_train_outputs[i])
            pred.append(det_pred_outputs[i])

        scores = torch.cat(scores)[go_back_frame_idxs]
        objectness = torch.cat(objectness)[go_back_frame_idxs]
        bboxes = torch.cat(bboxes)[go_back_frame_idxs]
        kpts = torch.cat(kpts)[go_back_frame_idxs]
        kpt_vis = torch.cat(kpt_vis)[go_back_frame_idxs]
        features = torch.cat(features)[go_back_frame_idxs]
        priors = torch.cat(priors)[go_back_frame_idxs]

        # TODO Truver une manière de passer feature maps et data_samples...

        with torch.no_grad():
            preds = postprocess_one_stage_detections(
                self.post_processor,
                scores,
                objectness,
                bboxes,
                kpts,
                kpt_vis,
                features,
                priors,
                data_samples,
            )

        # TODO ground truth tracking

        # TODO entrainer MART
        outputs = self.runner.model.train_step(outputs, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook("after_train_iter", batch_idx=idx, data_batch=data_batch, outputs=outputs)
        self._iter += 1
