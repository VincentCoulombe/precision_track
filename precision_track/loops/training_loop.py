from typing import Sequence
import torch
from mmengine.runner import EpochBasedTrainLoop
from mmengine.structures import InstanceData
from precision_track.registry import LOOPS
from precision_track.utils import PoseDataSample


@LOOPS.register_module()
class TrackingEpochBasedTrainLoop(EpochBasedTrainLoop):

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook("before_train_iter", batch_idx=idx, data_batch=data_batch)

        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.

        inputs = data_batch["inputs"]
        data_samples = data_batch["data_samples"]

        # TODO One-Stage vs Two-Stage Training
        B = len(inputs)
        T, C, W, H = inputs[0].shape

        if self.runner.one_stage:
            num_train_frames = min(self.runner.train_frames, T)
            if num_train_frames == 0:
                self.runner.logger.warn("One-Stage training is activated, but either train_frames is not set or your are training on empty sequences.")

            # Select the training and no_grad frames:
            random_frame_idxs = torch.randperm(T)  # use these random indices to select the frames.

            go_back_frame_idxs = torch.argsort(random_frame_idxs)  # use these indices to go back to the original order.

            # Split random_frame_idxs into training and no_grad frame indices:
            train_frame_idxs = random_frame_idxs[:num_train_frames]
            train_frames = []
            det_train_data_samples = []
            for input_i, data_sample_i in zip(inputs, data_samples):
                train_frames.append(input_i[train_frame_idxs])
                det_train_data_sample = PoseDataSample()
                det_train_data_sample.gt_instance_labels = data_sample_i.gt_instances[train_frame_idxs]
                det_train_data_samples.append(det_train_data_sample)

            train_frames = torch.cat(train_frames).view(num_train_frames * B, C, W, H)

            no_grad_frame_idxs = random_frame_idxs[num_train_frames:]
            no_grad_frames = torch.cat([i[no_grad_frame_idxs] for i in inputs]).view((T - num_train_frames) * B, C, W, H)

            det_train_outputs = self.runner.detector.train_step(
                dict(inputs=train_frames, data_samples=det_train_data_samples), optim_wrapper=self.runner.optim_wrapper
            )
        else:
            det_train_outputs = torch.tensor([])
            go_back_frame_idxs = torch.arange(T)
            no_grad_frames = torch.cat([i[no_grad_frame_idxs] for i in inputs]).view(T * B, C, W, H)

        det_pred_outputs = self.runner.detector.predict(dict(inputs=no_grad_frames, data_samples=data_samples))
        # TODO concat predictions + training_preds et replacer sequences selon idx

        # TODO ground truth tracking

        # TODO entrainer MART
        outputs = self.runner.model.train_step(outputs, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook("after_train_iter", batch_idx=idx, data_batch=data_batch, outputs=outputs)
        self._iter += 1
