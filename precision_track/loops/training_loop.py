from typing import Sequence, Union, Dict, Optional, List, Tuple
import itertools
import torch
from torch.optim.lr_scheduler import MultiStepLR
from collections import OrderedDict, defaultdict
from torch.utils.data import DataLoader
from mmengine.runner import EpochBasedTrainLoop
from mmengine.structures import InstanceData
from mmengine.config import Config
from precision_track.registry import LOOPS
from precision_track.utils import PoseDataSample, postprocess_one_stage_detections, unflatten_predictions
from precision_track.models.postprocessing.steps import PostProcessingSteps
from precision_track.tracking import OnlineGroundTruth
from precision_track.apis import AssociationStep


@LOOPS.register_module()
class OnlineTrainLoop(EpochBasedTrainLoop):
    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        post_processor: Config,
        max_epochs: int,
        metafile: str,
        val_begin: int = 1,
        val_interval: int = 1,
        dynamic_intervals: Optional[List[Tuple[int, int]]] = None,
    ) -> None:

        assert hasattr(runner, "detector")

        self.post_processor = PostProcessingSteps(post_processor)
        self.tracker = AssociationStep(
            tracking_algorithm=dict(
                type="OnlineGroundTruth",
            ),
            metafile=metafile,
            motion_algorithm=dict(
                type="DynamicKalmanFilterPytorch",
            ),
            memory_length=2,
        )

        super().__init__(
            runner,
            dataloader,
            max_epochs,
            val_begin,
            val_interval,
            dynamic_intervals,
        )

        self._cls_num_list = None

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
        T, C, H, W = sequences_inputs[0].shape

        num_train_frames = min(self.runner.train_frames, T)

        # Select the training and no_grad frames:
        random_frame_idxs = torch.randperm(T)  # use these random indices to select the frames.

        go_back_frame_idxs = torch.argsort(random_frame_idxs)  # use these indices to go back to the original order.

        train_frame_idxs = random_frame_idxs[:num_train_frames]
        train_frames = []
        det_train_data_samples = []

        no_grad_frame_idxs = random_frame_idxs[num_train_frames:]
        no_grad_frames = []
        no_grad_data_samples = []

        splitted_seq_data_samples = [OrderedDict() for _ in sequences_data_samples]

        for i, (sequence_inputs, sequence_data_samples) in enumerate(zip(sequences_inputs, sequences_data_samples)):

            if hasattr(sequence_data_samples, "action_label_counter") and self._cls_num_list is None:
                self._cls_num_list = sequence_data_samples.action_label_counter

            seq_train_frames = []
            seq_train_ds = []

            for train_idx in train_frame_idxs:
                seq_train_frames.append(sequence_inputs[train_idx])
                ds = self._load_data_sample(sequence_data_samples, train_idx)
                seq_train_ds.append(ds)
                splitted_seq_data_samples[i][train_idx] = ds

            if num_train_frames > 0:  # TODO temp Hack
                train_frames.append(torch.cat(seq_train_frames).view(num_train_frames, C, H, W))
            det_train_data_samples.append(seq_train_ds)

            seq_no_grad_frames = []
            seq_no_grad_ds = []
            for no_grad_idx in no_grad_frame_idxs:
                seq_no_grad_frames.append(sequence_inputs[no_grad_idx])
                ds = self._load_data_sample(sequence_data_samples, no_grad_idx)
                seq_no_grad_ds.append(ds)
                splitted_seq_data_samples[i][no_grad_idx] = ds

            no_grad_frames.append(torch.cat(seq_no_grad_frames).view(T - num_train_frames, C, H, W))
            no_grad_data_samples.append(seq_no_grad_ds)

        if num_train_frames > 0:  # TODO temp Hack
            train_frames = torch.cat(train_frames).view(num_train_frames * B, C, H, W)
        no_grad_frames = torch.cat(no_grad_frames).view((T - num_train_frames) * B, C, H, W)

        if num_train_frames > 0:
            self.runner.detector.train()
            flatten_ds = list(itertools.chain.from_iterable(det_train_data_samples))
            with self.runner.optim_wrapper.optim_context(self.runner.detector):
                data = self.runner.detector.data_preprocessor(dict(inputs=train_frames, data_samples=flatten_ds), True)
                out = self.runner.detector.loss(**data, return_preds=True)

            det_train_outputs = list(out.pop("detections", None))
            det_losses, det_log_vars = self.runner.detector.parse_losses(out)
            det_log_vars.pop("loss", None)
        else:
            det_train_outputs = None
            det_losses = None
            det_log_vars = dict()

        self.runner.detector.eval()
        with torch.inference_mode():
            data = self.runner.detector.data_preprocessor(
                dict(inputs=no_grad_frames, data_samples=list(itertools.chain.from_iterable(no_grad_data_samples))), False
            )
            det_pred_outputs = self.runner.detector.test_step(data)

        scores = []
        objectness = []
        bboxes = []
        kpts = []
        kpt_vis = []
        features = []
        priors = []
        for i, pred in enumerate([scores, objectness, bboxes, kpts, kpt_vis, features]):
            if det_train_outputs is not None:
                pred.append(det_train_outputs[i])
            pred.append(det_pred_outputs[i])

        feature_maps = []
        # mlvl_features_pred = unflatten_predictions(det_pred_outputs[5], [(80, 80), (40, 40), (20, 20)])
        # if num_train_frames > 0:
        #     mlvl_features_train = unflatten_predictions(det_train_outputs[5], [(80, 80), (40, 40), (20, 20)])
        # else:
        #     mlvl_features_train = [] * len(mlvl_features_pred)

        # for mlvl_f_t, mlvl_f_p, mlvl_s in zip(mlvl_features_train, mlvl_features_pred, [(80, 80), (40, 40), (20, 20)]):
        #     if isinstance(mlvl_f_t, torch.Tensor):
        #         mlvl_f = torch.cat((mlvl_f_t, mlvl_f_p))
        #     else:
        #         mlvl_f = mlvl_f_p
        #     feature_maps.append(mlvl_f.view(B, T, -1, mlvl_s[0], mlvl_s[1])[:, go_back_frame_idxs, :, :])

        priors = det_pred_outputs[-2]
        P = priors.shape[1]

        features = torch.cat(features).view(B, T, P, -1)[:, go_back_frame_idxs, :, :]

        # Detach to not update the detection head's weight using the sequence losses.
        scores = torch.cat(scores).view(B, T, P, -1)[:, go_back_frame_idxs, :, :].detach()
        objectness = torch.cat(objectness).view(B, T, P, -1)[:, go_back_frame_idxs, :, :].detach()
        bboxes = torch.cat(bboxes).view(B, T, P, 4)[:, go_back_frame_idxs, :, :].detach()
        kpts = torch.cat(kpts).view(B, T, P, -1, 2)[:, go_back_frame_idxs, :, :, :].detach()
        kpt_vis = torch.cat(kpt_vis).view(B, T, P, -1)[:, go_back_frame_idxs, :, :].detach()

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
            seq_data_sample = PoseDataSample()
            seq_anns = []
            for post_processed_frame, ds in zip(post_processed_seq, splitted_seq_data_samples[i]):
                post_processed_frame["img_id"] = ds.frame_id
                post_processed_frame["ori_shape"] = ds.img_shape
                frame_anns = self.tracker(post_processed_frame)

                nb_gts = len(post_processed_frame["gt_instances"])
                if ds.frame_id == 29 and nb_gts == 0:
                    stop = True

                seq_inputs = self.runner.model.data_preprocessor(
                    dict(inputs=feature_maps, data_samples=frame_anns),
                    return_buffers=ds.frame_id == 29,  # TODO hack, rework preprocessor....
                )
                data_samples = seq_inputs.pop("data_samples")
                seq_inputs.pop("block_ids")
                frame_anns = data_samples["gt_instances"]
                seq_anns.append(frame_anns)
            for seq_input in seq_inputs:
                inputs[seq_input].append(seq_inputs[seq_input])
                input_dims[seq_input] = tuple(seq_inputs[seq_input].shape[1:])
            seq_data_sample.gt_instances = InstanceData.cat(seq_anns)
            splitted_seq_data_samples[i] = seq_data_sample
        for in_ in inputs:
            in_dims = input_dims[in_]
            inputs[in_] = torch.cat(inputs[in_]).view(-1, *in_dims)

        with self.runner.optim_wrapper.optim_context(self.runner.model):

            # sanity_check = defaultdict(int)
            # block_actions = inputs["actions"].tolist()
            # for ac in block_actions:
            #     for a in ac:
            #         sanity_check[a[0]] += 1
            # from precision_track.outputs.display import display_class_balance

            # display_class_balance(sanity_check)
            batch_actions, counts = torch.unique(inputs["actions"], return_counts=True)
            batch_cls_num_list = []
            for cls_idx in range(len(self._cls_num_list)):
                batch_cls_idx = torch.where(cls_idx == batch_actions)[0]
                if batch_cls_idx.numel() > 0:
                    batch_cls_num_list.append(counts[batch_cls_idx].item())
                else:
                    batch_cls_num_list.append(0)
            outputs = self.runner.model.loss(**inputs, data_samples=splitted_seq_data_samples, cls_num_list=batch_cls_num_list)
        losses, log_vars = self.runner.model.parse_losses(outputs)

        if isinstance(det_losses, torch.Tensor):
            losses = losses + det_losses
        outputs = log_vars | det_log_vars
        outputs["loss"] = losses

        self.runner.optim_wrapper.update_params(losses)
        self.runner.call_hook("after_train_iter", batch_idx=idx, data_batch=data_batch, outputs=outputs)
        self._iter += 1

    @staticmethod
    def _load_data_sample(data_samples, idx):
        out = PoseDataSample(metainfo=data_samples.metainfo)
        mask = data_samples.gt_instances.frame_id == idx
        out.frame_id = idx.item()
        out.gt_instance_labels = data_samples.gt_instances[mask]
        out.ori_shape = None
        out.img_id = None
        out.img_path = None
        out.id = None
        # out.img_shape = data_samples.img_shape
        return out


@LOOPS.register_module()
class FeatureExtractionTrainLoop(EpochBasedTrainLoop):
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook("before_train_iter", batch_idx=idx, data_batch=data_batch)
        outputs = self.runner.model.train_step(data_batch, optim_wrapper=self.runner.optim_wrapper, return_preds=True)[0]

        self.runner.call_hook("after_train_iter", batch_idx=idx, data_batch=data_batch, outputs=outputs)
        self._iter += 1
