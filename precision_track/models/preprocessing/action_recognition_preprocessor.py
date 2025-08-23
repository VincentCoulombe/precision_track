from typing import Optional

import numpy as np
import torch
from mmengine.model import BaseDataPreprocessor

from precision_track.registry import MODELS
from precision_track.utils import get_device, kpts_to_poses, parse_pose_metainfo


class ActionTube:
    def __init__(self, block_size, embedding_dim, device=None, dtype=torch.float32):
        self.T = block_size
        self.E = embedding_dim
        self.device = device
        self.dtype = dtype
        self._buffer = torch.zeros((self.T, self.E), device=self.device, dtype=self.dtype)
        self.n_filled = 0

    def append(self, x):
        self._buffer = torch.roll(self._buffer, shifts=-1, dims=0)
        self._buffer[-1] = x
        self.n_filled += 1

    def is_valid(self):
        return self.n_filled > self.T

    def to_tensor(self):
        return self._buffer


@MODELS.register_module()
class ActionRecognitionPreprocessor(BaseDataPreprocessor):
    METAINFO_KEYS = ["skeleton_links"]
    SUPPORTED_MODES = ["predict", "loss", "sequence"]

    def __init__(
        self,
        metainfo: str,
        block_size: int,
        kpts_conf_thr: Optional[float] = 0.5,
        mode: Optional[str] = "predict",
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        assert 0 <= kpts_conf_thr < 1
        assert block_size >= 0

        self._device = device or get_device()
        self.kpts_conf_thr = kpts_conf_thr

        self.skeleton_links = parse_pose_metainfo(dict(from_file=metainfo)).get("skeleton_links")
        self.skeleton_sources = torch.tensor([s for s, _ in self.skeleton_links], device=self._device)
        self.skeleton_targets = torch.tensor([t for _, t in self.skeleton_links], device=self._device)

        self._mode = mode
        self.block_size = block_size

        self.feature_action_tube = dict()
        self.pose_action_tube = dict()
        self.velocity_action_tube = dict()

        self.instance_ids_to_action_tube_idx = dict()

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        assert mode in self.SUPPORTED_MODES
        self._mode = mode

    def predict(self, data: dict, *args, **kwargs) -> dict:
        data_samples = data["data_samples"]
        assert len(data_samples) == 1, "The action recognition pipeline does not support batched inference."
        data_sample = data_samples[0]

        data_sample["pred_track_instances"]["valid_action_recognition_context"] = np.zeros_like(
            data_sample["pred_track_instances"]["instances_id"], dtype=bool
        )

        features = data_sample["pred_track_instances"]["features"].to(self._device)
        vels = torch.from_numpy(data_sample["pred_track_instances"]["velocities"]).to(self._device).view(features.shape[0], -1)
        poses, scale = kpts_to_poses(
            torch.from_numpy(data_sample["pred_track_instances"]["keypoints"]).to(self._device),
            torch.from_numpy(data_sample["pred_track_instances"]["keypoint_scores"]).to(self._device),
            self.skeleton_sources,
            self.skeleton_targets,
            self.kpts_conf_thr,
            normalize=True,
        )
        vels /= scale
        poses = poses.view(features.shape[0], -1)

        feat_shape = features.shape[1]
        vel_shape = vels.shape[1]
        pose_shape = poses.shape[1]

        frame_id = data_sample["img_id"]

        dynamics = []
        skeletons = []
        feats = []
        active_ids = set()
        corrections = data_sample.get("corrected_instances_id", dict())

        for i, inst_id in enumerate(data_sample["pred_track_instances"]["instances_id"]):
            inst_id = int(inst_id)
            active_ids.add(inst_id)
            context_ok = np.zeros(3, dtype=bool)

            was_corrected = False
            cls_corr = corrections.get(data_sample["pred_track_instances"]["classes"][i], list())
            for corr_a, corr_b in cls_corr:
                if inst_id == corr_a or inst_id == corr_b:
                    was_corrected = True

            for j, (buf, new_input, dim, tube_dict) in enumerate(
                zip(
                    [skeletons, feats, dynamics],
                    [poses, features, vels],
                    [pose_shape, feat_shape, vel_shape],
                    [self.pose_action_tube, self.feature_action_tube, self.velocity_action_tube],
                )
            ):
                if inst_id not in tube_dict:
                    tube_dict[inst_id] = {"tube": ActionTube(self.block_size, dim, device=self._device), "last_seen": frame_id}
                else:
                    tube_dict[inst_id]["last_seen"] = frame_id

                if was_corrected:
                    tube_dict[inst_id]["tube"].n_filled = 0

                tube_dict[inst_id]["tube"].append(new_input[i])
                context_ok[j] = tube_dict[inst_id]["tube"].is_valid()
                buf.append(tube_dict[inst_id]["tube"].to_tensor())

            if np.all(context_ok):
                data_sample["pred_track_instances"]["valid_action_recognition_context"][i] = True

        all_ids = set(self.feature_action_tube.keys())
        inactive_ids = all_ids - active_ids
        dead_ids = set()

        for inst_id in inactive_ids:
            for tube_dict, dim in zip(
                [self.pose_action_tube, self.feature_action_tube, self.velocity_action_tube],
                [pose_shape, feat_shape, vel_shape],
            ):
                if inst_id not in tube_dict:
                    continue
                if frame_id - tube_dict[inst_id]["last_seen"] > self.block_size:
                    dead_ids.add(inst_id)
                else:
                    tube_dict[inst_id]["tube"].append(torch.zeros(dim, device=self._device))

        for inst_id in dead_ids:
            self.pose_action_tube.pop(inst_id, None)
            self.feature_action_tube.pop(inst_id, None)
            self.velocity_action_tube.pop(inst_id, None)

        return dict(
            features=torch.stack(feats, dim=0),
            poses=torch.stack(skeletons, dim=0),
            dynamics=torch.stack(dynamics, dim=0),
            data_samples=data["data_samples"],
        )

    def forward(
        self,
        data: dict,
        *args,
        **kwargs,
    ) -> dict:
        if self.mode == "loss":
            return self.loss(data, *args, **kwargs)
        elif self.mode == "sequence":
            return self.sequence(data, *args, **kwargs)
        return self.predict(data, *args, **kwargs)

    def loss(
        self,
        data: dict,
        *args,
        **kwargs,
    ) -> dict:
        dynamics = []
        skeletons = []
        labels = []
        positives = []
        scales = []
        for data_sample in data["data_samples"]:
            dynamics.append(data_sample.pred_track_instances.dynamics.view(self.block_size, 2))
            skeleton, scale = kpts_to_poses(
                data_sample.pred_track_instances.kpts.to(self.device),
                data_sample.pred_track_instances.kpt_vis.to(self.device),
                self.skeleton_sources,
                self.skeleton_targets,
                self.kpts_conf_thr,
                normalize=True,
            )
            skeletons.append(skeleton.view(self.block_size, -1))
            labels.append(data_sample.gt_instance_labels.action_labels)
            positives.append(data_sample.gt_instance_labels.positives)
            scales.append(scale)

        dynamics = torch.stack(dynamics, dim=0).to(self.device) / torch.stack(scales, dim=0).to(self.device)
        skeletons = torch.stack(skeletons, dim=0)
        features = torch.stack(data["inputs"], dim=0).to(self.device)
        labels = torch.stack(labels, dim=0).to(self.device).to(torch.int64)
        positives = torch.stack(positives, dim=0).to(self.device).to(torch.int64)

        return dict(features=features, poses=skeletons, dynamics=dynamics, labels=labels, binary_labels=positives)

    def sequence(
        self,
        data: dict,
        *args,
        **kwargs,
    ) -> dict:
        data_samples = data["data_samples"]
        assert len(data_samples) == 1, "The action recognition pipeline do not support batches during sequencing."
        data_sample = data_samples[0]

        dynamics = data_sample.pred_track_instances.dynamics
        N, T, _ = dynamics.shape
        assert T == self.block_size
        assert N == len(data_sample.pred_track_instances.kpts) == len(data_sample.pred_track_instances.kpt_vis)
        dynamics = dynamics.view(N, self.block_size, 2).to(self.device)
        inst_skeletons = []
        for i, (inst_pose, inst_vis) in enumerate(zip(data_sample.pred_track_instances.kpts, data_sample.pred_track_instances.kpt_vis)):
            skeleton, scale = kpts_to_poses(
                inst_pose.to(self.device),
                inst_vis.to(self.device),
                self.skeleton_sources,
                self.skeleton_targets,
                self.kpts_conf_thr,
                normalize=True,
            )
            inst_skeletons.append(skeleton.view(self.block_size, -1))
            dynamics[i, ...] /= scale
        skeletons = torch.stack(inst_skeletons, dim=0)
        labels = data_sample.gt_instance_labels.action_labels.view(N).to(self.device)
        features = data["inputs"][0].view(N, T, -1).to(self.device)

        return dict(features=features, poses=skeletons, dynamics=dynamics, labels=labels)
