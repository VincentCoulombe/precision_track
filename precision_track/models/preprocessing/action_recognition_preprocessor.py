from typing import Optional, Dict, Union, List, Union
import heapq
from itertools import chain
from collections import defaultdict
from mmengine.structures import InstanceData
import numpy as np
import torch
from mmengine.model import BaseDataPreprocessor

from precision_track.registry import MODELS
from precision_track.utils import get_device, kpts_to_poses, parse_pose_metainfo, BaseVelocityEncoder


class IdIndexMap:
    def __init__(self, max_size: int):
        self.id2idx: Dict[str, int] = {}
        self.free: List[int] = []
        self.next_idx: int = 0
        self.max_size = max_size

    def has(self, id_: str) -> bool:
        return id_ in self.id2idx

    def get(self, id_: str) -> Optional[int]:
        return self.id2idx.get(id_, None)

    def acquire(self, id_: str) -> int:
        """Map id_ to a stable index, reusing freed slots when possible."""
        if id_ in self.id2idx:
            return self.id2idx[id_]
        if self.free:
            idx = heapq.heappop(self.free)
        else:
            if self.next_idx >= self.max_size:
                raise RuntimeError("IdIndexMap at capacity")
            idx = self.next_idx
            self.next_idx += 1
        self.id2idx[id_] = idx
        return idx

    def release(self, id_: str) -> None:
        """Delete id_ and free its index for reuse."""
        idx = self.id2idx.pop(id_, None)
        if idx is not None:
            heapq.heappush(self.free, idx)

    def active_indices(self) -> List[int]:
        """Unordered active indices; use for indexing or mask building."""
        return list(self.id2idx.values())

    def available_count(self) -> int:
        """How many indices can be reused without growing."""
        return len(self.free)

    def size(self) -> int:
        """Number of active ids."""
        return len(self.id2idx)

    def capacity(self) -> int:
        """Highest issued length if you need a fixed-size mask/tensor."""
        return self.next_idx


@MODELS.register_module()
class ActionRecognitionPreprocessor(BaseDataPreprocessor):
    def __init__(
        self,
        metainfo: str,
        embd_size: int,
        block_size: int,
        max_size: Optional[int] = 100,
        kpts_conf_thr: Optional[float] = 0.5,
        device: Optional[str] = None,
        with_vels: Optional[bool] = False,
        velocity_encoder: Optional[dict] = None,
        with_kpts: Optional[bool] = False,
        with_kpt_vels: Optional[bool] = False,
        with_actions: Optional[bool] = False,
        ignore_index: Optional[int] = -100,
        **kwargs,
    ):
        super().__init__()
        assert 0 <= kpts_conf_thr < 1
        assert block_size > 0
        assert max_size > 0
        assert embd_size > 0

        self._device = device or get_device()
        self.kpts_conf_thr = kpts_conf_thr

        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        self.skeleton_links = metainfo.get("skeleton_links")
        self._init_graph()

        self.cls_map = {cls_: lbl for lbl, cls_ in enumerate(metainfo.get("classes"))}

        self._block_size = int(block_size)
        self._max_size = int(max_size)
        self._embd_size = int(embd_size)
        self._ignore_index = int(ignore_index)

        self._with_vels = with_vels
        self._with_kpts = with_kpts
        self._with_kpt_vels = with_kpt_vels
        self._with_actions = with_actions

        self.block_features = torch.zeros((self._max_size, self._block_size, self._embd_size), dtype=torch.float32, device=self._device).contiguous()

        self.vel_encoder = None
        if velocity_encoder is not None:
            self.vel_encoder = MODELS.build(velocity_encoder)

        self.block_vels = None
        if with_vels:
            assert isinstance(self.vel_encoder, BaseVelocityEncoder), f"Expected a velocity encoder, but got {self.vel_encoder}."
            self.block_vels = torch.zeros((self._max_size, self._block_size, self.vel_encoder.nb_dim), dtype=torch.float32, device=self._device).contiguous()

        self.block_poses = None
        if with_kpts:
            self.block_poses = torch.zeros(
                (self._max_size, self._block_size, len(self.skeleton_links) * 2), dtype=torch.float32, device=self._device
            ).contiguous()

        self.block_pose_vels = None
        if with_kpt_vels:
            self.block_pose_vels = torch.zeros(
                (self._max_size, self._block_size, len(self.skeleton_links) * 2), dtype=torch.float32, device=self._device
            ).contiguous()

        self.block_actions = None
        if with_actions:
            self.block_actions = torch.zeros((self._max_size, self._block_size, 1), dtype=torch.long, device=self._device).contiguous()

        self.ids2idx = IdIndexMap(max_size=self._max_size)
        self.time_no_see = defaultdict(int)
        self._head = torch.zeros(self._max_size, dtype=torch.long, device=self._device)

        self.roll = torch.zeros_like(self._head, dtype=bool)
        self.consecutive_hits = torch.zeros_like(self._head)

    def _init_graph(self):
        self.skeleton_sources = torch.as_tensor([s for s, _ in self.skeleton_links], device=self._device)
        self.skeleton_targets = torch.as_tensor([t for _, t in self.skeleton_links], device=self._device)

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        data_sample = data["data_samples"]
        if isinstance(data_sample, list):
            data_sample = data_sample[0]
        pred_track_instances = data_sample.get("pred_track_instances", None)
        if isinstance(pred_track_instances, InstanceData):
            pred_track_instances = pred_track_instances.to_dict()
        assert isinstance(pred_track_instances, dict), f"pred_track_instances must be a dict, not {pred_track_instances}."

        frame_id = int(data_sample.get("img_id", -1))
        assert frame_id >= 0, f"frame_id must be > 0, not: {frame_id}."

        ids = pred_track_instances["instances_id"]
        if isinstance(ids, np.ndarray):
            ids = torch.from_numpy(ids)
        ids = ids.to(self._device)

        labels = pred_track_instances["labels"]
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        unique_ids = [f"{label.item()}-{id_.item()}" for label, id_ in zip(labels, ids)]
        running_idxs = torch.zeros_like(ids, dtype=torch.long, device=self._device)
        hidden_idxs = self._register_ids(running_idxs, unique_ids)

        corrections = data_sample.get("corrected_instances_id", dict())
        for k, v in corrections.items():
            if v:
                lbl = self.cls_map[k]
                flatten_v = list(chain.from_iterable(v))
                for corrected_id in flatten_v:
                    corrected_idx = self.ids2idx.get(f"{lbl}-{corrected_id}")
                    if corrected_idx is not None:
                        self.consecutive_hits[corrected_idx] = 0
        self.roll = self.consecutive_hits[running_idxs] > self._block_size

        features = pred_track_instances["features"]
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        assert self._device != features.device, f"Expected tensors to be on {self._device}, got {features.device} instead."

        vels = pred_track_instances["velocities"]
        if isinstance(vels, np.ndarray):
            vels = torch.from_numpy(vels).to(self._device)
        vels = vels.view(features.shape[0], 2).to(self._device).to(features.dtype)

        pred_track_instances["valid_action_recognition_context"] = self.roll
        scale = 1.0

        out = {}
        self._ring_write(self.block_features, running_idxs, features)
        self._ring_write(self.block_features, hidden_idxs)

        if self._with_kpts:
            kpts = pred_track_instances["keypoints"]
            if isinstance(kpts, np.ndarray):
                kpts = torch.from_numpy(kpts)
            kpts = kpts.to(self._device)

            kpt_vis = pred_track_instances["keypoint_scores"]
            if isinstance(kpt_vis, np.ndarray):
                kpt_vis = torch.from_numpy(kpt_vis)
            kpt_vis = kpt_vis.to(self._device)

            if kpts.numel() > 0:
                poses, scale = kpts_to_poses(
                    kpts,
                    kpt_vis,
                    self.skeleton_sources,
                    self.skeleton_targets,
                    self.kpts_conf_thr,
                    normalize=True,
                )
            else:
                poses = kpts
            poses = poses.to(self.block_poses.dtype).view(features.shape[0], len(self.skeleton_links) * 2)
            self._ring_write(self.block_poses, running_idxs, poses)
            self._ring_write(self.block_poses, hidden_idxs)

        if self._with_vels:
            if scale is not None:
                vels = vels / scale
            vels = vels.to(self.block_vels.dtype).view(features.shape[0], 2)
            if self.vel_encoder is not None:
                vels = self.vel_encoder(vels)
            self._ring_write(self.block_vels, running_idxs, vels)
            self._ring_write(self.block_vels, hidden_idxs)

        if self._with_actions:
            actions = pred_track_instances["actions"]
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions)
            actions = actions.to(self._device).to(self.block_actions.dtype).view(features.shape[0], 1)
            self._ring_write(self.block_actions, running_idxs, actions)
            self._ring_write(self.block_actions, hidden_idxs, default_data_value=self._ignore_index)

        out["features"] = self.materialize(self.block_features, running_idxs)
        if self._with_kpts:
            out["poses"] = self.materialize(self.block_poses, running_idxs)
        if self._with_vels:
            out["dynamics"] = self.materialize(self.block_vels, running_idxs)
        if self._with_actions:
            out["actions"] = self.materialize(self.block_actions, running_idxs)
        out["data_samples"] = data["data_samples"]

        self._update_head(running_idxs)
        self._update_head(hidden_idxs)
        self._delete_ids()

        return out

    def materialize(self, block: torch.Tensor, rows: torch.Tensor):
        """Re-arrange the block chronologically, starting from the registered rolling position."""
        nb_insts = rows.size(0)
        idx_time = torch.arange(self._block_size, device=block.device).unsqueeze(0).expand(nb_insts, -1)

        if self.roll.any():
            start = self._head.index_select(0, rows)
            rolled_idx = (start.unsqueeze(1) + idx_time + 1) % self._block_size
            idx_time = torch.where(self.roll.unsqueeze(1), rolled_idx, idx_time)

        return block[rows.unsqueeze(1), idx_time, :]

    def _ring_write(self, block: torch.Tensor, rows: torch.Tensor, data: Optional[torch.Tensor] = None, default_data_value: Union[float, int] = 0.0):
        """Insert new data at the correct rolling position."""
        assert self._head.device == block.device
        pos = self._head.index_select(0, rows)
        if isinstance(data, torch.Tensor) and data.numel() > 0:
            B = data.shape[0]
            assert rows.dtype == torch.long
            assert rows.device == block.device, "rows/block device mismatch"
            assert B == rows.shape[0], "batch mismatch"
            block[rows, pos, ...] = data.view(B, -1)
        else:
            block[rows, pos, ...] = default_data_value
        return block

    def _update_head(self, rows: torch.Tensor):
        """Update the rolling position relative to its curent position and the block size."""
        assert rows.dtype == torch.long
        assert self._head.device == rows.device
        pos = self._head.index_select(0, rows)
        self._head.index_copy_(0, rows, (pos + 1) % self._block_size)

    def _register_ids(self, idxs: torch.Tensor, new_ids: List[str]):
        assert idxs.shape[0] == len(new_ids)

        seen_ids = set()
        for i, new_id in enumerate(new_ids):
            seen_ids.add(new_id)
            idx = self.ids2idx.get(new_id)
            if idx is None:
                idx = self.ids2idx.acquire(new_id)
            idxs[i] = idx
            self.consecutive_hits[idx] += 1

        for k in self.time_no_see:
            self.time_no_see[k] += 1

        for k in seen_ids:
            self.time_no_see[k] = 0

        all_ids = set(self.ids2idx.id2idx.keys())
        hidden_ids = all_ids - seen_ids
        hidden_idxs = torch.zeros(len(hidden_ids), dtype=idxs.dtype, device=idxs.device)
        for i, hidden_id in enumerate(hidden_ids):
            idx = self.ids2idx.get(hidden_id)
            hidden_idxs[i] = idx
            self.consecutive_hits[idx] += 1
        return hidden_idxs

    def _delete_ids(self):
        expired = [k for k, t in self.time_no_see.items() if t >= self._block_size]
        if not expired:
            return

        for id_ in expired:
            idx = self.ids2idx.get(id_)
            if idx is not None:
                self._head[idx] = 0
                self.consecutive_hits[idx] = 0
            self.ids2idx.release(id_)
            self.time_no_see.pop(id_, None)


@MODELS.register_module()
class ActionRecognitionTrainingPreprocessor(BaseDataPreprocessor):
    METAINFO_KEYS = ["skeleton_links"]
    SUPPORTED_MODES = ["loss", "sequence", "pretrain"]

    def __init__(
        self,
        metainfo: str,
        block_size: int,
        velocity_encoder: Optional[dict] = None,
        kpts_conf_thr: Optional[float] = 0.5,
        mode: Optional[str] = "loss",
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

        self.vel_encoder = None
        if velocity_encoder is not None:
            self.vel_encoder = MODELS.build(velocity_encoder)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        assert mode in self.SUPPORTED_MODES
        self._mode = mode

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
        elif self.mode == "pretrain":
            return self.pretrain(data, *args, **kwargs)

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
            dynamics.append(data_sample.pred_track_instances.velocities.view(self.block_size, 2))
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
        if self.vel_encoder is not None:
            dynamics = self.vel_encoder(dynamics)
        skeletons = torch.stack(skeletons, dim=0)
        features = torch.stack(data["inputs"], dim=0).to(self.device)
        labels = torch.stack(labels, dim=0).to(self.device).to(torch.int64)
        positives = torch.stack(positives, dim=0).to(self.device).to(torch.int64)

        return dict(features=features, poses=skeletons, dynamics=dynamics, labels=labels, binary_labels=positives)

    def pretrain(
        self,
        data: dict,
        *args,
        **kwargs,
    ) -> dict:
        dynamics = []
        skeletons = []
        scales = []
        for data_sample in data["data_samples"]:
            dynamics.append(data_sample.pred_track_instances.velocities.view(self.block_size, 2))
            skeleton, scale = kpts_to_poses(
                data_sample.pred_track_instances.kpts.to(self.device),
                data_sample.pred_track_instances.kpt_vis.to(self.device),
                self.skeleton_sources,
                self.skeleton_targets,
                self.kpts_conf_thr,
                normalize=True,
            )
            skeletons.append(skeleton.view(self.block_size, -1))
            scales.append(scale)

        dynamics = torch.stack(dynamics, dim=0).to(self.device) / torch.stack(scales, dim=0).to(self.device)
        if self.vel_encoder is not None:
            dynamics = self.vel_encoder(dynamics)
        skeletons = torch.stack(skeletons, dim=0)
        features = torch.stack(data["inputs"], dim=0).to(self.device)

        return dict(features=features, poses=skeletons, dynamics=dynamics)

    def sequence(
        self,
        data: dict,
        *args,
        **kwargs,
    ) -> dict:
        data_samples = data["data_samples"]
        assert len(data_samples) == 1, "The action recognition pipeline do not support batches during sequencing."
        data_sample = data_samples[0]

        dynamics = data_sample.pred_track_instances.velocities
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
        if self.vel_encoder is not None:
            dynamics = self.vel_encoder(dynamics)
        skeletons = torch.stack(inst_skeletons, dim=0)
        labels = data_sample.gt_instance_labels.action_labels.view(N).to(self.device)
        features = data["inputs"][0].view(N, T, -1).to(self.device)

        return dict(features=features, poses=skeletons, dynamics=dynamics, labels=labels)


@MODELS.register_module()
class GroupActionRecognitionTrainingPreprocessor(BaseDataPreprocessor):
    def __init__(
        self,
        metainfo: str,
        block_size: int,
        velocity_encoder: Optional[dict] = None,
        kpts_conf_thr: Optional[float] = 0.5,
        with_distance_priors: bool = True,
        with_vel_coherence: bool = True,
        with_vel_approach: bool = True,
        with_orientation_priors: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        assert 0 <= kpts_conf_thr < 1
        assert block_size > 0

        self._device = device or get_device()
        self.kpts_conf_thr = kpts_conf_thr
        self.block_size = block_size
        self.with_distance_priors = with_distance_priors
        self.with_vel_coherence = with_vel_coherence
        self.with_vel_approach = with_vel_approach
        self.with_orientation_priors = with_orientation_priors

        pose_meta = parse_pose_metainfo(dict(from_file=metainfo))
        self.skeleton_links = pose_meta.get("skeleton_links")
        self.skeleton_sources = torch.tensor([s for s, _ in self.skeleton_links], device=self._device)
        self.skeleton_targets = torch.tensor([t for _, t in self.skeleton_links], device=self._device)

        if self.with_orientation_priors:
            kpt_name_to_idx = pose_meta["keypoint_name2id"]
            named_pairs = pose_meta.get("orientation_keypoint_pairs", [])
            assert (
                named_pairs
            ), "If you want to train a group action recognition model with orientation priors, you must populate the 'orientation_keypoint_pairs' in your metadata file. Please refer to documentation."
            self.orientation_keypoint_pairs = [(kpt_name_to_idx[ant], kpt_name_to_idx[post]) for ant, post in named_pairs]

        self.vel_encoder = None
        if velocity_encoder is not None:
            self.vel_encoder = MODELS.build(velocity_encoder)

    def _get_orientation(
        self,
        kpts_t: torch.Tensor,
        kpt_vis_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = kpts_t.shape[0]
        body_orientation = torch.zeros(N, 2, device=kpts_t.device)
        assigned = torch.zeros(N, dtype=torch.bool, device=kpts_t.device)

        for ant_idx, post_idx in self.orientation_keypoint_pairs:
            if assigned.all():
                break
            valid = (kpt_vis_t[:, ant_idx] > 0) & (kpt_vis_t[:, post_idx] > 0) & ~assigned
            if valid.any():
                vec = kpts_t[valid, ant_idx] - kpts_t[valid, post_idx]
                body_orientation[valid] = vec / (vec.norm(dim=-1, keepdim=True) + 1e-6)
                assigned |= valid

        return body_orientation, assigned

    def forward(self, data: dict, *args, **kwargs) -> dict:
        return self.loss(data)

    def loss(self, data: dict, *args, **kwargs) -> dict:
        all_features, all_poses, all_dynamics, all_node_labels = [], [], [], []
        all_classification_matrices, all_relationship_matrices, all_masks = [], [], []
        all_distance_priors, all_coherence, all_vel_approach = [], [], []
        all_orientations_alignment, all_gate_ij = [], []

        for sample_idx, data_sample in enumerate(data["data_samples"]):
            inputs = data["inputs"][sample_idx].to(self._device)
            kpts = data_sample.pred_track_instances.kpts.to(self._device)
            kpt_vis = data_sample.pred_track_instances.kpt_vis.to(self._device)
            velocities = data_sample.pred_track_instances.velocities.to(self._device)
            group_matrix = data_sample.gt_instance_labels.group_action_matrix.to(self._device)
            valid_mask = data_sample.pred_track_instances.valid_mask.to(self._device)
            N = inputs.shape[0]

            inst_poses, inst_scales = [], []
            for i in range(N):
                skeleton, scale = kpts_to_poses(
                    kpts[i],
                    kpt_vis[i],
                    self.skeleton_sources,
                    self.skeleton_targets,
                    self.kpts_conf_thr,
                    normalize=True,
                )
                inst_poses.append(skeleton.view(self.block_size, -1))
                inst_scales.append(scale)

            poses = torch.stack(inst_poses, dim=0)
            scales = torch.stack(inst_scales, dim=0).view(N, self.block_size, 1)

            dynamics = velocities / scales
            if self.vel_encoder is not None:
                dynamics = self.vel_encoder(dynamics)

            # Relationship priors (based on the last timestep only)
            kpts_t = kpts[:, -1, :, :]
            vis_t = (kpt_vis[:, -1, :] > self.kpts_conf_thr).float()
            centroids = (kpts_t * vis_t.unsqueeze(-1)).sum(-2) / vis_t.sum(-1, keepdim=True).clamp(min=1)
            vels_t = velocities[:, -1, :]

            if self.with_distance_priors:
                dist = (centroids.unsqueeze(0) - centroids.unsqueeze(1)).norm(dim=-1)
                scale_t = scales[:, -1, 0]
                sij = (scale_t.unsqueeze(0) + scale_t.unsqueeze(1)) / 2
                d_norm = dist / (sij + 1e-6)
                all_distance_priors.append(2.0 / (1.0 + d_norm) - 1.0)  # Weight it back to the [-1, 1] range

            if self.with_vel_coherence or self.with_vel_approach:
                v_norm = vels_t / (vels_t.norm(dim=-1, keepdim=True) + 1e-6)

            if self.with_vel_coherence:
                all_coherence.append((v_norm.unsqueeze(0) * v_norm.unsqueeze(1)).sum(-1))

            if self.with_vel_approach or self.with_orientation_priors:
                centroid_edges = centroids.unsqueeze(0) - centroids.unsqueeze(1)
                disp_norm = centroid_edges / (centroid_edges.norm(dim=-1, keepdim=True) + 1e-6)

            if self.with_vel_approach:
                all_vel_approach.append((v_norm.unsqueeze(1) * disp_norm).sum(-1))

            if self.with_orientation_priors:
                body_orientation, assigned = self._get_orientation(kpts_t, vis_t)

                # 1 = j in front of i, 0= lateral, -1=behind
                orientations_alignment = (body_orientation.unsqueeze(1) * disp_norm).sum(-1)

                # Keep track of where the orientation could not be estimated (to differenciate with lateral)
                gate = assigned.float()
                gate_ij = gate.unsqueeze(1) * gate.unsqueeze(0)

                all_orientations_alignment.append(orientations_alignment * gate_ij)
                all_gate_ij.append(gate_ij)

            all_features.append(inputs)
            all_poses.append(poses)
            all_dynamics.append(dynamics)
            all_node_labels.append(data_sample.gt_instance_labels.node_labels.to(self._device))
            all_classification_matrices.append(group_matrix)
            all_relationship_matrices.append((group_matrix >= 0).to(torch.int64))
            all_masks.append(valid_mask)

        labels = torch.stack(all_classification_matrices, dim=0)
        out = dict(
            features=torch.stack(all_features, dim=0),
            poses=torch.stack(all_poses, dim=0),
            dynamics=torch.stack(all_dynamics, dim=0),
            node_labels=torch.stack(all_node_labels, dim=0),
            labels=labels,
            binary_labels=(labels >= 0).long(),
            valid_mask=torch.stack(all_masks, dim=0),
        )
        if self.with_distance_priors:
            out["distance_priors"] = torch.stack(all_distance_priors, dim=0)
        if self.with_vel_coherence:
            out["vel_coherence"] = torch.stack(all_coherence, dim=0)
        if self.with_vel_approach:
            out["vel_approach"] = torch.stack(all_vel_approach, dim=0)
        if self.with_orientation_priors:
            out["orientations_alignment"] = torch.stack(all_orientations_alignment, dim=0)
            out["orientations_valid"] = torch.stack(all_gate_ij, dim=0)
        return out


@MODELS.register_module()
class RelationshipDetectionBaselinePreprocessor(BaseDataPreprocessor):
    def __init__(self, actions_of_interest: list, non_blocking=False):
        super().__init__(non_blocking)
        self.actions_of_interest = torch.tensor(actions_of_interest, dtype=torch.int64)

    def forward(self, data: dict, *args, **kwargs) -> dict:
        return self.loss(data)

    def loss(self, data: dict, *args, **kwargs) -> dict:
        out = dict(
            query_idxs=[],
            bboxes=[],
        )
        for data_sample in data["data_samples"]:
            bboxes = data_sample.pred_track_instances.bboxes[:, -1, :]
            actions = data_sample.gt_instance_labels.node_labels[:, -1]
            query_idx = torch.where(actions == self.actions_of_interest)[0]
            out["query_idxs"].append(query_idx)
            out["bboxes"].append(bboxes)
        return dict(data_samples=out)
