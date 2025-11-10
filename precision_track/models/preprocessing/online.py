import heapq
from typing import Optional, Dict, Union, List, Union
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from mmengine.model import BaseDataPreprocessor
from mmengine.structures import InstanceData

from precision_track.registry import MODELS
from precision_track.utils import get_device, kpts_to_poses, parse_pose_metainfo

from .action_recognition_preprocessor import ActionTube


@MODELS.register_module()
class OnlinePreprocessor(BaseDataPreprocessor):
    def __init__(self, non_blocking=False, *args, **kwargs):
        super().__init__(non_blocking)

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        pass


@MODELS.register_module()
class FPVOnlinePreprocessor(OnlinePreprocessor):
    def __init__(
        self,
        metainfo: str,
        block_size: int,
        kpts_conf_thr: Optional[float] = 0.5,
        device: Optional[str] = None,
        with_kpt_vels: Optional[bool] = False,
        with_actions: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__()
        assert 0 <= kpts_conf_thr < 1
        assert block_size >= 0

        self._device = device or get_device()
        self.kpts_conf_thr = kpts_conf_thr

        self.skeleton_links = parse_pose_metainfo(dict(from_file=metainfo)).get("skeleton_links")
        self._init_graph()

        self.block_size = block_size

        self._with_kpt_vels = with_kpt_vels
        self._with_actions = with_actions

        self.action_tubes = [OrderedDict(), OrderedDict(), OrderedDict()]
        self.null_values = [0] * 3
        self.dtypes = [torch.float32, torch.float32, torch.float32]
        if with_kpt_vels:
            self.action_tubes += [OrderedDict()]
            self.null_values += [0]
            self.dtypes += [torch.float32]
        if with_actions:
            self.action_tubes += [OrderedDict()]
            self.null_values += [-100]
            self.dtypes += [torch.long]

    def _init_graph(self):
        self.skeleton_sources = torch.tensor([s for s, _ in self.skeleton_links], device=self._device)
        self.skeleton_targets = torch.tensor([t for _, t in self.skeleton_links], device=self._device)

    def forward(self, data: dict, return_buffers: bool = False) -> Union[dict, list]:
        data_samples = data["data_samples"]
        pred_track_instances = data_samples.get("pred_track_instances", None)
        frame_id = int(data_samples.get("img_id", -1))

        assert isinstance(pred_track_instances, dict), f"pred_track_instances must be a dict, not {pred_track_instances}."
        assert frame_id >= 0, f"frame_id must be > 0, not: {frame_id}."

        ids = pred_track_instances["instances_id"]
        labels = pred_track_instances["labels"]

        features = pred_track_instances["features"]
        if self._device == features.device:
            self._device = features.device
            self._init_graph()

        pred_track_instances["valid_action_recognition_context"] = torch.zeros_like(ids, dtype=bool, device=self._device)
        vels = pred_track_instances["velocities"].view(features.shape[0], 2).to(self._device).to(features.dtype)
        poses, scale = kpts_to_poses(
            pred_track_instances["keypoints"].to(self._device),
            pred_track_instances["keypoint_scores"].to(self._device),
            self.skeleton_sources,
            self.skeleton_targets,
            self.kpts_conf_thr,
            normalize=True,
        )
        vels /= scale
        poses = poses.view(features.shape[0], self.skeleton_sources.shape[0] * 2)

        new_data = [poses, features, vels]
        new_shapes = [poses.shape[1], features.shape[1], vels.shape[1]]
        buffers = [[], [], []]

        if self._with_kpt_vels:
            kpt_vels = pred_track_instances["keypoint_velocities"].view(features.shape[0], self.skeleton_sources.shape[0]) / scale
            new_data.append(kpt_vels)
            new_shapes.append(kpt_vels.shape[1])
            buffers.append([])

        if self._with_actions:
            actions = pred_track_instances["actions"].view(features.shape[0], 1)
            new_data.append(actions)
            new_shapes.append(1)
            buffers.append([])

        active_ids = set()
        corrections = pred_track_instances.get("corrected_instances_id", dict())

        for i, inst_id in enumerate(ids):
            inst_id = int(inst_id)
            active_ids.add(inst_id)
            context_ok = torch.zeros(len(buffers), dtype=bool)

            was_corrected = False
            cls_corr = corrections.get(labels[i], list())
            for corr_a, corr_b in cls_corr:
                if inst_id == corr_a or inst_id == corr_b:
                    was_corrected = True

            for j, (buf, new_input, dim, tube_dict, nv, dtype) in enumerate(
                zip(buffers, new_data, new_shapes, self.action_tubes, self.null_values, self.dtypes)
            ):
                if inst_id not in tube_dict:
                    tube_dict[inst_id] = {
                        "tube": ActionTube(
                            self.block_size,
                            dim,
                            device=self._device,
                            null_value=nv,
                            dtype=dtype,
                        ),
                        "last_seen": frame_id,
                    }
                else:
                    tube_dict[inst_id]["last_seen"] = frame_id

                if was_corrected:
                    tube_dict[inst_id]["tube"].n_filled = 0

                tube_dict[inst_id]["tube"].append(new_input[i])
                context_ok[j] = tube_dict[inst_id]["tube"].is_valid()
                buf.append(tube_dict[inst_id]["tube"].to_tensor())

            if torch.all(context_ok):
                pred_track_instances["valid_action_recognition_context"][i] = True

        all_ids = set(self.action_tubes[0].keys())
        inactive_ids = all_ids - active_ids
        dead_ids = set()

        for inst_id in inactive_ids:
            for tube_dict, dim in zip(self.action_tubes, new_shapes):
                if inst_id not in tube_dict:
                    continue
                if frame_id - tube_dict[inst_id]["last_seen"] > self.block_size:
                    dead_ids.add(inst_id)
                else:
                    tube_dict[inst_id]["tube"].append(torch.zeros(dim, device=self._device))

        for inst_id in dead_ids:
            for tube in self.action_tubes:
                tube.pop(inst_id, None)

        if not return_buffers:
            return dict(
                data_samples=data_samples,
                block_ids=all_ids,
            )
        out = dict(
            poses=torch.stack(buffers[0], dim=0),
            features=torch.stack(buffers[1], dim=0),
            dynamics=torch.stack(buffers[2], dim=0),
            data_samples=data_samples,
            block_ids=all_ids,
        )
        if self._with_kpt_vels:
            out["kpt_vels"] = torch.stack(buffers[3], dim=0)
            if self._with_actions:
                out["actions"] = torch.stack(buffers[4], dim=0)
        elif self._with_actions:
            out["actions"] = torch.stack(buffers[3], dim=0)

        return out


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
class TestPreprocessor(OnlinePreprocessor):
    def __init__(
        self,
        metainfo: str,
        embd_size: int,
        block_size: int,
        max_size: Optional[int] = 100,
        kpts_conf_thr: Optional[float] = 0.5,
        device: Optional[str] = None,
        with_vels: Optional[bool] = False,
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

        self.skeleton_links = parse_pose_metainfo(dict(from_file=metainfo)).get("skeleton_links")
        self._init_graph()

        self._block_size = int(block_size)
        self._max_size = int(max_size)
        self._embd_size = int(embd_size)
        self._ignore_index = int(ignore_index)

        self._with_vels = with_vels
        self._with_kpts = with_kpts
        self._with_kpt_vels = with_kpt_vels
        self._with_actions = with_actions

        self.block_features = torch.zeros((self._max_size, self._block_size, self._embd_size), dtype=torch.float32, device=self._device).contiguous()

        self.block_vels = None
        if with_vels:
            self.block_vels = torch.zeros((self._max_size, self._block_size, 2), dtype=torch.float32, device=self._device).contiguous()

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
        data_samples = data["data_samples"]
        if isinstance(data_samples, list):
            data_samples = data_samples[0]
        pred_track_instances = data_samples.get("pred_track_instances", None)
        frame_id = int(data_samples.get("img_id", -1))

        assert isinstance(pred_track_instances, dict), f"pred_track_instances must be a dict, not {pred_track_instances}."
        assert frame_id >= 0, f"frame_id must be > 0, not: {frame_id}."

        ids = pred_track_instances["instances_id"]
        if isinstance(ids, np.ndarray):
            ids = torch.from_numpy(ids)
        ids = ids.to(self._device)

        labels = pred_track_instances["labels"]
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        ids = ids.to(self._device)

        unique_ids = [f"{label.item()}-{id_.item()}" for label, id_ in zip(labels, ids)]
        running_idxs = torch.zeros_like(ids, dtype=torch.long, device=self._device)
        hidden_idxs = self._register_ids(running_idxs, unique_ids)

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
        if data is not None:
            assert rows.dtype == torch.long
            assert rows.device == block.device, "rows/block device mismatch"
            assert data.shape[0] == rows.shape[0], "batch mismatch"
            block[rows, pos, ...] = data
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
