from typing import Optional, Union
from collections import OrderedDict
import numpy as np
import torch
from mmengine.model import BaseDataPreprocessor

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
        self.skeleton_sources = torch.tensor([s for s, _ in self.skeleton_links], device=self._device)
        self.skeleton_targets = torch.tensor([t for _, t in self.skeleton_links], device=self._device)

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

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        instances = data["instances"]

        instances.valid_action_recognition_context = torch.zeros_like(instances.instances_id, dtype=bool)

        features = instances.features
        vels = instances.velocities.view(features.shape[0], -1)

        poses, scale = kpts_to_poses(
            instances.keypoints,
            instances.keypoint_scores,
            self.skeleton_sources,
            self.skeleton_targets,
            self.kpts_conf_thr,
            normalize=True,
        )
        vels /= scale
        poses = poses.view(features.shape[0], -1)

        frame_id = instances.frame_id[0]

        new_data = [poses, features, vels]
        new_shapes = [poses.shape[1], features.shape[1], vels.shape[1]]
        buffers = [[], [], []]

        if self._with_kpt_vels:
            kpt_vels = instances.keypoint_velocities.view(features.shape[0], -1) / scale
            new_data.append(kpt_vels)
            new_shapes.append(kpt_vels.shape[1])
            buffers.append([])

        if self._with_actions:
            actions = instances.actions.view(features.shape[0], 1)
            new_data.append(actions)
            new_shapes.append(1)
            buffers.append([])

        active_ids = set()
        corrections = hasattr(instances, "corrected_instances_id") or dict()

        for i, inst_id in enumerate(instances.instances_id):
            inst_id = int(inst_id)
            active_ids.add(inst_id)
            context_ok = torch.zeros(len(buffers), dtype=bool)

            was_corrected = False
            cls_corr = corrections.get(instances.labels[i], list())
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
                instances.valid_action_recognition_context[i] = True

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

        out = dict(
            poses=torch.stack(buffers[0], dim=0),
            features=torch.stack(buffers[1], dim=0),
            dynamics=torch.stack(buffers[2], dim=0),
            instances=instances,
            block_ids=all_ids,
        )
        if self._with_kpt_vels:
            out["kpt_vels"] = torch.stack(buffers[3], dim=0)
            if self._with_actions:
                out["actions"] = torch.stack(buffers[4], dim=0)
        elif self._with_actions:
            out["actions"] = torch.stack(buffers[3], dim=0)

        return out
