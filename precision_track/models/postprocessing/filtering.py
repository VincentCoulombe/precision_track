from collections.abc import Iterable
from typing import Optional

import numpy as np
import torch
from mmengine.registry import MODELS

from precision_track.utils import biou_batch, parse_pose_metainfo

from .base import BaseActionPostProcessor, BasePostProcessor


@MODELS.register_module()
class LowScoresFiltering(BasePostProcessor):

    def __init__(
        self,
        score_thr: Optional[float] = 0.1,
        max_topk: Optional[int] = 1000,
        filter_features: Optional[bool] = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        assert 0 <= score_thr <= 1
        self.score_thr = score_thr
        assert 0 <= max_topk and isinstance(max_topk, int)
        self.max_topk = max_topk
        assert isinstance(filter_features, bool)
        self.filter_features = filter_features

    def forward(self, bboxes, scores, keypoints, kpt_vis, labels, features, kept_idxs):
        valid_mask = scores > self.score_thr
        scores = scores[valid_mask]
        valid_idxs = torch.nonzero(valid_mask)
        num_topk = min(self.max_topk, valid_idxs.size(0))

        scores, idxs = scores.sort(descending=True)
        scores = scores[:num_topk]
        keep_idxs, _ = valid_idxs[idxs[:num_topk]].unbind(dim=1)

        bboxes = bboxes[keep_idxs]
        kpt_vis = kpt_vis[keep_idxs]
        keypoints = keypoints[keep_idxs]
        if self.filter_features:
            features = features[keep_idxs]
        labels = labels[keep_idxs]
        kept_idxs = kept_idxs[keep_idxs]

        return bboxes, scores, keypoints, kpt_vis, labels, features, kept_idxs


@MODELS.register_module()
class NearnessBasedActionFiltering(BaseActionPostProcessor):
    def __init__(self, concerned_labels: list, fallback_label: int, metainfo: str):
        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        self.actions = metainfo.get("actions", [])
        assert isinstance(concerned_labels, Iterable)
        for cl in concerned_labels:
            assert cl in self.actions, f"{cl} not in {self.actions.tolist()}."
        assert fallback_label in self.actions, f"{fallback_label} not in {self.actions.tolist()}."
        self.concerned_labels = np.array(concerned_labels)
        self.fallback_label = fallback_label
        super().__init__()

    def forward(self, data_sample: dict):
        actions = data_sample["pred_track_instances"]["actions"]
        bboxes = data_sample["pred_track_instances"]["bboxes"]

        action_mask = np.isin(actions, self.concerned_labels)
        relevant_bboxes = bboxes[action_mask]

        if relevant_bboxes.size > 0:
            bious = biou_batch(relevant_bboxes, bboxes.copy(), 0.25)
            isolated = (bious.sum(1) - bious.max(1)) == 0
            if any(isolated):
                update_indices = np.flatnonzero(action_mask)[isolated]
                actions[update_indices] = self.fallback_label
                data_sample["pred_track_instances"]["actions"] = actions

        return data_sample
