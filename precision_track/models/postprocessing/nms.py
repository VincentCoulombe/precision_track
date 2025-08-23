from typing import Optional

import torch
import torch.nn.functional as F
from mmengine.registry import MODELS
from torchvision.ops import nms

from .base import BasePostProcessor


@MODELS.register_module()
class NMSPostProcessor(BasePostProcessor):

    def __init__(
        self,
        score_thr: Optional[float] = 0.1,
        nms_thr: Optional[float] = 0.65,
        nms_pre: Optional[int] = 1000,
        pool_thr: Optional[float] = 0.9,
    ):
        super().__init__()
        assert 0 <= score_thr <= 1
        assert 0 <= nms_thr <= 1
        assert 0 <= pool_thr <= 1
        assert 0 <= nms_pre and isinstance(nms_pre, int)
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.nms_pre = nms_pre
        self.pool_thr = pool_thr

    def forward(self, bboxes, scores, keypoints, kpt_vis, labels, features, keep_idxs):
        valid_mask = scores > self.score_thr
        scores = scores[valid_mask]
        valid_idxs = torch.nonzero(valid_mask)
        num_topk = min(self.nms_pre, valid_idxs.size(0))

        scores, idxs = scores.sort(descending=True)
        scores = scores[:num_topk]
        keep_idxs, _ = valid_idxs[idxs[:num_topk]].unbind(dim=1)

        bboxes = bboxes[keep_idxs]
        features = F.normalize(features[keep_idxs], p=2, dim=-1, eps=1e-12)

        if bboxes.numel() > 0:
            features_nms = features
            scores_nms = scores
            if self.nms_thr < 1.0:
                keep_idxs_nms = nms(bboxes, scores, self.nms_thr)
                if keep_idxs_nms.numel() == 0:
                    return bboxes, scores_nms, keypoints, kpt_vis, labels, features
                keep_idxs = keep_idxs[keep_idxs_nms]
                bboxes = bboxes[keep_idxs_nms]
                scores_nms = scores[keep_idxs_nms]
                features_nms = features[keep_idxs_nms]
            if self.pool_thr < 1.0:
                pooling_costs = features_nms @ features.T
                pooling_mask = pooling_costs >= self.pool_thr
                for i in range(features_nms.shape[0]):
                    mask = pooling_mask[i]
                    if mask.sum() == 0:
                        continue
                    feats_i = features[mask]
                    scores_i = scores[mask]
                    weights = F.softmax(scores_i, dim=0)
                    pooled = (feats_i * weights[:, None]).sum(dim=0)
                    features_nms[i] = pooled
            features = features_nms
            scores = scores_nms

        labels = labels[keep_idxs]
        kpt_vis = kpt_vis[keep_idxs]
        keypoints = keypoints[keep_idxs]

        return bboxes, scores, keypoints, kpt_vis, labels, features, keep_idxs
