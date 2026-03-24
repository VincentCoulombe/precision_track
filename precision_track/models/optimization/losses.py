# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import random
from functools import partial
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import distances, losses, miners

from precision_track.registry import MODELS
from precision_track.utils import bbox_overlaps, parse_pose_metainfo


@MODELS.register_module()
class BCELoss(nn.Module):
    """Binary Cross Entropy loss.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of the loss. Default: 1.0.
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            before output. Defaults to False.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.0, reduction="mean", use_sigmoid=False):
        super().__init__()

        assert reduction in ("mean", "sum", "none"), f"the argument " f"`reduction` should be either 'mean', 'sum' or 'none', " f"but got {reduction}"

        self.reduction = reduction
        self.use_sigmoid = use_sigmoid
        criterion = F.binary_cross_entropy if use_sigmoid else F.binary_cross_entropy_with_logits
        self.criterion = partial(criterion, reduction="none")
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """

        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output, target)
            if target_weight.dim() == 1:
                target_weight = target_weight[:, None]
            loss = loss * target_weight
        else:
            loss = self.criterion(output, target)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss * self.loss_weight


@MODELS.register_module()
class IoULoss(nn.Module):
    """Binary Cross Entropy loss.

    Args:
        reduction (str): Options are "none", "mean" and "sum".
        eps (float): Epsilon to avoid log(0).
        loss_weight (float): Weight of the loss. Default: 1.0.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self, reduction="mean", mode="log", eps: float = 1e-16, loss_weight=1.0):
        super().__init__()

        assert reduction in ("mean", "sum", "none"), f"the argument " f"`reduction` should be either 'mean', 'sum' or 'none', " f"but got {reduction}"

        assert mode in ("linear", "square", "log"), f"the argument " f"`reduction` should be either 'linear', 'square' or " f"'log', but got {mode}"

        self.reduction = reduction
        self.criterion = partial(F.cross_entropy, reduction="none")
        self.loss_weight = loss_weight
        self.mode = mode
        self.eps = eps

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
        """
        ious = bbox_overlaps(output, target, is_aligned=True).clamp(min=self.eps)

        if self.mode == "linear":
            loss = 1 - ious
        elif self.mode == "square":
            loss = 1 - ious.pow(2)
        elif self.mode == "log":
            loss = -ious.log()
        else:
            raise NotImplementedError

        if target_weight is not None:
            for i in range(loss.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = loss * target_weight

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss * self.loss_weight


@MODELS.register_module()
class L1Loss(nn.Module):
    """L1Loss loss."""

    def __init__(self, reduction="mean", use_target_weight=False, loss_weight=1.0):
        super().__init__()

        assert reduction in ("mean", "sum", "none"), f"the argument " f"`reduction` should be either 'mean', 'sum' or 'none', " f"but got {reduction}"

        self.criterion = partial(F.l1_loss, reduction=reduction)
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output regression.
            target (torch.Tensor[N, K, 2]): Target regression.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            for _ in range(target.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = self.criterion(output * target_weight, target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@MODELS.register_module()
class OKSLoss(nn.Module):
    """A PyTorch implementation of the Object Keypoint Similarity (OKS) loss as
    described in the paper "YOLO-Pose: Enhancing YOLO for Multi Person Pose
    Estimation Using Object Keypoint Similarity Loss" by Debapriya et al.
    (2022).

    The OKS loss is used for keypoint-based object recognition and consists
    of a measure of the similarity between predicted and ground truth
    keypoint locations, adjusted by the size of the object in the image.

    The loss function takes as input the predicted keypoint locations, the
    ground truth keypoint locations, a mask indicating which keypoints are
    valid, and bounding boxes for the objects.

    Args:
        metainfo (Optional[str]): Path to a JSON file containing information
            about the dataset's annotations.
        reduction (str): Options are "none", "mean" and "sum".
        eps (float): Epsilon to avoid log(0).
        loss_weight (float): Weight of the loss. Default: 1.0.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'linear'
        norm_target_weight (bool): whether to normalize the target weight
            with number of visible keypoints. Defaults to False.
    """

    def __init__(self, metainfo: Optional[str] = None, reduction="mean", mode="linear", eps=1e-8, norm_target_weight=False, loss_weight=1.0):
        super().__init__()

        assert reduction in ("mean", "sum", "none"), f"the argument " f"`reduction` should be either 'mean', 'sum' or 'none', " f"but got {reduction}"

        assert mode in ("linear", "square", "log"), f"the argument " f"`reduction` should be either 'linear', 'square' or " f"'log', but got {mode}"

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.mode = mode
        self.norm_target_weight = norm_target_weight
        self.eps = eps

        if metainfo is not None:
            metainfo = parse_pose_metainfo(dict(from_file=metainfo))
            sigmas = metainfo.get("sigmas", None)
            if sigmas is not None:
                self.register_buffer("sigmas", torch.as_tensor(sigmas))

    def forward(self, output, target, target_weight=None, areas=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints coordinates.
            target (torch.Tensor[N, K, 2]): Target keypoints coordinates..
            target_weight (torch.Tensor[N, K]): Loss weight for each keypoint.
            areas (torch.Tensor[N]): Instance size which is adopted as
                normalization factor.
        """
        dist = torch.norm(output - target, dim=-1)
        if areas is not None:
            dist = dist / areas.pow(0.5).clip(min=self.eps).unsqueeze(-1)
        if hasattr(self, "sigmas"):
            sigmas = self.sigmas.reshape(*((1,) * (dist.ndim - 1)), -1)
            dist = dist / (sigmas * 2)

        oks = torch.exp(-dist.pow(2) / 2)

        if target_weight is not None:
            if self.norm_target_weight:
                target_weight = target_weight / target_weight.sum(dim=-1, keepdims=True).clip(min=self.eps)
            else:
                target_weight = target_weight / target_weight.size(-1)
            oks = oks * target_weight
        oks = oks.sum(dim=-1)

        if self.mode == "linear":
            loss = 1 - oks
        elif self.mode == "square":
            loss = 1 - oks.pow(2)
        elif self.mode == "log":
            loss = -oks.log()
        else:
            raise NotImplementedError()

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss * self.loss_weight


@MODELS.register_module()
def pairwise_circleloss(
    neg_similarities: torch.Tensor,
    pos_similarities: torch.Tensor,
    targets: torch.Tensor,
    margin: Optional[float] = 0.25,
    gamma: Optional[float] = 32,
) -> torch.Tensor:

    if targets.ndim == 1:
        N = neg_similarities.size(0)

        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t())
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t())
    elif targets.ndim == 2:
        is_pos = targets
        is_neg = 1 - targets
    else:
        raise ValueError(f"targets souhld either be 1D or 2D, not: {targets.ndim}D.")

    is_pos = is_pos.float()
    is_neg = is_neg.float()

    s_p = pos_similarities * is_pos
    s_n = neg_similarities * is_neg

    alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.0)
    alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.0)
    delta_p = 1 - margin
    delta_n = margin

    logit_p = -gamma * alpha_p * (s_p - delta_p) + (-99999999.0) * (1 - is_pos)
    logit_n = gamma * alpha_n * (s_n - delta_n) + (-99999999.0) * (1 - is_neg)

    loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

    return loss


def positive_circleloss(
    pos_similarities: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.25,
    gamma: float = 32.0,
) -> torch.Tensor:
    """
    Circle-loss variant that tightens *only* positive pairs.
    Works when the mini-batch contains a single identity.
    """
    if targets.ndim == 1:
        N = targets.size(0)
        is_pos = targets.view(N, 1).eq(targets.view(1, N))
    elif targets.ndim == 2:
        is_pos = targets.bool()
    else:
        raise ValueError("targets must be 1-D or 2-D")

    is_pos = is_pos.float()

    s_p = pos_similarities * is_pos
    alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, 0.0)
    delta_p = 1.0 - margin

    logit_p = -gamma * alpha_p * (s_p - delta_p) + (-1e8) * (1 - is_pos)

    loss = F.softplus(torch.logsumexp(logit_p, dim=1)).mean()
    return loss


@MODELS.register_module()
class CircleLoss(nn.Module):

    def __init__(self, margin: int = 0.25, gamma: int = 32, loss_weight=1.0):
        super().__init__()
        self.loss = losses.CircleLoss(
            m=margin,
            gamma=gamma,
        )
        self.loss_weight = loss_weight

    def forward(self, embeddings, y):
        return self.loss(embeddings, y) * self.loss_weight


@MODELS.register_module()
class ArcFaceLoss(nn.Module):

    def __init__(self, num_classes: int, embedding_size: int, margin: int = 0.5, scale: int = 64, loss_weight=1.0):
        super().__init__()
        self.loss = losses.ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_size,
            margin=57.3 * margin,
            scale=scale,
        )
        self.loss_weight = loss_weight

    def forward(self, embeddings, y):
        return self.loss(embeddings, y) * self.loss_weight


@MODELS.register_module()
class TripletLoss(nn.Module):

    def __init__(
        self,
        margin: int = 0.2,
        pos_strategy: str = "easy",
        neg_strategy: str = "semihard",
        distance: str = "cosine",
        loss_weight=1.0,
    ):
        super().__init__()
        if distance == "cosine":
            distance = distances.CosineSimilarity()
        elif distance == "l2":
            distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
        elif distance == "l2_squared":
            distance = distances.LpDistance(normalize_embeddings=True, p=2, power=2)
        else:
            raise ValueError(f"Invalid distance: {distance}")

        self.loss = losses.TripletMarginLoss(distance=distance, margin=margin)
        self.miner = miners.BatchEasyHardMiner(
            pos_strategy=pos_strategy,
            neg_strategy=neg_strategy,
            allowed_pos_range=None,
            allowed_neg_range=None,
            distance=distance,
        )
        self.loss_weight = loss_weight

    def forward(self, embeddings, y):
        if embeddings.numel() == 0:
            return 0
        indices_tuple = self.miner(embeddings, y)
        return self.loss(embeddings, y, indices_tuple) * self.loss_weight


@MODELS.register_module()
class LDAMWithDRW(nn.Module):
    """
    LDAM (Cao et al., NeurIPS'19) with optional DRW (Cui et al., CVPR'19),
    """

    def __init__(
        self,
        metainfo: str,
        max_m: float = 0.5,
        s: float = 30.0,
        use_drw: bool = True,
        drw_start_epoch: int = 0,
        beta: float = 0.9999,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        self.n_class = len(metainfo.get("actions", []))
        self.max_m = float(max_m)
        self.s = float(s)
        self.use_drw = bool(use_drw)
        self.drw_start_epoch = int(drw_start_epoch)
        self.beta = float(beta)
        self.label_smoothing = float(label_smoothing)
        self.ignore_index = int(ignore_index)

        self._cached_m_list = torch.tensor([])
        self._cached_cb_weight = torch.tensor([])
        self._cached_counts_key = None

    @torch.no_grad()
    def _maybe_update_cache(self, cls_num_list: Sequence[int], device, dtype):
        key = (tuple(map(int, cls_num_list)), device, dtype)
        if key == self._cached_counts_key:
            return

        counts = torch.tensor(cls_num_list, device=device, dtype=torch.float32).clamp_min(1.0)

        m_list = 1.0 / counts.sqrt().sqrt()
        m_list = m_list * (self.max_m / m_list.max())
        self._cached_m_list = m_list.to(device=device, dtype=dtype)

        eff_num = 1.0 - torch.pow(torch.tensor(self.beta, device=device, dtype=torch.float32), counts)
        cb_w = (1.0 - self.beta) / eff_num
        cb_w = cb_w / cb_w.mean()
        self._cached_cb_weight = cb_w.to(device=device, dtype=dtype)

        self._cached_counts_key = key

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        epoch: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"logits must be [B, C], got {tuple(logits.shape)}")
        if target.ndim != 1:
            raise ValueError(f"target must be [B], got {tuple(target.shape)}")

        B, C = logits.shape
        device, dtype = logits.device, logits.dtype

        cls_num_list = torch.zeros(self.n_class, dtype=torch.long)
        cls_idx, cls_counts = torch.unique(target, return_counts=True)
        acknowledged = cls_idx != self.ignore_index
        cls_num_list[cls_idx[acknowledged].long().detach().cpu()] = cls_counts[acknowledged].long().detach().cpu()
        cls_num_list = cls_num_list.tolist()

        if not cls_num_list:
            return F.cross_entropy(
                logits,
                target,
                reduction="mean",
                label_smoothing=self.label_smoothing,
                ignore_index=self.ignore_index,
            )

        self._maybe_update_cache(cls_num_list, device, dtype)

        valid = target != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0

        t_valid = target[valid]
        if not (t_valid.dtype == torch.long and t_valid.numel() > 0):
            raise ValueError("target must be torch.long indices (with possible ignore_index).")
        if t_valid.min() < 0 or t_valid.max() >= self._cached_m_list.numel():
            raise ValueError("target out of range for number of classes.")

        adjusted = logits.clone()
        margins = torch.zeros(B, device=device, dtype=dtype)
        margins_valid = self._cached_m_list.gather(0, t_valid.to(torch.long))
        margins[valid] = margins_valid
        rows = torch.arange(B, device=device)[valid]
        adjusted[rows, t_valid] -= margins_valid
        adjusted = self.s * adjusted

        weight = None
        if self.use_drw and (epoch is not None) and (epoch >= self.drw_start_epoch):
            weight = self._cached_cb_weight

        return F.cross_entropy(
            adjusted,
            target,
            weight=weight,
            reduction="mean",
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )


@MODELS.register_module()
class MSELoss(nn.Module):
    """Mean Squared Error loss.

    Args:
        reduction (str): Options are "none", "mean" and "sum".
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, reduction="mean", use_target_weight=False, loss_weight=1.0):
        super().__init__()

        assert reduction in ("mean", "sum", "none"), f"the argument " f"`reduction` should be either 'mean', 'sum' or 'none', " f"but got {reduction}"

        self.criterion = partial(F.mse_loss, reduction=reduction)
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, ...]): Output regression.
            target (torch.Tensor[N, K, ...]): Target regression.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            for _ in range(target.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = self.criterion(output * target_weight, target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


def focal_loss_multiclass(
    logits,
    targets,
    gamma=2.0,
    alpha=None,
    label_smoothing=0.0,
    loss_weight=1.0,
    ignore_index=-100,
):
    N, C = logits.shape
    log_pt_full = F.log_softmax(logits, dim=1)
    log_pt = log_pt_full.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()

    focal_weight = (1.0 - pt) ** gamma

    if alpha is not None:
        if isinstance(alpha, (float, int)):
            alpha_t = torch.full((N,), alpha, device=logits.device)
        else:
            alpha = alpha.to(logits.device)
            alpha_t = alpha[targets]
        focal_weight = focal_weight * alpha_t

    mask = targets != ignore_index

    if label_smoothing > 0.0:
        smooth = torch.full_like(log_pt_full, label_smoothing / (C - 1))
        smooth.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
        loss = -(smooth * log_pt_full).sum(dim=-1)
    else:
        loss = -log_pt

    return (focal_weight * loss)[mask].mean() * loss_weight


def focal_loss(logits, targets, gamma=2.0, alpha=0.95, loss_weight=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = alpha_t * (1 - pt) ** gamma
    return (focal_weight * bce).mean() * loss_weight


def weighted_bce_loss(logits, targets, weight_power=0.25, loss_weight=1.0):
    pos_ratio = targets.mean().clamp(min=1e-6)
    pos_weight = ((1.0 - pos_ratio) / pos_ratio) ** weight_power
    return F.binary_cross_entropy_with_logits(logits.view(-1), targets.view(-1), pos_weight=pos_weight.expand_as(targets)) * loss_weight


def _mine_negative(neg_stack, s_pos, strategy):
    if strategy == "hard":
        return neg_stack.max()
    elif strategy == "easy":
        return neg_stack.min()
    else:  # semihard
        mask = neg_stack < s_pos
        return neg_stack[mask].max() if mask.any() else neg_stack.max()


_NEG_STRATEGIES = ["hard", "semihard", "easy"]


def mined_bce_loss(bce_logits, binary_labels, valid_mask, loss_weight=1.0):
    B = bce_logits.shape[0]
    total_loss = bce_logits.new_zeros(())
    n_pairs = 0

    for b in range(B):
        logits = bce_logits[b]
        labels = binary_labels[b]
        valid = valid_mask[b]

        pos_mask = (labels == 1) & valid.unsqueeze(0) & valid.unsqueeze(1)
        pos_i, pos_j = torch.where(torch.triu(pos_mask, diagonal=1))

        for pi, pj in zip(pos_i.tolist(), pos_j.tolist()):
            s_pos = logits[pi, pj]

            neg_i_list, neg_j_list = [], []
            for k in range(logits.shape[0]):
                if not valid[k] or k == pi or k == pj:
                    continue
                if labels[pi, k] == 0:
                    neg_i_list.append(logits[pi, k])
                if labels[pj, k] == 0:
                    neg_j_list.append(logits[pj, k])

            if not neg_i_list and not neg_j_list:
                continue

            loss_pos = F.binary_cross_entropy_with_logits(s_pos.unsqueeze(0), s_pos.new_ones(1))
            pair_loss = 2.0 * loss_pos

            if neg_i_list:
                s_neg_i = _mine_negative(torch.stack(neg_i_list), s_pos, random.choice(_NEG_STRATEGIES))
                pair_loss = pair_loss + F.binary_cross_entropy_with_logits(s_neg_i.unsqueeze(0), s_neg_i.new_zeros(1))

            if neg_j_list:
                s_neg_j = _mine_negative(torch.stack(neg_j_list), s_pos, random.choice(_NEG_STRATEGIES))
                pair_loss = pair_loss + F.binary_cross_entropy_with_logits(s_neg_j.unsqueeze(0), s_neg_j.new_zeros(1))

            total_loss = total_loss + pair_loss
            n_pairs += 1

    if n_pairs == 0:
        return bce_logits.sum() * 0.0
    return (total_loss / n_pairs) * loss_weight
