# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.structures import InstanceData
from torch import Tensor

from precision_track.registry import TASK_UTILS
from precision_track.utils import bbox_overlaps, parse_pose_metainfo

INF = 100000.0
EPS = 1.0e-7


def cast_tensor_type(x, scale=1.0, dtype=None):
    if dtype == "fp16":
        # scale is for preventing overflows
        x = (x / scale).half()
    return x


@TASK_UTILS.register_module()
class BBoxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, scale=1.0, dtype=None):
        self.scale = scale
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, bboxes1, bboxes2, mode="iou", is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
                in <x1, y1, x2, y2> format, or shape (m, 5) in <x1, y1, x2,
                y2, score> format.
            bboxes2 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
                in <x1, y1, x2, y2> format, shape (m, 5) in <x1, y1, x2, y2,
                score> format, or be empty. If ``is_aligned `` is ``True``,
                then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]

        if self.dtype == "fp16":
            # change tensor type to save cpu and cuda memory and keep speed
            bboxes1 = cast_tensor_type(bboxes1, self.scale, self.dtype)
            bboxes2 = cast_tensor_type(bboxes2, self.scale, self.dtype)
            overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                # resume cpu float32
                overlaps = overlaps.float()
            return overlaps

        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + f"(" f"scale={self.scale}, dtype={self.dtype})"
        return repr_str


@TASK_UTILS.register_module()
class PoseOKS:
    """OKS score Calculator."""

    def __init__(self, metainfo: Optional[str] = "configs/_base_/datasets/coco.py"):

        if metainfo is not None:
            metainfo = parse_pose_metainfo(dict(from_file=metainfo))
            sigmas = metainfo.get("sigmas", None)
            if sigmas is not None:
                self.sigmas = torch.as_tensor(sigmas)

    @torch.no_grad()
    def __call__(self, output: Tensor, target: Tensor, target_weights: Tensor, areas: Tensor, eps: float = 1e-8) -> Tensor:

        dist = torch.norm(output - target, dim=-1)
        areas = areas.reshape(*((1,) * (dist.ndim - 2)), -1, 1)
        dist = dist / areas.pow(0.5).clip(min=eps)

        if hasattr(self, "sigmas"):
            if self.sigmas.device != dist.device:
                self.sigmas = self.sigmas.to(dist.device)
            sigmas = self.sigmas.reshape(*((1,) * (dist.ndim - 1)), -1)
            dist = dist / (sigmas * 2)

        target_weights = target_weights / target_weights.sum(dim=-1, keepdims=True).clip(min=eps)
        oks = (torch.exp(-dist.pow(2) / 2) * target_weights).sum(dim=-1)
        return oks


@TASK_UTILS.register_module()
class SimOTAAssigner:
    """Computes matching between predictions and ground truth.

    Args:
        center_radius (float): Radius of center area to determine
            if a prior is in the center of a gt. Defaults to 2.5.
        candidate_topk (int): Top-k ious candidates to calculate dynamic-k.
            Defaults to 10.
        iou_weight (float): Weight of bbox iou cost. Defaults to 3.0.
        cls_weight (float): Weight of classification cost. Defaults to 1.0.
        oks_weight (float): Weight of keypoint OKS cost. Defaults to 3.0.
        vis_weight (float): Weight of keypoint visibility cost. Defaults to 0.0
        dynamic_k_indicator (str): Cost type for calculating dynamic-k,
            either 'iou' or 'oks'. Defaults to 'iou'.
        use_keypoints_for_center (bool): Whether to use keypoints to determine
            if a prior is in the center of a gt. Defaults to False.
        iou_calculator (dict): Config of IoU calculation method.
            Defaults to dict(type='BBoxOverlaps2D').
        oks_calculator (dict): Config of OKS calculation method.
            Defaults to dict(type='PoseOKS').
    """

    def __init__(
        self,
        center_radius: float = 2.5,
        candidate_topk: int = 10,
        iou_weight: float = 3.0,
        cls_weight: float = 1.0,
        oks_weight: float = 3.0,
        vis_weight: float = 0.0,
        dynamic_k_indicator: str = "iou",
        use_keypoints_for_center: bool = False,
        iou_calculator: Config = dict(type="BBoxOverlaps2D"),
        oks_calculator: Config = dict(type="PoseOKS"),
    ):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.oks_weight = oks_weight
        self.vis_weight = vis_weight
        assert dynamic_k_indicator in ("iou", "oks"), (
            f"the argument " f"`dynamic_k_indicator` should be either 'iou' or 'oks', " f"but got {dynamic_k_indicator}"
        )
        self.dynamic_k_indicator = dynamic_k_indicator

        self.use_keypoints_for_center = use_keypoints_for_center
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.oks_calculator = TASK_UTILS.build(oks_calculator)

    def assign(self, pred_instances: InstanceData, gt_instances: InstanceData, **kwargs) -> dict:
        """Assign gt to priors using SimOTA.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
        Returns:
            dict: Assignment result containing assigned gt indices,
                max iou overlaps, assigned labels, etc.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_keypoints = gt_instances.keypoints
        gt_keypoints_visible = gt_instances.keypoints_visible
        gt_areas = gt_instances.areas
        num_gt = gt_bboxes.size(0)

        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        priors = pred_instances.priors
        keypoints = pred_instances.keypoints
        keypoints_visible = pred_instances.keypoints_visible
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return dict(num_gts=num_gt, gt_inds=assigned_gt_inds, max_overlaps=max_overlaps, labels=assigned_labels)

        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(priors, gt_bboxes, gt_keypoints, gt_keypoints_visible)
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        valid_pred_kpts = keypoints[valid_mask]
        valid_pred_kpts_vis = keypoints_visible[valid_mask]

        num_valid = valid_decoded_bbox.size(0)
        if num_valid == 0:
            # No valid bboxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return dict(num_gts=num_gt, gt_inds=assigned_gt_inds, max_overlaps=max_overlaps, labels=assigned_labels)

        cost_matrix = (~is_in_boxes_and_center) * INF

        # calculate iou
        pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
        if self.iou_weight > 0:
            iou_cost = -torch.log(pairwise_ious + EPS)
            cost_matrix = cost_matrix + iou_cost * self.iou_weight

        # calculate oks
        if self.oks_weight > 0 or self.dynamic_k_indicator == "oks":
            pairwise_oks = self.oks_calculator(
                valid_pred_kpts.unsqueeze(1),  # [num_valid, 1, k, 2]
                target=gt_keypoints.unsqueeze(0),  # [1, num_gt, k, 2]
                target_weights=gt_keypoints_visible.unsqueeze(0),  # [1, num_gt, k]
                areas=gt_areas.unsqueeze(0),  # [1, num_gt]
            )  # -> [num_valid, num_gt]

            oks_cost = -torch.log(pairwise_oks + EPS)
            cost_matrix = cost_matrix + oks_cost * self.oks_weight

        # calculate cls
        if self.cls_weight > 0:
            gt_onehot_label = F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1]).float().unsqueeze(0).repeat(num_valid, 1, 1)
            valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
            # disable AMP autocast to avoid overflow
            with torch.cuda.amp.autocast(enabled=False):
                cls_cost = (
                    F.binary_cross_entropy(
                        valid_pred_scores.to(dtype=torch.float32),
                        gt_onehot_label,
                        reduction="none",
                    )
                    .sum(-1)
                    .to(dtype=valid_pred_scores.dtype)
                )
            cost_matrix = cost_matrix + cls_cost * self.cls_weight
        # calculate vis
        if self.vis_weight > 0:
            valid_pred_kpts_vis = valid_pred_kpts_vis.unsqueeze(1).repeat(1, num_gt, 1)  # [num_valid, 1, k]
            gt_kpt_vis = gt_keypoints_visible.unsqueeze(0).float()  # [1, num_gt, k]
            with torch.cuda.amp.autocast(enabled=False):
                vis_cost = (
                    F.binary_cross_entropy(
                        valid_pred_kpts_vis.to(dtype=torch.float32),
                        gt_kpt_vis.repeat(num_valid, 1, 1),
                        reduction="none",
                    )
                    .sum(-1)
                    .to(dtype=valid_pred_kpts_vis.dtype)
                )
            cost_matrix = cost_matrix + vis_cost * self.vis_weight

        if self.dynamic_k_indicator == "iou":
            matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(cost_matrix, pairwise_ious, num_gt, valid_mask)
        elif self.dynamic_k_indicator == "oks":
            matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(cost_matrix, pairwise_oks, num_gt, valid_mask)

        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes,), -INF, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious.to(max_overlaps)
        return dict(num_gts=num_gt, gt_inds=assigned_gt_inds, max_overlaps=max_overlaps, labels=assigned_labels)

    def get_in_gt_and_in_center_info(
        self,
        priors: Tensor,
        gt_bboxes: Tensor,
        gt_keypoints: Optional[Tensor] = None,
        gt_keypoints_visible: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Get the information of which prior is in gt bboxes and gt center
        priors."""
        num_gt = gt_bboxes.size(0)

        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        # is prior centers in gt bboxes, shape: [n_prior, n_gt]
        l_ = repeated_x - gt_bboxes[:, 0]
        t_ = repeated_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - repeated_x
        b_ = gt_bboxes[:, 3] - repeated_y

        deltas = torch.stack([l_, t_, r_, b_], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        # is prior centers in gt centers
        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        if self.use_keypoints_for_center and gt_keypoints_visible is not None:
            gt_kpts_cts = (gt_keypoints * gt_keypoints_visible.unsqueeze(-1)).sum(dim=-2) / gt_keypoints_visible.sum(dim=-1, keepdims=True).clip(min=0)
            gt_kpts_cts = gt_kpts_cts.to(gt_bboxes)
            valid_mask = gt_keypoints_visible.sum(-1) > 0
            gt_cxs[valid_mask] = gt_kpts_cts[valid_mask][..., 0]
            gt_cys[valid_mask] = gt_kpts_cts[valid_mask][..., 1]

        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y

        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y

        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers, shape: [num_priors]
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

        # both in boxes and centers, shape: [num_fg, num_gt]
        is_in_boxes_and_centers = is_in_gts[is_in_gts_or_centers, :] & is_in_cts[is_in_gts_or_centers, :]
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(self, cost: Tensor, pairwise_ious: Tensor, num_gt: int, valid_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets."""
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds


@TASK_UTILS.register_module()
class TaskAlignedAssigner(SimOTAAssigner):
    def __init__(self, alpha: float = 1.0, beta: float = 6.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def assign(self, pred_instances: InstanceData, gt_instances: InstanceData, **kwargs) -> dict:
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_keypoints = gt_instances.keypoints
        gt_keypoints_visible = gt_instances.keypoints_visible
        gt_areas = gt_instances.areas
        num_gt = gt_bboxes.size(0)

        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        priors = pred_instances.priors
        keypoints = pred_instances.keypoints
        num_bboxes = decoded_bboxes.size(0)

        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return dict(num_gts=num_gt, gt_inds=assigned_gt_inds, max_overlaps=max_overlaps, labels=assigned_labels)

        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(priors, gt_bboxes, gt_keypoints, gt_keypoints_visible)
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        valid_pred_kpts = keypoints[valid_mask]

        num_valid = valid_decoded_bbox.size(0)
        if num_valid == 0:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return dict(num_gts=num_gt, gt_inds=assigned_gt_inds, max_overlaps=max_overlaps, labels=assigned_labels)

        # Compute pairwise IoU or OKS
        if self.dynamic_k_indicator == "iou":
            quality = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
        else:
            quality = self.oks_calculator(
                valid_pred_kpts.unsqueeze(1),
                target=gt_keypoints.unsqueeze(0),
                target_weights=gt_keypoints_visible.unsqueeze(0),
                areas=gt_areas.unsqueeze(0),
            )

        # Classification scores
        cls_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
        gt_onehot_label = F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1]).float()
        gt_onehot_label = gt_onehot_label.unsqueeze(0).repeat(num_valid, 1, 1)
        alignment = (cls_scores * gt_onehot_label).sum(-1)

        # Alignment metric
        alignment_metric = (alignment**self.alpha) * (quality**self.beta)

        # Top-k selection
        topk = min(self.candidate_topk, num_valid)
        assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
        matched_pred_ious = decoded_bboxes.new_zeros((num_bboxes,), dtype=torch.float32)

        for gt_idx in range(num_gt):
            topk_vals, topk_idxs = alignment_metric[:, gt_idx].topk(topk, largest=True)
            valid_idxs = valid_mask.nonzero().squeeze(1)[topk_idxs]
            assigned_gt_inds[valid_idxs] = gt_idx + 1
            assigned_labels[valid_idxs] = gt_labels[gt_idx].to(assigned_labels.dtype)
            matched_pred_ious[valid_idxs] = quality[topk_idxs, gt_idx]

        return dict(num_gts=num_gt, gt_inds=assigned_gt_inds, max_overlaps=matched_pred_ious, labels=assigned_labels)


def center_of_mass(masks: Tensor, eps: float = 1e-7) -> Tensor:
    """Compute the masks center of mass.

    Args:
        masks: Mask tensor, has shape (num_masks, H, W).
        eps: a small number to avoid normalizer to be zero.
            Defaults to 1e-7.
    Returns:
        Tensor: The masks center of mass. Has shape (num_masks, 2).
    """
    n, h, w = masks.shape
    grid_h = torch.arange(h, device=masks.device)[:, None]
    grid_w = torch.arange(w, device=masks.device)
    normalizer = masks.sum(dim=(1, 2)).float().clamp(min=eps)
    center_y = (masks * grid_h).sum(dim=(1, 2)) / normalizer
    center_x = (masks * grid_w).sum(dim=(1, 2)) / normalizer
    center = torch.cat([center_x[:, None], center_y[:, None]], dim=1)
    return center


@TASK_UTILS.register_module()
class DynamicSoftLabelAssigner:
    """Computes matching between predictions and ground truth with dynamic soft
    label assignment.

    Args:
        soft_center_radius (float): Radius of the soft center prior.
            Defaults to 3.0.
        topk (int): Select top-k predictions to calculate dynamic k
            best matches for each gt. Defaults to 13.
        iou_weight (float): The scale factor of iou cost. Defaults to 3.0.
        iou_calculator (Config): Config of overlaps Calculator.
            Defaults to dict(type='BboxOverlaps2D').
    """

    def __init__(
        self,
        soft_center_radius: float = 3.0,
        topk: int = 13,
        iou_weight: float = 3.0,
        iou_calculator: Config = dict(type="BboxOverlaps2D"),
        *args,
        **kwargs,
    ) -> None:
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        gt_instances_ignore: Optional[InstanceData] = None,
        **kwargs,
    ) -> dict:
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_gt = gt_bboxes.size(0)

        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        priors = pred_instances.priors
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return dict(num_gts=num_gt, gt_inds=assigned_gt_inds, max_overlaps=max_overlaps, labels=assigned_labels)

        prior_center = priors[:, :2]
        lt_ = prior_center[:, None] - gt_bboxes[:, :2]
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]

        deltas = torch.cat([lt_, rb_], dim=-1)
        is_in_gts = deltas.min(dim=-1).values > 0

        valid_mask = is_in_gts.sum(dim=1) > 0

        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        if num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return dict(num_gts=num_gt, gt_inds=assigned_gt_inds, max_overlaps=max_overlaps, labels=assigned_labels)
        if hasattr(gt_instances, "masks"):
            gt_center = center_of_mass(gt_instances.masks, eps=EPS)
        gt_center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.0
        valid_prior = priors[valid_mask]
        strides = valid_prior[:, 2]
        distance = (valid_prior[:, None, :2] - gt_center[None, :, :]).pow(2).sum(-1).sqrt() / strides[:, None]
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight

        gt_onehot_label = F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1]).float().unsqueeze(0).repeat(num_valid, 1, 1)
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores.sigmoid()
        soft_cls_cost = F.binary_cross_entropy_with_logits(valid_pred_scores, soft_label, reduction="none") * scale_factor.abs().pow(2.0)
        soft_cls_cost = soft_cls_cost.sum(dim=-1)

        cost_matrix = soft_cls_cost + iou_cost + soft_center_prior

        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(cost_matrix, pairwise_ious, num_gt, valid_mask)

        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes,), -INF, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious.to(max_overlaps)
        return dict(num_gts=num_gt, gt_inds=assigned_gt_inds, max_overlaps=max_overlaps, labels=assigned_labels)

    def dynamic_k_matching(self, cost: Tensor, pairwise_ious: Tensor, num_gt: int, valid_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets. Same as SimOTA.

        Args:
            cost (Tensor): Cost matrix.
            pairwise_ious (Tensor): Pairwise iou matrix.
            num_gt (int): Number of gt.
            valid_mask (Tensor): Mask for valid bboxes.

        Returns:
            tuple: matched ious and gt indexes.
        """
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
