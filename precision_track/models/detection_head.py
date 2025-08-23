# Copyright (c) OpenMMLab. All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from precision_track.registry import MODELS, TASK_UTILS
from precision_track.utils import PoseDataSample

from .modules.heads.yolox import YOLOXPoseHeadModule


def reduce_mean(tensor):
    """ "Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


@MODELS.register_module()
class DetectionHead(BaseModule):

    def __init__(
        self,
        num_keypoints: int,
        in_channels: Union[int, Sequence],
        prior_generator: Config,
        num_classes: int = 1,
        widen_factor: float = 1.0,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        featmap_strides: Sequence[int] = [8, 16, 32],
        conv_bias: Union[bool, str] = "auto",
        conv_cfg: Optional[Config] = None,
        norm_cfg: Config = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: Config = dict(type="SiLU", inplace=True),
        init_cfg: Optional[Config] = None,
        assigner: Config = None,
        loss_cls: Optional[Config] = None,
        loss_obj: Optional[Config] = None,
        loss_bbox: Optional[Config] = None,
        loss_oks: Optional[Config] = None,
        loss_vis: Optional[Config] = None,
        overlaps_power: float = 1.0,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.head_module = YOLOXPoseHeadModule(
            num_keypoints,
            in_channels,
            num_classes,
            widen_factor,
            feat_channels,
            stacked_convs,
            featmap_strides,
            conv_bias,
            conv_cfg,
            norm_cfg,
            act_cfg,
            init_cfg,
        )
        self.featmap_sizes = None
        self.featmap_strides = featmap_strides
        self.num_keypoints = num_keypoints
        self.num_classes = self.head_module.num_classes
        self.overlaps_power = overlaps_power
        self.feat_channels = self.head_module.feat_channels

        self.assigner = TASK_UTILS.build(assigner) if assigner is not None else None
        self.loss_cls = MODELS.build(loss_cls) if loss_cls is not None else None
        self.loss_obj = MODELS.build(loss_obj) if loss_obj is not None else None
        self.loss_bbox = MODELS.build(loss_bbox) if loss_bbox is not None else None
        self.loss_oks = MODELS.build(loss_oks) if loss_oks is not None else None
        self.loss_vis = MODELS.build(loss_vis) if loss_vis is not None else None

        self.temperature = 1

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            flatten_cls_scores (Tensor): Classification scores for each level.
            flatten_objectness (Tensor): Objectness scores for each level.
            flatten_bbox_preds (Tensor): Bounding box predictions for each level.
            flatten_kpt_reg (Tensor): Keypoint predictions for each level.
            flatten_kpt_vis (Tensor): Keypoint visibilities for each level.
        """
        cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis = self.head_module.forward(x)

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)
        mlvl_strides = [flatten_priors.new_full((featmap_size.numel(),), stride) for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)]
        flatten_stride = torch.cat(mlvl_strides)

        flatten_prior_features = self._flatten_predictions(x)
        flatten_cls_scores = self._flatten_predictions(cls_scores) * self.temperature
        flatten_bbox_preds = self._flatten_predictions(bbox_preds)
        flatten_objectness = self._flatten_predictions(objectnesses)
        flatten_kpt_offsets = self._flatten_predictions(kpt_offsets)
        flatten_kpt_vis = self._flatten_predictions(kpt_vis)
        flatten_bbox_preds = self.decode_bbox(flatten_bbox_preds, flatten_priors, flatten_stride)
        flatten_kpt_reg = self.decode_kpt_reg(flatten_kpt_offsets, flatten_priors, flatten_stride)

        return (
            flatten_cls_scores.contiguous(),
            flatten_objectness.contiguous(),
            flatten_bbox_preds.contiguous(),
            flatten_kpt_reg.contiguous(),
            flatten_kpt_vis.contiguous(),
            flatten_prior_features.contiguous(),
            flatten_priors.contiguous().view(1, -1, 2),
            flatten_stride.contiguous().view(1, -1, 1),
        )

    def decode_bbox(
        self,
        pred_bboxes: torch.Tensor,
        priors: torch.Tensor,
        stride: Union[torch.Tensor, int],
    ) -> torch.Tensor:
        """Decode regression results (delta_x, delta_y, log_w, log_h) to
        bounding boxes (tl_x, tl_y, br_x, br_y).

        Note:
            - batch size: B
            - token number: N

        Args:
            pred_bboxes (torch.Tensor): Encoded boxes with shape (B, N, 4),
                representing (delta_x, delta_y, log_w, log_h) for each box.
            priors (torch.Tensor): Anchors coordinates, with shape (N, 2).
            stride (torch.Tensor | int): Strides of the bboxes. It can be a
                single value if the same stride applies to all boxes, or it
                can be a tensor of shape (N, ) if different strides are used
                for each box.

        Returns:
            torch.Tensor: Decoded bounding boxes with shape (N, 4),
                representing (tl_x, tl_y, br_x, br_y) for each box.
        """
        stride = stride.view(1, stride.size(0), 1)
        priors = priors.view(1, priors.size(0), 2)

        xys = (pred_bboxes[..., :2] * stride) + priors
        whs = pred_bboxes[..., 2:].exp() * stride

        # Calculate bounding box corners
        tl_x = xys[..., 0] - whs[..., 0] / 2
        tl_y = xys[..., 1] - whs[..., 1] / 2
        br_x = xys[..., 0] + whs[..., 0] / 2
        br_y = xys[..., 1] + whs[..., 1] / 2

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes

    def decode_kpt_reg(self, pred_kpt_offsets: torch.Tensor, priors: torch.Tensor, stride: torch.Tensor) -> torch.Tensor:
        """Decode regression results (delta_x, delta_y) to keypoints
        coordinates (x, y).

        Args:
            pred_kpt_offsets (torch.Tensor): Encoded keypoints offsets with
                shape (batch_size, num_anchors, num_keypoints, 2).
            priors (torch.Tensor): Anchors coordinates with shape
                (num_anchors, 2).
            stride (torch.Tensor): Strides of the anchors.

        Returns:
            torch.Tensor: Decoded keypoints coordinates with shape
                (batch_size, num_boxes, num_keypoints, 2).
        """
        stride = stride.view(1, stride.size(0), 1, 1)
        priors = priors.view(1, priors.size(0), 1, 2)
        pred_kpt_offsets = pred_kpt_offsets.reshape(*pred_kpt_offsets.shape[:-1], self.num_keypoints, 2)

        decoded_kpts = pred_kpt_offsets * stride + priors
        return decoded_kpts

    def _flatten_predictions(self, preds: List[Tensor]):
        """Flattens the predictions from a list of tensors to a single
        tensor."""
        if len(preds) == 0:
            return None

        preds = [x.permute(0, 2, 3, 1).flatten(1, 2) for x in preds]
        return torch.cat(preds, dim=1)

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: Optional[List[PoseDataSample]],
        train_cfg: Config = {},
        return_features: Optional[bool] = False,
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """

        # 1. collect predictions
        (
            flatten_cls_scores,
            flatten_objectness,
            flatten_bbox_decoded,
            flatten_kpt_decoded,
            flatten_kpt_vis,
            flatten_priors_features,
            flatten_priors,
            flatten_strides,
        ) = self.forward(feats)

        flatten_priors = torch.hstack((flatten_priors.squeeze(0), flatten_strides.squeeze(0).repeat(1, 2)))

        # 2. generate targets
        targets = self._get_targets(
            flatten_priors,
            flatten_cls_scores.detach(),
            flatten_objectness.detach(),
            flatten_bbox_decoded.detach(),
            flatten_kpt_decoded.detach(),
            flatten_kpt_vis.detach(),
            batch_data_samples,
        )
        (
            pos_masks,
            cls_targets,
            obj_targets,
            obj_weights,
            bbox_targets,
            bbox_aux_targets,
            kpt_targets,
            kpt_aux_targets,
            vis_targets,
            vis_weights,
            pos_areas,
            pos_priors,
            batch_gt_indices,
            batch_pos_masks,
            num_fg_imgs,
        ) = targets

        num_pos = torch.tensor(sum(num_fg_imgs), dtype=torch.float, device=flatten_cls_scores.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        # 3. calculate loss
        # 3.1 objectness loss
        losses = dict()

        obj_preds = flatten_objectness.view(-1, 1)
        losses["loss_obj"] = self.loss_obj(obj_preds, obj_targets, obj_weights) / num_total_samples

        if num_pos > 0:
            # 3.2 bbox loss
            bbox_preds = flatten_bbox_decoded.view(-1, 4)[pos_masks]
            losses["loss_bbox"] = self.loss_bbox(bbox_preds, bbox_targets) / num_total_samples

            # 3.3 keypoint loss
            kpt_preds = flatten_kpt_decoded.view(-1, self.num_keypoints, 2)[pos_masks]
            losses["loss_kpt"] = self.loss_oks(kpt_preds, kpt_targets, vis_targets, pos_areas)

            # 3.4 keypoint visibility loss
            kpt_vis_preds = flatten_kpt_vis.view(-1, self.num_keypoints)[pos_masks]
            losses["loss_vis"] = self.loss_vis(kpt_vis_preds, vis_targets, vis_weights)

            # 3.5 classification loss
            cls_preds = flatten_cls_scores.view(-1, self.num_classes)[pos_masks]
            losses["overlaps"] = cls_targets
            cls_targets = cls_targets.pow(self.overlaps_power).detach()
            losses["loss_cls"] = self.loss_cls(cls_preds, cls_targets) / num_total_samples

            feat_preds = flatten_priors_features.detach().view(-1, self.feat_channels)[pos_masks]
        else:
            bbox_preds = bbox_targets[0]
            kpt_preds = kpt_targets[0]
            kpt_vis_preds = vis_targets[0]
            cls_preds = cls_targets[0]
            feat_preds = cls_preds.new_zeros((0, self.feat_channels))

        if return_features:
            return losses, (
                cls_preds,
                obj_preds,
                bbox_preds,
                kpt_preds,
                kpt_vis_preds,
                feat_preds,
                batch_pos_masks,
                batch_gt_indices,
            )

        return losses

    @torch.no_grad()
    def _get_targets(
        self,
        priors: Tensor,
        batch_cls_scores: Tensor,
        batch_objectness: Tensor,
        batch_decoded_bboxes: Tensor,
        batch_decoded_kpts: Tensor,
        batch_kpt_vis: Tensor,
        batch_data_samples: List[PoseDataSample],
    ):
        num_imgs = len(batch_data_samples)

        # use clip to avoid nan
        batch_cls_scores = batch_cls_scores.clip(min=-1e4, max=1e4).sigmoid()
        batch_objectness = batch_objectness.clip(min=-1e4, max=1e4).sigmoid()
        batch_kpt_vis = batch_kpt_vis.clip(min=-1e4, max=1e4).sigmoid()
        batch_cls_scores[torch.isnan(batch_cls_scores)] = 0
        batch_objectness[torch.isnan(batch_objectness)] = 0

        targets_each = []
        for i in range(num_imgs):
            target = self._get_targets_single(
                priors,
                batch_cls_scores[i],
                batch_objectness[i],
                batch_decoded_bboxes[i],
                batch_decoded_kpts[i],
                batch_kpt_vis[i],
                batch_data_samples[i],
            )
            targets_each.append(target + (target[0],))

        targets = list(zip(*targets_each))
        for i, target in enumerate(targets[:-3]):
            if torch.is_tensor(target[0]):
                target = tuple(filter(lambda x: x.size(0) > 0, target))
                if len(target) > 0:
                    targets[i] = torch.cat(target)
        (
            foreground_masks,
            cls_targets,
            obj_targets,
            obj_weights,
            bbox_targets,
            kpt_targets,
            vis_targets,
            vis_weights,
            pos_areas,
            pos_priors,
            gt_indices,
            num_pos_per_img,
            imgs_pos_masks,
        ) = targets

        bbox_aux_targets, kpt_aux_targets = None, None

        return (
            foreground_masks,
            cls_targets,
            obj_targets,
            obj_weights,
            bbox_targets,
            bbox_aux_targets,
            kpt_targets,
            kpt_aux_targets,
            vis_targets,
            vis_weights,
            pos_areas,
            pos_priors,
            gt_indices,
            imgs_pos_masks,
            num_pos_per_img,
        )

    @torch.no_grad()
    def _get_targets_single(
        self,
        priors: Tensor,
        cls_scores: Tensor,
        objectness: Tensor,
        decoded_bboxes: Tensor,
        decoded_kpts: Tensor,
        kpt_vis: Tensor,
        data_sample: PoseDataSample,
    ) -> tuple:
        """Compute classification, bbox, keypoints and objectness targets for
        priors in a single image.

        Args:
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            cls_scores (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in xyxy format.
            decoded_kpts (Tensor): Decoded keypoints predictions of one image,
                a 3D-Tensor with shape [num_priors, num_keypoints, 2].
            kpt_vis (Tensor): Keypoints visibility predictions of one image,
                a 2D-Tensor with shape [num_priors, num_keypoints].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            data_sample (PoseDataSample): Data sample that contains the ground
                truth annotations for current image.

        Returns:
            tuple: A tuple containing various target tensors for training:
                - foreground_mask (Tensor): Binary mask indicating foreground
                    priors.
                - cls_target (Tensor): Classification targets.
                - obj_target (Tensor): Objectness targets.
                - obj_weight (Tensor): Weights for objectness targets.
                - bbox_target (Tensor): BBox targets.
                - kpt_target (Tensor): Keypoints targets.
                - vis_target (Tensor): Visibility targets for keypoints.
                - vis_weight (Tensor): Weights for keypoints visibility
                    targets.
                - pos_areas (Tensor): Areas of positive samples.
                - pos_priors (Tensor): Priors corresponding to positive
                    samples.
                - group_index (List[Tensor]): Indices of groups for positive
                    samples.
                - num_pos_per_img (int): Number of positive samples.
        """
        # TODO: change the shape of objectness to [num_priors]
        num_priors = priors.size(0)
        gt_instances = data_sample.gt_instance_labels
        gt_fields = data_sample.get("gt_fields", dict())
        num_gts = len(gt_instances)

        # No target
        if num_gts == 0:
            cls_target = cls_scores.new_zeros((0, self.num_classes))
            bbox_target = cls_scores.new_zeros((0, 4))
            obj_target = cls_scores.new_zeros((num_priors, 1))
            obj_weight = cls_scores.new_ones((num_priors, 1))
            kpt_target = cls_scores.new_zeros((0, self.num_keypoints, 2))
            vis_target = cls_scores.new_zeros((0, self.num_keypoints))
            vis_weight = cls_scores.new_zeros((0, self.num_keypoints))
            pos_areas = cls_scores.new_zeros((0,))
            pos_priors = priors[:0]
            foreground_mask = cls_scores.new_zeros(num_priors).bool()
            return (
                foreground_mask,
                cls_target,
                obj_target,
                obj_weight,
                bbox_target,
                kpt_target,
                vis_target,
                vis_weight,
                pos_areas,
                pos_priors,
                [],
                0,
            )

        # assign positive samples
        scores = cls_scores * objectness
        pred_instances = InstanceData(
            bboxes=decoded_bboxes,
            scores=scores.sqrt_(),
            priors=priors,
            keypoints=decoded_kpts,
            keypoints_visible=kpt_vis,
        )
        assign_result = self.assigner.assign(pred_instances=pred_instances, gt_instances=gt_instances)

        # sampling
        pos_inds = torch.nonzero(assign_result["gt_inds"] > 0, as_tuple=False).squeeze(-1).unique()
        num_pos_per_img = pos_inds.size(0)
        pos_gt_labels = assign_result["labels"][pos_inds]
        pos_assigned_gt_inds = assign_result["gt_inds"][pos_inds] - 1

        # bbox target
        bbox_target = gt_instances.bboxes[pos_assigned_gt_inds.long()]

        # cls target
        max_overlaps = assign_result["max_overlaps"][pos_inds]
        cls_target = F.one_hot(pos_gt_labels, self.num_classes) * max_overlaps.unsqueeze(-1)

        # pose targets
        kpt_target = gt_instances.keypoints[pos_assigned_gt_inds]
        vis_target = gt_instances.keypoints_visible[pos_assigned_gt_inds]
        if "keypoints_visible_weights" in gt_instances:
            vis_weight = gt_instances.keypoints_visible_weights[pos_assigned_gt_inds]
        else:
            vis_weight = vis_target.new_ones(vis_target.shape)
        pos_areas = gt_instances.areas[pos_assigned_gt_inds]

        # obj target
        obj_target = torch.zeros_like(objectness)
        obj_target[pos_inds] = 1

        invalid_mask = gt_fields.get("heatmap_mask", None)
        if invalid_mask is not None and (invalid_mask != 0.0).any():
            # ignore the tokens that predict the unlabled instances
            pred_vis = (kpt_vis.unsqueeze(-1) > 0.3).float()
            mean_kpts = (decoded_kpts * pred_vis).sum(dim=1) / pred_vis.sum(dim=1).clamp(min=1e-8)
            mean_kpts = mean_kpts.reshape(1, -1, 1, 2)
            wh = invalid_mask.shape[-1]
            grids = mean_kpts / (wh - 1) * 2 - 1
            mask = invalid_mask.unsqueeze(0).float()
            weight = F.grid_sample(mask, grids, mode="bilinear", padding_mode="zeros")
            obj_weight = 1.0 - weight.reshape(num_priors, 1)
        else:
            obj_weight = obj_target.new_ones(obj_target.shape)

        # misc
        foreground_mask = torch.zeros_like(objectness.squeeze()).to(torch.bool)
        foreground_mask[pos_inds] = 1
        pos_priors = priors[pos_inds]
        # group_index = [torch.where(pos_assigned_gt_inds == num)[0] for num in torch.unique(pos_assigned_gt_inds)]

        return (
            foreground_mask,
            cls_target,
            obj_target,
            obj_weight,
            bbox_target,
            kpt_target,
            vis_target,
            vis_weight,
            pos_areas,
            pos_priors,
            pos_assigned_gt_inds.long(),
            num_pos_per_img,
        )
