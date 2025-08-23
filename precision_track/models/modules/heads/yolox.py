from typing import List, Optional, Sequence, Tuple, Union

import torch.nn as nn
from mmengine.config import Config
from mmengine.model import BaseModule, bias_init_with_prob
from torch import Tensor

from precision_track.registry import MODELS

from ..blocks.cnn import ConvModule


@MODELS.register_module()
class YOLOXPoseHeadModule(BaseModule):
    """YOLOXPose head module for one-stage human pose estimation.

    This module predicts classification scores, bounding boxes, keypoint
    offsets and visibilities from multi-level feature maps.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        num_keypoints (int): Number of keypoints defined for one instance.
         in_channels (Union[int, Sequence]): Number of channels in the input
             feature map.
        feat_channels (int): Number of channels in the classification score
            and objectness prediction branch. Defaults to 256.
         widen_factor (float): Width multiplier, multiply number of
             channels in each layer by this amount. Defaults to 1.0.
        num_groups (int): Group number of group convolution layers in keypoint
            regression branch. Defaults to 8.
        channels_per_group (int): Number of channels for each group of group
            convolution layers in keypoint regression branch. Defaults to 32.
        featmap_strides (Sequence[int]): Downsample factor of each feature
            map. Defaults to [8, 16, 32].
        conv_bias (bool or str): If specified as `auto`, it will be decided
            by the norm_cfg. Bias of conv will be set as True if `norm_cfg`
            is None, otherwise False. Defaults to "auto".
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        num_keypoints: int,
        in_channels: Union[int, Sequence],
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
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.feat_channels = int(feat_channels * widen_factor)
        self.stacked_convs = stacked_convs
        assert conv_bias == "auto" or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.featmap_strides = featmap_strides

        if isinstance(in_channels, int):
            in_channels = int(in_channels * widen_factor)
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints

        self._init_layers()

    def _init_layers(self):
        """Initialize heads for all level feature maps."""
        self._init_cls_branch()
        self._init_reg_branch()
        self._init_pose_branch()

    def _init_cls_branch(self):
        """Initialize classification branch for all level feature maps."""
        self.conv_cls = nn.ModuleList()
        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                stacked_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=self.conv_bias,
                    )
                )
            self.conv_cls.append(nn.Sequential(*stacked_convs))

        # output layers
        self.out_cls = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_cls.append(nn.Conv2d(self.feat_channels, self.num_classes, 1))

    def _init_reg_branch(self):
        """Initialize classification branch for all level feature maps."""
        self.conv_reg = nn.ModuleList()
        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                stacked_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=self.conv_bias,
                    )
                )
            self.conv_reg.append(nn.Sequential(*stacked_convs))

        # output layers
        self.out_bbox = nn.ModuleList()
        self.out_obj = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_bbox.append(nn.Conv2d(self.feat_channels, 4, 1))
            self.out_obj.append(nn.Conv2d(self.feat_channels, 1, 1))

    def _init_pose_branch(self):
        self.conv_pose = nn.ModuleList()

        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs * 2):
                in_chn = self.in_channels if i == 0 else self.feat_channels
                stacked_convs.append(
                    ConvModule(
                        in_chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=self.conv_bias,
                    )
                )
            self.conv_pose.append(nn.Sequential(*stacked_convs))

        # output layers
        self.out_kpt = nn.ModuleList()
        self.out_kpt_vis = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_kpt.append(nn.Conv2d(self.feat_channels, self.num_keypoints * 2, 1))
            self.out_kpt_vis.append(nn.Conv2d(self.feat_channels, self.num_keypoints, 1))

    def init_weights(self):
        """Initialize weights of the head."""
        # Use prior in model initialization to improve stability
        super().init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(self.out_cls, self.out_obj):
            conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            cls_scores (List[Tensor]): Classification scores for each level.
            objectnesses (List[Tensor]): Objectness scores for each level.
            bbox_preds (List[Tensor]): Bounding box predictions for each level.
            kpt_offsets (List[Tensor]): Keypoint offsets for each level.
            kpt_vis (List[Tensor]): Keypoint visibilities for each level.
        """

        cls_scores, bbox_preds, objectnesses = [], [], []
        kpt_offsets, kpt_vis = [], []

        for i in range(len(x)):

            cls_feat = self.conv_cls[i](x[i])
            reg_feat = self.conv_reg[i](x[i])
            pose_feat = self.conv_pose[i](x[i])

            cls_scores.append(self.out_cls[i](cls_feat))
            objectnesses.append(self.out_obj[i](reg_feat))
            bbox_preds.append(self.out_bbox[i](reg_feat))
            kpt_offsets.append(self.out_kpt[i](pose_feat))
            kpt_vis.append(self.out_kpt_vis[i](pose_feat))

        return cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis


@MODELS.register_module()
class RTMDetPoseHeadModule(BaseModule):
    """RTMDetPose head module for one-stage human pose estimation.

    This module predicts classification scores, bounding boxes, keypoint
    offsets and visibilities from multi-level feature maps.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        num_keypoints (int): Number of keypoints defined for one instance.
         in_channels (Union[int, Sequence]): Number of channels in the input
             feature map.
        feat_channels (int): Number of channels in the classification score
            and objectness prediction branch. Defaults to 256.
         widen_factor (float): Width multiplier, multiply number of
             channels in each layer by this amount. Defaults to 1.0.
        num_groups (int): Group number of group convolution layers in keypoint
            regression branch. Defaults to 8.
        channels_per_group (int): Number of channels for each group of group
            convolution layers in keypoint regression branch. Defaults to 32.
        featmap_strides (Sequence[int]): Downsample factor of each feature
            map. Defaults to [8, 16, 32].
        conv_bias (bool or str): If specified as `auto`, it will be decided
            by the norm_cfg. Bias of conv will be set as True if `norm_cfg`
            is None, otherwise False. Defaults to "auto".
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
        share_conv (bool, optional): Whether to share conv layers between stages.
            Defaults to True.
    """

    def __init__(
        self,
        num_keypoints: int,
        in_channels: Union[int, Sequence],
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
        share_conv: Optional[bool] = False,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.feat_channels = int(feat_channels * widen_factor)
        self.stacked_convs = stacked_convs
        assert conv_bias == "auto" or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.featmap_strides = featmap_strides

        if isinstance(in_channels, int):
            in_channels = int(in_channels * widen_factor)
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.share_conv = share_conv

        self._init_layers()

    def _init_layers(self):
        """Initialize heads for all level feature maps."""
        self._init_bbox_branch()
        self._init_pose_branch()

    def _init_bbox_branch(self):
        """Initialize classification branch for all level feature maps."""
        self.conv_cls = nn.ModuleList()
        self.conv_obj = nn.ModuleList()
        self.conv_bbox = nn.ModuleList()

        self.out_cls = nn.ModuleList()
        self.out_obj = nn.ModuleList()
        self.out_bbox = nn.ModuleList()

        for _ in self.featmap_strides:
            cls = []
            obj = []
            bbox = []
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=self.conv_bias,
                    )
                )
                obj.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=self.conv_bias,
                    )
                )
                bbox.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=self.conv_bias,
                    )
                )

            self.conv_cls.append(nn.Sequential(*cls))
            self.conv_obj.append(nn.Sequential(*obj))
            self.conv_bbox.append(nn.Sequential(*bbox))

            self.out_cls.append(nn.Conv2d(self.feat_channels, self.num_classes, 1))
            self.out_obj.append(nn.Conv2d(self.feat_channels, 1, 1))
            self.out_bbox.append(nn.Conv2d(self.feat_channels, 4, 1))

        if self.share_conv:
            for n in range(len(self.featmap_strides)):
                for i in range(self.stacked_convs):
                    self.conv_cls[n][i].conv = self.conv_cls[0][i].conv
                    self.conv_obj[n][i].conv = self.conv_cls[n][i].conv
                    self.conv_bbox[n][i].conv = self.conv_bbox[0][i].conv

    def _init_pose_branch(self):
        self.conv_pose = nn.ModuleList()
        self.conv_pose_vis = nn.ModuleList()

        self.out_pose = nn.ModuleList()
        self.out_pose_vis = nn.ModuleList()

        for _ in self.featmap_strides:
            pose = []
            vis = []
            for i in range(self.stacked_convs * 2):
                in_chn = self.in_channels if i == 0 else self.feat_channels
                pose.append(
                    ConvModule(
                        in_chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=self.conv_bias,
                    )
                )
                vis.append(
                    ConvModule(
                        in_chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=self.conv_bias,
                    )
                )
            self.conv_pose.append(nn.Sequential(*pose))
            self.conv_pose_vis.append(nn.Sequential(*vis))

            self.out_pose.append(nn.Conv2d(self.feat_channels, self.num_keypoints * 2, 1))
            self.out_pose_vis.append(nn.Conv2d(self.feat_channels, self.num_keypoints, 1))

        if self.share_conv:
            for n in range(len(self.featmap_strides)):
                for i in range(self.stacked_convs):
                    self.conv_pose[n][i].conv = self.conv_pose[0][i].conv
                    self.conv_pose_vis[n][i].conv = self.conv_pose[n][i].conv

    def init_weights(self):
        """Initialize weights of the head."""
        # Use prior in model initialization to improve stability
        super().init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(self.out_cls, self.out_obj):
            conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            cls_scores (List[Tensor]): Classification scores for each level.
            objectnesses (List[Tensor]): Objectness scores for each level.
            bbox_preds (List[Tensor]): Bounding box predictions for each level.
            kpt_offsets (List[Tensor]): Keypoint offsets for each level.
            kpt_vis (List[Tensor]): Keypoint visibilities for each level.
        """

        cls_scores, bbox_preds, objectnesses = [], [], []
        kpt_offsets, kpt_vis = [], []

        for i in range(len(x)):

            cls_feat = self.conv_cls[i](x[i])
            obj_feat = self.conv_obj[i](x[i])
            bbox_feat = self.conv_bbox[i](x[i])
            pose_feat = self.conv_pose[i](x[i])
            pose_vis_feat = self.conv_pose_vis[i](x[i])

            cls_scores.append(self.out_cls[i](cls_feat))
            objectnesses.append(self.out_obj[i](obj_feat))
            bbox_preds.append(self.out_bbox[i](bbox_feat))
            kpt_offsets.append(self.out_pose[i](pose_feat))
            kpt_vis.append(self.out_pose_vis[i](pose_vis_feat))

        return cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis
