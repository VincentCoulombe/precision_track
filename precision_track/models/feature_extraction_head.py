from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.model import BaseModule, bias_init_with_prob
from torch import Tensor

from precision_track.registry import MODELS
from precision_track.utils import PoseDataSample

from .modules.blocks.cnn import ConvModule


@MODELS.register_module()
class FeatureExtractionHead(BaseModule):

    def __init__(
        self,
        in_channels: Union[int, Sequence],
        widen_factor: float = 1.0,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        featmap_strides: Sequence[int] = [8, 16, 32],
        conv_bias: Union[bool, str] = "auto",
        conv_cfg: Optional[Config] = None,
        norm_cfg: Config = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: Config = dict(type="SiLU", inplace=True),
        init_cfg: Optional[Config] = None,
        loss_features: Optional[Config] = None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.feat_channels = int(feat_channels * widen_factor)
        self.stacked_convs = stacked_convs
        assert conv_bias == "auto" or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.featmap_sizes = None
        self.featmap_strides = featmap_strides

        if isinstance(in_channels, int):
            in_channels = int(in_channels * widen_factor)
        self.in_channels = in_channels

        self.conv_feats = nn.ModuleList()
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
            self.conv_feats.append(nn.Sequential(*stacked_convs))

        # output layers
        self.out_feats = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_feats.append(nn.Conv2d(self.feat_channels, self.feat_channels, 1))

        self.loss_features = MODELS.build(loss_features) if loss_features is not None else None

    def init_weights(self):
        """Initialize weights of the head."""
        super().init_weights()
        bias_init = bias_init_with_prob(0.01)
        for out_feat in self.out_feats:
            out_feat.bias.data.fill_(bias_init)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        features = []
        for i in range(len(x)):
            feats = self.conv_feats[i](x[i])
            features.append(self.out_feats[i](feats))
        return self._flatten_predictions(features)

    def loss(
        self,
        priors_features: Tuple[Tensor],
        detection_head_features: Tuple[Tensor],
        batch_data_samples: Optional[List[PoseDataSample]],
        return_features: Optional[bool] = False,
        train_cfg: Config = {},
    ) -> dict:
        (
            _,
            _,
            _,
            _,
            _,
            _,
            batch_pos_masks,
            batch_gt_indices,
        ) = detection_head_features
        assert len(batch_pos_masks) == len(batch_gt_indices)
        B = len(batch_pos_masks)
        if B == 1 and len(batch_gt_indices[0]) == 0:
            if return_features:
                return dict(), tuple()
            return dict()
        batch_extracted_features = self.forward(priors_features)
        assert batch_extracted_features.size(0) == B
        batch_loss = torch.tensor(0.0, dtype=torch.float32, device=batch_extracted_features.device)
        batch_pos_extracted_features = []
        batch_pos_labels = []
        for i, (pos_masks, gt_indices) in enumerate(zip(batch_pos_masks, batch_gt_indices)):
            extracted_features = batch_extracted_features[i]
            if isinstance(gt_indices, list):
                gt_indices = torch.tensor(gt_indices, device=extracted_features.device, dtype=extracted_features.dtype)
            pos_extracted_features = extracted_features.view(-1, self.feat_channels)[pos_masks]
            loss_features = self.loss_features(pos_extracted_features, gt_indices)
            batch_loss = batch_loss + loss_features
            batch_pos_extracted_features.append(pos_extracted_features)
            batch_pos_labels.append(gt_indices)
        loss = dict(loss_feature_clustering=batch_loss / B)
        if return_features:
            return loss, (batch_pos_extracted_features, batch_pos_labels)
        return loss

    def _flatten_predictions(self, preds: List[Tensor]):
        """Flattens the predictions from a list of tensors to a single
        tensor."""
        if len(preds) == 0:
            return None

        preds = [x.permute(0, 2, 3, 1).flatten(1, 2) for x in preds]
        return torch.cat(preds, dim=1)
