from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.model import BaseModule, bias_init_with_prob
from torch import Tensor

from precision_track.registry import MODELS
from precision_track.utils import PoseDataSample, postprocess_one_stage_detections, iou_batch, reformat, linear_assignment
from precision_track.models.postprocessing.steps import PostProcessingSteps

from .modules.blocks.cnn import ConvModule


@MODELS.register_module()
class FeatureExtractionHead(BaseModule):

    def __init__(
        self,
        data_postprocessor: dict,
        loss_config: dict,
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

        self._loss = MODELS.build(loss_config)
        self.post_processor = data_postprocessor
        if data_postprocessor is not None:
            self.post_processor = PostProcessingSteps(data_postprocessor)
        else:
            self.post_processor = []

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
        features_map: Tuple[Tensor],
        detection_head_output: Tuple[Tensor],
        batch_data_samples: Optional[List[PoseDataSample]],
        train_cfg: Config = {},
        *args,
        **kwargs,
    ) -> dict:
        B = len(batch_data_samples) // 2
        (
            scores,
            objectness,
            bboxes,
            kpts,
            kpt_vis,
            features,
            priors,
            _,
        ) = detection_head_output

        extracted_features = self.forward(features_map)

        postprocessed_outputs = postprocess_one_stage_detections(
            self.post_processor,
            scores,
            objectness,
            bboxes,
            kpts,
            kpt_vis,
            features,
            priors,
            batch_data_samples,
            0.0,
        )

        frames = dict()
        for i, postprocessed_output in enumerate(postprocessed_outputs):
            kept_idxs = postprocessed_output["pred_instances"].kept_idxs

            i_extracted_features = extracted_features[i][kept_idxs]
            target_features = postprocessed_output["pred_instances"]["features"]
            assert i_extracted_features.shape == target_features.shape

            matches = dict()
            if postprocessed_output["img_id"] not in frames:
                frames[postprocessed_output["img_id"]] = []

            lbls_bboxes = reformat(postprocessed_output["gt_instances"].bboxes.detach().cpu().numpy(), "xyxy", "cxcywh")
            predicted_bboxes = postprocessed_output["pred_instances"].bboxes.detach().cpu().numpy()
            ious = iou_batch(predicted_bboxes, lbls_bboxes)
            matched_preds, matched_lbls = linear_assignment(ious, thresh=0.9)

            matches["extracted_features"] = i_extracted_features[matched_preds]
            matches["target_features"] = target_features[matched_preds].detach()

            matches["instances_id"] = postprocessed_output["gt_instances"].instances_id[matched_lbls]
            frames[postprocessed_output["img_id"]].append(matches)

        feat_loss = 0

        for frame_data in frames.values():
            assert len(frame_data) == 2
            frame_a = frame_data[0]
            frame_b = frame_data[1]

            instances_id_a = frame_a["instances_id"]
            instances_id_b = frame_b["instances_id"]

            mask_a = instances_id_a.unsqueeze(1) == instances_id_b.unsqueeze(0)
            idx_a, idx_b = torch.where(mask_a)

            extracted_a = frame_a["extracted_features"][idx_a]
            extracted_b = frame_b["extracted_features"][idx_b]
            target_a = frame_a["target_features"][idx_a]
            target_b = frame_b["target_features"][idx_b]

            feat_loss += self._loss(extracted_a, target_b)
            feat_loss += self._loss(extracted_b, target_a)

        return feat_loss / B, extracted_features

    def _flatten_predictions(self, preds: List[Tensor]):
        """Flattens the predictions from a list of tensors to a single
        tensor."""
        if len(preds) == 0:
            return None

        preds = [x.permute(0, 2, 3, 1).flatten(1, 2) for x in preds]
        return torch.cat(preds, dim=1)
