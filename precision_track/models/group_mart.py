from typing import NamedTuple, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from torch import Tensor

from precision_track.registry import MODELS
from precision_track.utils import PoseDataSample, load_checkpoint, parse_pose_metainfo, iou_batch
from precision_track.models.optimization.losses import focal_loss

from .modules.blocks.transformers import SelfAttention, TransformerMLP


class GMARTPredictions(NamedTuple):
    node_pred: Tensor
    node_emb: Tensor
    edge_probs: Tensor
    edge_class_probs: Tensor


class CrossNodeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.norm2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = TransformerMLP(config)

    def forward(self, x: Tensor, attn_bias: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x


@MODELS.register_module()
class GMART(BaseModel):
    def __init__(
        self,
        mart_config: dict,
        mart_checkpoint: str,
        metainfo: str,
        n_gnn_layers: int = 2,
        with_distance_prior: bool = True,
        with_vel_coherence: bool = True,
        with_vel_approach: bool = True,
        with_orientation_priors: bool = True,
        refine_nodes: bool = False,
        relationship_loss_weight: int = 1.0,
        classification_loss: int = 1.0,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(data_preprocessor, init_cfg)

        self.mart = MODELS.build(mart_config)

        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        self.n_group_classes = len(metainfo.get("actions", []))

        self.with_distance_prior = with_distance_prior
        self.with_vel_coherence = with_vel_coherence
        self.with_vel_approach = with_vel_approach
        self.with_orientation_priors = with_orientation_priors

        n_priors = (
            int(with_distance_prior)
            + int(with_vel_coherence)
            + int(with_vel_approach)
            + 2 * int(with_orientation_priors)  # orientations_alignment & orientations_valid
        )

        self.refine_nodes = refine_nodes

        n_embd = self.mart.n_embd
        gnn_config = self.mart.config.copy()
        gnn_config.causal = False

        self.prior_proj = nn.Linear(n_priors, 1, bias=False)
        self.gnn_layers = nn.ModuleList([CrossNodeAttention(gnn_config) for _ in range(n_gnn_layers)])
        self.edge_proj = nn.Sequential(nn.Linear(2 * n_embd, n_embd, bias=self.mart.config.bias), nn.GELU())
        self.bce_head = nn.Linear(n_embd, 1)
        self.ce_head = nn.Linear(n_embd, self.n_group_classes)
        if self.refine_nodes:
            self.node_head = nn.Sequential(
                TransformerMLP(gnn_config),
                nn.Linear(n_embd, self.mart.n_class, bias=gnn_config.bias),
            )

        self.apply(self._init_weights)
        load_checkpoint(self.mart, mart_checkpoint)

        self._prep_mart()

        self.relationship_loss_weight = max(int(relationship_loss_weight), 1.0)
        self.classification_loss = max(int(classification_loss), 1.0)

    def _prep_mart(self):
        self.mart.requires_grad_(False)
        self.mart.eval()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self._prep_mart()
        return self

    def forward(
        self,
        features: Tensor,
        poses: Tensor,
        dynamics: Tensor,
        node_labels: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        binary_labels: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
        distance_priors: Optional[Tensor] = None,
        vel_coherence: Optional[Tensor] = None,
        vel_approach: Optional[Tensor] = None,
        orientations_alignment: Optional[Tensor] = None,
        orientations_valid: Optional[Tensor] = None,
        data_samples: Optional[List[PoseDataSample]] = None,
        mode: Optional[str] = "tensor",
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor], dict]:
        if isinstance(features, list):
            features = torch.stack(features)
        if mode == "loss":
            return self.loss(
                features,
                poses,
                dynamics,
                node_labels,
                labels,
                binary_labels,
                valid_mask,
                distance_priors,
                vel_coherence,
                vel_approach,
                orientations_alignment,
                orientations_valid,
            )
        elif mode == "predict":
            return self.predict(
                features,
                poses,
                dynamics,
                valid_mask,
                distance_priors,
                vel_coherence,
                vel_approach,
                orientations_alignment,
                orientations_valid,
            )
        else:
            raise RuntimeError(f'Invalid mode "{mode}". Only supports loss and predict.')

    def _forward(
        self,
        batch_size: int,
        nb_subjects: int,
        nb_timesteps: int,
        features: Tensor,
        poses: Tensor,
        dynamics: Tensor,
        distance_priors: Optional[Tensor] = None,
        vel_coherence: Optional[Tensor] = None,
        vel_approach: Optional[Tensor] = None,
        orientations_alignment: Optional[Tensor] = None,
        orientations_valid: Optional[Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        with torch.no_grad():
            node_emb_full = self.mart._get_transformations(
                self.mart._get_projections(
                    features=features.reshape(batch_size * nb_subjects, nb_timesteps, -1),
                    poses=poses.reshape(batch_size * nb_subjects, nb_timesteps, -1),
                    dynamics=dynamics.reshape(batch_size * nb_subjects, nb_timesteps, -1),
                )
            )
            node_emb_last = node_emb_full[:, -1, :]
            if not self.refine_nodes:
                node_logits = self.mart.classification_head(node_emb_last)
                node_logits = node_logits.reshape(batch_size, nb_subjects, -1)

        node_emb = node_emb_last.reshape(batch_size, nb_subjects, -1)

        prior_list = []
        if self.with_distance_prior and distance_priors is not None:
            prior_list.append(distance_priors.unsqueeze(-1))
        if self.with_vel_coherence and vel_coherence is not None:
            prior_list.append(vel_coherence.unsqueeze(-1))
        if self.with_vel_approach and vel_approach is not None:
            prior_list.append(vel_approach.unsqueeze(-1))
        if self.with_orientation_priors and orientations_alignment is not None:
            prior_list.append(orientations_alignment.unsqueeze(-1))
            prior_list.append(orientations_valid.unsqueeze(-1))

        attn_bias = None
        if prior_list:
            priors = torch.cat(prior_list, dim=-1)
            attn_bias = self.prior_proj(priors).squeeze(-1)
            attn_bias = attn_bias.unsqueeze(1)

        x = node_emb
        for layer in self.gnn_layers:
            x = layer(x, attn_bias=attn_bias)

        if self.refine_nodes:
            node_logits = self.node_head(x)

        hi = x.unsqueeze(2).expand(-1, -1, nb_subjects, -1)
        hj = x.unsqueeze(1).expand(-1, nb_subjects, -1, -1)
        pairs = self.edge_proj(torch.cat([hi, hj], dim=-1))

        return node_logits, F.normalize(x, p=2, dim=-1), self.bce_head(pairs).squeeze(-1), self.ce_head(pairs)

    def loss(
        self,
        features: Tensor,
        poses: Tensor,
        dynamics: Tensor,
        node_labels: Optional[Tensor],
        labels: Tensor,
        binary_labels: Tensor,
        valid_mask: Tensor,
        distance_priors: Optional[Tensor] = None,
        vel_coherence: Optional[Tensor] = None,
        vel_approach: Optional[Tensor] = None,
        orientations_alignment: Optional[Tensor] = None,
        orientations_valid: Optional[Tensor] = None,
    ) -> dict:
        B, N, T = features.shape[:3]
        node_logits, _, bce_logits, ce_logits = self._forward(
            B,
            N,
            T,
            features,
            poses,
            dynamics,
            distance_priors,
            vel_coherence,
            vel_approach,
            orientations_alignment,
            orientations_valid,
        )

        pair_valid = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        diag_mask = ~torch.eye(N, dtype=torch.bool, device=bce_logits.device).unsqueeze(0).expand(B, -1, -1)
        pair_valid = pair_valid & diag_mask

        relationship_loss = focal_loss(
            bce_logits[pair_valid],
            binary_labels[pair_valid].float(),
        )

        ce_mask = pair_valid & (labels >= 0)
        classification_loss = F.cross_entropy(
            ce_logits[ce_mask],
            labels[ce_mask],
        )

        losses = dict(relationship_loss=relationship_loss * self.relationship_loss_weight, classification_loss=classification_loss)

        if self.refine_nodes:
            node_classification_loss = F.cross_entropy(
                node_logits[valid_mask],
                node_labels[:, :, -1][valid_mask],
            )
            losses["node_classification_loss"] = node_classification_loss

        return losses

    def predict(
        self,
        features: Tensor,
        poses: Tensor,
        dynamics: Tensor,
        valid_mask: Optional[Tensor] = None,
        distance_priors: Optional[Tensor] = None,
        vel_coherence: Optional[Tensor] = None,
        vel_approach: Optional[Tensor] = None,
        orientations_alignment: Optional[Tensor] = None,
        orientations_valid: Optional[Tensor] = None,
        data_samples=None,
    ) -> GMARTPredictions:
        B, N, T = features.shape[:3]
        node_logits, node_emb, edge_logits, ce_logits = self._forward(
            B,
            N,
            T,
            features,
            poses,
            dynamics,
            distance_priors,
            vel_coherence,
            vel_approach,
            orientations_alignment,
            orientations_valid,
        )

        node_pred = F.softmax(node_logits, dim=-1)
        edge_probs = torch.sigmoid(edge_logits)
        edge_class_probs = F.softmax(ce_logits, dim=-1)

        diag = torch.eye(N, dtype=torch.bool, device=edge_probs.device).unsqueeze(0)
        edge_probs = edge_probs.masked_fill(diag, 0.0)
        edge_class_probs = edge_class_probs.masked_fill(diag.unsqueeze(-1), 0.0)

        if valid_mask is not None:
            pair_valid = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
            node_pred = node_pred * valid_mask.unsqueeze(-1)
            node_emb = node_emb * valid_mask.unsqueeze(-1)
            edge_probs = edge_probs * pair_valid
            edge_class_probs = edge_class_probs * pair_valid.unsqueeze(-1)

        return GMARTPredictions(node_pred, node_emb, edge_probs, edge_class_probs)


@MODELS.register_module()
class RelationshipDetectionBaselineModel(BaseModel):
    def __init__(self, metainfo: str, data_preprocessor=None, *args, **kwargs):
        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        self.n_group_classes = len(metainfo.get("actions", []))
        super().__init__(data_preprocessor=data_preprocessor)

    def forward(
        self,
        data_samples: dict,
        mode: Optional[str] = "tensor",
        *args,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor], dict]:
        if mode in ("loss", "predict"):
            return self.predict(data_samples)
        raise RuntimeError(f'Invalid mode "{mode}". Only supports loss and predict.')

    def predict(self, data_samples: dict) -> GMARTPredictions:
        all_edge_probs = []
        for query_idxs, bboxes in zip(data_samples["query_idxs"], data_samples["bboxes"]):
            N = bboxes.shape[0]
            edge_probs = torch.zeros(N, N, device=bboxes.device)
            if len(query_idxs) > 0:
                cx, cy = bboxes[:, 0], bboxes[:, 1]
                dist = ((cx.unsqueeze(0) - cx.unsqueeze(1)) ** 2 + (cy.unsqueeze(0) - cy.unsqueeze(1)) ** 2).sqrt()
                dist.fill_diagonal_(float("inf"))
                for i in query_idxs:
                    j = dist[i].argmin()
                    edge_probs[i, j] = 1.0
                    edge_probs[j, i] = 1.0
            all_edge_probs.append(edge_probs)

        N_max = max(e.shape[0] for e in all_edge_probs)
        B = len(all_edge_probs)
        device = all_edge_probs[0].device
        edge_probs_batch = torch.zeros(B, N_max, N_max, device=device)
        for b, ep in enumerate(all_edge_probs):
            N = ep.shape[0]
            edge_probs_batch[b, :N, :N] = ep

        # This is only a baseline for the relationship detection.
        node_pred = torch.zeros(B, N_max, 1, device=device)
        node_emb = torch.zeros(B, N_max, 1, device=device)
        edge_class_probs = torch.rand(B, N_max, N_max, self.n_group_classes, device=device)

        return [GMARTPredictions(node_pred, node_emb, edge_probs_batch, edge_class_probs)]
