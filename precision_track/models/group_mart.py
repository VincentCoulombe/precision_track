from typing import NamedTuple, Optional, Tuple, Union, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from torch import Tensor
from addict import Dict

from precision_track.registry import MODELS
from precision_track.utils import PoseDataSample, load_checkpoint, parse_pose_metainfo
from precision_track.models.optimization.losses import focal_loss, focal_loss_multiclass

from .modules.blocks.transformers import TransformerMLP, TransformerBlock


class GMARTPredictions(NamedTuple):
    class_logits: Tensor
    action_embeddings: Tensor
    interaction_logits: Tensor
    social_logits: Tensor


@MODELS.register_module()
class GMART(BaseModel):
    def __init__(
        self,
        mart_config: dict,
        mart_checkpoint: str,
        metainfo: str,
        with_vel_coherence: bool = True,
        with_vel_approach: bool = True,
        with_orientation_priors: bool = True,
        with_keypoint_priors: bool = False,
        radius: float = 10.0,
        relationship_loss_weight: float = 1.0,
        classification_loss_weight: float = 1.0,
        classification_alpha: Optional[List[float]] = None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(data_preprocessor, init_cfg)

        self.mart = MODELS.build(mart_config)

        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        self.n_group_classes = len(metainfo.get("social_actions", [])) + 1  # Account for added null class
        n_keypoint_priors = len(metainfo.get("distance_keypoint_pairs", []))

        self.with_vel_coherence = with_vel_coherence
        self.with_vel_approach = with_vel_approach
        self.with_orientation_priors = with_orientation_priors
        self.n_keypoint_priors = n_keypoint_priors

        n_priors = (
            1
            + int(with_vel_coherence)
            + int(with_vel_approach)
            + 2 * int(with_orientation_priors)  # orientations_alignment & orientations_valid
            + n_keypoint_priors * int(with_keypoint_priors)
        )

        self.radius = radius

        n_embd = self.mart.n_embd

        self.edge_mlp = nn.Sequential(
            nn.Linear(n_priors, n_embd // 2, bias=self.mart.config.bias),
            nn.LayerNorm(n_embd // 2, bias=self.mart.config.bias),
            nn.GELU(),
            nn.Linear(n_embd // 2, n_embd // 2, bias=self.mart.config.bias),
            nn.LayerNorm(n_embd // 2, bias=self.mart.config.bias),
            nn.GELU(),
        )

        self.edge_proj = nn.Linear(n_embd // 2, 1, bias=self.mart.config.bias)

        encoder_config = self.mart.config.copy()
        encoder_config.causal = False
        self.encoder = nn.ModuleList([TransformerBlock(encoder_config) for _ in range(encoder_config.n_block)])

        self.bce_head = nn.Sequential(
            nn.Linear(2 * n_embd + n_embd // 2, n_embd, bias=self.mart.config.bias),
            nn.LayerNorm(n_embd, bias=self.mart.config.bias),
            nn.GELU(),
            nn.Linear(n_embd, n_embd // 2, bias=self.mart.config.bias),
            nn.LayerNorm(n_embd // 2, bias=self.mart.config.bias),
            nn.GELU(),
            nn.Linear(n_embd // 2, n_embd // 4, bias=self.mart.config.bias),
            nn.LayerNorm(n_embd // 4, bias=self.mart.config.bias),
            nn.GELU(),
            nn.Linear(n_embd // 4, 1, bias=self.mart.config.bias),
        )

        self.ce_head = nn.Sequential(
            nn.Linear(n_embd, n_embd // 2, bias=self.mart.config.bias),
            nn.LayerNorm(n_embd // 2, bias=self.mart.config.bias),
            nn.GELU(),
            nn.Linear(n_embd // 2, n_embd // 2, bias=self.mart.config.bias),
            nn.LayerNorm(n_embd // 2, bias=self.mart.config.bias),
            nn.GELU(),
            nn.Linear(n_embd // 2, self.n_group_classes),
        )

        self.apply(self._init_weights)
        load_checkpoint(self.mart, mart_checkpoint)

        self._prep_mart()

        self.relationship_loss_weight = max(relationship_loss_weight, 1.0)
        self.classification_loss_weight = classification_loss_weight
        self.classification_alpha = torch.tensor(classification_alpha, dtype=torch.float32) if classification_alpha is not None else None

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
        keypoint_priors: Optional[Tensor] = None,
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
                keypoint_priors,
            )
        elif mode == "predict":
            return self.predict(
                inputs=(
                    features,
                    poses,
                    dynamics,
                    valid_mask,
                    distance_priors,
                    vel_coherence,
                    vel_approach,
                    orientations_alignment,
                    orientations_valid,
                    keypoint_priors,
                ),
                data_samples=data_samples,
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
        distance_priors: Tensor,
        vel_coherence: Optional[Tensor] = None,
        vel_approach: Optional[Tensor] = None,
        orientations_alignment: Optional[Tensor] = None,
        orientations_valid: Optional[Tensor] = None,
        keypoint_priors: Optional[Tensor] = None,
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
            node_logits = self.mart.classification_head(node_emb_last).reshape(batch_size, nb_subjects, -1)

        node_emb = node_emb_last.reshape(batch_size, nb_subjects, -1)

        prior_list = [distance_priors.unsqueeze(-1)]
        if self.with_vel_coherence and vel_coherence is not None:
            prior_list.append(vel_coherence.unsqueeze(-1))
        if self.with_vel_approach and vel_approach is not None:
            prior_list.append(vel_approach.unsqueeze(-1))
        if self.with_orientation_priors and orientations_alignment is not None:
            prior_list.append(orientations_alignment.unsqueeze(-1))
            prior_list.append(orientations_valid.unsqueeze(-1))
        if self.n_keypoint_priors > 0 and keypoint_priors is not None:
            prior_list.append(keypoint_priors)

        priors = torch.cat(prior_list, dim=-1) if prior_list else None

        projected_edges = self.edge_mlp(priors).squeeze(-1)
        edge_weights = self.edge_proj(projected_edges).squeeze(-1)
        edge_weights = F.softmin(edge_weights, dim=-1)
        edge_weights = edge_weights.unsqueeze(1).log()
        for encoder_layer in self.encoder:
            node_emb = encoder_layer(node_emb, attn_mask=edge_weights)

        h_i = node_emb.unsqueeze(2).expand(-1, -1, nb_subjects, -1)
        h_j = node_emb.unsqueeze(1).expand(-1, nb_subjects, -1, -1)
        graph = torch.cat([h_i, projected_edges, h_j], dim=-1)

        bce_logits = self.bce_head(graph).squeeze(-1)

        ce_logits = self.ce_head(node_emb)

        return (
            node_logits,
            F.normalize(node_emb, p=2, dim=-1),
            bce_logits,
            ce_logits,
        )

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
        keypoint_priors: Optional[Tensor] = None,
    ) -> dict:
        B, N, T = features.shape[:3]
        _, _, bce_logits, ce_logits = self._forward(
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
            keypoint_priors,
        )

        pair_valid = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        diag_mask = ~torch.eye(N, dtype=torch.bool, device=bce_logits.device).unsqueeze(0).expand(B, -1, -1)
        pair_valid = pair_valid & diag_mask

        relationship_loss = focal_loss(
            bce_logits[pair_valid],
            binary_labels[pair_valid].float(),
            loss_weight=self.relationship_loss_weight,
        )

        ce_mask = labels >= 0
        classification_loss = focal_loss_multiclass(
            ce_logits[ce_mask],
            labels[ce_mask],
            alpha=self.classification_alpha.to(ce_logits.device) if self.classification_alpha is not None else None,
            loss_weight=self.classification_loss_weight,
        )

        losses = dict(relationship_loss=relationship_loss, classification_loss=classification_loss)

        return losses

    def predict(self, inputs: Tuple[Tensor], data_samples: List[PoseDataSample] = None) -> GMARTPredictions:
        (
            features,
            poses,
            dynamics,
            valid_mask,
            distance_priors,
            vel_coherence,
            vel_approach,
            orientations_alignment,
            orientations_valid,
            keypoint_priors,
        ) = inputs
        if features.ndim == 3:
            features = features.unsqueeze(0)
            if distance_priors is not None:
                distance_priors = distance_priors.unsqueeze(0)
            if vel_coherence is not None:
                vel_coherence = vel_coherence.unsqueeze(0)
            if vel_approach is not None:
                vel_approach = vel_approach.unsqueeze(0)
            if orientations_alignment is not None:
                orientations_alignment = orientations_alignment.unsqueeze(0)
            if orientations_valid is not None:
                orientations_valid = orientations_valid.unsqueeze(0)
            if keypoint_priors is not None:
                keypoint_priors = keypoint_priors.unsqueeze(0)
        B, N, T, _ = features.shape
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
            keypoint_priors,
        )

        edge_probs = torch.sigmoid(edge_logits)
        node_pred = F.softmax(node_logits, dim=-1)
        social_pred = F.softmax(ce_logits, dim=-1)

        diag = torch.eye(N, dtype=torch.bool, device=edge_probs.device).unsqueeze(0)
        edge_probs = edge_probs.masked_fill(diag, 0.0)

        if valid_mask is not None:
            pair_valid = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
            node_pred = node_pred * valid_mask.unsqueeze(-1)
            node_emb = node_emb * valid_mask.unsqueeze(-1)
            edge_probs = edge_probs * pair_valid
            social_pred = social_pred * valid_mask.unsqueeze(-1)

        return GMARTPredictions(node_pred, node_emb, edge_probs, social_pred)


@MODELS.register_module()
class RelationshipDetectionBaselineModel(BaseModel):
    def __init__(self, metainfo: str, data_preprocessor=None, *args, **kwargs):
        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        self.n_group_classes = len(metainfo.get("social_actions", [])) + 1
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

        node_pred = torch.zeros(B, N_max, 1, device=device)
        node_emb = torch.zeros(B, N_max, 1, device=device)

        return [GMARTPredictions(node_pred, node_emb, edge_probs_batch, node_pred)]


@MODELS.register_module()
class RelationshipDetectionPoseBaselineModel(RelationshipDetectionBaselineModel):
    def __init__(self, mart_config: dict, mart_checkpoint: str, metainfo: str, data_preprocessor=None, *args, **kwargs):
        super().__init__(metainfo=metainfo, data_preprocessor=data_preprocessor)
        self.mart = MODELS.build(mart_config)
        load_checkpoint(self.mart, mart_checkpoint)
        self.group_classes = self.mart.metainfo.get("social_actions", [])
        classes = self.mart.metainfo.get("actions", [])
        actions_of_interest = []
        for i, cls_ in enumerate(classes):
            if cls_ in self.group_classes:
                actions_of_interest.append(i)
        self.actions_of_interest = torch.tensor(actions_of_interest, dtype=torch.int64)

        self.action2social = dict()
        for i, social_action in enumerate(self.group_classes):
            idx = np.where(np.array(classes) == social_action)[0]
            if idx:
                self.action2social[idx.item()] = i + 1

    def forward(
        self,
        features: Tensor,
        poses: Tensor,
        dynamics: Tensor,
        keypoint_priors: Tensor,
        distance_priors: Tensor,
        mode: Optional[str] = "tensor",
        *args,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor], dict]:
        if mode in ("loss", "predict"):
            return self.predict((features, poses, dynamics, keypoint_priors, distance_priors))
        raise RuntimeError(f'Invalid mode "{mode}". Only supports loss and predict.')

    def predict(self, inputs: Tuple[Tensor], data_samples: List[PoseDataSample] = None) -> GMARTPredictions:
        features, poses, dynamics, keypoint_priors, _ = inputs
        if len(features.shape) == 3:
            N, T, _ = features.shape
            B = 1
        else:
            B, N, T, _ = features.shape
        device = features.device
        with torch.no_grad():
            probs, _ = self.mart.predict(
                inputs=(
                    features.reshape(B * N, T, -1),
                    poses.reshape(B * N, T, -1),
                    dynamics.reshape(B * N, T, -1),
                )
            )

        preds = torch.argmax(probs, dim=-1).to("cpu")
        query_idx = torch.where(preds == self.actions_of_interest)[0].to(device)
        edge_probs = torch.zeros(B, N, N, device=device)
        diag = torch.eye(N, dtype=torch.bool, device=device)
        keypoint_priors = keypoint_priors.masked_fill(diag.unsqueeze(0).unsqueeze(-1), float("inf"))
        if len(query_idx) > 0:
            keypoint_priors_of_interest = keypoint_priors[:, query_idx, ...]
            rel_idx = keypoint_priors_of_interest.min(dim=-1)[0].argmin(dim=-1)

            if query_idx.dim() == 1:
                query_idx = query_idx.unsqueeze(0)
            for batch in range(B):
                for b_rel_idx, b_query_idx in zip(rel_idx[batch], query_idx[batch]):
                    edge_probs[batch, b_query_idx, b_rel_idx] = 1

        node_emb = torch.zeros(B, N, 1, device=device)

        social_probs = torch.zeros((B, N, len(self.group_classes) + 1), dtype=torch.float32, device=probs.device)
        preds = preds.view(B, N)
        for b in range(B):
            for i, pred in enumerate(preds[b]):
                social_idx = self.action2social.get(pred.item(), 0)
                social_probs[b, i, social_idx] = 1.0

        return [GMARTPredictions(probs.view(B, N, -1), node_emb, edge_probs, social_probs)]
