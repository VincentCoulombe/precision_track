from typing import NamedTuple, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from torch import Tensor
from addict import Dict

from precision_track.registry import MODELS
from precision_track.utils import PoseDataSample, load_checkpoint, parse_pose_metainfo
from precision_track.models.optimization.losses import weighted_bce_loss

from .modules.blocks.transformers import TransformerMLP


class GMARTPredictions(NamedTuple):
    node_pred: Tensor
    node_emb: Tensor
    edge_probs: Tensor
    edge_class_probs: Tensor


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
        refine_nodes: bool = False,
        radius: float = 5.0,
        relationship_loss_weight: float = 1.0,
        classification_loss_weight: float = 1.0,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(data_preprocessor, init_cfg)

        self.mart = MODELS.build(mart_config)

        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        self.n_group_classes = len(metainfo.get("actions", []))
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

        self.refine_nodes = refine_nodes
        self.radius = radius

        n_embd = self.mart.n_embd

        self.edge_mlp = nn.Sequential(
            nn.Linear(n_priors, n_embd // 2, bias=self.mart.config.bias),
            nn.LayerNorm(n_embd // 2, bias=self.mart.config.bias),  # stabilise after first projection
            nn.GELU(),
            nn.Linear(n_embd // 2, n_embd // 2, bias=self.mart.config.bias),
            nn.LayerNorm(n_embd // 2, bias=self.mart.config.bias),  # stabilise after second projection
            nn.GELU(),
            nn.Linear(n_embd // 2, n_embd // 2, bias=self.mart.config.bias),
            nn.LayerNorm(n_embd // 2, bias=self.mart.config.bias),  # stabilise after second projection
            nn.GELU(),
            nn.Linear(n_embd // 2, 1, bias=self.mart.config.bias),  # collapse to scalar logit
        )

        self.graph_conv = nn.Linear(n_embd, n_embd, bias=self.mart.config.bias)
        self.norm = nn.LayerNorm(n_embd)

        self.graph_conv_row = nn.Linear(n_embd, n_embd // 2, bias=self.mart.config.bias)
        self.graph_conv_col = nn.Linear(n_embd, n_embd // 2, bias=self.mart.config.bias)
        self.norm_prime = nn.LayerNorm(n_embd)

        self.rel_mlp = nn.Sequential(
            nn.Linear(3 * n_embd, n_embd, bias=self.mart.config.bias),
            nn.LayerNorm(n_embd, bias=self.mart.config.bias),  # stabilise after first projection
            nn.GELU(),
            nn.Linear(n_embd, n_embd, bias=self.mart.config.bias),
            nn.LayerNorm(n_embd, bias=self.mart.config.bias),  # stabilise after second projection
            nn.GELU(),
        )

        self.bce_head = nn.Linear(n_embd, 1)
        self.ce_head = nn.Sequential(
            nn.Linear(2 * n_embd, n_embd, bias=self.mart.config.bias),
            nn.LayerNorm(n_embd, bias=self.mart.config.bias),
            nn.GELU(),
            nn.Linear(n_embd, self.n_group_classes, bias=self.mart.config.bias),
        )

        if self.refine_nodes:
            self.node_head = nn.Sequential(
                nn.LayerNorm(self.mart.config.n_embd, bias=self.mart.config.bias),
                TransformerMLP(self.mart.config),
                nn.Linear(n_embd, self.mart.n_class, bias=self.mart.config.bias),
            )

        self.apply(self._init_weights)
        load_checkpoint(self.mart, mart_checkpoint)

        self._prep_mart()

        self.relationship_loss_weight = max(relationship_loss_weight, 1.0)
        self.classification_loss_weight = max(classification_loss_weight, 1.0)

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
        distance_priors: Tensor,  # (N, S, S)  pairwise distances
        vel_coherence: Optional[Tensor] = None,  # (N, S, S)
        vel_approach: Optional[Tensor] = None,  # (N, S, S)
        orientations_alignment: Optional[Tensor] = None,  # (N, S, S)
        orientations_valid: Optional[Tensor] = None,  # (N, S, S)
        keypoint_priors: Optional[Tensor] = None,  # (N, S, S, K)
    ) -> Tuple[torch.Tensor]:

        # ── 1. MART node embeddings (frozen) ─────────────────────────────
        with torch.no_grad():
            node_emb_full = self.mart._get_transformations(
                self.mart._get_projections(
                    features=features.reshape(batch_size * nb_subjects, nb_timesteps, -1),
                    poses=poses.reshape(batch_size * nb_subjects, nb_timesteps, -1),
                    dynamics=dynamics.reshape(batch_size * nb_subjects, nb_timesteps, -1),
                )
            )
            node_emb_last = node_emb_full[:, -1, :]
            mart_node_logits = self.mart.classification_head(node_emb_last).reshape(batch_size, nb_subjects, -1)

        node_logits = mart_node_logits
        node_emb = node_emb_last.reshape(batch_size, nb_subjects, -1)  # (N, S, n_embd)

        # ── 2. Stack geometric/kinematic priors → edge feature tensor ────
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

        priors = torch.cat(prior_list, dim=-1) if prior_list else None  # (N, S, S, n_priors)

        # ── 3. EdgeMLP → learned proximity adjacency A_hat ───────────────

        # (a) MLP maps edge feature vectors to scalar logits
        logits = self.edge_mlp(priors).squeeze(-1)  # (N, S, S)

        # (b) Distance mask — distance_priors IS the (N, S, S) pairwise
        #     distance matrix, so use it directly instead of recomputing.
        #     FIX: was incorrectly calling torch.norm on distance_priors itself.
        mask = distance_priors < self.radius  # (N, S, S)
        logits = logits.masked_fill(~mask, -1e3)

        # (c) Self-loop injection — every node must always attend to itself.
        #     FIX: was referencing undefined `pos`; use nb_subjects + node_emb.device.
        eye = torch.eye(nb_subjects, device=node_emb.device).unsqueeze(0)  # (1, S, S)
        logits = logits + eye * 1e3

        # (d) Row-wise softmax → normalised adjacency A_hat
        A_hat = F.softmax(logits, dim=-1)  # (N, S, S)

        # ── 4. First GraphConv — proximity-based spatial aggregation ─────
        #
        #   A_hat @ node_emb   : (N,S,S) @ (N,S,F) → (N,S,F)
        #     each subject aggregates a weighted average of its neighbours'
        #     feature vectors according to A_hat.
        #   self.graph_conv    : linear projection F → C
        #   LayerNorm + GELU   : post-aggregation normalisation
        #
        x_agg = A_hat @ node_emb  # (N, S, F)
        h = self.graph_conv(x_agg)  # (N, S, C)
        h = self.norm(h)  # (N, S, C)
        h = F.gelu(h)  # (N, S, C)

        # ── 5. Directed pairwise features → interaction logits ───────────
        #
        #   For each ordered pair (i→j) we build a vector that encodes:
        #     h[i]        : initiator's spatially-refined representation
        #     h[j]        : receiver's  spatially-refined representation
        #     h[i] - h[j] : role asymmetry — swapping i↔j flips the sign,
        #                   so the MLP can distinguish "A grooms B" from
        #                   "B grooms A" purely from the node features.
        #
        #   FIX: was concatenating h with itself on dim=-1, yielding (N,S,2C)
        #        with no cross-subject information. Correct shape is (N,S,S,3C).
        #
        h_i = h.unsqueeze(2).expand(-1, -1, nb_subjects, -1)  # (N, S, S, C)
        h_j = h.unsqueeze(1).expand(-1, nb_subjects, -1, -1)  # (N, S, S, C)
        relation_map = torch.cat([h_i, h_j, h_i - h_j], dim=-1)  # (N, S, S, 3C)

        # InteractionMLP + BCE head → directed interaction logits
        # bce_logits[n, i, j] = raw score for "subject i is acting on subject j"
        relationally_encoded = self.rel_mlp(relation_map)  # (N, S, S, C')
        bce_logits = self.bce_head(relationally_encoded).squeeze(-1)  # (N, S, S)

        # Mask diagonal — a subject cannot interact with itself
        diag_mask = eye.bool()  # (1, S, S)
        bce_logits = bce_logits.masked_fill(diag_mask, -1e3)

        # ── 6. Directed dual GCN on the interaction graph (Option C) ─────
        #
        #   I = sigmoid(bce_logits) : soft directed interaction matrix (N, S, S)
        #
        #   We run two separate GraphConvs:
        #
        #   Row-normalised  I_row[i,:] sums to 1
        #   → h_row[i] aggregates from subjects i IS ACTING UPON (initiator view)
        #   → captures the social context of being an initiator
        #
        #   Col-normalised  I_col[:,j] sums to 1
        #   → h_col[j] aggregates from subjects ACTING UPON j  (receiver view)
        #   → captures the social context of being a receiver
        #
        #   h_prime = concat(h_row, h_col) : (N, S, 2*C')
        #   Each node's representation now jointly encodes both its initiator
        #   and receiver roles in the current frame's interaction structure.
        #
        I = torch.sigmoid(bce_logits)  # (N, S, S)
        eps = 1e-6

        # Row normalisation (initiator view)
        I_row = I / I.sum(dim=-1, keepdim=True).clamp(min=eps)  # (N, S, S)
        # Col normalisation (receiver view)
        I_col = I / I.sum(dim=-2, keepdim=True).clamp(min=eps)  # (N, S, S)

        # Two independent GraphConvs share the same input h but use
        # different normalised adjacencies
        h_row = F.gelu(self.graph_conv_row(I_row @ h))  # (N, S, C')  initiator
        h_col = F.gelu(self.graph_conv_col(I_col @ h))  # (N, S, C')  receiver

        h_prime = torch.cat([h_row, h_col], dim=-1)  # (N, S, 2*C')
        h_prime = self.norm_prime(h_prime)  # (N, S, 2*C')

        # ── 7. GAR — per-pair classification using dual GCN embeddings ────
        #
        #   For each directed pair (i→j), build a pair feature vector by
        #   averaging the dual-GCN embeddings of both subjects.  Additive
        #   pooling keeps the input dim at 2*C' and ensures GAR gradients
        #   flow back through h_prime → I → bce_logits, giving the
        #   interaction head auxiliary supervision from the GAR task.
        #
        h_prime_i = h_prime.unsqueeze(2).expand(-1, -1, nb_subjects, -1)  # (N, S, S, 2*C')
        h_prime_j = h_prime.unsqueeze(1).expand(-1, nb_subjects, -1, -1)  # (N, S, S, 2*C')
        pair_ce_input = torch.cat((h_prime_i, h_prime_j), dim=-1)
        ce_logits = self.ce_head(pair_ce_input)  # (N, S, S, n_classes)

        return (
            node_logits,
            mart_node_logits,
            F.normalize(node_emb, p=2, dim=-1),
            bce_logits,  # (N, S, S)  — raw logits, apply sigmoid + focal loss outside
            ce_logits,  # (N, S, S, n_classes)
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
        node_logits, mart_node_logits, _, bce_logits, ce_logits = self._forward(
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

        relationship_loss = weighted_bce_loss(
            bce_logits[pair_valid],
            binary_labels[pair_valid].float(),
            loss_weight=self.relationship_loss_weight,
        )

        ce_mask = pair_valid & (labels >= 0)
        classification_loss = F.cross_entropy(
            ce_logits[ce_mask],
            labels[ce_mask],
        )

        losses = dict(relationship_loss=relationship_loss, classification_loss=classification_loss * self.classification_loss_weight)

        if self.refine_nodes:
            has_relationship = (binary_labels == 1).any(dim=-1) | (binary_labels == 1).any(dim=-2)  # (B, N)
            refine_mask = valid_mask & has_relationship
            node_classification_loss = F.cross_entropy(
                node_logits[refine_mask],
                node_labels[:, :, -1][refine_mask],
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
        keypoint_priors: Optional[Tensor] = None,
        data_samples=None,
    ) -> GMARTPredictions:
        B, N, T = features.shape[:3]
        node_logits, mart_node_logits, node_emb, edge_logits, ce_logits = self._forward(
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

        node_logits = mart_node_logits
        if self.refine_nodes:
            has_relationship = edge_probs.gt(0.5).any(dim=-1) | edge_probs.gt(0.5).any(dim=-2)
            node_ce_logits = ce_logits.mean(dim=2).values
            node_logits = torch.where(has_relationship.unsqueeze(-1), node_ce_logits, mart_node_logits)

        node_pred = F.softmax(node_logits, dim=-1)
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
