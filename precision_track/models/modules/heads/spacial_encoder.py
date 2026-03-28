from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import Config
from mmengine.model import BaseModel
from torch import Tensor

from precision_track.models.modules.transformers.blocks import TransformerBlock, TransformerMLP
from precision_track.registry import MODELS
from precision_track.utils import PoseDataSample, reformat


class SpatialRBFAttentionMask(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_rbf = int(config.num_rbf)
        self.eps = 1e-6
        self.max_rel_dist = 16

        mu = torch.linspace(0, self.max_rel_dist, self.num_rbf)
        sigma = torch.full((self.num_rbf,), 0.5)

        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)

        self.Wb = nn.Parameter(torch.zeros(config.n_head, self.num_rbf))
        nn.init.normal_(self.Wb, mean=0.0, std=0.02)

        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, bboxes):
        """
        Args:
            bboxes (torch.Tensor): Bounding Boxes of the cxcywh format

        Returns:
            torch.Tensor: The spatial RBF attention biases
        """
        if bboxes.ndim == 2:
            bboxes = bboxes.unsqueeze(0)

        centers = bboxes[..., :2]
        diff = centers[:, :, None, :] - centers[:, None, :, :]
        euc_dist = diff.norm(p=2, dim=-1)

        widths = bboxes[..., 2]
        heights = bboxes[..., 3]
        sizes = torch.sqrt(widths**2 + heights**2)
        size_norm = 0.5 * (sizes[:, :, None] + sizes[:, None, :]) + self.eps
        d_ij = (euc_dist / size_norm).clamp(max=self.max_rel_dist)

        phi = torch.exp(-0.5 * ((d_ij.unsqueeze(-1) - self.mu) / self.sigma) ** 2)

        bias = torch.einsum("bijk,hk->bijh", phi, self.Wb)
        return (self.beta * bias).permute(0, 3, 1, 2).contiguous()


class FlexibleRBFAttentionMask(nn.Module):
    # TODO Tester!
    # TODO 1) doit être capable de donner la même chose que SpatialRBFAttentionMask si j'utilise juste la distance
    # TODO 2) Comprendre les autres métriques (Noter dans Canva) + refactor velocities (pas besoin des recalculer) + s'assurer que tout est OK!
    """
    Flexible spatio-temporal RBF attention bias.

    Supports any combination of:
      - d: normalized pairwise distance
      - a: approach speed (i -> j) along line of centers
      - f: velocity alignment (cosine similarity of v_i and v_j)

    Output:
      bias: (B, n_head, N, N) to be added to attention logits.
    """

    def __init__(self, config):
        super().__init__()

        self.n_head = int(config.n_head)
        self.eps = 1e-6

        # Which relational features to use
        self.use_dist = getattr(config, "use_dist", True)
        self.use_approach = getattr(config, "use_approach", False)
        self.use_align = getattr(config, "use_align", False)

        # RBF configuration
        self.num_rbf_dist = int(getattr(config, "num_rbf_dist", getattr(config, "num_rbf", 8)))
        self.num_rbf_approach = int(getattr(config, "num_rbf_approach", getattr(config, "num_rbf", 8)))
        self.num_rbf_align = int(getattr(config, "num_rbf_align", getattr(config, "num_rbf", 8)))

        self.max_rel_dist = float(getattr(config, "max_rel_dist", 16.0))
        # Max absolute approach speed before clamping (in pixels / normed units)
        self.max_rel_vel = float(getattr(config, "max_rel_vel", 1.0))

        # --- RBF banks ---
        def make_rbf_bank(num_rbf, low, high, sigma):
            if num_rbf <= 0:
                return None, None
            mu = torch.linspace(low, high, num_rbf)
            sigma = torch.full((num_rbf,), sigma)
            return mu, sigma

        # Distance RBF: d in [0, max_rel_dist]
        if self.use_dist:
            mu_d, sigma_d = make_rbf_bank(
                self.num_rbf_dist,
                low=0.0,
                high=self.max_rel_dist,
                sigma=getattr(config, "sigma_dist", 0.5),
            )
            self.register_buffer("mu_dist", mu_d)
            self.register_buffer("sigma_dist", sigma_d)

        # Approach RBF: a_normalized in [-1, 1]
        if self.use_approach:
            mu_a, sigma_a = make_rbf_bank(
                self.num_rbf_approach,
                low=-1.0,
                high=1.0,
                sigma=getattr(config, "sigma_approach", 0.25),
            )
            self.register_buffer("mu_approach", mu_a)
            self.register_buffer("sigma_approach", sigma_a)

        # Alignment RBF: f in [-1, 1]
        if self.use_align:
            mu_f, sigma_f = make_rbf_bank(
                self.num_rbf_align,
                low=-1.0,
                high=1.0,
                sigma=getattr(config, "sigma_align", 0.25),
            )
            self.register_buffer("mu_align", mu_f)
            self.register_buffer("sigma_align", sigma_f)

        # Total RBF feature dimension for MLP input
        in_dim = 0
        if self.use_dist:
            in_dim += self.num_rbf_dist
        if self.use_approach:
            in_dim += self.num_rbf_approach
        if self.use_align:
            in_dim += self.num_rbf_align

        if in_dim == 0:
            raise ValueError("At least one of use_dist/use_approach/use_align must be True.")

        hidden_dim = int(getattr(config, "rbf_mlp_hidden_dim", 64))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.n_head),
        )

        # Global scale for the whole bias block
        self.beta = nn.Parameter(torch.tensor(0.1))

    def _rbf(self, x, mu, sigma):
        """
        x: (B, N, N)
        mu, sigma: (M,)
        returns: (B, N, N, M)
        """
        x = x.unsqueeze(-1)  # (B, N, N, 1)
        return torch.exp(-0.5 * ((x - mu) / (sigma + self.eps)) ** 2)

    def _compute_distance(self, bboxes):
        """
        bboxes: (B, N, 4) in cx, cy, w, h
        returns: d_ij: (B, N, N)
        """
        centers = bboxes[..., :2]  # (B, N, 2)
        widths = bboxes[..., 2]
        heights = bboxes[..., 3]
        sizes = torch.sqrt(widths**2 + heights**2)  # (B, N)

        # Pairwise center distance
        ci = centers[:, :, None, :]  # (B, N, 1, 2)
        cj = centers[:, None, :, :]  # (B, 1, N, 2)
        diff = ci - cj  # (B, N, N, 2)
        euc_dist = diff.norm(p=2, dim=-1)  # (B, N, N)

        # Size normalization (avg diag)
        si = sizes[:, :, None]  # (B, N, 1)
        sj = sizes[:, None, :]  # (B, 1, N)
        size_norm = 0.5 * (si + sj) + self.eps  # (B, N, N)

        d_ij = (euc_dist / size_norm).clamp(max=self.max_rel_dist)
        return d_ij, centers

    def _compute_velocity_features(self, centers, prev_bboxes=None, velocities=None):
        """
        centers: (B, N, 2)
        prev_bboxes: optional (B, N, 4) or (N, 4)
        velocities: optional (B, N, 2) or (N, 2)

        returns:
          velocities: (B, N, 2) or None
        """
        if velocities is not None:
            if velocities.ndim == 2:
                velocities = velocities.unsqueeze(0)
            return velocities

        if prev_bboxes is not None:
            if prev_bboxes.ndim == 2:
                prev_bboxes = prev_bboxes.unsqueeze(0)
            prev_centers = prev_bboxes[..., :2]
            v = centers - prev_centers  # (B, N, 2)
            return v

        return None  # No velocity available

    def _compute_approach(self, centers, velocities):
        """
        centers: (B, N, 2)
        velocities: (B, N, 2)
        returns: a_ij_normalized in [-1, 1], (B, N, N)
        """
        B, N, _ = centers.shape

        ci = centers[:, :, None, :]  # (B, N, 1, 2)
        cj = centers[:, None, :, :]  # (B, 1, N, 2)
        r_ij = cj - ci  # (B, N, N, 2)

        r_norm = r_ij.norm(p=2, dim=-1, keepdim=True)  # (B, N, N, 1)
        r_hat = r_ij / (r_norm + self.eps)

        v_i = velocities[:, :, None, :]  # (B, N, 1, 2) broadcast over j

        a_ij = (v_i * r_hat).sum(dim=-1)  # (B, N, N)

        # Clamp and normalize to [-1, 1] for RBF
        a_ij = a_ij.clamp(-self.max_rel_vel, self.max_rel_vel)
        a_ij_norm = a_ij / (self.max_rel_vel + self.eps)
        return a_ij_norm

    def _compute_alignment(self, velocities):
        """
        velocities: (B, N, 2)
        returns: f_ij in [-1, 1], (B, N, N)
        """
        vi = velocities[:, :, None, :]  # (B, N, 1, 2)
        vj = velocities[:, None, :, :]  # (B, 1, N, 2)

        dot = (vi * vj).sum(dim=-1)  # (B, N, N)
        ni = vi.norm(p=2, dim=-1)  # (B, N, 1)
        nj = vj.norm(p=2, dim=-1)  # (B, 1, N)

        denom = ni * nj + self.eps
        cos_sim = (dot / denom).clamp(-1.0, 1.0)  # (B, N, N)
        return cos_sim

    def forward(self, bboxes, prev_bboxes=None, velocities=None):
        """
        Args:
            bboxes: (B, N, 4) or (N, 4) in cx, cy, w, h
            prev_bboxes: optional (B, N, 4) or (N, 4) if you want the module
                         to compute velocities internally.
            velocities: optional (B, N, 2) or (N, 2) precomputed.

        Returns:
            bias: (B, n_head, N, N)
        """
        if bboxes.ndim == 2:
            bboxes = bboxes.unsqueeze(0)

        # Distance (and centers)
        d_ij, centers = self._compute_distance(bboxes)

        # Velocity tensor if any velocity-based feature is used
        need_vel = self.use_approach or self.use_align
        vel = None
        if need_vel:
            vel = self._compute_velocity_features(centers, prev_bboxes, velocities)
            if vel is None:
                raise ValueError("use_approach/use_align is True but no prev_bboxes or velocities were provided.")

        # Collect RBF features
        feats = []

        if self.use_dist:
            phi_d = self._rbf(d_ij, self.mu_dist, self.sigma_dist)  # (B, N, N, M_d)
            feats.append(phi_d)

        if self.use_approach:
            a_ij = self._compute_approach(centers, vel)  # (B, N, N)
            phi_a = self._rbf(a_ij, self.mu_approach, self.sigma_approach)  # (B, N, N, M_a)
            feats.append(phi_a)

        if self.use_align:
            f_ij = self._compute_alignment(vel)  # (B, N, N)
            phi_f = self._rbf(f_ij, self.mu_align, self.sigma_align)  # (B, N, N, M_f)
            feats.append(phi_f)

        # Concatenate along last dim → (B, N, N, D_in)
        phi_cat = torch.cat(feats, dim=-1)

        B, N, _, D_in = phi_cat.shape
        phi_flat = phi_cat.view(B * N * N, D_in)  # (B*N*N, D_in)
        bias_flat = self.mlp(phi_flat)  # (B*N*N, n_head)
        bias = bias_flat.view(B, N, N, self.n_head)  # (B, N, N, H)

        bias = self.beta * bias  # global scale
        bias = bias.permute(0, 3, 1, 2).contiguous()  # (B, H, N, N)

        return bias


if __name__ == "__main__":
    from addict import Dict

    spatial_encoder = SpatialRBFAttentionMask(Dict(dict(num_rbf=32, n_head=8)))

    # # 1) exemple facile avec souris
    # mice_easy_ids = [[9], [16], [17], [22], [25]]
    # mice_easy_bboxes = reformat(
    #     torch.tensor(
    #         [
    #             [620.001098632813, 958.89404296875, 93.9957275390625, 141.101379394531],
    #             [723.748413085938, 716.857299804687, 173.719116210938, 88.2179565429687],
    #             [872.577941894531, 974.242553710937, 95.0311279296875, 107.758728027344],
    #             [921.390502929688, 829.545166015625, 125.867614746094, 157.823547363281],
    #             [1612.30895996094, 1044.890625, 206.4599609375, 91.9466552734375],
    #         ],
    #         dtype=torch.float32,
    #     ),
    #     "xywh",
    #     "cxcywh",
    # )
    # phi = spatial_encoder(mice_easy_bboxes)

    # 2) exemple difficile avec souris
    # mice_hard_ids = [[0], [2], [3], [9], [14]]
    # mice_hard_bboxes = reformat(
    #     torch.tensor(
    #         [
    #             [2088.2197265625, 816.319458007813, 81.937744140625, 195.530700683594],
    #             [2074.154296875, 746.004272460938, 157.987548828125, 111.623352050781],
    #             [1687.47338867188, 1028.50708007813, 223.953979492188, 120.658081054688],
    #             [1891.48413085938, 798.775756835938, 191.316528320312, 86.4910888671875],
    #             [1988.943359375, 660.394409179688, 116.8046875, 134.460021972656],
    #         ],
    #         dtype=torch.float32,
    #     ),
    #     "xywh",
    #     "cxcywh",
    # )
    # phi = spatial_encoder(mice_hard_bboxes)

    # 3) exemple large avec AP
    ap_large_ids = [[1], [2]]
    ap_large_bboxes = reformat(
        torch.tensor(
            [
                [4.0546875, 211.833801269531, 1570.27111816406, 852.598693847656],
                [1813.619140625, 359.148651123047, 106.283203125, 202.410064697266],
            ],
            dtype=torch.float32,
        ),
        "xywh",
        "cxcywh",
    )
    phi = spatial_encoder(ap_large_bboxes)

    # 3) exemple medium avec AP
    ap_large_ids = [[4], [2], [1], [6], [3], [5]]
    ap_large_bboxes = reformat(
        torch.tensor(
            [
                [22.5882352941176, 113.411764705882, 256.470588235294, 642.352941176471],
                [556.705882352941, 732.235294117647, 311.764705882353, 265.882352941177],
                [696.705882352941, 399.294117647059, 315.294117647059, 383.529411764706],
                [949.647058823529, 658.117647058824, 265.882352941176, 418.823529411765],
                [1126, 0, 298.941176470588, 308.705882352941],
                [1518.52690863579, 16.828535669587, 244.705882352941, 428.235294117647],
            ],
            dtype=torch.float32,
        ),
        "xywh",
        "cxcywh",
    )
    phi = spatial_encoder(ap_large_bboxes)


@MODELS.register_module()
class SpacialEncoder(BaseModel):
    METAINFO_KEYS = [
        "skeleton_links",
    ]

    def __init__(
        self,
        config: Config,
        metainfo: str,
        data_preprocessor: Optional[Union[dict, nn.Module]] = None,
        loss_actions: Optional[Config] = None,
        *args,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor)

        self.encoder = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_block)])
        self.spacial_embeddings = nn.Embedding(self.block_size, config.n_embd)  # TODO Un par prior

        self.n_class = config.n_output
        self.classification_head = nn.Sequential(
            TransformerMLP(config),
            nn.Linear(config.n_embd, self.n_class, bias=config.bias),
        )

        self.proj = TransformerMLP(config)

        self.loss_actions = loss_actions
        if loss_actions is not None:
            self.loss_actions = MODELS.build(loss_actions)
        self.dropout = nn.Dropout(config.dropout)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        poses: torch.Tensor,
        dynamics: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        data_samples: Optional[List[PoseDataSample]] = None,
        mode: Optional[str] = "tensor",
        *args,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor], dict]:
        if isinstance(features, list):
            features = torch.stack(features)
        if mode == "loss":
            return self.loss(features, poses, dynamics, labels)
        elif mode == "predict":
            return self.predict((features, poses, dynamics), data_samples)
        elif mode == "tensor":
            return self._forward(features, poses, dynamics)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". ' "Only supports loss, predict and tensor mode.")

    def loss(self, inputs: List[dict], data_samples: List[PoseDataSample], *args, **kwargs) -> dict:
        batched_inputs = self._build_batch(inputs=inputs)
        features = batched_inputs["features"]
        dynamics = batched_inputs["dynamics"]
        poses = batched_inputs["poses"]

        features = self.dropout(features)
        dynamics = self.dropout(dynamics)
        poses = self.dropout(poses)

        class_logits, decoder_embs = self._forward(features, poses, dynamics, return_embs=True)
        N, T, _ = decoder_embs.shape
        losses = dict()

        action_labels = batched_inputs.get("actions")
        if action_labels is not None:
            action_labels = action_labels.reshape(N * T).long()
            losses["classification_loss"] = self.loss_actions(class_logits.reshape(N * T, self.n_class), action_labels, *args, **kwargs)
        return losses

    @staticmethod
    def _build_batch(inputs: List[PoseDataSample]) -> dict:
        out = dict()
        features = []
        poses = []
        dynamics = []
        actions = []
        for seq_input in inputs:
            for k, v in seq_input.pred_track_instances.items():
                if isinstance(v, torch.Tensor):
                    if k == "features":
                        features.append(v)
                    if k == "poses":
                        poses.append(v)
                    if k == "dynamics":
                        dynamics.append(v)
                    if k == "actions":
                        actions.append(v)

        for k, list_of_tensor in zip(["features", "poses", "dynamics", "actions"], [features, poses, dynamics, actions]):
            if list_of_tensor:
                out[k] = torch.concat(list_of_tensor, dim=0)
        return out

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        return self.test_step(data)

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        batched_inputs = self._build_batch(inputs=data)
        return self.predict(inputs=batched_inputs, data_samples=data)

    def predict(self, inputs: Tuple[Tensor], data_samples: List[PoseDataSample] = None) -> Tuple[Tensor]:
        class_logits, action_embeddings = self._forward(**inputs, data_samples=data_samples, return_embs=True)
        return F.softmax(class_logits[:, -1, :], dim=-1), action_embeddings[:, -1, :]

    def _forward(
        self,
        features: Tensor,
        poses: torch.Tensor,
        dynamics: torch.Tensor,
        data_samples: Optional[List[PoseDataSample]] = None,
        return_embs: bool = False,
        *args,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor]]:

        pose_embs = self.pose_encoder(poses.reshape(-1, self.block_size, self.n_pose * 2))
        dyns_embs = self.velocity_encoder(dynamics)
        feat_embs = self.feature_encoder(features)
        x = torch.cat((feat_embs, pose_embs, dyns_embs), dim=-1)  # TODO attention-based fusion, rester en 128!

        x = self.proj(x)
        x = x + self.pe(torch.arange(0, self.block_size, device=x.device, dtype=torch.long))

        for block in self.decoder:
            x = block(x)

        class_logits = self.classification_head(x)

        if return_embs:
            return class_logits, x
        else:
            return class_logits
