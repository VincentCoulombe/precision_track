from typing import Dict, Optional, Tuple, Union, List


import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import Config
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from torch import Tensor

from precision_track.registry import MODELS
from precision_track.utils import PoseDataSample, parse_pose_metainfo

from .modules.blocks.transformers import ProjLN, TransformerBlock, TransformerMLP


@MODELS.register_module()
class MART(BaseModel):
    METAINFO_KEYS = [
        "skeleton_links",
    ]
    SUPPORTED_MODES = ["loss", "pretrain"]

    def __init__(
        self,
        config: Config,
        metainfo: str,
        data_preprocessor: Optional[Union[dict, nn.Module]] = None,
        loss_actions: Optional[Config] = None,
        mask_ratio: Optional[float] = 0.5,
        mode: Optional[str] = "loss",
        *args,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor)
        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        self.n_pose = len(metainfo.get("skeleton_links", []))
        self.n_kpts = metainfo.get("num_keypoints", 0)
        self.n_embd_feats = config.n_embd
        self.block_size = config.block_size

        n_embd_dynamics = config.n_embd_dynamics
        config.n_embd = n_embd_dynamics
        self.n_encoded_dynamics = config.get("n_encoded_dynamics", 2)
        self.velocity_encoder = nn.Sequential(
            ProjLN(self.n_encoded_dynamics, config.n_embd, bias=config.bias),
            TransformerMLP(config),
            nn.LayerNorm(config.n_embd, bias=config.bias),
        )

        n_embd_pose = config.n_embd_pose
        config.n_embd = n_embd_pose
        self.pose_encoder = nn.Sequential(
            ProjLN(self.n_pose * 2, config.n_embd, bias=config.bias),
            TransformerMLP(config),
            nn.LayerNorm(config.n_embd, bias=config.bias),
        )

        config.n_embd = self.n_embd_feats
        self.feature_encoder = nn.Sequential(
            TransformerMLP(config),
            nn.LayerNorm(config.n_embd, bias=config.bias),
        )

        config.n_embd = self.n_embd_feats + n_embd_dynamics + n_embd_pose

        self.decoder = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_block)])
        self.pe = nn.Embedding(self.block_size, config.n_embd)

        self.register_buffer("mask_token", torch.zeros(1, 1, config.n_embd))
        self.mask_ratio = mask_ratio
        self.reconstruction_head = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

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

        assert mode in self.SUPPORTED_MODES
        self._mode = mode

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
        elif mode == "pretrain":
            return self.pretrain(features, poses, dynamics)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". ' "Only supports loss, predict and pretrain mode.")

    def train_step(self, data: Union[dict, tuple, list], optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode=self._mode)
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def pretrain(self, features: Tensor, poses: torch.Tensor, dynamics: torch.Tensor):
        x = self._get_projections(
            features=features.reshape(-1, self.block_size, self.n_embd_feats),
            poses=poses.reshape(-1, self.block_size, self.n_pose * 2),
            dynamics=dynamics.reshape(-1, self.block_size, self.n_encoded_dynamics),
        )
        target = x.detach()
        B, T, E = x.shape
        mask = torch.rand(B, T, device=x.device) < self.mask_ratio

        # Mask the last time step more often
        if torch.rand(()) < self.mask_ratio:
            mask[:, -1] = True

        x_masked = x.clone()
        x_masked[mask] = 0.0

        x = self._get_transformations(x_masked)

        recon = self.reconstruction_head(x)
        loss = F.mse_loss(recon[mask], target[mask])
        return dict(mse_loss=loss)

    def loss(self, features: Tensor, poses: torch.Tensor, dynamics: torch.Tensor, labels: torch.Tensor) -> dict:
        x = self._get_projections(
            features=features.reshape(-1, self.block_size, self.n_embd_feats),
            poses=poses.reshape(-1, self.block_size, self.n_pose * 2),
            dynamics=dynamics.reshape(-1, self.block_size, self.n_encoded_dynamics),
        )

        x = self._get_transformations(x)
        class_logits = self.classification_head(x)

        N, T, _ = x.shape
        labels = labels.reshape(N * T).long()
        loss = F.cross_entropy(class_logits.reshape(N * T, self.n_class), labels)
        return dict(classification_loss=loss)

    def predict(self, inputs: Tuple[Tensor], data_samples: List[PoseDataSample] = None) -> Tuple[Tensor]:
        features, poses, dynamics = inputs
        x = self._get_projections(
            features=features.reshape(-1, self.block_size, self.n_embd_feats),
            poses=poses.reshape(-1, self.block_size, self.n_pose * 2),
            dynamics=dynamics.reshape(-1, self.block_size, self.n_encoded_dynamics),
        )
        x = self._get_transformations(x)
        class_logits = self.classification_head(x)
        return F.softmax(class_logits[:, -1, :], dim=-1), F.normalize(x[:, -1, :], p=2, dim=-1)

    def _get_projections(
        self,
        features: Tensor,
        poses: torch.Tensor,
        dynamics: torch.Tensor,
    ):
        features = self.dropout(features)
        dynamics = self.dropout(dynamics)
        poses = self.dropout(poses)

        pose_embs = self.pose_encoder(poses)
        dyns_embs = self.velocity_encoder(dynamics)
        feat_embs = self.feature_encoder(features)

        return self.proj(torch.cat((feat_embs, pose_embs, dyns_embs), dim=-1))

    def _get_transformations(
        self,
        x: Tensor,
    ):
        x = x + self.pe(torch.arange(0, self.block_size, device=x.device, dtype=torch.long))
        for block in self.decoder:
            x = block(x)
        return x
