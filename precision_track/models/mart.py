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


class BaseMARModel(BaseModel):
    METAINFO_KEYS = [
        "skeleton_links",
    ]
    SUPPORTED_MODES = ["loss"]

    def __init__(
        self,
        config: Config,
        metainfo: str,
        data_preprocessor: Optional[Union[dict, nn.Module]] = None,
        loss_actions: Optional[Config] = None,
        mode: Optional[str] = "loss",
        *args,
        **kwargs,
    ):
        assert hasattr(config, "block_size") and config.block_size > 0, "config.block_size must be positive"
        assert hasattr(config, "n_output") and config.n_output > 0, "config.n_output must be positive"
        assert hasattr(config, "n_block"), "config must have n_block attribute"

        super().__init__(
            data_preprocessor=data_preprocessor,
        )

        self.block_size = config.block_size

        self.n_embd = 0

        self.n_embd_feats = 0
        if config.get("with_features"):
            self.n_embd_feats = config.get("n_embd_features", 0)
            self.n_embd += self.n_embd_feats

        self.n_embd_dynamics = 0
        if config.get("with_dynamics"):
            self.n_embd_dynamics = config.get("n_embd_dynamics", 0)
            self.n_embd += self.n_embd_dynamics

        self.n_embd_poses = 0
        if config.get("with_poses"):
            self.n_embd_poses = config.get("n_embd_poses", 0)
            self.n_embd += self.n_embd_poses

        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        self.n_pose = len(metainfo.get("skeleton_links", []))
        self.n_kpts = metainfo.get("num_keypoints", 0)

        self.loss_actions = loss_actions
        if loss_actions is not None:
            self.loss_actions = MODELS.build(loss_actions)

        self._mode = mode

        self.dropout = nn.Dropout(config.dropout)

        if self.n_embd_dynamics > 0:
            dynamics_config = config.copy()
            dynamics_config.n_embd = self.n_embd_dynamics
            self.n_encoded_dynamics = config.get("n_encoded_dynamics", 2)
            self.velocity_encoder = nn.Sequential(
                ProjLN(self.n_encoded_dynamics, dynamics_config.n_embd, bias=dynamics_config.bias),
                TransformerMLP(dynamics_config),
                nn.LayerNorm(dynamics_config.n_embd, bias=dynamics_config.bias),
            )
        else:
            self.velocity_encoder = None

        if self.n_embd_poses > 0:
            pose_config = config.copy()
            pose_config.n_embd = self.n_embd_poses
            self.pose_encoder = nn.Sequential(
                ProjLN(self.n_pose * 2, pose_config.n_embd, bias=pose_config.bias),
                TransformerMLP(pose_config),
                nn.LayerNorm(pose_config.n_embd, bias=pose_config.bias),
            )
        else:
            self.pose_encoder = None

        if self.n_embd_feats > 0:
            feature_config = config.copy()
            feature_config.n_embd = self.n_embd_feats
            self.feature_encoder = nn.Sequential(
                ProjLN(feature_config.n_embd, feature_config.n_embd, bias=feature_config.bias),
                TransformerMLP(feature_config),
                nn.LayerNorm(feature_config.n_embd, bias=feature_config.bias),
            )
        else:
            self.feature_encoder = None

        config.n_embd = self.n_embd

        self.proj = TransformerMLP(config)
        self.config = config

    def train_step(self, data: Union[dict, tuple, list], optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode=self._mode)
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def _get_projections(
        self,
        features: Tensor,
        poses: torch.Tensor,
        dynamics: torch.Tensor,
    ):
        embeddings = []

        if self.feature_encoder is not None:
            feat = features.reshape(-1, self.block_size, self.n_embd_feats)
            feat = self.dropout(feat)
            assert self.feature_encoder(feat).shape[-1] == 128, f"{self.feature_encoder(feat).shape}"
            embeddings.append(self.feature_encoder(feat))

        if self.pose_encoder is not None:
            pose = poses.reshape(-1, self.block_size, self.n_pose * 2)
            pose = self.dropout(pose)
            assert self.pose_encoder(pose).shape[-1] == 96, f"{self.pose_encoder(pose).shape}"
            embeddings.append(self.pose_encoder(pose))

        if self.velocity_encoder is not None:
            dyn = dynamics.reshape(-1, self.block_size, self.n_encoded_dynamics)
            dyn = self.dropout(dyn)
            assert self.velocity_encoder(dyn).shape[-1] == 32, f"{self.velocity_encoder(dyn).shape}"
            embeddings.append(self.velocity_encoder(dyn))

        return self.proj(torch.cat(embeddings, dim=-1))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(param)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)


@MODELS.register_module()
class MART(BaseMARModel):
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
        super().__init__(
            config=config,
            metainfo=metainfo,
            data_preprocessor=data_preprocessor,
            loss_actions=loss_actions,
            mode=mode,
        )

        assert mode in self.SUPPORTED_MODES, f"Invalid mode '{mode}'. Must be one of {self.SUPPORTED_MODES}"
        assert 0.0 <= mask_ratio <= 1.0, f"mask_ratio must be between 0 and 1, got {mask_ratio}"

        self.decoder = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.n_block)])
        self.pe = nn.Embedding(self.block_size, self.config.n_embd)
        self.register_buffer("pe_indices", torch.arange(0, self.block_size, dtype=torch.long))

        self.register_buffer("mask_token", torch.zeros(1, 1, self.config.n_embd))
        self.mask_ratio = mask_ratio
        self.reconstruction_head = nn.Linear(self.config.n_embd, self.config.n_embd, bias=self.config.bias)

        self.n_class = self.config.n_output
        self.classification_head = nn.Sequential(
            TransformerMLP(self.config),
            nn.Linear(self.config.n_embd, self.n_class, bias=self.config.bias),
        )

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

    def pretrain(self, features: Tensor, poses: torch.Tensor, dynamics: torch.Tensor):
        x = self._get_projections(
            features=features,
            poses=poses,
            dynamics=dynamics,
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
        if mask.any():
            loss = F.mse_loss(recon[mask], target[mask])
        else:
            loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        return dict(mse_loss=loss)

    def loss(self, features: Tensor, poses: torch.Tensor, dynamics: torch.Tensor, labels: torch.Tensor) -> dict:
        x = self._get_projections(
            features=features,
            poses=poses,
            dynamics=dynamics,
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
            features=features,
            poses=poses,
            dynamics=dynamics,
        )
        x = self._get_transformations(x)
        class_logits = self.classification_head(x)
        return F.softmax(class_logits[:, -1, :], dim=-1), F.normalize(x[:, -1, :], p=2, dim=-1)

    def _get_transformations(
        self,
        x: Tensor,
    ):
        x = x + self.pe(self.pe_indices)
        for block in self.decoder:
            x = block(x)
        return x


@MODELS.register_module()
class MLPAnalyzer(BaseMARModel):
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
        super().__init__(
            config=config,
            metainfo=metainfo,
            data_preprocessor=data_preprocessor,
            loss_actions=loss_actions,
            mode="loss",
        )
        self.n_class = self.config.n_output
        self.classification_head = nn.Sequential(
            TransformerMLP(self.config),
            nn.Linear(self.config.n_embd, self.n_class, bias=self.config.bias),
        )

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
        else:
            raise RuntimeError(f'Invalid mode "{mode}". ' "Only supports loss and predict mode.")

    def loss(self, features: Tensor, poses: torch.Tensor, dynamics: torch.Tensor, labels: torch.Tensor) -> dict:
        x = self._get_projections(
            features=features,
            poses=poses,
            dynamics=dynamics,
        )
        class_logits = self.classification_head(x)

        N, T, _ = x.shape
        labels = labels.reshape(N * T).long()
        loss = F.cross_entropy(class_logits.reshape(N * T, self.n_class), labels)
        return dict(classification_loss=loss)

    def predict(self, inputs: Tuple[Tensor], data_samples: List[PoseDataSample] = None) -> Tuple[Tensor]:
        features, poses, dynamics = inputs
        x = self._get_projections(
            features=features,
            poses=poses,
            dynamics=dynamics,
        )
        class_logits = self.classification_head(x)
        return F.softmax(class_logits[:, -1, :], dim=-1), F.normalize(x[:, -1, :], p=2, dim=-1)


@MODELS.register_module()
class LSTMAnalyzer(BaseMARModel):
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
        super().__init__(
            config=config,
            metainfo=metainfo,
            data_preprocessor=data_preprocessor,
            loss_actions=loss_actions,
            mode="loss",
        )

        self.lstm = nn.LSTM(
            input_size=self.config.n_embd,
            hidden_size=self.config.n_embd,
            num_layers=self.config.n_block,
            batch_first=True,
            dropout=self.config.dropout if self.config.n_block > 1 else 0.0,
            bidirectional=False,
        )

        self.ln = nn.LayerNorm(self.config.n_embd, bias=config.bias)

        self.n_class = self.config.n_output
        self.classification_head = nn.Sequential(
            TransformerMLP(self.config),
            nn.Linear(self.config.n_embd, self.n_class, bias=self.config.bias),
        )

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
        else:
            raise RuntimeError(f'Invalid mode "{mode}". ' "Only supports loss and predict mode.")

    def loss(self, features: Tensor, poses: torch.Tensor, dynamics: torch.Tensor, labels: torch.Tensor) -> dict:
        x = self._get_projections(
            features=features,
            poses=poses,
            dynamics=dynamics,
        )
        x = self._get_lstm_output(x)
        class_logits = self.classification_head(x)

        N, T, _ = x.shape
        labels = labels.reshape(N * T).long()
        loss = F.cross_entropy(class_logits.reshape(N * T, self.n_class), labels)
        return dict(classification_loss=loss)

    def predict(self, inputs: Tuple[Tensor], data_samples: List[PoseDataSample] = None) -> Tuple[Tensor]:
        features, poses, dynamics = inputs
        x = self._get_projections(
            features=features,
            poses=poses,
            dynamics=dynamics,
        )
        x = self._get_lstm_output(x)
        class_logits = self.classification_head(x)
        return F.softmax(class_logits[:, -1, :], dim=-1), F.normalize(x[:, -1, :], p=2, dim=-1)

    def _get_lstm_output(self, x: Tensor) -> Tensor:
        lstm_out, _ = self.lstm(x)
        return self.ln(lstm_out)
