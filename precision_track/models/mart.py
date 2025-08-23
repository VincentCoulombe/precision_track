from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import Config
from mmengine.model import BaseModel
from torch import Tensor

from precision_track.registry import MODELS
from precision_track.utils import PoseDataSample, parse_pose_metainfo

from .modules.blocks.transformers import ProjLN, TransformerBlock, TransformerMLP


@MODELS.register_module()
class MART(BaseModel):
    METAINFO_KEYS = [
        "skeleton_links",
    ]

    def __init__(
        self,
        config: Config,
        metainfo: str,
        data_preprocessor: Optional[Union[dict, nn.Module]] = None,
        loss_embs: Optional[Config] = None,
        *args,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor)
        self.n_pose = len(parse_pose_metainfo(dict(from_file=metainfo)).get("skeleton_links"))
        n_embd_feats = config.n_embd
        self.block_size = config.block_size

        n_embd_dynamics = config.n_embd_dynamics
        config.n_embd = n_embd_dynamics
        self.velocity_encoder = nn.Sequential(
            ProjLN(2, config.n_embd, bias=config.bias),
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

        config.n_embd = n_embd_feats
        self.feature_encoder = nn.Sequential(
            TransformerMLP(config),
            nn.LayerNorm(config.n_embd, bias=config.bias),
        )

        config.n_embd = n_embd_feats + n_embd_dynamics + n_embd_pose

        self.decoder = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_block)])
        self.pe = nn.Embedding(self.block_size, config.n_embd)

        self.n_class = config.n_output
        self.classification_head = nn.Sequential(
            TransformerMLP(config),
            nn.Linear(config.n_embd, self.n_class, bias=config.bias),
        )

        self.proj = TransformerMLP(config)

        self.loss_embs = MODELS.build(loss_embs) if loss_embs is not None else None
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

    def loss(self, features: Tensor, poses: torch.Tensor, dynamics: torch.Tensor, labels: torch.Tensor) -> dict:
        features = self.dropout(features)
        dynamics = self.dropout(dynamics)
        poses = self.dropout(poses)

        class_logits, decoder_embs = self._forward(features, poses, dynamics, return_embs=True)
        N, T, E = decoder_embs.shape
        losses = dict()

        labels_flat = labels.reshape(N * T)
        losses["classification_loss"] = F.cross_entropy(class_logits.reshape(N * T, self.n_class), labels_flat)

        if self.loss_embs is not None:
            decoder_embs = decoder_embs.reshape(N * T, E)
            losses["embedding_loss"] = self.loss_embs(decoder_embs, labels_flat)

        return losses

    def predict(self, inputs: Tuple[Tensor], data_samples: List[PoseDataSample] = None) -> Tuple[Tensor]:
        class_logits, action_embeddings = self._forward(*inputs, data_samples, return_embs=True)
        return F.softmax(class_logits[:, -1, :], dim=-1), action_embeddings[:, -1, :]

    def _forward(
        self, features: Tensor, poses: torch.Tensor, dynamics: torch.Tensor, data_samples: Optional[List[PoseDataSample]] = None, return_embs: bool = False
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
