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
        loss_actions: Optional[Config] = None,
        *args,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor)
        metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        self.n_pose = len(metainfo.get("skeleton_links", []))
        self.n_kpts = metainfo.get("num_keypoints", 0)
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
        # this is a quick hack to temporaly accomodate the old preprocessor
        features = inputs
        poses = data_samples
        dynamics = args[0]
        action_labels = args[1]
        # batched_inputs = self._build_batch(inputs=inputs)
        # features = batched_inputs["features"]
        # dynamics = batched_inputs["dynamics"]
        # poses = batched_inputs["poses"]

        features = self.dropout(features)
        dynamics = self.dropout(dynamics)
        poses = self.dropout(poses)

        class_logits, decoder_embs = self._forward(features, poses, dynamics, return_embs=True)
        N, T, _ = decoder_embs.shape
        losses = dict()

        # action_labels = batched_inputs.get("actions")
        if action_labels is not None:
            action_labels = action_labels.reshape(N * T).long()
            # losses["classification_loss"] = self.loss_actions(class_logits.reshape(N * T, self.n_class), action_labels, *args, **kwargs)
            losses["classification_loss"] = F.cross_entropy(class_logits.reshape(N * T, self.n_class), action_labels)
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

    # def val_step(self, data: Union[tuple, dict, list]) -> list:
    #     return self.test_step(data)

    # def test_step(self, data: Union[dict, tuple, list]) -> list:
    #     batched_inputs = self._build_batch(inputs=data)
    #     return self.predict(inputs=batched_inputs, data_samples=data)

    def predict(self, inputs: Tuple[Tensor], data_samples: List[PoseDataSample] = None) -> Tuple[Tensor]:
        # class_logits, action_embeddings = self._forward(**inputs, data_samples=data_samples, return_embs=True)
        class_logits, action_embeddings = self._forward(*inputs, data_samples=data_samples, return_embs=True)
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
