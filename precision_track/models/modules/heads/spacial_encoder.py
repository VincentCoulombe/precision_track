from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import Config
from mmengine.model import BaseModel
from torch import Tensor

from precision_track.registry import MODELS
from precision_track.utils import PoseDataSample, reformat, parse_pose_metainfo

# from .modules.blocks.transformers import ProjLN, TransformerBlock, TransformerMLP

import torch
import torch.nn as nn
import torch.nn.functional as F


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
