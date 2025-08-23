# Copyright (c) Meta Platforms, Inc. and affiliates and NanoGPT.
# All rights reserved.

# Modifications made by:
# Copyright (c) Vincent Coulombe

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from logging import WARNING
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor


class HieraMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x: Tensor):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


def get_alibi_slopes(n_heads: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    def slopes_power_of_2(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio**i) for i in range(n)]

    if (math.log2(n_heads)).is_integer():
        slopes = slopes_power_of_2(n_heads)
    else:
        m = 2 ** math.floor(math.log2(n_heads))
        slopes = slopes_power_of_2(m)
        slopes += slopes_power_of_2(2 * m)[0::2][: n_heads - m]
    return torch.tensor(slopes, device=device, dtype=dtype)


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.causal = config.causal

        self.use_alibi = getattr(config, "use_alibi", False)
        if not self.causal and self.use_alibi:
            print_log(
                (
                    "Vanilla ALiBi self-attention does not yield good results when paired with an encoder."
                    "See: https://github.com/ofirpress/attention_with_linear_biases/issues/5"
                ),
                logger="current",
                level=WARNING,
            )
        if self.use_alibi:
            slopes = get_alibi_slopes(self.n_head, dtype=torch.float32)
            self.register_buffer("alibi_slopes", slopes, persistent=False)
        self._alibi_cache_T = None
        self._alibi_cache = None

    def _alibi_bias(self, T: int, device, dtype):
        if self._alibi_cache is not None and self._alibi_cache_T == T:
            return self._alibi_cache.to(device=device, dtype=dtype)  # For speed

        i = torch.arange(T, device=device)
        j = torch.arange(T, device=device)
        dist = (i[:, None] - j[None, :]).clamp(min=0).to(dtype)

        slopes = self.alibi_slopes.to(device=device, dtype=dtype).view(1, self.n_head, 1, 1)
        bias = -slopes * dist.view(1, 1, T, T)

        self._alibi_cache_T = T
        self._alibi_cache = bias.detach()
        return bias

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None):
        B, T, E = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, E // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, E // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, E // self.n_head).transpose(1, 2)

        float_mask = None

        if self.causal:
            causal = torch.full((T, T), float("-inf"), device=x.device, dtype=q.dtype)
            causal = torch.triu(causal, diagonal=1)
            float_mask = causal.view(1, 1, T, T)

        if self.use_alibi:
            alibi = self._alibi_bias(T, device=x.device, dtype=q.dtype)
            float_mask = alibi if float_mask is None else (float_mask + alibi)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                add = torch.zeros_like(attn_mask, dtype=q.dtype)
                add = add.masked_fill(~attn_mask, float("-inf"))
            else:
                add = attn_mask.to(dtype=q.dtype)
            float_mask = add if float_mask is None else (float_mask + add)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=float_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, E)
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransformerMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.norm2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = TransformerMLP(config)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ProjLN(nn.Module):
    def __init__(self, in_dim: int, d_model: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model, bias=bias)
        self.ln = nn.LayerNorm(d_model, bias=bias)

    def forward(self, x):
        return self.ln(self.proj(x))


class APVAttentionMixer(nn.Module):

    def __init__(
        self,
        in_dims: tuple[int, int, int],
        n_embd: int = 256,
        nheads: int = 4,
        nlayers: int = 1,
        dropout: float = 0.1,
        use_fuse_token: bool = True,
        modality_dropout_p: float = 0.0,
    ):
        super().__init__()
        da, dp, dv = in_dims
        self.pa, self.pp, self.pv = ProjLN(da, n_embd), ProjLN(dp, n_embd), ProjLN(dv, n_embd)
        self.use_fuse_token = use_fuse_token
        self.modality_dropout_p = modality_dropout_p

        # Learnable modality/type embeddings
        self.mod_embed = nn.Parameter(torch.randn(3, n_embd) * 0.02)  # [a, p, v]
        self.fuse_token = nn.Parameter(torch.randn(1, n_embd) * 0.02) if use_fuse_token else None

        enc_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=nheads, dim_feedforward=4 * n_embd, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.out_proj = ProjLN(n_embd, n_embd)

    def _apply_modality_dropout(self, tokens: torch.Tensor, has_fuse: bool) -> torch.Tensor:
        # tokens: [B, L, D]; if fuse is present, it's at index 0 (don’t drop it).
        if not self.training or self.modality_dropout_p <= 0.0:
            return tokens
        B, L, D = tokens.shape
        start = 1 if has_fuse else 0
        keep = (torch.rand(B, L - start, device=tokens.device) > self.modality_dropout_p).unsqueeze(-1).float()
        tokens[:, start:, :] = tokens[:, start:, :] * keep
        return tokens

    def forward(self, x_a: torch.Tensor, x_p: torch.Tensor, x_v: torch.Tensor):

        za = self.pa(x_a) + self.mod_embed[0]
        zp = self.pp(x_p) + self.mod_embed[1]
        zv = self.pv(x_v) + self.mod_embed[2]
        tokens = torch.stack([za, zp, zv], dim=1)  # [B, 3, D]

        if self.use_fuse_token:
            B, _, D = tokens.shape
            fuse = self.fuse_token.expand(B, 1, D)  # [B,1,D]
            tokens = torch.cat([fuse, tokens], dim=1)  # [B, 4, D]

        tokens = self._apply_modality_dropout(tokens, has_fuse=self.use_fuse_token)
        tokens = self.encoder(tokens)  # self-attn over 3 (or 4) tokens
        z_fused = tokens[:, 0, :] if self.use_fuse_token else tokens.mean(dim=1)
        return self.out_proj(z_fused), tokens


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with " "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1 :: 2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with " "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe
