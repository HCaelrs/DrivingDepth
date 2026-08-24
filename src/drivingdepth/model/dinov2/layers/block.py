# flake8: noqa: F821
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable

from torch import Tensor, nn

from .attention import Attention, CrossAttention
from .layer_scale import LayerScale
from .mlp import Mlp


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope=None,
        ln_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim, eps=ln_eps)
        # Cross-attention blocks normalize the context stream separately
        if attn_class == CrossAttention:
            self.norm_ctx = norm_layer(dim, eps=ln_eps)
        else:
            self.norm_ctx = nn.Identity()
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim, eps=ln_eps)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(
        self,
        x: Tensor,
        context: Tensor = None,
        pos=None,
        context_pos=None,
        attn_mask=None,
    ) -> Tensor:
        """`context` / `context_pos` are only consumed by CrossAttention blocks."""
        x = x + self.ls1(
            self.attn(
                self.norm1(x),
                context=self.norm_ctx(context) if context is not None else context,
                pos=pos,
                context_pos=context_pos,
                attn_mask=attn_mask,
            )
        )
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x
