# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger("dinov2")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, attn_mask=None, **kwargs) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=(
                    (attn_mask)[:, None].repeat(1, self.num_heads, 1, 1)
                    if attn_mask is not None
                    else None
                ),
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.fused_attn = fused_attn
        self.rope = rope

        context_dim = context_dim if context_dim is not None else dim

        # Q and KV are projected separately: they come from different sources
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, context: Tensor, pos=None, context_pos=None, attn_mask=None) -> Tensor:
        """
        Args:
            x: image features [B, N, dim]
            context: sparse depth features [B, M, context_dim]
            pos: RoPE position indices for the image tokens
            context_pos: RoPE position indices for the depth tokens; without it
                the keys are left unrotated, which only makes sense if the depth
                tokens carry no image-space position
        """
        B, N, C = x.shape
        M = context.shape[1]

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = (
            self.kv(context)
            .reshape(B, M, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            if pos is not None:
                q = self.rope(q, pos)
            if context_pos is not None:
                k = self.rope(k, context_pos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=attn_mask,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossViewFrameAttention(nn.Module):
    """Self-attention with adjacency constraint: each image attends to
    (1) ring-adjacent views in the same frame (self + left + right)
    (2) same camera across all frames (cross-frame)

    Input: (B, S, N, C) where S = num_frames * num_views
    K/V are gathered per-image based on precomputed adjacency indices.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.fused_attn = fused_attn
        self.rope = rope

        # Q and KV are projected separately: they hold different token counts
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # Cache the kv indices: they only depend on (num_frames, num_views)
        self._kv_indices = None
        self._cached_fv = (None, None)

    @staticmethod
    def build_kv_indices(num_frames: int, num_views: int) -> Tensor:
        """Build K/V image indices for each query image.

        For image (fi, vj), K/V includes:
          - same-frame ring neighbors: (fi, vj-1), (fi, vj), (fi, vj+1)
          - cross-frame same view: (f*, vj) for all f*

        Returns:
            (S, K_count) LongTensor where S = F*V, K_count = F+2 (after dedup)
        """
        S = num_frames * num_views
        indices = []
        for fi in range(num_frames):
            for vj in range(num_views):
                kv_set = set()
                # ring-adjacent views in same frame
                for dv in [-1, 0, 1]:
                    kv_set.add(fi * num_views + (vj + dv) % num_views)
                # same view across all frames
                for fk in range(num_frames):
                    kv_set.add(fk * num_views + vj)
                indices.append(sorted(kv_set))

        # Every query image must end up with the same number of K/V images
        k_counts = [len(idx) for idx in indices]
        assert len(set(k_counts)) == 1, f"K counts not uniform: {set(k_counts)}"

        return torch.tensor(indices, dtype=torch.long)  # (S, K_count)

    def _get_kv_indices(self, num_frames: int, num_views: int, device) -> Tensor:
        """Get or build cached kv indices."""
        if self._cached_fv != (num_frames, num_views) or self._kv_indices is None:
            self._kv_indices = self.build_kv_indices(num_frames, num_views).to(device)
            self._cached_fv = (num_frames, num_views)
        return self._kv_indices

    def forward(
        self,
        x: Tensor,
        num_frames: int,
        num_views: int,
        pos: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """
        Args:
            x: (B, S, N, C) where S = num_frames * num_views
            num_frames: number of temporal frames
            num_views: number of camera views
            pos: (B, S, N, 2) RoPE position indices (optional)
        Returns:
            (B, S, N, C) — same shape as input
        """
        B, S, N, C = x.shape
        assert S == num_frames * num_views, f"S={S} != F*V={num_frames}*{num_views}"
        H = self.num_heads
        head_dim = C // H

        # 1. kv indices: (S, K_count)
        kv_idx = self._get_kv_indices(num_frames, num_views, x.device)
        K_count = kv_idx.shape[1]

        # 2. Gather KV tokens: x[:, kv_idx] -> (B, S, K_count, N, C)
        kv_gathered = x[:, kv_idx]  # (B, S, K_count, N, C)
        kv_gathered = kv_gathered.reshape(B, S, K_count * N, C)  # (B, S, K_count*N, C)

        # 3. Flatten batch: Q=(B*S, N, C), KV=(B*S, K_count*N, C)
        x_q = x.reshape(B * S, N, C)
        x_kv = kv_gathered.reshape(B * S, K_count * N, C)

        # 4. Q projection
        q = self.q_proj(x_q).reshape(B * S, N, H, head_dim).permute(0, 2, 1, 3)
        # (B*S, H, N, head_dim)

        # 5. KV projection
        kv = (
            self.kv_proj(x_kv)
            .reshape(B * S, K_count * N, 2, H, head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]  # (B*S, H, K_count*N, head_dim)

        # 6. QK Norm
        q, k = self.q_norm(q), self.k_norm(k)

        # 7. RoPE (optional)
        if self.rope is not None and pos is not None:
            # pos: (B, S, N, 2) -> q_pos: (B*S, N, 2)
            q_pos = pos.reshape(B * S, N, 2)
            q = self.rope(q, q_pos)
            # Gather pos for KV
            kv_pos = pos[:, kv_idx]  # (B, S, K_count, N, 2)
            kv_pos = kv_pos.reshape(B, S, K_count * N, 2)
            kv_pos = kv_pos.reshape(B * S, K_count * N, 2)
            k = self.rope(k, kv_pos)

        # 8. Attention
        if self.fused_attn:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        # 9. Output projection
        out = out.transpose(1, 2).reshape(B * S, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # 10. Reshape back
        return out.reshape(B, S, N, C)