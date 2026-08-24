# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import math
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from drivingdepth.utils.logger import logger

from .layers import (
    Block,
    LayerScale,
    Mlp,
    PatchEmbed,
    PositionGetter,
    RotaryPositionEmbedding2D,
    SwiGLUFFNFused,
)
from .layers.attention import CrossAttention, CrossViewFrameAttention


class CrossViewFrameBlock(nn.Module):
    """Block wrapping CrossViewFrameAttention + FFN for adjacency-constrained
    cross-view and cross-frame self-attention.

    Input/Output: (B, S, N, C) where S = num_frames * num_views.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        ffn_layer=None,
        init_values=None,
        qk_norm: bool = False,
        rope=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossViewFrameAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            rope=rope,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if ffn_layer is None:
            ffn_layer = Mlp
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=0.0,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x, num_frames, num_views, pos=None):
        """
        Args:
            x: (B, S, N, C) where S = num_frames * num_views
            num_frames: number of temporal frames
            num_views: number of camera views
            pos: (B, S, N, 2) optional RoPE positions
        Returns:
            (B, S, N, C)
        """
        x = x + self.ls1(
            self.attn(self.norm1(x), num_frames=num_frames, num_views=num_views, pos=pos)
        )
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        init_values=1.0,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        alt_start=-1,
        qknorm_start=-1,
        rope_start=-1,
        rope_freq=100,
        cat_token=True,
        with_sp_depth=False,
        channel_prompt_da: int=8,
        cross_attn_version: str="v1",
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating
                positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating
                positional embeddings
        """
        super().__init__()
        self.patch_start_idx = 1
        norm_layer = nn.LayerNorm
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.img_size = img_size
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.alt_start != -1:
            self.camera_token = nn.Parameter(torch.randn(1, 2, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        if self.rope_start != -1:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
            self.position_getter = PositionGetter() if self.rope is not None else None
        else:
            self.rope = None
        self.block_fn = block_fn
        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                qk_norm=i >= qknorm_start if qknorm_start != -1 else False,
                rope=self.rope if i >= rope_start and rope_start != -1 else None,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        if with_sp_depth:
            self.channel_prompt_da = channel_prompt_da
            if cross_attn_version != "v2":
                raise ValueError(f"Unknown cross_attn_version: {cross_attn_version}")
            self.cross_attn_blocks_v2 = nn.ModuleList([
                # block[0]: cross-modal attention (img -> depth)
                self.block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    ffn_layer=ffn_layer,
                    init_values=init_values,
                    attn_class=CrossAttention,
                    qk_norm=True,
                    rope=self.rope,
                ),
                # block[1]: adjacency-constrained cross-view + cross-frame self-attention
                CrossViewFrameBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    ffn_layer=ffn_layer,
                    init_values=init_values,
                    qk_norm=True,
                    rope=self.rope,
                ),
            ])
            self._build_depth_encoder()

    def _build_depth_encoder(self):
        """Lightweight CNN that turns the packed sparse depth into DINO-sized tokens.

        The final PatchEmbed projection is zero-initialized so an untrained model
        starts out behaving exactly like the pure-RGB one.
        """
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(self.channel_prompt_da, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            PatchEmbed(
                img_size=self.img_size,
                patch_size=self.patch_size,
                in_chans=256,
                embed_dim=self.embed_dim,
            ),
        )
        target_conv = self.depth_encoder[-1].proj
        nn.init.constant_(target_conv.weight, 0.0)
        if target_conv.bias is not None:
            nn.init.constant_(target_conv.bias, 0.0)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the
            # interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using
            # both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_cls_token(self, B, S):
        cls_token = self.cls_token.expand(B, S, -1)
        cls_token = cls_token.reshape(B * S, -1, self.embed_dim)
        return cls_token

    def prepare_tokens_with_masks(self, x, cls_token=None, **kwargs):
        B, S, nc, w, h = x.shape
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.patch_embed(x)
        cls_token = self.prepare_cls_token(B, S)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        x = rearrange(x, "(b s) n c -> b s n c", b=B, s=S)
        return x

    def _prepare_rope(self, B, S, H, W, device):
        pos = None
        pos_nodiff = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=device
            )
            pos = rearrange(pos, "(b s) n c -> b s n c", b=B)
            pos_nodiff = torch.zeros_like(pos).to(pos.dtype)
            if self.patch_start_idx > 0:
                pos = pos + 1
                pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(device).to(pos.dtype)
                pos_special = rearrange(pos_special, "(b s) n c -> b s n c", b=B)
                pos = torch.cat([pos_special, pos], dim=2)
                pos_nodiff = pos_nodiff + 1
                pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
        return pos, pos_nodiff

    def _get_intermediate_layers_not_chunked(self, x, n=1, **kwargs):
        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        depth_fusion_feat = []
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)

        # Encode the packed sparse depth (B, S, C, H, W) into per-patch tokens
        sp_depth = kwargs.get("sp_depth", None)
        depth_tokens = None
        if sp_depth is not None and hasattr(self, "depth_encoder"):
            d_in = rearrange(sp_depth, "b s c h w -> (b s) c h w")
            d_feat = self.depth_encoder(d_in)
            depth_tokens = rearrange(d_feat, "(b s) n c -> b s n c", b=B, s=S)

        for i, blk in enumerate(self.blocks):
            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos

            if self.alt_start != -1 and i == self.alt_start:
                if kwargs.get("cam_token", None) is not None:
                    cam_token = kwargs.get("cam_token")
                else:
                    ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                    src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                    cam_token = torch.cat([ref_token, src_token], dim=1)
                x[:, :, 0] = cam_token

            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x = self.process_attention(
                    x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask", None)
                )
            else:
                x = self.process_attention(x, blk, "local", pos=l_pos)
                local_x = x

            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                output.append((out_x[:, :, 0], out_x))
                if hasattr(self, "cross_attn_blocks_v2"):
                    views = kwargs.get('views', None)
                    assert views is not None, "Views must be provided when using cross_attn_blocks_v2"
                    f, v = x.shape[1] // views, views
                    feat_dims = {"f": f, "v": v}
                    n_tokens = x.shape[2]  # tokens per image (cls + patches)

                    # block[0]: cross-modal attention (img -> depth)
                    cross_x = x.clone()
                    cross_x = rearrange(cross_x, "b (f v) n c -> (b f) (v n) c", **feat_dims)
                    cross_pos_cm = pos.clone()
                    cross_pos_cm = rearrange(cross_pos_cm, "b (f v) n c -> (b f) (v n) c", **feat_dims)
                    depth_tokens_cam = torch.cat([cam_token[:, :, None], depth_tokens], dim=2)
                    cross_depth_tokens = rearrange(depth_tokens_cam, "b (f v) n c -> (b f) (v n) c", **feat_dims)
                    cross_x = self.cross_attn_blocks_v2[0](
                        cross_x, cross_depth_tokens, pos=cross_pos_cm, context_pos=cross_pos_cm
                    )

                    # reshape back to (B, F*V, N, C) for block[1]
                    cross_x = rearrange(cross_x, "(b f) (v n) c -> b (f v) n c", b=B, f=f, v=v, n=n_tokens)

                    # block[1]: adjacency-constrained cross-view + cross-frame self-attention
                    cross_x = self.cross_attn_blocks_v2[1](cross_x, num_frames=f, num_views=v, pos=pos)

                    depth_fusion_feat.append(torch.cat([local_x, cross_x], dim=-1)[:, :, 1:])
        return output, depth_fusion_feat

    def process_attention(self, x, block, attn_type="global", pos=None, attn_mask=None):
        b, s, n = x.shape[:3]
        if attn_type == "local":
            x = rearrange(x, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
        elif attn_type == "global":
            x = rearrange(x, "b s n c -> b (s n) c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")

        x = block(x, pos=pos, attn_mask=attn_mask)

        if attn_type == "local":
            x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
        elif attn_type == "global":
            x = rearrange(x, "b (s n) c -> b s n c", b=b, s=s)
        return x

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        outputs, depth_fusion_feat = self._get_intermediate_layers_not_chunked(x, n, **kwargs)
        camera_tokens = [out[0] for out in outputs]
        depth_fusion_feat = [
            [torch.cat(
                [out[..., : self.embed_dim], self.norm(out[..., self.embed_dim :])],
                dim=-1,
            ), None]
            for out in depth_fusion_feat
        ]
        if outputs[0][1].shape[-1] == self.embed_dim:
            outputs = [self.norm(out[1]) for out in outputs]
        elif outputs[0][1].shape[-1] == (self.embed_dim * 2):
            outputs = [
                torch.cat(
                    [out[1][..., : self.embed_dim], self.norm(out[1][..., self.embed_dim :])],
                    dim=-1,
                )
                for out in outputs
            ]
        else:
            raise ValueError(f"Invalid output shape: {outputs[0][1].shape}")
        outputs = [out[..., 1 + self.num_register_tokens :, :] for out in outputs]
        return tuple(zip(outputs, camera_tokens)), depth_fusion_feat


def vit_large(patch_size=16, num_register_tokens=0, depth=24, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=depth,
        num_heads=16,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, depth=40, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=depth,
        num_heads=24,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
