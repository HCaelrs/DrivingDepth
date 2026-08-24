# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py


from typing import List
import torch.nn as nn

from drivingdepth.model.dinov2.vision_transformer import vit_giant2, vit_large


class DinoV2(nn.Module):
    def __init__(
        self,
        name: str,
        out_layers: List[int],
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = True,
        with_sp_depth: bool = False,
        cross_attn_version: str = "v1",
        channel_prompt_da: int = 8,
        **kwargs,
    ):
        super().__init__()
        assert name in {"vitl", "vitg", "vitg_sp"}
        self.name = name
        self.out_layers = out_layers
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        encoder_map = {
            "vitl": vit_large,
            "vitg": vit_giant2,
            "vitg_sp": vit_giant2,
        }
        encoder_fn = encoder_map[self.name]
        ffn_layer = "swiglufused" if "vitg" in self.name else "mlp"
        self.pretrained = encoder_fn(
            img_size=518,
            patch_size=14,
            ffn_layer=ffn_layer,
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
            with_sp_depth=with_sp_depth,
            cross_attn_version=cross_attn_version,
            channel_prompt_da=channel_prompt_da,
        )

    def forward(self, x, **kwargs):
        return self.pretrained.get_intermediate_layers(
            x,
            self.out_layers,
            **kwargs,
        )
