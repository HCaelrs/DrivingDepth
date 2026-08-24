# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import torch.nn as nn
from addict import Dict
from omegaconf import DictConfig, OmegaConf

from drivingdepth.cfg import create_object
from drivingdepth.utils.alignment import (
    ROE_align_scale,
    compute_sky_mask,
    set_sky_regions_to_max_depth,
)


def _wrap_cfg(cfg_obj):
    return OmegaConf.create(cfg_obj)


class DepthAnything3Net(nn.Module):
    """One depth-estimation branch: DinoV2 backbone + DPT/DualDPT head.

    Optional pieces:
    - cam_enc: encodes the given extrinsics/intrinsics into a camera token that
      is injected into the backbone at `alt_start`.
    - pix_scale_head: sparse-depth-prompted DPT predicting a per-pixel scale
      correction (and its confidence).

    forward returns an addict Dict with at least `depth`; with a pix_scale_head
    it also carries `org_depth`, `pix_scale` and `pix_scale_conf`.
    """

    def __init__(self, net, head, cam_enc=None, pix_scale_head=None):
        super().__init__()
        self.backbone = net if isinstance(net, nn.Module) else create_object(_wrap_cfg(net))
        self.head = head if isinstance(head, nn.Module) else create_object(_wrap_cfg(head))

        self.pix_scale_head = None
        if pix_scale_head is not None:
            self.pix_scale_head = (
                pix_scale_head
                if isinstance(pix_scale_head, nn.Module)
                else create_object(_wrap_cfg(pix_scale_head))
            )

        self.cam_enc = None
        if cam_enc is not None:
            self.cam_enc = (
                cam_enc if isinstance(cam_enc, nn.Module) else create_object(_wrap_cfg(cam_enc))
            )

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        sp_depth: torch.Tensor | None = None,
        views: int = 6,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: input images (B, N, 3, H, W)
            extrinsics: camera extrinsics (B, N, 4, 4); None disables the camera token
            intrinsics: camera intrinsics (B, N, 3, 3)
            sp_depth: packed sparse depth prompt (B, N, C, H, W)
            views: number of cameras per frame, so the backbone can split N = frames x views
        """
        if extrinsics is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
        else:
            cam_token = None

        feats, depth_fusion_feats = self.backbone(
            x, cam_token=cam_token, sp_depth=sp_depth, views=views
        )
        H, W = x.shape[-2], x.shape[-1]

        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self.head(feats, H, W, patch_start_idx=0)
            if self.pix_scale_head is not None and sp_depth is not None:
                depth_feats = depth_fusion_feats if depth_fusion_feats != [] else feats
                prompt_da = torch.concat([sp_depth, output.depth.unsqueeze(2)], dim=2)
                output = self._process_pix_scale_head(
                    depth_feats, H, W, output, prompt_da=prompt_da, views=views
                )
                output.org_depth = output.depth
                output.depth = output.depth * output.pix_scale.view(output.depth.shape)

        return output

    def _process_pix_scale_head(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        output: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run the pix_scale head and merge its outputs into `output`."""
        pix_scale_output = self.pix_scale_head(feats, H, W, patch_start_idx=0, **kwargs)
        output.pix_scale = pix_scale_output.pix_scale
        # Use `in` rather than attribute access: addict.Dict fabricates missing keys
        if "pix_scale_conf" in pix_scale_output and isinstance(
            pix_scale_output.pix_scale_conf, torch.Tensor
        ):
            output.pix_scale_conf = pix_scale_output.pix_scale_conf
        return output


class NestedDepthAnything3Net(nn.Module):
    """Two branches: `da3` predicts the depth, `da3_metric` only supplies the sky mask.

    Depth is composed as `org_depth x pix_scale x scale_factor`, where
    `scale_factor` comes from a ROE fit of the dense prediction against the
    sparse LiDAR depth over non-sky pixels. Sky pixels are then flattened to the
    99th percentile of the non-sky depth (capped at 400 m).
    """

    def __init__(self, anyview: DictConfig, metric: DictConfig):
        super().__init__()
        self.da3 = create_object(anyview)
        self.da3_metric = create_object(metric)

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        sp_depth: torch.Tensor | None = None,
        ref_sp_depth: torch.Tensor | None = None,
        views: int = 6,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: input images (B, N, 3, H, W)
            extrinsics: camera extrinsics (B, N, 4, 4)
            intrinsics: camera intrinsics (B, N, 3, 3)
            sp_depth: packed sparse depth prompt (B, N, C, H, W)
            ref_sp_depth: raw sparse depth (B, N, 1, H, W) used for ROE scale alignment
            views: number of cameras per frame
        """
        output = self.da3(x, extrinsics, intrinsics, sp_depth=sp_depth, views=views)
        metric_output = self.da3_metric(x)

        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)
        # Least-squares (ROE) fit against the sparse depth gives one global scale
        output.depth, output.scale_factor = self._align_scale_to_sparse_depth(
            output.depth, ref_sp_depth.view(output.depth.shape), non_sky_mask
        )
        output.org_depth = output.org_depth * output.scale_factor.view(-1, 1, 1, 1)
        output = self._handle_sky_regions(output, non_sky_mask)

        return output

    def _align_scale_to_sparse_depth(
        self,
        dense_depth: torch.Tensor,
        sp_depth: torch.Tensor,
        non_sky_mask: torch.Tensor | None = None,
    ):
        """Scale the dense depth to the sparse depth with a per-sample ROE fit."""
        align_mask = sp_depth > 0.2
        if non_sky_mask is not None:
            align_mask = align_mask & non_sky_mask.view(align_mask.shape)
        scale_factor = ROE_align_scale(dense_depth, sp_depth, align_mask).detach()
        return scale_factor.view(-1, 1, 1, 1).expand_as(dense_depth) * dense_depth, scale_factor

    def _handle_sky_regions(
        self,
        output: Dict[str, torch.Tensor],
        non_sky_mask: torch.Tensor,
        sky_depth_def: float = 400.0,
    ) -> Dict[str, torch.Tensor]:
        """Flatten sky pixels to the 99th percentile of the non-sky depth."""
        output.sky_mask = ~non_sky_mask
        # Subsample before torch.quantile, which refuses very large inputs
        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = min(torch.quantile(sampled_depth, 0.99), sky_depth_def)

        output.depth, output.depth_conf = set_sky_regions_to_max_depth(
            output.depth, output.depth_conf, non_sky_mask, max_depth=non_sky_max
        )
        return output
