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

"""
Alignment utilities for depth estimation and metric scaling.
"""

from typing import Tuple
import torch
from drivingdepth.utils.moge_alignment import align_depth_scale
import utils3d


def ROE_align_scale(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: torch.Tensor
):
    bs = pred_depth.shape[0]
    pred_points_lr, gt_points_lr, lr_mask = utils3d.pt.masked_nearest_resize(pred_depth, gt_depth, mask=mask, size=(128, 128))
    scale = align_depth_scale(pred_points_lr.view(bs, -1), gt_points_lr.view(bs, -1), lr_mask.view(bs, -1).float(), trunc=1)
    return scale


def compute_sky_mask(sky_prediction: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """
    Compute non-sky mask from sky prediction.

    Args:
        sky_prediction: Sky prediction tensor
        threshold: Threshold for sky classification

    Returns:
        Boolean mask where True indicates non-sky regions
    """
    return sky_prediction < threshold


def set_sky_regions_to_max_depth(
    depth: torch.Tensor,
    depth_conf: torch.Tensor,
    non_sky_mask: torch.Tensor,
    max_depth: float = 400.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Set sky regions to maximum depth and high confidence.

    Args:
        depth: Depth tensor
        depth_conf: Depth confidence tensor
        non_sky_mask: Non-sky region mask
        max_depth: Maximum depth value for sky regions

    Returns:
        Tuple of (updated_depth, updated_depth_conf)
    """
    depth = depth.clone()

    # Set sky regions to max depth and high confidence
    depth[~non_sky_mask] = max_depth
    if depth_conf is not None:
        depth_conf = depth_conf.clone()
        depth_conf[~non_sky_mask] = 1.0
        return depth, depth_conf
    else:
        return depth, None
