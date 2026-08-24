import utils3d
import torch
import torch.nn.functional as F


def multi_resolution_depth_pack(sp_depth, mask, level=[0, 1, 2, 3], norm=True, resize=True):
    """Stack a sparse depth map at several resolutions.

    Args:
        sp_depth: [B, V, 1, H, W] sparse depth, 0 = invalid
        mask: validity mask, same shape as `sp_depth`
        level: downsampling exponents; level 0 is the input resolution
        norm: divide by the per-sample mean of the valid depths
        resize: bilinearly upsample every level back to H x W and concat on
            the channel axis; when False, return the raw per-level lists
    Returns:
        (depth, mask) — concatenated tensors if `resize`, else lists of tensors
    """
    assert len(sp_depth.shape) == 5 and sp_depth.shape[-3] == 1
    if norm:
        norm_coff = (sp_depth * (sp_depth > 0)).view(sp_depth.shape[0], -1).sum(dim=1) / \
            (sp_depth > 0).view(sp_depth.shape[0], -1).sum(dim=1)
        depth = sp_depth / norm_coff.view(sp_depth.shape[0], 1, 1, 1, 1)
    else:
        depth = sp_depth
    out_depth = []
    out_mask = []
    size = depth.shape[-2:]
    for i in level:
        if i == 0:
            out_depth.append(depth)
            out_mask.append(mask)
            continue
        processed_depth, processed_mask = utils3d.pt.masked_nearest_resize(
            depth, mask=mask, size=[size[0] // 2**i, size[1] // 2**i]
        )
        if resize:
            processed_depth = F.interpolate(
                processed_depth.view(-1, *processed_depth.shape[-3:]),
                size, mode='bilinear').view(depth.shape)
            processed_mask = F.interpolate(
                processed_mask.float().view(-1, *processed_mask.shape[-3:]),
                size, mode='bilinear').view(depth.shape)
        out_depth.append(processed_depth)
        out_mask.append(processed_mask)
    if not resize:
        return out_depth, out_mask
    return torch.cat(out_depth, dim=-3), torch.cat(out_mask, dim=-3)
