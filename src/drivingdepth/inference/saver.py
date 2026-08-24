import os
import json
import glob
import numpy as np
import torch
import cv2
from pathlib import Path
from collections import defaultdict
from drivingdepth.utils.visualize import visualize_depth


def _fast_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Fast Spearman rank correlation using argsort (5-6x faster than scipy)."""
    n = len(a)
    if n < 2:
        return float("nan")
    rank_a = np.empty(n, dtype=np.float64)
    rank_b = np.empty(n, dtype=np.float64)
    rank_a[np.argsort(a)] = np.arange(1, n + 1)
    rank_b[np.argsort(b)] = np.arange(1, n + 1)
    d = rank_a - rank_b
    return 1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1))


def _sobel_magnitude(arr: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)


def compute_grad_metrics(pred_m: np.ndarray, img_rgb: np.ndarray, edge_pct: float = 90.0,
                         mask: np.ndarray = None):
    """
    Gradient metrics between predicted depth and image.
    Args:
        pred_m: float32 (H, W) predicted depth in metres
        img_rgb: float32 (H, W, 3) normalized RGB image (from dataloader)
        edge_pct: percentile for edge threshold
        mask: bool (H, W), True = pixels to EXCLUDE (e.g. sky)
    Returns:
        dict with spearman, edge_cr, rev_edge_cr
    """
    # Convert normalized RGB to grayscale uint8
    img_gray = np.mean(img_rgb, axis=-1)
    if img_gray.max() <= 1.0:
        img_gray = (img_gray * 255).astype(np.float32)

    if img_gray.shape != pred_m.shape:
        img_gray = cv2.resize(img_gray, (pred_m.shape[1], pred_m.shape[0]),
                              interpolation=cv2.INTER_LINEAR)

    grad_i = _sobel_magnitude(img_gray)
    grad_d = _sobel_magnitude(pred_m)

    gi = grad_i.ravel()
    gd = grad_d.ravel()

    # Apply mask: exclude sky pixels
    if mask is not None:
        valid = ~mask.ravel()
        gi = gi[valid]
        gd = gd[valid]

    if gi.size == 0:
        return {"spearman": float("nan"), "edge_cr": float("nan"), "rev_edge_cr": float("nan")}

    rho = _fast_spearman(gi, gd)

    tau = np.percentile(gi, edge_pct)
    edge_mask = gi >= tau
    non_edge = ~edge_mask
    mean_edge = float(gd[edge_mask].mean()) if edge_mask.sum() > 0 else float("nan")
    mean_non = float(gd[non_edge].mean()) if non_edge.sum() > 0 else float("nan")
    edge_cr = mean_edge / (mean_non + 1e-8) if not np.isnan(mean_non) else float("nan")

    tau_d = np.percentile(gd, edge_pct)
    dedge_mask = gd >= tau_d
    non_dedge = ~dedge_mask
    mean_i_edge = float(gi[dedge_mask].mean()) if dedge_mask.sum() > 0 else float("nan")
    mean_i_nonedge = float(gi[non_dedge].mean()) if non_dedge.sum() > 0 else float("nan")
    rev_edge_cr = mean_i_edge / (mean_i_nonedge + 1e-8) if not np.isnan(mean_i_nonedge) else float("nan")

    return {"spearman": float(rho), "edge_cr": float(edge_cr), "rev_edge_cr": float(rev_edge_cr)}


def compute_per_view_metrics(pred: torch.Tensor, gt: torch.Tensor,
                             sky_mask: torch.Tensor = None,
                             img_rgb: np.ndarray = None):
    """
    Compute depth metrics + gradient metrics for a single view.
    Args:
        pred: [H, W] predicted depth (meters)
        gt: [H, W] ground truth depth (meters)
        sky_mask: [H, W] bool, True = sky (excluded from depth metrics)
        img_rgb: [H, W, 3] float32 normalized RGB (for gradient metrics)
    Returns:
        dict with absrel, delta1, delta0.5, absin02, n_pixels,
        spearman, edge_cr, rev_edge_cr, or None if no valid pixels
    """
    p = pred.float().reshape(-1)
    g = gt.float().reshape(-1)
    valid = (g > 0) & (p > 0)
    if sky_mask is not None:
        valid = valid & (~sky_mask.reshape(-1))

    p, g = p[valid], g[valid]
    n = int(valid.sum().item())
    if n == 0:
        return None

    abs_rel = float((torch.abs(p - g) / g.clamp(min=1e-6)).mean().item())
    ratio = torch.max(p / g.clamp(min=1e-6), g / p.clamp(min=1e-6))
    delta1 = float((ratio < 1.25).float().mean().item() * 100)
    delta05 = float((ratio < np.sqrt(1.25)).float().mean().item() * 100)
    abs_in02 = float((torch.abs(p - g) < 0.2).float().mean().item() * 100)

    result = {
        "absrel": abs_rel,
        "delta1": delta1,
        "delta0.5": delta05,
        "absin02": abs_in02,
        "n_pixels": n,
    }

    # Gradient metrics
    if img_rgb is not None:
        pred_np = pred.cpu().float().numpy()
        sky_np = sky_mask.cpu().numpy() if sky_mask is not None else None
        grad_m = compute_grad_metrics(pred_np, img_rgb, mask=sky_np)
        result.update(grad_m)

    return result


def save_sparse_depth(sparse_depth: torch.Tensor, path: str):
    """Save sparse GT depth as compressed npz. sparse_depth in meters, [H, W]."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = sparse_depth.cpu().float().numpy()
    np.savez_compressed(path, sparse_depth=d)


def save_confidence(conf: torch.Tensor, path: str):
    """Save confidence as uint16 PNG. conf in [0,1], stored as conf*60000."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c = conf.cpu().float().numpy()
    c_uint16 = np.clip(c * 60000, 0, 65534).astype(np.uint16)
    cv2.imwrite(path, c_uint16)


def save_depth_uint16(depth: torch.Tensor, path: str, scale: float = 100.0):
    """Save depth map as uint16 PNG. depth in meters, stored as depth*scale."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = depth.cpu().float().numpy()
    d_uint16 = np.clip(d * scale, 0, 65534).astype(np.uint16)
    cv2.imwrite(path, d_uint16)


def save_pix_scale_uint16(pix_scale: torch.Tensor, path: str, scale: float = 1000.0):
    """Save pix_scale as uint16 PNG. pix_scale stored as pix_scale*scale."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ps = pix_scale.cpu().float().numpy()
    ps_uint16 = np.clip(ps * scale, 0, 65534).astype(np.uint16)
    cv2.imwrite(path, ps_uint16)


def save_sky_mask(sky_mask: torch.Tensor, path: str):
    """Save sky_mask as uint8 PNG (0 or 255)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m = (sky_mask.cpu().bool().numpy() * 255).astype(np.uint8)
    cv2.imwrite(path, m)


def save_depth_vis(depth: torch.Tensor, path: str):
    """Save depth visualization as PNG using Spectral colormap."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = depth.cpu().float().numpy()
    vis = visualize_depth(d)  # returns uint8 RGB (H, W, 3)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, vis_bgr)


_GRAD_KEYS = {"spearman", "edge_cr", "rev_edge_cr"}


def merge_inference_jsons(save_dir: str, ds_name: str):
    """Merge all per-batch JSONs into a single summary.json."""
    ds_dir = os.path.join(save_dir, ds_name)
    batch_json_dir = os.path.join(ds_dir, "batch_jsons")
    batch_jsons = sorted(glob.glob(os.path.join(batch_json_dir, "rank*_batch*.json")))

    all_samples = []
    for jf in batch_jsons:
        with open(jf, "r") as f:
            data = json.load(f)
        rel_json = os.path.relpath(jf, ds_dir)
        for s in data.get("samples", []):
            s["source_json"] = rel_json
            all_samples.append(s)

    # Aggregate metrics
    per_camera = defaultdict(list)
    per_frame_idx = defaultdict(list)
    all_metrics = []
    for s in all_samples:
        m = s.get("metrics")
        if m is None:
            continue
        all_metrics.append(m)
        # Extract camera name from rel_path (last dir before filename)
        rel = s.get("rel_path", "")
        parts = Path(rel).parts
        cam_name = parts[-2] if len(parts) >= 2 else "unknown"
        per_camera[cam_name].append(m)
        per_frame_idx[s.get("frame_idx", -1)].append(m)

    def weighted_avg(metrics_list):
        if not metrics_list:
            return {}
        total_n = sum(m["n_pixels"] for m in metrics_list)
        if total_n == 0:
            return {}
        depth_keys = ["absrel", "delta1", "delta0.5", "absin02"]
        result = {k: sum(m[k] * m["n_pixels"] for m in metrics_list) / total_n for k in depth_keys}
        # Gradient metrics: simple mean (not pixel-weighted)
        for k in _GRAD_KEYS:
            vals = [m[k] for m in metrics_list if k in m and not np.isnan(m[k])]
            if vals:
                result[k] = float(np.mean(vals))
        result["n_samples"] = len(metrics_list)
        return result

    summary = {
        "dataset": ds_name,
        "total_samples": len(all_samples),
        "metrics_avg": weighted_avg(all_metrics),
        "per_camera_metrics": {cam: weighted_avg(ms) for cam, ms in sorted(per_camera.items())},
        "per_frame_idx_metrics": {str(fi): weighted_avg(ms) for fi, ms in sorted(per_frame_idx.items())},
        "samples": all_samples,
    }

    summary_path = os.path.join(ds_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary_path
