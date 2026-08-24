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

import os
import time
import json
import uuid
import torch
from pathlib import Path
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from typing import Dict

from addict import Dict as AdDict
from safetensors.torch import load_file

from drivingdepth.cfg import create_object, load_config
from drivingdepth.inference.saver import (
    compute_per_view_metrics,
    save_depth_uint16,
    save_sparse_depth,
    save_confidence,
    save_pix_scale_uint16,
    save_sky_mask,
    save_depth_vis,
    merge_inference_jsons,
)


def _is_str_field(v):
    if isinstance(v, str):
        return True
    if isinstance(v, (list, tuple)) and len(v) > 0:
        flat = v[0]
        while isinstance(flat, (list, tuple)) and len(flat) > 0:
            flat = flat[0]
        return isinstance(flat, str)
    return False


RUN_ID = time.strftime("%Y%m%d%H%M%S", time.localtime()) + f"_{uuid.getnode()}"


class DepthAnything3Inferencer:
    """Depth Anything 3 inference / evaluation driver, data-parallel via accelerate."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.cfg = config.inference
        self.accelerator = Accelerator(
            mixed_precision=self.cfg.get("mixed_precision", "bf16")
        )
        self.device = self.accelerator.device

        self.save_dir = self.cfg.get("save_dir", None)
        if not self.save_dir:
            raise ValueError(
                "inference.save_dir is required: metrics are computed offline by merging "
                "the per-batch JSONs written there, not during the forward pass"
            )
        log_root = os.path.join(self.save_dir, "logs")
        os.makedirs(log_root, exist_ok=True)
        self.log_path = os.path.join(log_root, f"{RUN_ID}.log")

        self._setup_model()
        self._setup_data()

    def _log2file_print(self, msg):
        timestamp = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S', time.localtime())}]"
        msg = f"{timestamp} {msg}"
        print(msg)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")


    def _setup_model(self):
        """Build the model and load weights (.pth from finetuning, or raw .safetensors)."""
        model_config = load_config(self.config.model.config_path)
        self.model = create_object(model_config)

        file_name = self.cfg.load_from
        if file_name.endswith(".pth"):
            pretrained_state_dict = torch.load(file_name, map_location='cpu')["model_state_dict"]
        elif file_name.endswith(".safetensors"):
            pretrained_state_dict = load_file(file_name)
            for key in list(pretrained_state_dict.keys()):
                if key.startswith("model."):
                    pretrained_state_dict[key.replace("model.", "")] = pretrained_state_dict.pop(key)
        else:
            raise ValueError(f"unsupported checkpoint format: {file_name}")

        missing, unexpected = self.model.load_state_dict(pretrained_state_dict, strict=False)

        for param in self.model.parameters():
            param.requires_grad = False

        if self.accelerator.is_main_process:
            total_params = sum(p.numel() for p in self.model.parameters())
            self._log2file_print(
                f"loaded weights: {file_name} | {total_params:,} params | "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

    def _setup_data(self):
        """Build one val DataLoader per entry in config.data.datasets."""
        self.val_loaders: Dict[str, DataLoader] = {}
        num_workers = self.cfg.get("num_workers", 16)

        for ds_name, ds_cfg in self.config.data.datasets.items():
            dataset_module, dataset_class = ds_cfg.dataset_class.rsplit('.', 1)
            module = __import__(dataset_module, fromlist=[dataset_class])
            DatasetClass = getattr(module, dataset_class)

            val_ds = DatasetClass(**dict(ds_cfg.val))
            self.val_loaders[ds_name] = DataLoader(
                val_ds,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            if self.accelerator.is_main_process:
                self._log2file_print(f"[Data] {ds_name}: val={len(val_ds)} samples")
    def _prepare_batch_data(self, batch):
        """Move a batch onto the device, keeping string fields as-is."""
        batch_tensors = AdDict()
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_tensors[key] = value.to(self.device)
            elif _is_str_field(value):
                batch_tensors[key] = value
            elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                batch_tensors[key] = [elem.to(self.device) for elem in value]
            else:
                batch_tensors[key] = torch.tensor(value, device=self.device)
                
        return batch_tensors


    def _validate_single_loader(self, loader, desc="Validation", ds_name=""):
        """Run inference over one DataLoader and write per-batch results to disk.

        No metric is averaged here on purpose: a per-batch mean weights every batch
        equally, so it drifts with batch_size / world size. The reported numbers come
        from merging the per-batch JSONs once the loop is done.
        """
        squeeze_frames_idx = lambda x, n: x.view(x.shape[0], -1, *x.shape[n:])

        zero_prompt_da = self.cfg.get('zero_prompt_da', False)

        with torch.no_grad():
            iterator = tqdm(loader, desc=desc, disable=not self.accelerator.is_main_process)
            for batch_idx, batch in enumerate(iterator):
                batch_data = self._prepare_batch_data(batch)

                sp_depth_input = squeeze_frames_idx(batch_data.packed_sparse_depths, -3)
                if zero_prompt_da:
                    sp_depth_input = torch.zeros_like(sp_depth_input)
                model_output = self.model(
                    squeeze_frames_idx(batch_data.rgbs, -3).permute(0, 1, 4, 2, 3),
                    extrinsics  = squeeze_frames_idx(batch_data.extrinsics_w2c, -2),
                    intrinsics  = squeeze_frames_idx(batch_data.intrinsics, -2),
                    sp_depth    = sp_depth_input,
                    ref_sp_depth = squeeze_frames_idx(batch_data.sparse_depths, -2).unsqueeze(-3),
                    views       = batch_data.sparse_depths.shape[2],
                )

                self._save_inference_batch(
                    model_output, batch_data, batch_idx, ds_name, self.save_dir
                )

                self.accelerator.wait_for_everyone()

        self.accelerator.wait_for_everyone()

        metrics = {}
        if self.accelerator.is_main_process:
            summary_path = merge_inference_jsons(self.save_dir, ds_name)
            with open(summary_path) as f:
                summary = json.load(f)
            metrics = dict(summary.get("metrics_avg", {}))
            n_views = metrics.pop("n_samples", 0)
            self._log2file_print(
                f"[InferenceSave] Summary saved: {summary_path} ({n_views} views)"
            )
        self.accelerator.wait_for_everyone()
        return metrics

    def _save_inference_batch(self, model_output, batch_data, batch_idx, ds_name, save_dir):
        """Save inference results for one batch on this rank."""
        rank = self.accelerator.process_index
        # model_output.depth: [B, N, H, W] (N = n_frames * n_views after squeeze)
        B = model_output.depth.shape[0]
        N = model_output.depth.shape[1]

        # Determine n_frames and n_views from batch_data
        # sparse_depths shape before squeeze: [B, n_frames, n_views, H, W]
        n_frames = batch_data.sparse_depths.shape[1]
        n_views = batch_data.sparse_depths.shape[2]

        # GT depth for metrics: sparse_depths is [B, n_frames, n_views, H, W]
        gt_depth_raw = batch_data.sparse_depths
        gt_depth = gt_depth_raw.view(B, -1, *gt_depth_raw.shape[-2:])  # [B, N, H, W]

        # RGB images: batch_data.rgbs is [B, n_frames, n_views, H, W, 3] (normalized)
        rgbs = batch_data.rgbs  # [B, n_frames, n_views, H, W, 3]

        # rel_paths: after default collate, indexed as [frame][view][batch]
        rel_paths = batch_data.get('rel_paths', None)

        ds_dir = os.path.join(save_dir, ds_name)
        samples = []

        for b in range(B):
            for n in range(N):
                frame_idx = n // n_views
                view_idx = n % n_views

                # Get rel_path
                if rel_paths is not None:
                    try:
                        rel_path = rel_paths[frame_idx][view_idx][b]
                    except (IndexError, TypeError):
                        rel_path = f"rank{rank}_b{batch_idx}_s{b}_f{frame_idx}_v{view_idx}"
                else:
                    rel_path = f"rank{rank}_b{batch_idx}_s{b}_f{frame_idx}_v{view_idx}"

                stem = str(Path(rel_path).with_suffix(""))

                # Save files
                depth_path = os.path.join(ds_dir, "depth", f"{stem}.png")
                depth_vis_path = os.path.join(ds_dir, "depth_vis", f"{stem}.png")
                sky_mask_path = os.path.join(ds_dir, "sky_mask", f"{stem}.png")
                pix_scale_path = os.path.join(ds_dir, "pix_scale", f"{stem}.png")
                conf_path = os.path.join(ds_dir, "confidence", f"{stem}.png")
                sparse_gt_path = os.path.join(ds_dir, "sparse_gt", f"{stem}.npz")

                pred_depth = model_output.depth[b, n]
                save_depth_uint16(pred_depth, depth_path)
                save_depth_vis(pred_depth, depth_vis_path)

                # Save sparse GT depth for offline ROE evaluation
                save_sparse_depth(gt_depth[b, n], sparse_gt_path)

                if model_output.get('sky_mask') is not None:
                    save_sky_mask(model_output.sky_mask[b, n], sky_mask_path)

                if model_output.get('pix_scale') is not None:
                    save_pix_scale_uint16(model_output.pix_scale[b, n], pix_scale_path)

                if model_output.get('pix_scale_conf') is not None:
                    conf_tensor = model_output.pix_scale_conf[b, n]
                    if conf_tensor.dim() == 3:
                        conf_tensor = conf_tensor[0]
                    save_confidence(conf_tensor, conf_path)

                # Get RGB image for gradient metrics (denormalize from ImageNet norm)
                _img = rgbs[b, frame_idx, view_idx].cpu().float()  # [H, W, 3]
                _mean = torch.tensor([0.485, 0.456, 0.406])
                _std = torch.tensor([0.229, 0.224, 0.225])
                img_rgb = (_img * _std + _mean).clamp(0, 1).numpy()  # [H, W, 3] in [0, 1]

                # Per-view metrics (depth + gradient)
                sky = model_output.sky_mask[b, n] if model_output.get('sky_mask') is not None else None
                metrics = compute_per_view_metrics(pred_depth, gt_depth[b, n], sky, img_rgb=img_rgb)

                # Intrinsics / extrinsics (prefer origin_extrinsics_w2c for absolute coords)
                intrinsic = batch_data.intrinsics[b, frame_idx, view_idx].cpu().numpy().tolist()
                origin_w2c = batch_data.get('origin_extrinsics_w2c')
                if origin_w2c is not None:
                    extrinsic_w2c = origin_w2c[b, frame_idx, view_idx].cpu().numpy().tolist()
                else:
                    extrinsic_w2c = batch_data.extrinsics_w2c[b, frame_idx, view_idx].cpu().numpy().tolist()

                samples.append({
                    "batch_i": b,
                    "frame_idx": frame_idx,
                    "view_idx": view_idx,
                    "rel_path": rel_path,
                    "depth_path": os.path.relpath(depth_path, save_dir),
                    "depth_vis_path": os.path.relpath(depth_vis_path, save_dir),
                    "sky_mask_path": os.path.relpath(sky_mask_path, save_dir) if model_output.get('sky_mask') is not None else None,
                    "pix_scale_path": os.path.relpath(pix_scale_path, save_dir) if model_output.get('pix_scale') is not None else None,
                    "conf_path": os.path.relpath(conf_path, save_dir) if model_output.get('pix_scale_conf') is not None else None,
                    "sparse_gt_path": os.path.relpath(sparse_gt_path, save_dir),
                    "intrinsic": intrinsic,
                    "extrinsic_w2c": extrinsic_w2c,
                    "metrics": metrics,
                })

        # Write per-batch JSON into batch_jsons/ subfolder
        batch_json_dir = os.path.join(ds_dir, "batch_jsons")
        os.makedirs(batch_json_dir, exist_ok=True)
        json_path = os.path.join(batch_json_dir, f"rank{rank}_batch{batch_idx:04d}.json")
        with open(json_path, "w") as f:
            json.dump({"dataset": ds_name, "rank": rank, "batch_idx": batch_idx, "samples": samples}, f, indent=2)

    def validate(self):
        """Evaluate each dataset; returns {ds_name/metric: value} plus a cross-dataset mean.

        Every number comes from the dataset's summary.json (pixel-weighted over all
        views); delta1 / delta0.5 / absin02 are percentages.
        """
        self.model.eval()

        all_results = {}
        for ds_name, loader in self.val_loaders.items():
            ds_metrics = self._validate_single_loader(loader, desc=f"Val-{ds_name}", ds_name=ds_name)
            all_results[ds_name] = ds_metrics

            if self.accelerator.is_main_process:
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in ds_metrics.items())
                self._log2file_print(f"[{ds_name}] summary.json: {metrics_str}")

        merged = {}
        for ds_name, ds_metrics in all_results.items():
            for k, v in ds_metrics.items():
                merged[f"{ds_name}/{k}"] = v

        # With more than one dataset, also report the per-metric mean across them
        all_keys = set()
        for ds_metrics in all_results.values():
            all_keys.update(ds_metrics.keys())
        for k in sorted(all_keys):
            vals = [dm[k] for dm in all_results.values() if k in dm]
            if vals:
                merged[k] = sum(vals) / len(vals)

        return merged

    def run(self):
        """Run inference + evaluation; returns a metrics dict."""
        # autocast / device placement only; no DDP wrap (inference needs no gradient sync)
        self.model = self.accelerator.prepare_model(self.model, evaluation_mode=True)
        for ds_name in list(self.val_loaders.keys()):
            self.val_loaders[ds_name] = self.accelerator.prepare(self.val_loaders[ds_name])

        metrics = self.validate()

        if self.accelerator.is_main_process:
            self._log2file_print(f"evaluation done: {metrics}")
            self._log2file_print(f"output dir: {self.save_dir}")
        return metrics
