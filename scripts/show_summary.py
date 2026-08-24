#!/usr/bin/env python3
"""
Print per-camera metrics table from a summary.json file.

Usage:
    python scripts/show_summary.py inference_output-4f1s2i/nuscenes/summary.json
    python scripts/show_summary.py inference_output/ddad/summary.json
"""

import argparse
import json
import sys
from collections import defaultdict

import numpy as np


_GRAD_KEYS = {"spearman", "edge_cr", "rev_edge_cr"}


def _weighted_avg(metrics_list):
    if not metrics_list:
        return {}
    total_n = sum(m.get("n_pixels", 0) for m in metrics_list)
    if total_n == 0:
        return {}
    depth_keys = ["absrel", "delta1", "delta0.5", "absin02"]
    result = {k: sum(m.get(k, 0) * m.get("n_pixels", 0) for m in metrics_list) / total_n for k in depth_keys}
    for k in _GRAD_KEYS:
        vals = [m[k] for m in metrics_list if k in m and m[k] is not None and not np.isnan(m[k])]
        if vals:
            result[k] = float(np.mean(vals))
    result["n_samples"] = len(metrics_list)
    return result


def _make_header(has_grad: bool) -> str:
    h = (f"{'Camera':<28} {'N_samples':>9} {'AbsRel':>8} "
         f"{'delta1(%)':>10} {'delta0.5(%)':>12} {'AbsIn02(%)':>11}")
    if has_grad:
        h += f"  {'Spearman':>9}  {'EdgeCR':>8}  {'RevEdgeCR':>10}"
    return h


def _fmt_row(name: str, m: dict, has_grad: bool) -> str:
    if not m:
        return f"{name:<28} {'N/A':>9}"
    n_samples = m.get("n_samples", 0)
    row = (f"{name:<28} {n_samples:>9d} "
           f"{m.get('absrel', 0):>8.4f} "
           f"{m.get('delta1', 0):>10.2f} "
           f"{m.get('delta0.5', 0):>12.2f} "
           f"{m.get('absin02', 0):>11.2f}")
    if has_grad:
        sp = m.get("spearman", float("nan"))
        ec = m.get("edge_cr", float("nan"))
        rec = m.get("rev_edge_cr", float("nan"))
        sp_str = f"{sp:>9.4f}" if not (sp is None or np.isnan(sp)) else f"{'N/A':>9}"
        ec_str = f"{ec:>8.3f}" if not (ec is None or np.isnan(ec)) else f"{'N/A':>8}"
        rec_str = f"{rec:>10.3f}" if not (rec is None or np.isnan(rec)) else f"{'N/A':>10}"
        row += f"  {sp_str}  {ec_str}  {rec_str}"
    return row


def main():
    parser = argparse.ArgumentParser(description="Print metrics from summary.json")
    parser.add_argument("summary_json", type=str, help="Path to summary.json")
    args = parser.parse_args()

    with open(args.summary_json) as f:
        data = json.load(f)

    ds_name = data.get("dataset", "unknown").upper()
    total = data.get("total_samples", 0)
    avg = data.get("metrics_avg", {})
    per_cam = data.get("per_camera_metrics", {})

    # Compute per_frame_idx from samples (works regardless of summary version)
    per_frame = data.get("per_frame_idx_metrics", {})
    if not per_frame:
        frame_groups = defaultdict(list)
        for s in data.get("samples", []):
            m = s.get("metrics")
            if m is None:
                continue
            fi = s.get("frame_idx", -1)
            frame_groups[fi].append(m)
        if frame_groups:
            per_frame = {str(fi): _weighted_avg(ms) for fi, ms in sorted(frame_groups.items())}

    has_grad = any(k in avg for k in _GRAD_KEYS)

    header = _make_header(has_grad)
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print(f"  Dataset: {ds_name}  |  Total samples: {total}")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    for cam in sorted(per_cam.keys()):
        print(_fmt_row(cam, per_cam[cam], has_grad))

    print(sep)
    print(_fmt_row("ALL", avg, has_grad))

    # Per frame_idx breakdown
    if per_frame:
        print(f"\n{sep}")
        print(f"  Per frame_idx breakdown:")
        print(sep)
        print(header)
        print(sep)
        for fi in sorted(per_frame.keys(), key=lambda x: int(x) if x.lstrip('-').isdigit() else 0):
            print(_fmt_row(f"frame={fi}", per_frame[fi], has_grad))
        print(sep)

    print()


if __name__ == "__main__":
    main()
