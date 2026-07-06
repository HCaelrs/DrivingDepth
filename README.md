# DrivingDepth: Sparse-Prompted Pixel-wise Scale Correction for Driving Depth Estimation

<div align="center">


[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://hcaelrs.github.io/DrivingDepth-page/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.31488-b31b1b)](https://arxiv.org/pdf/2606.31488)

</div>

<p align="center">
  <img src="assets/teaser.png" width="90%" alt="DrivingDepth teaser">
</p>

DrivingDepth is a sparse-prompted metric depth framework for autonomous driving. It resolves the **geometry--scale conflict** between dense visual geometry from depth foundation models and sparse, noisy metric anchors from projected LiDAR by learning a residual pixel-wise scale correction on top of a frozen foundation prior.

## 📰 News

- **2026-06-30:** 🎬 Paper is released: [DrivingDepth arXiv](https://arxiv.org/pdf/2606.31488). Code will be released very soon.
- **2026-05-14:** 🎳 Demo is released: [DrivingDepth-page](https://hcaelrs.github.io/DrivingDepth-page/). Paper under writing, code to be organized soon.

## ✨ Highlights

- **Geometry-preserving metric calibration:** treats sparse LiDAR as geometric prompts rather than dense supervision targets.
- **Residual pixel-wise scale correction:** keeps the frozen DA3 prior as dense visual geometry and learns only the per-pixel scale needed for metric alignment.
- **Low-cost adaptation:** freezes the foundation model and trains only a lightweight scale head and feature adapter, fitting the full framework on a single 8-GPU node.
- **Sparse-aware prompting:** injects sparse-depth cues through a Geometry-Preserving Feature Adapter and a Sparse-Aware Pixel-Scale Head.
- **Robust driving depth:** uses learned confidence, surface-normal regularization, and scale smoothness to handle noisy or misaligned LiDAR projections.
- **Strong performance:** on nuScenes with 4-frame surround-view input, DrivingDepth achieves **11.19 AbsRel** and **5.741 EdgeCR**, outperforming MapAnything (**11.99 / 1.914**) in both metric accuracy and geometric consistency.

## 📌 Abstract

Dense depth estimation for autonomous driving faces a *geometry--scale conflict*: depth foundation models deliver pixel-aligned dense visual geometry without reliable metric scale, while projected LiDAR provides metric anchors that are sparse, noisy, and misaligned with image structures. Existing sparse-prompted methods incorporate LiDAR by regenerating depth from scratch, overriding the foundation model's coherent geometry and producing structural artifacts on visually continuous surfaces. Our key insight is that foundation models already capture geometrically coherent relative depth; no additional surface structure learning is required—only a per-pixel scale factor mapping relative geometry to metric coordinates. Based on this, we propose DrivingDepth, which treats sparse LiDAR as *geometric prompts* that locally calibrate a frozen foundation prior through residual pixel-wise scale correction, preserving dense visual geometry by construction. On nuScenes with 4-frame surround-view input, DrivingDepth achieves an AbsRel of 11.19 and an EdgeCR of 5.741, outperforming MapAnything (11.99/1.914) by simultaneously delivering SOTA metric accuracy and geometric consistency.

## 🧠 Method Overview

<p align="center">
  <img src="assets/structure.png" width="95%" alt="DrivingDepth architecture">
</p>

DrivingDepth starts from a frozen Depth Anything 3 (DA3) prior and performs minimal-intervention metric calibration:

1. **Frozen visual geometry prior:** DA3 maps surround-view images and camera parameters to dense, coherent depth priors.
2. **Geometry-Preserving Feature Adapter:** sparse-depth tokens are fused with image tokens and propagated along constrained frame-view connections.
3. **Sparse-Aware Pixel-Scale Head:** multi-scale features and LiDAR prompts predict a pixel-wise correction map and confidence map.
4. **Metric output:** the corrected depth is produced by multiplying the frozen prior with the learned scale map and a clip-level global scale factor.

## 🖼️ Qualitative Results

<p align="center">
  <img src="assets/comparison_stacked.png" width="95%" alt="Qualitative comparison on nuScenes (rows 1–2) and DDAD (rows 3–4). Columns: RGB, MOGE-2, PriorDA, MapAnything, DA3, scale map, DrivingDepth.">
</p>

DrivingDepth preserves RGB-depth consistency and produces metrically calibrated depth maps with fewer structural artifacts than sparse-prompted methods that regenerate depth from fused features.

## 📚 Citation

If you find this work useful, please consider citing:

```bibtex
@article{huang2026drivingdepth,
  title={DrivingDepth: Sparse-Prompted Pixel-wise Scale Correction for Driving Depth Estimation},
  author={Huang, Chi and Zhang, Wenhao and Yin, Hang and Wang, YuAn and Li, Hao and Wang, Bosheng and Sun, Xun and Wang, Liang},
  journal={arXiv preprint arXiv:2606.31488},
  year={2026}
}
```
