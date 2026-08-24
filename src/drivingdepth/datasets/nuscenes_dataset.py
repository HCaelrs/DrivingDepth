import os
import warnings
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from einops import rearrange
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from drivingdepth.utils.depth_utils import multi_resolution_depth_pack
from drivingdepth.utils.geometry import affine_inverse

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

LIDAR_NAME = "LIDAR_TOP"


def get_packed_depth(sp_depth):
    """Build the multi-resolution sparse depth prompt: (n, m, 8, H, W)."""
    multi_reso_depth = multi_resolution_depth_pack(sp_depth, sp_depth > 0)
    return torch.concat([multi_reso_depth[0], multi_reso_depth[1]], dim=2)

class NuScenesMultiViewDataset(Dataset):
    """Multi-view + temporal nuScenes clips.

    For a clip of n frames x m cameras, __getitem__ returns:
        rgbs: (n, m, H, W, 3) float32, ImageNet-normalized
        intrinsics: (n, m, 3, 3)
        extrinsics_w2c: (n, m, 4, 4) world_to_camera, scene-normalized
        origin_extrinsics_w2c: (n, m, 4, 4) world_to_camera, absolute
        sparse_depths: (n, m, H, W) float32, 0 = invalid
        packed_sparse_depths: (n, m, 8, H, W) multi-resolution depth prompt
        rel_paths: (n, m) nuScenes image filenames, one per view
    """

    def __init__(
        self,
        nusc: str,
        data_root: str,
        cameras: List[str],
        target_size: Tuple[int, int] = (1008, 574),  # (W, H)
        num_frames: int = 10,   # consecutive frames sampled per scene
        stride: int = 2,        # step between clip start indices
        interval: int = 1,      # step between frames inside a clip
        min_depth: float = 1.0,
        max_depth: float = 80.0,
    ):
        self.nusc = NuScenes(version=nusc, dataroot=data_root, verbose=False)
        self.cameras = cameras
        self.m = len(cameras)
        self.target_w, self.target_h = target_size
        self.num_frames = num_frames
        self.stride = stride
        self.interval = interval
        self.min_depth = min_depth
        self.max_depth = max_depth

        # Each clip is a list of num_frames sample tokens
        self.clips: List[List[str]] = []
        self._build_clips()
        print(
            f"[NuScenesMultiViewDataset] Loaded {nusc} for {len(self.clips)} clips "
            f"(stride={stride}, num_frames={num_frames}, interval={interval})"
        )
    def _normalize_extrinsics(self, ex_t: torch.Tensor) -> torch.Tensor:
        """Put the clip in the first camera's frame and scale it to unit median baseline."""
        transform = affine_inverse(ex_t[:, :1])
        ex_t_norm = ex_t @ transform
        c2ws = affine_inverse(ex_t_norm)
        translations = c2ws[..., :3, 3]
        dists = translations.norm(dim=-1)
        median_dist = torch.median(dists)
        median_dist = torch.clamp(median_dist, min=1e-1)
        ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / median_dist
        return ex_t_norm

    def _build_clips(self):
        for scene in self.nusc.scene:
            samples = []
            token = scene["first_sample_token"]
            while token != "":
                samples.append(token)
                token = self.nusc.get("sample", token)["next"]

            if len(samples) < self.num_frames:
                continue  # skip short scenes

            required_span = (self.num_frames - 1) * self.interval + 1
            for start in range(0, len(samples) - required_span + 1, self.stride):
                indices = list(
                    range(start, start + self.num_frames * self.interval, self.interval)
                )
                indices = [i for i in indices if i < len(samples)]
                if len(indices) != self.num_frames:
                    continue
                clip = [samples[i] for i in indices]

                # Keep the clip only if every frame has LiDAR plus all requested cameras
                if all(
                    LIDAR_NAME in self.nusc.get("sample", t)["data"]
                    and all(cam in self.nusc.get("sample", t)["data"] for cam in self.cameras)
                    for t in clip
                ):
                    self.clips.append(clip)

    def __len__(self):
        return len(self.clips)
    def _get_camera_data(self, sample_token: str, cam_name: str, lidar_pc: LidarPointCloud):
        """Load one (frame, camera): RGB, scaled intrinsic, cam2world, sparse depth."""
        sample = self.nusc.get("sample", sample_token)
        cam_data = self.nusc.get("sample_data", sample["data"][cam_name])

        # --- Load & Resize Image ---
        img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        img_resized = cv2.resize(
            img, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA
        )
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # --- Intrinsic (scaled to target_size) ---
        cam_calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        K_scaled = np.array(cam_calib["camera_intrinsic"], dtype=np.float32)
        K_scaled[0, :] *= self.target_w / img.shape[1]
        K_scaled[1, :] *= self.target_h / img.shape[0]

        # --- Extrinsic: camera -> ego -> global ---
        lidar_data = self.nusc.get("sample_data", sample["data"][LIDAR_NAME])
        lidar_calib = self.nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        ego_pose_cam = self.nusc.get("ego_pose", cam_data["ego_pose_token"])

        def get_transform(translation, rotation):
            T_mat = np.eye(4, dtype=np.float32)
            T_mat[:3, :3] = Quaternion(rotation).rotation_matrix
            T_mat[:3, 3] = np.array(translation)
            return T_mat

        T_cam_to_ego = get_transform(cam_calib["translation"], cam_calib["rotation"])
        T_ego_to_global_cam = get_transform(ego_pose_cam["translation"], ego_pose_cam["rotation"])
        T_cam_to_global = T_ego_to_global_cam @ T_cam_to_ego
        # --- Sparse depth: project the LiDAR sweep into this camera ---
        depth_map = np.zeros((self.target_h, self.target_w), dtype=np.float32)
        pc = LidarPointCloud(lidar_pc.points.copy())

        # LiDAR -> ego -> camera
        pc.rotate(Quaternion(lidar_calib["rotation"]).rotation_matrix)
        pc.translate(np.array(lidar_calib["translation"]))
        cam_from_ego_rot = Quaternion(cam_calib["rotation"]).inverse
        pc.rotate(cam_from_ego_rot.rotation_matrix)
        pc.translate(-cam_from_ego_rot.rotate(np.array(cam_calib["translation"])))

        pts = pc.points[:3, :]
        depths = pts[2, :]
        mask = (depths > self.min_depth) & (depths < self.max_depth)
        pts, depths = pts[:, mask], depths[mask]

        if pts.shape[1] > 0:
            pts_2d = view_points(pts, K_scaled, normalize=True)
            u, v = pts_2d[0, :].astype(int), pts_2d[1, :].astype(int)
            valid = (u >= 0) & (u < self.target_w) & (v >= 0) & (v < self.target_h)
            u, v, depths = u[valid], v[valid], depths[valid]
            # Keep the nearest hit per pixel
            for ui, vi, di in zip(u, v, depths):
                if depth_map[vi, ui] == 0 or di < depth_map[vi, ui]:
                    depth_map[vi, ui] = di

        return {
            "rgb": img_rgb,                    # (H, W, 3)
            "intrinsic": K_scaled,             # (3, 3)
            "extrinsic_c2w": T_cam_to_global,  # (4, 4), camera -> world (global)
            "sparse_depth": depth_map,         # (H, W)
            "rel_path": cam_data["filename"],  # e.g. "samples/CAM_FRONT/xxx.jpg"
        }
    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        clip_tokens = self.clips[idx]
        n = len(clip_tokens)

        rgbs = np.empty((n, self.m, self.target_h, self.target_w, 3), dtype=np.float32)
        intrinsics = np.empty((n, self.m, 3, 3), dtype=np.float32)
        extrinsics_c2w = np.empty((n, self.m, 4, 4), dtype=np.float32)
        sparse_depths = np.zeros((n, self.m, self.target_h, self.target_w), dtype=np.float32)
        rel_paths = [["" for _ in range(self.m)] for _ in range(n)]

        # Load the LiDAR sweep once per frame and reuse it across cameras
        for t, sample_token in enumerate(clip_tokens):
            sample = self.nusc.get("sample", sample_token)
            lidar_token = sample["data"][LIDAR_NAME]
            lidar_path = os.path.join(
                self.nusc.dataroot, self.nusc.get("sample_data", lidar_token)["filename"]
            )
            try:
                lidar_pc = LidarPointCloud.from_file(lidar_path)
            except Exception as e:
                warnings.warn(f"Failed to load LiDAR {lidar_path}: {e}")
                lidar_pc = LidarPointCloud(np.zeros((4, 0), dtype=np.float32))

            for v, cam_name in enumerate(self.cameras):
                try:
                    data = self._get_camera_data(sample_token, cam_name, lidar_pc)
                    rgbs[t, v] = data["rgb"]
                    intrinsics[t, v] = data["intrinsic"]
                    extrinsics_c2w[t, v] = data["extrinsic_c2w"]
                    sparse_depths[t, v] = data["sparse_depth"]
                    rel_paths[t][v] = data["rel_path"]
                except Exception as e:
                    warnings.warn(f"Failed processing {sample_token}/{cam_name}: {e}")
                    # Leave zeros
        n, m = rgbs.shape[:2]
        w2c_tensor = torch.inverse(torch.tensor(extrinsics_c2w).clone())
        extrinsics_w2c = self._normalize_extrinsics(
            w2c_tensor.view(-1, 4, 4)[None].cpu()
        ).view(n, m, 4, 4)

        img_tensor = rearrange(torch.tensor(rgbs), "n m h w c -> (n m) c h w")
        rgbs = rearrange(NORMALIZE(img_tensor), "(n m) c h w -> n m h w c", n=n, m=m)
        sparse_depths = torch.tensor(sparse_depths)

        return {
            "rgbs": rgbs.contiguous(),
            "intrinsics": intrinsics,
            "extrinsics_w2c": extrinsics_w2c.contiguous(),
            "origin_extrinsics_w2c": w2c_tensor.contiguous(),
            "sparse_depths": sparse_depths,
            "packed_sparse_depths": get_packed_depth(sparse_depths.unsqueeze(2)).contiguous(),
            "rel_paths": rel_paths,
        }
