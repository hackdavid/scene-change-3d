"""Gravity alignment: find the floor and make +z point up, origin on the floor."""
from __future__ import annotations

import numpy as np

from .recon import Scene


def floor_transform(xyz: np.ndarray, scene: Scene, dist: float = 0.03) -> tuple[np.ndarray, dict]:
    import open3d as o3d

    o3d.utility.random.seed(0)
    up0 = -np.mean([v.R[1] for v in scene.views], 0)  # phones are held roughly upright
    up0 /= np.linalg.norm(up0)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pc = pc.voxel_down_sample(0.03)
    best = None  # largest roughly-horizontal plane (floor, bed or table top): gives the up axis
    rest = pc
    for _ in range(6):
        if len(rest.points) < 500:
            break
        (a, b, c, _d), inl = rest.segment_plane(dist, 3, 2000)
        n = np.array([a, b, c])
        n = -n if n @ up0 < 0 else n
        if n @ up0 > 0.8 and (best is None or len(inl) > best[1]):
            best = (n, len(inl))
        rest = rest.select_by_index(inl, invert=True)
    n = up0 if best is None else best[0]
    # floor height = lowest horizontal layer that holds a real share of the points (the biggest
    # plane is often a bed or table, and a few depth outliers poke below the floor)
    h = np.asarray(pc.points) @ n
    top = float(np.median([v.center @ n for v in scene.views])) - 0.5
    bins = np.arange(np.percentile(h, 0.2), top, 0.03)
    hist, _ = np.histogram(h, bins)
    dense = np.where(hist >= max(0.01 * len(h), 0.15 * hist.max()))[0]
    h0 = float(bins[dense[0]] + 0.015) if len(dense) else float(np.percentile(h, 2))
    info = {"up_axis": "largest horizontal RANSAC plane" if best is not None else "camera up (fallback)",
            "floor": "lowest dense horizontal layer"}
    z = n / np.linalg.norm(n)
    fwd = scene.views[0].R[2]
    x = np.cross(fwd, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rw = np.stack([x, y, z])  # rows: new axes in old coordinates
    c0 = scene.views[0].center
    tw = -Rw @ c0
    tw[2] = -h0  # floor at z = 0
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rw, tw
    info["tilt_from_camera_up_deg"] = float(np.degrees(np.arccos(np.clip(z @ up0, -1, 1))))
    return T, info


def apply_transform(scene: Scene, T: np.ndarray) -> None:
    Rw, tw = T[:3, :3], T[:3, 3]
    for v in scene.views:
        R_new = v.R @ Rw.T
        v.t = v.t - R_new @ tw
        v.R = R_new
    scene.sparse_xyz = scene.sparse_xyz @ Rw.T + tw
