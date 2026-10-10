"""CPU reconstruction: joint SfM over both videos' keyframes + metric monocular depth.

All keyframes (before and after) go into ONE structure-from-motion problem, so every camera
lands in the same world frame. Depth Anything V2 (metric, indoor) then gives a dense depth map
per keyframe, which also fixes the global scale: the world ends up (approximately) in metres.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEPTH_REPO = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


@dataclass
class View:
    name: str
    video: str  # "before" | "after"
    t_ms: int
    image: np.ndarray  # HxWx3 uint8 RGB
    K: np.ndarray  # 3x3
    R: np.ndarray  # 3x3 world->camera
    t: np.ndarray  # 3   world->camera (metres)
    depth: np.ndarray | None = None  # HxW metres, 0 = invalid

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass
class Scene:
    views: list[View]
    sparse_xyz: np.ndarray
    sparse_rgb: np.ndarray
    unregistered: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    metres_per_unit: float = 1.0

    def by_video(self, video: str) -> list[View]:
        return [v for v in self.views if v.video == video]


def _parse_name(name: str) -> tuple[str, int]:
    video, t = Path(name).stem.split("_")
    return video, int(t)


def run_sfm(image_dir: Path, names: list[str], work: Path, timings: dict):
    import pycolmap

    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    db.unlink(missing_ok=True)
    t0 = time.time()
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = 8192
    ext.sift.estimate_affine_shape = False
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_PINHOLE"  # phone video is already undistorted in-camera
    pycolmap.extract_features(db, image_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=ext)
    timings["sfm_features_s"] = time.time() - t0
    t0 = time.time()
    pycolmap.match_exhaustive(db)
    timings["sfm_matching_s"] = time.time() - t0
    t0 = time.time()
    recs = pycolmap.incremental_mapping(db, image_dir, work / "sparse")
    timings["sfm_mapping_s"] = time.time() - t0
    if not recs:
        raise RuntimeError("Structure-from-motion failed: no images could be registered. "
                           "Record with more sideways motion and more overlap.")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    timings["sfm_num_models"] = len(recs)
    return rec


class DepthModel:
    def __init__(self, repo: str = DEPTH_REPO, input_size: int = 392):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(repo)
        self.model = AutoModelForDepthEstimation.from_pretrained(repo).eval()
        # 392 instead of the default 518: ~3.4x faster on CPU, ~5-7% median depth difference
        self.proc.size = {"height": input_size, "width": input_size}

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            out = self.model(**self.proc(images=rgb, return_tensors="pt")).predicted_depth
        d = torch.nn.functional.interpolate(out[None], size=rgb.shape[:2], mode="bilinear", align_corners=False)
        return d[0, 0].numpy().astype(np.float32)


def _edge_mask(d: np.ndarray, rel: float = 0.06) -> np.ndarray:
    """True where depth is smooth (removes 'flying pixels' at depth discontinuities)."""
    import cv2

    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    g = np.sqrt(gx**2 + gy**2) / np.maximum(d, 1e-3)
    ok = g < rel
    ok = cv2.erode(ok.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return ok


def reconstruct(image_paths: list[Path], work: Path, max_depth: float = 6.0) -> Scene:
    import cv2

    timings: dict = {}
    image_dir = image_paths[0].parent
    names = [p.name for p in image_paths]
    rec = run_sfm(image_dir, names, work, timings)

    t0 = time.time()
    depth_model = DepthModel()
    timings["depth_model_load_s"] = time.time() - t0

    raw = []
    for img in rec.images.values():
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        M = cfw.matrix()
        R, t = M[:, :3].copy(), M[:, 3].copy()
        cam = rec.cameras[img.camera_id]
        K = cam.calibration_matrix()
        rgb = cv2.cvtColor(cv2.imread(str(image_dir / img.name)), cv2.COLOR_BGR2RGB)
        # undistort with the SfM camera so depth back-projection uses a clean pinhole model
        params = np.array(cam.params)
        dist = np.zeros(5)
        if cam.model.name in ("SIMPLE_RADIAL", "RADIAL"):
            dist[0] = params[3]
            if cam.model.name == "RADIAL":
                dist[1] = params[4]
        if np.any(dist):
            rgb = cv2.undistort(rgb, K, dist)
        obs = []  # (u, v, z_sfm)
        for p2 in img.points2D:
            if p2.has_point3D():
                X = rec.points3D[p2.point3D_id].xyz
                z = (R @ X + t)[2]
                if z > 0:
                    obs.append((p2.xy[0], p2.xy[1], z))
        raw.append((img.name, R, t, K, rgb, np.array(obs)))

    t0 = time.time()
    depths = [depth_model(r[4]) for r in raw]
    timings["depth_inference_s"] = time.time() - t0
    timings["depth_per_frame_s"] = timings["depth_inference_s"] / max(1, len(raw))

    # global scale: SfM units per metre, then per-view correction of the monocular scale
    ratios, per_view = [], []
    for (_, _, _, _, rgb, obs), d in zip(raw, depths):
        r = np.array([])
        if len(obs):
            u = np.clip(np.round(obs[:, 0]).astype(int), 0, d.shape[1] - 1)
            v = np.clip(np.round(obs[:, 1]).astype(int), 0, d.shape[0] - 1)
            r = obs[:, 2] / np.maximum(d[v, u], 1e-3)
        per_view.append(r)
        ratios.append(r)
    allr = np.concatenate([r for r in ratios if len(r)])
    units_per_m = float(np.median(allr))

    views = []
    for (name, R, t, K, rgb, obs), d, r in zip(raw, depths, per_view):
        corr = float(np.median(r)) / units_per_m if len(r) >= 30 else 1.0
        corr = float(np.clip(corr, 0.7, 1.4))
        dm = d * corr
        dm[~_edge_mask(dm)] = 0
        dm[(dm > max_depth) | (dm < 0.1)] = 0
        video, t_ms = _parse_name(name)
        views.append(View(name, video, t_ms, rgb, K, R, t / units_per_m, dm))
    views.sort(key=lambda v: (v.video != "before", v.t_ms))

    xyz = np.array([p.xyz for p in rec.points3D.values()]) / units_per_m
    rgbs = np.array([p.color for p in rec.points3D.values()]) / 255.0
    registered = {v.name for v in views}
    return Scene(views, xyz, rgbs, [n for n in names if n not in registered], timings, 1.0 / units_per_m)


def harmonize_depth_scales(scene: Scene, iters: int = 3, stride: int = 6, max_rel: float = 0.25) -> list[float]:
    """Make monocular depth maps agree with each other.

    Monocular depth has a per-frame scale error (worst on blank walls). Because most of the room
    is static, the median ratio between a view's depth and what *all other views* say the depth
    should be at the same pixels is a robust per-view scale correction. Returns the residual
    median relative disagreement per view (used as a per-view noise estimate)."""
    from .change import backproject

    resid = [0.0] * len(scene.views)
    for _ in range(iters):
        clouds = [backproject(v, stride)[0] for v in scene.views]
        new_scales = []
        for i, v in enumerate(scene.views):
            others = np.concatenate([c for j, c in enumerate(clouds) if j != i])
            Xc = others @ v.R.T + v.t
            z = Xc[:, 2]
            ok = z > 0.1
            u = v.K[0, 0] * Xc[ok, 0] / z[ok] + v.K[0, 2]
            w = v.K[1, 1] * Xc[ok, 1] / z[ok] + v.K[1, 2]
            h, wd = v.depth.shape
            inb = (u >= 0) & (u < wd) & (w >= 0) & (w < h)
            zz = z[ok][inb]
            d = v.depth[w[inb].astype(int), u[inb].astype(int)]
            m = d > 0
            zz, d = zz[m], d[m]
            r = zz / np.maximum(d, 1e-3)
            near = np.abs(r - 1) < max_rel  # only roughly-corresponding surfaces vote (ignores occlusions)
            if near.sum() < 200:
                new_scales.append(1.0)
                resid[i] = max_rel
                continue
            s = float(np.median(r[near]))
            new_scales.append(s)
            resid[i] = float(np.median(np.abs(r[near] - s)))
        for v, s in zip(scene.views, new_scales):
            v.depth = v.depth * float(np.clip(s, 0.8, 1.25))
    return resid
