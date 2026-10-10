"""Benchmark CPU reconstruction backends on a before/after pair with ground-truth cameras.

  python bench/bench_recon.py --data data/synth01 --method vggt --n 16
  python bench/bench_recon.py --data data/synth01 --method sfm  --n 16

Writes bench_out/<method>_n<N>/{result.json, topdown.png, cloud.ply}.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from whatchanged.keyframes import save_keyframes, select_keyframes  # noqa: E402


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Sim(3) s,R,t minimising ||dst - (s R src + t)||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum()
    t = mu_d - s * R @ mu_s
    return s, R, t


def gt_cameras(data: Path, variant: str):
    j = json.loads((data / f"{variant}_cameras.json").read_text())
    out = {}
    for k, f in enumerate(j["frames"]):
        R, t = np.array(f["R_w2c"]), np.array(f["t_w2c"])
        out[k] = (R, t, -R.T @ t)
    return out


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def run_vggt(paths, timings):
    import torch

    sys.path.insert(0, str(Path.home() / "third_party/vggt"))
    from huggingface_hub import hf_hub_download
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    t0 = time.time()
    model = VGGT()
    model.load_state_dict(torch.load(hf_hub_download("facebook/VGGT-1B", "model.pt"), map_location="cpu"))
    model.point_head = None
    model.track_head = None
    model.eval()
    timings["load_model_s"] = time.time() - t0
    images = load_and_preprocess_images([str(p) for p in paths])
    t0 = time.time()
    with torch.inference_mode():
        pred = model(images)
    timings["inference_s"] = time.time() - t0
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    extr, intr = extr[0].numpy(), intr[0].numpy()
    depth, conf = pred["depth"][0].numpy(), pred["depth_conf"][0].numpy()
    pts = unproject_depth_map_to_point_map(depth, extr, intr)
    cols = images.permute(0, 2, 3, 1).numpy()
    keep = conf > np.quantile(conf, 0.5)
    poses = {i: (extr[i][:, :3], extr[i][:, 3]) for i in range(len(paths))}
    return poses, pts[keep], cols[keep], None


def run_sfm(paths, timings, work: Path):
    import pycolmap
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    img_dir = paths[0].parent
    db = work / "db.db"
    db.unlink(missing_ok=True)
    t0 = time.time()
    pycolmap.extract_features(db, img_dir, image_names=[p.name for p in paths], camera_mode=pycolmap.CameraMode.SINGLE)
    timings["sfm_features_s"] = time.time() - t0
    t0 = time.time()
    pycolmap.match_exhaustive(db)
    timings["sfm_matching_s"] = time.time() - t0
    t0 = time.time()
    recs = pycolmap.incremental_mapping(db, img_dir, work / "sparse")
    timings["sfm_mapping_s"] = time.time() - t0
    if not recs:
        raise RuntimeError("SfM failed: no reconstruction")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    timings["sfm_models"] = len(recs)
    name_to_i = {p.name: i for i, p in enumerate(paths)}

    t0 = time.time()
    repo = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    proc = AutoImageProcessor.from_pretrained(repo)
    dmodel = AutoModelForDepthEstimation.from_pretrained(repo).eval()
    timings["load_depth_model_s"] = time.time() - t0

    poses, ratios, depth_maps, cams = {}, [], {}, {}
    t_depth = 0.0
    for img in rec.images.values():
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        M = cfw.matrix()
        i = name_to_i[img.name]
        poses[i] = (M[:, :3], M[:, 3])
        pil = Image.open(img_dir / img.name).convert("RGB")
        t1 = time.time()
        with torch.inference_mode():
            out = dmodel(**proc(images=pil, return_tensors="pt")).predicted_depth
        d = torch.nn.functional.interpolate(out[None], size=pil.size[::-1], mode="bicubic")[0, 0].numpy()
        t_depth += time.time() - t1
        depth_maps[i] = (d, np.asarray(pil) / 255.0)
        cam = rec.cameras[img.camera_id]
        cams[i] = cam.calibration_matrix()
        for p2 in img.points2D:
            if p2.has_point3D():
                X = rec.points3D[p2.point3D_id].xyz
                z = (M[:, :3] @ X + M[:, 3])[2]
                u, v = np.round(p2.xy).astype(int)
                if z > 0 and 0 <= v < d.shape[0] and 0 <= u < d.shape[1]:
                    ratios.append(z / d[v, u])
    timings["depth_total_s"] = t_depth
    s = float(np.median(ratios))  # SfM units per metre
    pts, cols = [], []
    for i, (d, rgb) in depth_maps.items():
        R, t = poses[i]
        K = cams[i]
        h, w = d.shape
        uu, vv = np.meshgrid(np.arange(0, w, 2), np.arange(0, h, 2))
        z = d[vv, uu] * s
        x = (uu - K[0, 2]) / K[0, 0] * z
        y = (vv - K[1, 2]) / K[1, 1] * z
        Xc = np.stack([x, y, z], -1).reshape(-1, 3)
        pts.append((Xc - t) @ R)  # R^T (Xc - t)
        cols.append(rgb[vv, uu].reshape(-1, 3))
    return poses, np.concatenate(pts), np.concatenate(cols), 1.0 / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth01")
    ap.add_argument("--method", choices=["vggt", "sfm"], required=True)
    ap.add_argument("--n", type=int, default=16, help="keyframes per video")
    a = ap.parse_args()
    data = ROOT / a.data
    out = ROOT / "bench_out" / f"{a.method}_n{a.n}"
    out.mkdir(parents=True, exist_ok=True)
    timings: dict = {}

    t0 = time.time()
    paths, meta = [], []
    for v in ["before", "after"]:
        kfs = select_keyframes(data / f"{v}.mp4", n=a.n)
        paths += save_keyframes(kfs, out / "keyframes", v)
        meta += [(v, k.index) for k in kfs]
    timings["keyframes_s"] = time.time() - t0

    t0 = time.time()
    if a.method == "vggt":
        poses, pts, cols, metric_scale = run_vggt(paths, timings)
    else:
        poses, pts, cols, metric_scale = run_sfm(paths, timings, out)
    timings["recon_total_s"] = time.time() - t0

    gts = {v: gt_cameras(data, v) for v in ["before", "after"]}
    idx = sorted(poses)
    est_c = np.array([-poses[i][0].T @ poses[i][1] for i in idx])
    gt_c = np.array([gts[meta[i][0]][meta[i][1]][2] for i in idx])
    s, R, t = umeyama(est_c, gt_c)
    aligned = (s * (R @ est_c.T)).T + t
    ate = float(np.sqrt(((aligned - gt_c) ** 2).sum(1).mean()))
    rot_err = []
    for i in idx:
        R_est_c2w = poses[i][0].T
        R_gt_c2w = gts[meta[i][0]][meta[i][1]][0].T
        dR = (R @ R_est_c2w).T @ R_gt_c2w
        rot_err.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    n_before = sum(1 for i in idx if meta[i][0] == "before")
    res = {
        "method": a.method,
        "keyframes_per_video": a.n,
        "frames_in": len(paths),
        "registered": len(idx),
        "registered_before": n_before,
        "registered_after": len(idx) - n_before,
        "ate_rmse_m": ate,
        "rot_err_median_deg": float(np.median(rot_err)),
        "scale_to_metric_true": float(s),
        "scale_to_metric_estimated": metric_scale,
        "timings": {k: round(v, 2) if isinstance(v, float) else v for k, v in timings.items()},
        "peak_rss_gb": round(peak_rss_gb(), 2),
    }
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    # top-down plot of the fused cloud (aligned to GT world) with both camera paths
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = (s * (R @ pts.T)).T + t
    sel = np.random.default_rng(0).choice(len(P), min(len(P), 150_000), replace=False)
    P, C = P[sel], np.clip(cols[sel], 0, 1)
    m = (P[:, 2] > 0.05) & (P[:, 2] < 2.0)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(P[m, 0], P[m, 1], c=C[m], s=0.3)
    for v, col in [("before", "tab:blue"), ("after", "tab:red")]:
        g = np.array([c for _, _, c in gts[v].values()])
        ax.plot(g[:, 0], g[:, 1], "--", color=col, lw=1, label=f"GT path {v}")
        e = np.array([aligned[k] for k, i in enumerate(idx) if meta[i][0] == v])
        if len(e):
            ax.scatter(e[:, 0], e[:, 1], color=col, marker="^", s=25, label=f"estimated {v}")
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.3, 2.3)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(f"{a.method}: {len(idx)}/{len(paths)} frames, ATE {ate*100:.1f} cm, recon {timings['recon_total_s']:.0f}s")
    fig.savefig(out / "topdown.png", dpi=110, bbox_inches="tight")

    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.colors = o3d.utility.Vector3dVector(C)
    o3d.io.write_point_cloud(str(out / "cloud.ply"), pc)


if __name__ == "__main__":
    main()
