"""Visibility-aware change detection (ray-consistency / free-space test).

A point seen in video A is called *gone* only if cameras of video B looked at that exact spot
and saw *through* it (observed depth clearly behind the point). If B's surface is in front of
it, the point is merely occluded; if no B camera covered it, it is unobserved. Neither of those
counts as a change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .recon import Scene, View


@dataclass
class PointSet:
    xyz: np.ndarray  # N x 3 world
    rgb: np.ndarray  # N x 3 in [0, 1]
    view_idx: np.ndarray  # N (index into Scene.views of the source view)
    uv: np.ndarray  # N x 2 pixel in the source view


def backproject(view: View, stride: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = view.depth
    h, w = d.shape
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]
    z = d[vv, uu]
    m = z > 0
    uu, vv, z = uu[m], vv[m], z[m]
    x = (uu - view.K[0, 2]) / view.K[0, 0] * z
    y = (vv - view.K[1, 2]) / view.K[1, 1] * z
    Xc = np.stack([x, y, z], -1)
    Xw = (Xc - view.t) @ view.R
    return Xw, view.image[vv, uu] / 255.0, np.stack([uu, vv], -1)


def points_for(scene: Scene, video: str, stride: int = 3) -> PointSet:
    xs, cs, vs, uvs = [], [], [], []
    for i, v in enumerate(scene.views):
        if v.video != video:
            continue
        X, C, uv = backproject(v, stride)
        xs.append(X)
        cs.append(C)
        vs.append(np.full(len(X), i))
        uvs.append(uv)
    return PointSet(np.concatenate(xs), np.concatenate(cs), np.concatenate(vs), np.concatenate(uvs))


def texture_map(img: np.ndarray, k: int = 15) -> np.ndarray:
    """Local grey-level standard deviation. Monocular depth on blank surfaces (white walls) is
    unreliable, so such pixels may not *originate* a change claim."""
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = cv2.blur(g, (k, k))
    m2 = cv2.blur(g * g, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0))


def tau(z: np.ndarray, abs_tol: float, rel_tol: float) -> np.ndarray:
    return abs_tol + rel_tol * z


def ray_votes(xyz: np.ndarray, views: list[View], abs_tol=0.06, rel_tol=0.08, margin=4, skip: np.ndarray | None = None):
    """Return per-point counts: (free, occupied, occluded, in_view) against the given views.
    `skip[i, j]` = True means view j must not vote for point i (e.g. its own source view)."""
    n = len(xyz)
    free = np.zeros(n, np.int32)
    occ = np.zeros(n, np.int32)
    hidden = np.zeros(n, np.int32)
    inview = np.zeros(n, np.int32)
    for j, v in enumerate(views):
        h, w = v.depth.shape
        Xc = xyz @ v.R.T + v.t
        z = Xc[:, 2]
        ok = z > 0.1
        u = np.full(n, -1.0)
        vv = np.full(n, -1.0)
        u[ok] = v.K[0, 0] * Xc[ok, 0] / z[ok] + v.K[0, 2]
        vv[ok] = v.K[1, 1] * Xc[ok, 1] / z[ok] + v.K[1, 2]
        ok &= (u >= margin) & (u < w - margin) & (vv >= margin) & (vv < h - margin)
        if skip is not None:
            ok &= ~skip[:, j]
        idx = np.where(ok)[0]
        ui, vi = u[idx].astype(int), vv[idx].astype(int)
        d_obs = v.depth[vi, ui]
        valid = d_obs > 0
        idx, d_obs, zz = idx[valid], d_obs[valid], z[idx][valid]
        tt = tau(zz, abs_tol, rel_tol)
        inview[idx] += 1
        free[idx[d_obs > zz + tt]] += 1
        occ[idx[np.abs(d_obs - zz) <= tt]] += 1
        hidden[idx[d_obs < zz - tt]] += 1
    return free, occ, hidden, inview


@dataclass
class ChangeMap:
    points: PointSet
    state: np.ndarray  # 0 static, 1 changed, 2 occluded/uncertain, 3 unobserved
    free: np.ndarray
    occ: np.ndarray
    inview: np.ndarray


STATIC, CHANGED, UNCERTAIN, UNOBSERVED = 0, 1, 2, 3


def detect(scene: Scene, src: str, dst: str, stride=3, min_free=2, free_ratio=0.7, self_free_max=0.25,
           abs_tol=0.06, rel_tol=0.08, min_texture=3.0) -> ChangeMap:
    """Points of video `src` that video `dst` saw through (= disappeared from src's point of view)."""
    pts = points_for(scene, src, stride)
    dst_views = scene.by_video(dst)
    free, occ, _hid, inv = ray_votes(pts.xyz, dst_views, abs_tol, rel_tol)
    # self-consistency against the other views of the same video: noisy depth that even the
    # same recording disagrees with is not trusted.
    src_idx = [i for i, v in enumerate(scene.views) if v.video == src]
    skip = pts.view_idx[:, None] == np.array(src_idx)[None, :]
    sfree, _socc, _, sinv = ray_votes(pts.xyz, [scene.views[i] for i in src_idx], abs_tol, rel_tol, skip=skip)
    self_bad = (sinv > 0) & (sfree > self_free_max * np.maximum(sinv, 1))
    state = np.full(len(pts.xyz), UNCERTAIN, np.int8)
    state[inv == 0] = UNOBSERVED
    decided = free + occ
    ratio = free / np.maximum(decided, 1)
    state[(decided > 0) & (ratio < 0.5)] = STATIC
    textured = np.zeros(len(pts.xyz), bool)
    for i in src_idx:
        m = pts.view_idx == i
        tex = texture_map(scene.views[i].image)
        textured[m] = tex[pts.uv[m, 1], pts.uv[m, 0]] >= min_texture
    changed = (free >= min_free) & (ratio >= free_ratio) & ~self_bad & textured
    state[changed] = CHANGED
    return ChangeMap(pts, state, free, occ, inv)
