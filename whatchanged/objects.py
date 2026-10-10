"""Turn change clusters into objects: best evidence views, image crops, 2D boxes."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .change import ChangeMap
from .clusters import Cluster
from .recon import Scene, View


@dataclass
class Evidence:
    view: int  # index into scene.views
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in the view image
    n_points: int


@dataclass
class ChangeObject:
    id: str
    kind: str  # gone | new
    cluster: Cluster
    evidence: list[Evidence]  # views from the video where the object IS visible
    counter_evidence: list[Evidence]  # views from the other video looking at the same spot
    label: str = "object"
    description: str = ""
    verified: bool | None = None
    embedding: np.ndarray | None = None
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def project(view: View, X: np.ndarray):
    Xc = X @ view.R.T + view.t
    z = Xc[:, 2]
    ok = z > 0.1
    u = view.K[0, 0] * Xc[:, 0] / np.maximum(z, 1e-6) + view.K[0, 2]
    v = view.K[1, 1] * Xc[:, 1] / np.maximum(z, 1e-6) + view.K[1, 2]
    h, w = view.depth.shape
    ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, ok


def _box(u, v, w, h, pad=0.25, min_size=48):
    x0, x1, y0, y1 = np.percentile(u, 2), np.percentile(u, 98), np.percentile(v, 2), np.percentile(v, 98)
    bw, bh = max(x1 - x0, min_size), max(y1 - y0, min_size)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = bw * (1 + 2 * pad), bh * (1 + 2 * pad)
    return (int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2)), int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2)))


def gather_evidence(scene: Scene, cm: ChangeMap, c: Cluster, k: int = 3) -> tuple[list[Evidence], list[Evidence]]:
    src_video = "before" if c.kind == "gone" else "after"
    ev = []
    for vi in np.unique(cm.points.view_idx[c.idx]):
        m = cm.points.view_idx[c.idx] == vi
        uv = cm.points.uv[c.idx][m]
        h, w = scene.views[vi].depth.shape
        ev.append(Evidence(int(vi), _box(uv[:, 0], uv[:, 1], w, h), int(m.sum())))
    ev.sort(key=lambda e: -e.n_points)
    X = cm.points.xyz[c.idx]
    counter = []
    for vi, v in enumerate(scene.views):
        if v.video == src_video:
            continue
        u, vv, ok = project(v, X)
        if ok.mean() < 0.5:
            continue
        h, w = v.depth.shape
        # only views that looked *through* the spot (free space) are valid counter-evidence;
        # a view where something stands in front of it would show a different object.
        ui, vi_ = u[ok].astype(int), vv[ok].astype(int)
        z = (X[ok] @ v.R.T + v.t)[:, 2]
        d = v.depth[vi_, ui]
        free = (d > 0) & (d > z + np.maximum(0.06, 0.08 * z))
        if free.mean() < 0.5:
            continue
        counter.append(Evidence(vi, _box(u[ok], vv[ok], w, h), int(free.sum())))
    counter.sort(key=lambda e: -e.n_points)
    return ev[:k], counter[:k]


def crop(scene: Scene, e: Evidence) -> np.ndarray:
    x0, y0, x1, y1 = e.box
    return scene.views[e.view].image[y0:y1, x0:x1]


def build_objects(scene: Scene, gone: ChangeMap, new: ChangeMap, clusters: list[Cluster]) -> list[ChangeObject]:
    objs = []
    for i, c in enumerate(clusters):
        cm = gone if c.kind == "gone" else new
        ev, cev = gather_evidence(scene, cm, c)
        lo, hi = c.bbox(cm.points.xyz)
        objs.append(ChangeObject(f"c{i}", c.kind, c, ev, cev, lo=lo, hi=hi))
    return objs


def merge_duplicates(objs: list[ChangeObject], gap: float = 0.3) -> None:
    """One physical object can produce several change clusters (e.g. screen and keyboard of a
    laptop, or a gone and a new cluster at the same spot). Merge verified ones with the same
    direction and label whose 3D boxes are within `gap` metres."""
    keep = sorted((o for o in objs if o.verified), key=lambda o: -len(o.cluster.idx))
    for i, a in enumerate(keep):
        if not a.verified:
            continue
        for b in keep[i + 1:]:
            if not (b.verified and b.kind == a.kind and b.label == a.label):
                continue
            if np.linalg.norm(np.maximum(0, np.maximum(a.lo - b.hi, b.lo - a.hi))) < gap:
                a.lo, a.hi = np.minimum(a.lo, b.lo), np.maximum(a.hi, b.hi)
                b.verified = False
                b.extra["merged_into"] = a.id
                b.description = f"same {b.label} as {a.id} (merged into one event)"
