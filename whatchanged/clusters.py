"""Group changed points into compact 3D change proposals."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .change import CHANGED, ChangeMap


@dataclass
class Cluster:
    kind: str  # "gone" (seen before, not after) | "new" (seen after, not before)
    idx: np.ndarray  # indices into ChangeMap.points
    center: np.ndarray
    extent: np.ndarray  # axis-aligned size (x, y, z) in metres
    n_views: int
    votes: int

    def bbox(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P = pts[self.idx]
        return np.percentile(P, 3, 0), np.percentile(P, 97, 0)


def cluster_changes(cm: ChangeMap, kind: str, eps=0.07, min_points=12, min_size=0.06, min_views=3,
                    min_count=60, min_height=0.04) -> list[Cluster]:
    import open3d as o3d

    idx = np.where(cm.state == CHANGED)[0]
    if len(idx) < min_count:
        return []
    P = cm.points.xyz[idx]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    labels = np.asarray(pc.cluster_dbscan(eps=eps, min_points=min_points))
    out = []
    for lab in range(labels.max() + 1):
        sel = idx[labels == lab]
        if len(sel) < min_count:
            continue
        Q = cm.points.xyz[sel]
        lo, hi = np.percentile(Q, 3, 0), np.percentile(Q, 97, 0)
        ext = hi - lo
        nv = len(np.unique(cm.points.view_idx[sel]))
        # an object must be seen in several keyframes and stand out of its support surface;
        # flat, rarely-seen blobs are almost always monocular-depth noise on printed fabric
        if ext.max() < min_size or ext[2] < min_height or nv < min_views:
            continue
        out.append(Cluster(kind, sel, (lo + hi) / 2, ext, nv, int(cm.free[sel].sum())))
    out.sort(key=lambda c: -len(c.idx))
    return out
