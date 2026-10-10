"""Pair removed and added objects into 'moved' events."""
from __future__ import annotations

import numpy as np

from .change import ChangeMap
from .objects import ChangeObject


def _icp(src: np.ndarray, dst: np.ndarray, init_t: np.ndarray) -> tuple[float, float]:
    import open3d as o3d

    a = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
    b = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst))
    T = np.eye(4)
    T[:3, 3] = init_t
    r = o3d.pipelines.registration.registration_icp(
        a, b, 0.05, T, o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return float(r.fitness), float(r.inlier_rmse)


def _same_label(g: ChangeObject, n: ChangeObject) -> bool:
    a, b = g.label.strip().lower(), n.label.strip().lower()
    return a == b or a in b or b in a


def pair_moves(objs: list[ChangeObject], gone: ChangeMap, new: ChangeMap,
               same_object=None) -> list[tuple[ChangeObject, ChangeObject, dict]]:
    """Greedy matching of verified gone/new objects, best appearance match first.

    `same_object(g, n, sim) -> (ok, label, reason)` confirms identity (VLM or CLIP rule)."""
    G = [o for o in objs if o.kind == "gone" and o.verified]
    N = [o for o in objs if o.kind == "new" and o.verified]
    cands = []
    for g in G:
        for n in N:
            sim = float(g.embedding @ n.embedding / (np.linalg.norm(g.embedding) * np.linalg.norm(n.embedding)))
            sg, sn = np.sort(g.hi - g.lo), np.sort(n.hi - n.lo)
            ratio = float(np.max(np.maximum(sg, sn) / np.maximum(np.minimum(sg, sn), 0.03)))
            # the semantic labels must agree before identity is even asked: a removed bottle and
            # an added charger are two changes, not one move. Size is not a gate, because change
            # clusters often cover only part of an object.
            if _same_label(g, n):
                cands.append((sim, g, n, sim, ratio))
    cands.sort(key=lambda c: -c[0])
    used, out = set(), []
    for _score, g, n, sim, ratio in cands:
        if g.id in used or n.id in used:
            continue
        ok, label, reason = same_object(g, n, sim) if same_object else (g.label == n.label, n.label, "")
        if not ok:
            continue
        used |= {g.id, n.id}
        g.label = n.label = label
        # a cluster's points live in the map it was found in, even if semantics flipped its direction
        Pg, Pn = ((gone if o.cluster.kind == "gone" else new).points.xyz[o.cluster.idx] for o in (g, n))
        fit, rmse = _icp(Pg, Pn, Pn.mean(0) - Pg.mean(0))
        out.append((g, n, {"appearance_similarity": round(sim, 3), "identity_check": reason,
                           "size_ratio": round(ratio, 2), "icp_fitness": round(fit, 3), "icp_rmse_m": round(rmse, 3)}))
    return out
