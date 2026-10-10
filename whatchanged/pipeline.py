"""End-to-end pipeline: two videos -> changes.json + report.html (CPU only)."""
from __future__ import annotations

import json
import pickle
import resource
import time
from pathlib import Path

import numpy as np

from . import report
from .align import apply_transform, floor_transform
from .change import detect, points_for
from .clusters import cluster_changes
from .keyframes import save_keyframes, select_keyframes
from .match import pair_moves
from .objects import build_objects, merge_duplicates
from .recon import harmonize_depth_scales, reconstruct
from .semantics import Labeller

MAX_HEIGHT_M = 1.6


def _log(msg: str) -> None:
    print(f"[whatchanged] {msg}", flush=True)


def _region(o) -> dict:
    return {"center": report._r((o.lo + o.hi) / 2), "size": report._r(o.hi - o.lo), "min": report._r(o.lo),
            "max": report._r(o.hi)}


def run(before: str, after: str, out: str | Path, n_keyframes: int = 20, semantics: str = "auto",
        reuse: bool = True) -> dict:
    out = Path(out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    T: dict[str, float] = {}
    t_all = time.time()

    t0 = time.time()
    kf_dir = work / "keyframes"
    if not (reuse and kf_dir.exists() and len(list(kf_dir.glob("*.png"))) >= 4):
        for video, tag in ((before, "before"), (after, "after")):
            kfs = select_keyframes(video, n=n_keyframes)
            save_keyframes(kfs, kf_dir, tag)
            _log(f"{tag}: {len(kfs)} keyframes")
    paths = sorted(kf_dir.glob("*.png"))
    T["keyframes_s"] = time.time() - t0

    t0 = time.time()
    cache = work / "scene.pkl"
    if reuse and cache.exists():
        scene = pickle.loads(cache.read_bytes())
        _log("reusing cached reconstruction")
    else:
        _log(f"reconstructing {len(paths)} keyframes (SfM + metric depth, CPU)...")
        scene = reconstruct(paths, work / "sfm")
        harmonize_depth_scales(scene)
        cache.write_bytes(pickle.dumps(scene))
    T["reconstruction_s"] = time.time() - t0
    n_before, n_after = len(scene.by_video("before")), len(scene.by_video("after"))
    _log(f"registered {n_before} before + {n_after} after views ({len(scene.unregistered)} unregistered)")
    if n_before < 3 or n_after < 3:
        raise RuntimeError("Too few views of one recording registered into the shared reconstruction; "
                           "re-record with more overlap between the two videos.")

    t0 = time.time()
    allp = np.concatenate([points_for(scene, v, 6).xyz for v in ("before", "after")])
    Tw, floor_info = floor_transform(allp, scene)
    apply_transform(scene, Tw)
    gone = detect(scene, "before", "after")
    new = detect(scene, "after", "before")
    clusters = cluster_changes(gone, "gone") + cluster_changes(new, "new")
    # robot-relevant height band: ceilings, skylights and the tops of tall wardrobes give noisy
    # monocular depth and are not obstacles, so changes centred above MAX_HEIGHT_M are ignored
    high = [c for c in clusters if c.center[2] > MAX_HEIGHT_M]
    clusters = [c for c in clusters if c.center[2] <= MAX_HEIGHT_M]
    if high:
        _log(f"ignoring {len(high)} proposal(s) centred above {MAX_HEIGHT_M} m (ceiling/skylight)")
    objs = build_objects(scene, gone, new, clusters)
    T["change_detection_s"] = time.time() - t0
    _log(f"{len(objs)} geometric change proposals")

    t0 = time.time()
    lab = Labeller(semantics)
    for o in objs:
        lab(scene, o)
        _log(f"  {o.id} {o.kind:4s} -> {'KEEP' if o.verified else 'reject'} {o.label}: {o.description}")
    T["semantics_s"] = time.time() - t0

    merge_duplicates(objs)
    (work / "stage.pkl").write_bytes(pickle.dumps((scene, gone, new, objs)))
    moves = pair_moves(objs, gone, new, lambda g, n, sim: lab.same_object(scene, g, n, sim))
    ev_dir = out / "evidence"
    events, used = [], set()
    for g, n, info in moves:
        used |= {g.id, n.id}
        fc, tc = (g.lo + g.hi) / 2, (n.lo + n.hi) / 2
        events.append({"type": "moved", "label": n.label, "from_object": g.id, "to_object": n.id,
                       "from": _region(g), "to": _region(n),
                       "displacement_m": round(float(np.linalg.norm((tc - fc)[:2])), 2),
                       "confidence": round(min(g.extra.get("confidence", .5), n.extra.get("confidence", .5)), 2),
                       "description": f"Before: {g.description} After: {n.description}", "match": info,
                       "from_evidence": report.evidence_images(scene, g, ev_dir),
                       "to_evidence": report.evidence_images(scene, n, ev_dir)})
    for o in objs:
        if o.verified and o.id not in used:
            events.append({"type": "removed" if o.kind == "gone" else "added", "label": o.label, "object": o.id,
                           "region": _region(o), "confidence": round(o.extra.get("confidence", .5), 2),
                           "description": o.description, "evidence": report.evidence_images(scene, o, ev_dir)})
    for i, e in enumerate(events):
        e["id"] = f"E{i + 1}"
    by_id = {o.id: o for o in objs}
    for e in events:
        ids = [e.get(k) for k in ("object", "from_object", "to_object") if e.get(k)]
        e["support"] = {"points": sum(len(by_id[i].cluster.idx) for i in ids),
                        "views": sum(by_id[i].cluster.n_views for i in ids)}
    rejected = [{"id": o.id, "kind": o.kind, "label": o.label, "region": _region(o),
                 "reason": o.description or "semantic check found the same content before and after",
                 "clip_before_after_similarity": o.extra.get("clip_before_after_similarity")}
                for o in objs if not o.verified]

    T["total_s"] = time.time() - t_all
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result = {
        "summary": report.summarize(events),
        "events": events,
        "rejected": rejected,
        "coordinate_frame": "metres; z up from the detected floor plane (z=0); x/y horizontal",
        "run": {"before": str(before), "after": str(after), "device": "cpu", "semantics": lab.mode,
                "labeller": lab.cfg[2] if lab.cfg else "openai/clip-vit-base-patch32",
                "keyframes": {"before": n_before, "after": n_after, "unregistered": scene.unregistered},
                "floor": floor_info, "timings_s": {k: round(v, 1) for k, v in {**scene.timings, **T}.items()},
                "peak_rss_mb": round(peak_mb),
                "points": {"before": {s: int((gone.state == i).sum()) for i, s in
                                      enumerate(["static", "changed", "occluded", "unobserved"])},
                           "after": {s: int((new.state == i).sum()) for i, s in
                                     enumerate(["static", "changed", "occluded", "unobserved"])}}},
    }
    (out / "changes.json").write_text(json.dumps(result, indent=2))
    (out / "summary.txt").write_text(result["summary"] + "\n")
    report.topdown(scene, gone, new, events, out / "topdown.jpg")
    report.evidence_sheet(events, out, out / "evidence_sheet.jpg")
    report.html_report(scene, gone, new, {o.id: o for o in objs}, result, out)
    _log(f"done in {T['total_s']:.0f}s -> {out}/report.html")
    _log(result["summary"])
    return result
