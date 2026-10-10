"""Outputs: changes.json, summary, evidence images, top-down map and a self-contained 3D viewer."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from ..change import STATIC, ChangeMap
from ..recon import Scene
from ..semantics import context_image

VENDOR = Path(__file__).parent / "vendor"
COLORS = {"removed": (230, 60, 60), "added": (60, 200, 90), "moved": (255, 150, 30)}


def _r(a, n=2):
    return [round(float(x), n) for x in a]


def _where(c: np.ndarray) -> str:
    h = float(c[2])
    if h < 0.25:
        return "on the floor"
    if h < 1.2:
        return f"on a surface about {h:.1f} m above the floor"
    return f"about {h:.1f} m above the floor"


def summarize(events: list[dict]) -> str:
    if not events:
        return "No confident changes were found between the two recordings."
    lines = []
    for e in events:
        if e["type"] == "moved":
            lines.append(f"The {e['label']} moved {e['displacement_m']:.2f} m: from {_where(np.array(e['from']['center']))} "
                         f"(x={e['from']['center'][0]:.2f}, y={e['from']['center'][1]:.2f}) to {_where(np.array(e['to']['center']))} "
                         f"(x={e['to']['center'][0]:.2f}, y={e['to']['center'][1]:.2f}).")
        else:
            b = e["region"]
            verb = "was removed from" if e["type"] == "removed" else "was added"
            lines.append(f"A {e['label']} {verb} {_where(np.array(b['center']))} "
                         f"(x={b['center'][0]:.2f}, y={b['center'][1]:.2f}, "
                         f"size {b['size'][0]:.2f}x{b['size'][1]:.2f}x{b['size'][2]:.2f} m).")
    return " ".join(lines)


def _save(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def evidence_images(scene: Scene, o, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    own = o.evidence[0]
    other = o.counter_evidence[0] if o.counter_evidence else None
    before, after = (own, other) if o.cluster.kind == "gone" else (other, own)
    files = {}
    for tag, e in (("before", before), ("after", after)):
        if e is None:
            continue
        p = out / f"{o.id}_{tag}.jpg"
        _save(p, context_image(scene, e, size=320))
        files[tag] = f"evidence/{p.name}"
        files[f"{tag}_frame"] = scene.views[e.view].name
    return files


def topdown(scene: Scene, gone: ChangeMap, new: ChangeMap, events: list[dict], path: Path, px: float = 0.02) -> None:
    pts = np.concatenate([gone.points.xyz[gone.state == STATIC], new.points.xyz[new.state == STATIC]])
    pts = pts[(pts[:, 2] > 0.05) & (pts[:, 2] < 2.0)]
    lo = np.percentile(pts[:, :2], 1, 0) - 0.3
    hi = np.percentile(pts[:, :2], 99, 0) + 0.3
    W, H = (np.ceil((hi - lo) / px)).astype(int)
    occ = np.zeros((H, W), np.float32)
    ij = ((pts[:, :2] - lo) / px).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
    np.add.at(occ, (H - 1 - ij[ok, 1], ij[ok, 0]), 1)
    img = np.full((H, W, 3), 255, np.uint8)
    img[occ > 2] = (170, 170, 170)

    def to_px(xy):
        return int((xy[0] - lo[0]) / px), int(H - 1 - (xy[1] - lo[1]) / px)

    for v in scene.views:
        c = v.center
        col = (90, 90, 230) if v.video == "before" else (40, 160, 200)
        cv2.circle(img, to_px(c), 3, col, -1)
    for e in events:
        regions = [("from", e["from"]), ("to", e["to"])] if e["type"] == "moved" else [("", e["region"])]
        for tag, r in regions:
            col = COLORS["removed"] if tag == "from" else COLORS["added"] if tag == "to" else COLORS[e["type"]]
            a = to_px(np.array(r["min"])[:2])
            b = to_px(np.array(r["max"])[:2])
            cv2.rectangle(img, a, b, col, 3)
        if e["type"] == "moved":
            cv2.arrowedLine(img, to_px(e["from"]["center"]), to_px(e["to"]["center"]), COLORS["moved"], 4, tipLength=0.08)
        anchor = e["to"]["center"] if e["type"] == "moved" else e["region"]["center"]
        p = to_px(anchor)
        cv2.putText(img, f"{e['id']} {e['label']} ({e['type']})", (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2)
    bar = int(1.0 / px)
    cv2.line(img, (20, H - 20), (20 + bar, H - 20), (0, 0, 0), 3)
    cv2.putText(img, "1 m", (20, H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "top-down: grey = static surfaces, dots = cameras (blue before, teal after)", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    _save(path, img)


def evidence_sheet(events: list[dict], out: Path, path: Path) -> None:
    rows = []
    for e in events:
        ims = []
        for key in (["from_evidence", "to_evidence"] if e["type"] == "moved" else ["evidence"]):
            for tag in ("before", "after"):
                f = e[key].get(tag)
                im = cv2.imread(str(out / f)) if f else None
                im = np.zeros((240, 320, 3), np.uint8) if im is None else cv2.resize(im, (320, 240))
                cv2.putText(im, tag.upper(), (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                ims.append(im)
        row = np.concatenate(ims, 1)
        head = np.full((40, row.shape[1], 3), 255, np.uint8)
        cv2.putText(head, f"{e['id']}: {e['label']} {e['type']}  (conf {e['confidence']:.2f})", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        rows.append(np.concatenate([head, row], 0))
    if not rows:
        return
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    cv2.imwrite(str(path), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])


def _pack(xyz: np.ndarray, rgb: np.ndarray) -> dict:
    return {"xyz": base64.b64encode(xyz.astype(np.float32).tobytes()).decode(),
            "rgb": base64.b64encode((np.clip(rgb, 0, 1) * 255).astype(np.uint8).tobytes()).decode()}


def html_report(scene: Scene, gone: ChangeMap, new: ChangeMap, objs_by_id: dict, result: dict, out: Path,
                max_points: int = 70000) -> Path:
    rng = np.random.default_rng(0)
    layers = {}
    for name, cm in (("before", gone), ("after", new)):
        m = np.where(cm.state != 1)[0]
        m = m[(cm.points.xyz[m, 2] > -0.2) & (cm.points.xyz[m, 2] < 2.6)]
        m = rng.choice(m, min(max_points, len(m)), replace=False)
        layers[name] = _pack(cm.points.xyz[m], cm.points.rgb[m])
    changed = []
    for e in result["events"]:
        for oid, col in ((e.get("from_object"), COLORS["removed"]), (e.get("to_object"), COLORS["added"]),
                         (e.get("object"), COLORS.get(e["type"]))):
            if not oid:
                continue
            o = objs_by_id[oid]
            cm = gone if o.cluster.kind == "gone" else new
            X = cm.points.xyz[o.cluster.idx]
            X = X[rng.choice(len(X), min(6000, len(X)), replace=False)]
            changed.append({"event": e["id"], **_pack(X, np.tile(np.array(col) / 255, (len(X), 1)))})
    evid = {}
    for e in result["events"]:
        for key in ("evidence", "from_evidence", "to_evidence"):
            for tag in ("before", "after"):
                f = e.get(key, {}).get(tag)
                if f:
                    evid[f] = "data:image/jpeg;base64," + base64.b64encode((out / f).read_bytes()).decode()
    cams = [{"video": v.video, "c": _r(v.center, 3), "f": _r(v.R[2], 3)} for v in scene.views]
    data = {"layers": layers, "changed": changed, "result": result, "evidence": evid, "cams": cams}
    tpl = (Path(__file__).parent / "template.html").read_text()
    html = (tpl.replace("/*THREE_SRC*/", (VENDOR / "three.module.min.js").read_text().replace("</script", "<\\/script"))
            .replace("/*ORBIT_SRC*/", (VENDOR / "OrbitControls.js").read_text().replace("</script", "<\\/script"))
            .replace("/*DATA*/", json.dumps(data)))
    p = out / "report.html"
    p.write_text(html)
    return p
