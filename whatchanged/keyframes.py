"""Adaptive keyframe selection: pick sharp frames spaced evenly in *camera motion*, not time."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Keyframe:
    index: int  # timestamp in the source video (ms)
    image: np.ndarray  # BGR
    sharpness: float


def _decode_candidates(video: str | Path, cand_fps: float, long_side: int):
    """Decode low-res candidate frames with ffmpeg (handles HEVC + phone rotation metadata)."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(probe.stdout)["streams"][0]
    w, h = info["width"], info["height"]
    rot = abs(int(info.get("tags", {}).get("rotate", 0) or 0))
    for sd in info.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = abs(int(sd["rotation"]))
    if rot in (90, 270):
        w, h = h, w
    s = min(1.0, long_side / max(w, h))
    ow, oh = int(round(w * s / 2) * 2), int(round(h * s / 2) * 2)
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={cand_fps},scale={ow}:{oh}:flags=area",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, oh, ow, 3)
    src_fps = cand_fps
    return [(round(i * 1000 / src_fps), f.copy()) for i, f in enumerate(frames)]  # index = timestamp in ms


def _motion(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Median sparse optical-flow magnitude (normalised by image diagonal) plus a penalty for lost tracks."""
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=8)
    if pts is None or len(pts) < 10:
        return 0.05
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
    st = st.reshape(-1).astype(bool)
    diag = float(np.hypot(*gray.shape))
    if st.sum() < 10:
        return 0.1
    flow = np.linalg.norm((nxt - pts).reshape(-1, 2)[st], axis=1) / diag
    lost = 1.0 - st.mean()
    return float(np.median(flow) + 0.05 * lost)


def select_keyframes(
    video: str | Path, n: int = 24, cand_fps: float = 5.0, long_side: int = 640
) -> list[Keyframe]:
    """Decode cheap low-res candidates, measure cumulative camera motion with LK flow,
    then take n frames evenly spaced along that motion curve (snapped to the sharpest
    neighbour). Static or slow parts of the video contribute few frames; fast pans many."""
    cands = _decode_candidates(video, cand_fps, long_side)
    if not cands:
        raise ValueError(f"no frames decoded from {video}")
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for _, f in cands]
    small = [cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2)) for g in grays]
    sharp = np.array([cv2.Laplacian(g, cv2.CV_64F).var() for g in grays])
    motion = np.zeros(len(cands))
    for i in range(1, len(cands)):
        motion[i] = motion[i - 1] + _motion(small[i - 1], small[i])
    n = min(n, len(cands))
    targets = np.linspace(0, motion[-1], n)
    chosen: list[int] = []
    for t in targets:
        c = int(np.argmin(np.abs(motion - t)))
        window = [j for j in range(c - 1, c + 2) if 0 <= j < len(cands) and j not in chosen]
        if not window:
            continue
        best = max(window, key=lambda j: sharp[j])
        chosen.append(best)
    chosen = sorted(set(chosen))
    return [Keyframe(cands[j][0], cands[j][1], float(sharp[j])) for j in chosen]


def save_keyframes(kfs: list[Keyframe], out_dir: str | Path, prefix: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in kfs:
        p = out_dir / f"{prefix}_{k.index:05d}.png"
        cv2.imwrite(str(p), k.image)
        paths.append(p)
    return paths
