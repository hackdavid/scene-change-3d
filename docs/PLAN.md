# WhatChanged: visibility-aware 3D change detection from two phone videos

**Goal:** Take two short phone videos of the same room (before and after). Produce one shared 3D scene, find the objects that were **moved, added or removed**, and explain each change in words. Every claim must be backed by geometry and by the frames that show it. Viewpoint changes, occlusion and unseen areas must not be reported as changes.

```
whatchanged run before.mp4 after.mp4 -o out/
  -> out/report.html     (one self-contained 3D viewer: open by double-click, no server)
  -> out/changes.json    (structured change events)
  -> out/evidence/*.jpg  (before/after crops for each event)
  -> out/topdown.png     (robot-style 2D occupancy diff map)
```

**CPU-first:** the default pipeline runs on an ordinary laptop/VM CPU, with no GPU. A GPU, if available, only makes the optional high-quality reconstruction mode faster (see §2a).

---

## 1. The four ideas behind this design

These are the choices that make this approach stronger than "reconstruct twice, then diff the point clouds".

### Idea 1: Reconstruct both videos together in one pass, so they share coordinates automatically
Keyframes from **both** videos go into a **single** reconstruction, so all cameras and depth maps come out in **one coordinate frame** and there is no separate registration step to get wrong. The unchanged room usually makes up most of the pixels, so it anchors the solution.
- **CPU default:** joint SfM (structure-from-motion: COLMAP/pycolmap, SIFT or SuperPoint+LightGlue) over the keyframes of both videos gives the camera poses and sparse 3D points. A small **metric** monocular depth model (Depth Anything V2 Metric-Indoor Small, ~25M params) gives one dense depth map per keyframe. Each depth map is scaled to agree with the SfM points. Bonus: this gives the scene in **metres**.
- **Optional mode:** one forward pass of a feed-forward multi-view model (VGGT; π³ / MapAnything as alternatives). It's slower and needs a lot of RAM on CPU, and is fast on a GPU. Both modes share the same `Reconstructor` interface.
- **Refinement:** estimate a robust Sim(3) between the before-points and the after-points using only *static-consistent* points. Repeat once after removing the changed regions.
- **Fallback if joint reconstruction gets confused by large changes:** reconstruct each video separately, then register them with cross-video feature matches (e.g. LightGlue) + RANSAC Sim(3) + ICP on static points. Both paths share the same interface, and an experiment (§7) decides which one is the default.

### Idea 2: A change is a free-space violation, not a large point distance
This is the core of the system. Raw point-cloud distance cannot tell apart *"the chair is gone"*, *"the chair is hidden behind the sofa"* and *"the after video never pointed there"*. A ray-consistency test can:

For each 3D point **p** from the before scene, and each after camera whose view contains **p**:
- `d_obs` = depth the after camera observed at p's pixel; `d_p` = depth of p in that camera.
- `d_obs > d_p + τ`: the camera **sees through** where p was, so p is **gone** (evidence for removed / moved-from).
- `|d_obs − d_p| ≤ τ`: p is **still there** (static evidence).
- `d_obs < d_p − τ`: p is **occluded** in this view, so this view gives **no evidence**.
- Not inside any after camera's view, or low depth confidence: **unobserved**.

`τ` scales with depth (e.g. `τ = a + b·d`) because depth error grows with distance. Votes are added up over all after views, giving each point one of four states: **static / changed / occluded / unobserved**, plus an evidence count. Running the same test from after to before finds **added / moved-to** points. All of this is a vectorised projection (points × cameras) in NumPy/PyTorch and takes seconds.

The result: the system only says "removed" when the after video **actually looked at that spot and saw empty space**. Places it never saw are reported as **unknown** and drawn as such in the viewer.

### Idea 3: Run semantics only where the geometry says something changed
Large models do not run over the video. Geometry first proposes a small number of change clusters. For each cluster:
1. Pick the 1–3 keyframes that see it best: most visible points, closest to front-on, sharpest.
2. Project the cluster into those frames and use the projection as a point/box prompt for **SAM 2** (or SAM 3 if it works better), giving a clean object mask.
3. Lift the mask back to 3D using the depth map. This gives the full object extent, including parts of the object that did not change.
4. Label it with a **VLM** on the crop: "what is this object?", answered with a short open-vocabulary label plus attributes (colour, material). An open-vocabulary detector (Grounding DINO / CLIP) serves as an offline fallback.

So heavy models see **about 10–30 crops in total, not hundreds of frames**. This is the main speed lever on the semantic side.

### Idea 4: "Moved" is a matching problem with geometric proof
Removed clusters (before) and added clusters (after) are paired with a Hungarian assignment. The cost combines:
`C = λ1·(1 − cos(DINOv2 crop embeddings)) + λ2·shape/size difference (3D box dims, FPFH) + λ3·label mismatch + λ4·implausible distance`
Each candidate pair is then **checked with ICP**: if the before-object's points, after a rigid transform, fit the after-object's points well, the pair becomes one `moved` event with translation (m) and yaw (°). Otherwise the pair stays as separate `removed` and `added` events. An optional VLM yes/no check ("same object in both crops?") breaks ties.

---

## 2. Pipeline and speed strategy (no per-frame analysis)

| Stage | What happens | Cost |
|---|---|---|
| 0. Decode | `ffmpeg` decodes at low resolution (~5 fps candidates, model input size). Decoding is cheap; inference is what's expensive. | CPU, seconds |
| 1. Adaptive keyframes | Score each candidate frame: **sharpness** (Laplacian variance), **parallax** since the last keyframe (sparse LK optical flow at ¼ res), **novelty** (small global embedding), **coverage gain**. Greedily keep a frame when it adds enough parallax/novelty and is sharp. **Overlap-aware:** make sure the after keyframes cover the regions seen in the before video (global-descriptor retrieval), because the change test needs views that cover the same areas. Target **16–32 keyframes per video**. | CPU, seconds |
| 2. Joint reconstruction | **CPU default:** joint SfM over all keyframes of both videos (poses + intrinsics + sparse points), then metric depth per keyframe (Depth Anything V2 Metric-Indoor Small), scaled to the SfM points. Pixels whose depth disagrees with the SfM points or neighbouring views are masked out. **Optional:** a single VGGT pass. | CPU (benchmark in §2a) |
| 3. Gravity + scale | Fit the floor plane with RANSAC to get the up axis (makes yaw angles and the top-down map meaningful). Metric scale comes from the median ratio between metric depth and SfM depth. Fallback: an assumed camera height. | CPU, seconds |
| 4. Ray-consistency change test | Idea 2, run in both directions, giving per-point state + evidence. | CPU, seconds |
| 5. Clustering | Voxelise, keep changed points with enough votes, DBSCAN, then drop tiny or thin clusters (below the depth-noise floor). | CPU, seconds |
| 6. Object lifting + semantics | Idea 3 on ~10–30 crops (SAM 2 tiny/small, DINOv2-small, CLIP on CPU; VLM via API). | CPU + API |
| 7. Move matching | Idea 4. | CPU, seconds |
| 8. Report | JSON, then natural language, then a self-contained `report.html`. | CPU, seconds |

All intermediate outputs (keyframes, reconstruction `.npz`, clusters) are **cached**. Changing thresholds or the viewer reruns only stages 4–8, which takes seconds and needs no GPU.

Illustrative target (not measured yet): two 30 s clips (~1,800 frames) → ~300 decoded candidates → ~32–50 keyframes for reconstruction → ~20 crops for semantics. Target: a few minutes end-to-end on an 8-core CPU. The real per-stage runtime breakdown goes in the README.

### 2a. CPU benchmark (measured on this VM: 8-core Xeon 8375C, 31 GB RAM, no GPU)
The primary test data is a real before/after pair recorded on a phone (`examples/room01`: the laptop was moved from the bedside table onto the bed and a cushion was removed). An early synthetic Blender room with known poses and changes (`tools/make_synthetic_room.py`: 1 moved chair, 1 removed bin, 2 added objects) was used for pipeline development; `bench/bench_recon.py` measures time, peak RAM, pose accuracy (ATE) and metric scale on it.

Measured on the real example `examples/room01` (39 keyframes, see README for the full table):

- **SfM + Depth Anything V2 Metric-Indoor Small (default):** all 39 keyframes registered in one model;
  SfM ~15 s, depth ~0.7–1.3 s per frame at 392 px input (6 s per frame at the default 518 px, with only
  ~7% median depth difference), whole pipeline ~1.5–2 min, ~2 GB peak RAM.
- **VGGT-1B on CPU:** the checkpoint loads, but a joint pass over ~40 frames is far too slow and
  memory-hungry for an interactive demo on this machine, so it stays the optional GPU backend.

---

## 3. Output format

`changes.json` (example schema; values are placeholders):
```json
{
  "units": "meters", "up_axis": "+z", "frame": "shared_world",
  "events": [
    {"id": "e1", "type": "moved", "label": "wooden chair",
     "before": {"center": [0, 0, 0], "bbox_3d": {"...": "..."}, "yaw_deg": 0},
     "after":  {"center": [0, 0, 0], "bbox_3d": {"...": "..."}, "yaw_deg": 0},
     "displacement_m": 0.0, "rotation_deg": 0.0,
     "confidence": 0.0, "evidence": {"before_frames": [], "after_frames": [], "votes_seen_through": 0},
     "description": "..."},
    {"id": "e2", "type": "added", "...": "..."},
    {"id": "e3", "type": "removed", "...": "..."}
  ],
  "unobserved_regions": [{"bbox_3d": {"...": "..."}, "note": "not seen in after video"}]
}
```
- **Natural language:** an LLM writes the summary **only from this JSON**. It is not allowed to invent numbers. Example: *"The wooden chair moved 1.2 m toward the window and turned ~40°. A cardboard box now sits in the walkway near the door. The backpack on the bed was removed."* A template-based summary works without any API.
- **Viewer (`report.html`, Three.js, one file):** grey static room; **red** = removed, **green** = added, **orange** box + arrow (before → after ghost) = moved; dim hatched blue = unobserved/uncertain. Includes a before/after/both toggle and a confidence slider. Clicking an event shows the evidence crops and the frames that support it. Camera frustums for both videos can be toggled.
- **Top-down occupancy diff (`topdown.png`):** a 2D map at robot height showing new obstacles, cleared space and moved objects. This connects the result directly to robot navigation.

---

## 4. Tech stack
- **Python 3.12 + uv**, NumPy/PyTorch, OpenCV, Open3D (RANSAC/ICP/DBSCAN/FPFH), SciPy (Hungarian)
- **Reconstruction (CPU default):** pycolmap SfM (SIFT, optionally SuperPoint+LightGlue) + Depth Anything V2 Metric-Indoor Small, behind a `Reconstructor` interface; **optional** VGGT / π³ / MapAnything mode for GPU users
- **Semantics:** SAM 2 tiny/small (masks from prompts), DINOv2-small (appearance embeddings), VLM via an OpenAI-compatible API for labels and summary; CLIP + template summary as a fully offline fallback
- **Compute:** everything runs on CPU by default; no GPU or cloud GPU needed
- **Viewer:** a single-file Three.js template with the point cloud embedded (decimated, quantised) so it stays a few MB
- Model licences checked before final submission (some weights are non-commercial; fine for a challenge, but noted in the README)

## 5. Repo layout
```
what-changed-in-the-room/
  README.md               # run instructions, examples, results, runtime table
  pyproject.toml / uv.lock
  whatchanged/
    cli.py                # `whatchanged run before.mp4 after.mp4 -o out/`
    video.py              # decode
    keyframes.py          # adaptive + overlap-aware selection
    recon/                # base.py (interface), vggt.py, separate_register.py
    align.py              # gravity, metric scale, static-consensus Sim(3)
    visibility.py         # ray-consistency change test (core)
    clusters.py
    objects.py            # mask prompting, lifting, 3D boxes, move matching + ICP
    semantics.py          # VLM labels, NL summary, offline fallback
    report/               # build.py, template.html, topdown.py
  examples/room01/        # before.mp4, after.mp4, ground_truth.yaml, output/
  eval/                   # metrics + experiments
  docs/PLAN.md, docs/DESIGN.md  # design notes & tradeoffs (submission note)
```

## 6. Milestones (in order; each ends with something you can look at)
1. **M0 Data:** you record 2–3 before/after pairs (guide in §8) and write down the real changes in `ground_truth.yaml`. Until then, I develop with a quick test clip I make myself.
2. **M1 Shared 3D:** keyframes + joint CPU reconstruction (SfM + metric depth), producing a viewer that shows **one** fused room with the camera paths of both videos. Proves the shared coordinate frame. *(Most important checkpoint.)*
3. **M2 Change proposals:** ray-consistency test + clusters. The viewer shows red/green/unknown regions.
4. **M3 Objects + moves:** SAM 2 lifting, 3D boxes, move matching + ICP, gravity + metric scale. The viewer shows arrows with displacement in metres.
5. **M4 Semantics:** labels + natural-language report + evidence crops.
6. **M5 Presentation:** final `report.html`, top-down map, README with one-command run, committed example outputs, `DESIGN.md`.
7. **M6 Experiments:** sparse vs dense keyframes, joint vs separate reconstruction, visibility test on vs off (false "removed" rate), geometry-only vs object-aware matching, per-stage runtime. Precision/recall for moved/added/removed; translation error for moved objects.

## 7. Risks and mitigations
| Risk | Mitigation |
|---|---|
| SfM fails to put both videos in one model (too little overlap/parallax) | Overlap-aware keyframes; exhaustive matching across videos; detect split models and warn; fall back to registering two models with cross-video matches |
| Monocular depth is inconsistent between views | Scale/shift each depth map to the SfM points; mask pixels that disagree across views; larger τ in the change test |
| Joint reconstruction gets distorted by changed objects | Static parts dominate; confidence masking; static-consensus Sim(3) refinement; separate-then-register fallback (compared in M6) |
| Textureless walls / glossy or glass surfaces give bad depth | Use depth confidence; these points count as "unobserved", not "changed" |
| Scale ambiguity makes "moved 1.2 m" meaningless | Metric depth prior + floor-plane gravity; scale source reported in JSON |
| Small objects fall below depth noise | Minimum object size stated (~10–15 cm, to validate); semantic crop check can rescue borderline cases |
| Not enough overlap between videos | Overlap-aware keyframes; coverage check that warns the user and marks unobserved areas |
| Lighting changes between recordings | Detection is geometric; colour is used only for identity matching |
| VLM label errors | Labels are descriptive only and never decide whether a change happened; geometry decides |

## 8. Capture guide (for the example videos)
- 15–30 s per video, 1080p, landscape, steady and slow; **walk in an arc** (sideways movement gives parallax; avoid only spinning in place)
- Good, even lighting; include stable references (walls, door, bed, desk)
- Between recordings, make **3–5 clear changes**: move a chair ~1 m, add a box/bag on the floor, remove an object, add a small obstacle (≥ ~15 cm)
- Record the second video along a **different path or start angle**, keeping most of the room in common
- Write down the real changes (that is the ground truth)

## 9. Open decisions
1. **Reconstruction default:** decided CPU-first: SfM + metric monocular depth as the default; VGGT stays optional (see §2a benchmark).
2. **Semantics:** VLM through an OpenAI-compatible API (best labels/descriptions) plus an offline open-source fallback (recommended), or fully open-source only.
3. ~~GPU~~: decided CPU-only.
