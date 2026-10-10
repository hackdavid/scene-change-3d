# What Changed in the Room?

Give it two short phone videos of the same room (**before** and **after**). It builds one shared
3D model from both, finds what was **moved, added or removed**, names each object, and writes
an interactive 3D report. **It runs on a CPU; no GPU is needed.**

```bash
uv sync                                   # Python 3.12, CPU-only PyTorch
uv run python -m whatchanged examples/room01/before.mp4 examples/room01/after.mp4 --out out/room01
open out/room01/report.html               # single self-contained file
```

## Example: `examples/room01`

Two handheld phone clips (5 s and 8 s) of a bedroom. Between the recordings the **laptop was
moved** from the bedside table onto the double bed and the **cushion was removed** from the single bed.
The clips are stored at 1080p (`before.mp4`, `after.mp4`) to keep the repo small; the original 4K recordings (not in git) give a similar but noisier result, see Limitations.

Output (`examples/room01/output/`, produced by the command above):

> The laptop moved 1.20 m: from on a surface about 0.5 m above the floor (x=-1.94, y=0.31) to on a surface about 0.5 m above the floor (x=-0.74, y=0.24). A pillow was removed from on a surface about 0.5 m above the floor (x=0.11, y=1.76, size 0.59x0.76x0.16 m).

| File | What it is |
|---|---|
| `report.html` | Three.js 3D viewer: the room as a point cloud, red = removed, green = added, orange arrow = moved; click an event to fly to it and see the before/after evidence |
| `changes.json` | Structured result: events with 3D boxes (metres, z up from the floor), displacement, labels, confidence, evidence frames, rejected proposals, timings |
| `summary.txt` | The one-paragraph natural-language summary above (written from `changes.json`, no free-form invention) |
| `evidence_sheet.jpg` | Before/after crops for each event, with the 3D region projected into the frames |
| `topdown.jpg` | Top-down map (robot view): static surfaces, cameras, change boxes and the move arrow |

![evidence](examples/room01/output/evidence_sheet.jpg)
![topdown](examples/room01/output/topdown.jpg)
![report](examples/room01/output/report.png)

The run also shows the false-positive filtering: geometry proposed 14 candidate regions,
the semantic check kept 4 (the laptop's old place, two clusters on its new place that were merged into one, and the cushion) and rejected the rest
(blank wall, floor, a ceiling fan, a backpack that was there in both videos...). They are listed under
"Rejected proposals" in the report and in `changes.json`.

## How it works

```
2 videos ─► adaptive keyframes ─► ONE joint SfM (both videos) ─► metric depth per keyframe
         ─► floor/gravity alignment ─► ray-consistency change test (both directions)
         ─► 3D clustering ─► semantic verification + labels ─► move pairing ─► report
```

1. **Adaptive keyframes, not frame-by-frame** (`keyframes.py`). ffmpeg decodes ~5 fps at 640 px;
   cheap optical flow measures how far the camera moved; ~20 frames per video are picked evenly
   along that motion curve, snapped to the sharpest neighbour. Of ~800 source frames only ~40 reach a model.
2. **One shared reconstruction** (`recon.py`). Keyframes of *both* videos go into a single
   pycolmap SfM run, so all cameras land in one coordinate frame and there is no separate
   alignment step to get wrong. Depth Anything V2 (metric, indoor, small) then gives a dense depth map per
   keyframe, scaled to the SfM points (so units are ~metres) and harmonised across views using the static room.
3. **Gravity** (`align.py`). The largest horizontal plane gives "up"; the lowest dense horizontal layer is the floor (z = 0).
4. **Visibility-aware change test** (`change.py`). A before-point is called *gone* only if
   after-cameras looked at that exact spot and saw **through** it (measured depth clearly behind it).
   If something is in front, it is only *occluded*; if no after-camera covered it, it is *unobserved*.
   Neither counts as a change. This is what separates real changes from viewpoint differences,
   occlusion and incomplete coverage. Same test after→before gives *new* points. Blank, textureless
   pixels (white walls) may not start a change claim, because monocular depth there is unreliable.
5. **Clustering** (`clusters.py`). DBSCAN on changed points; a cluster must be seen in ≥ 3 keyframes
   and stick out ≥ 4 cm from its support surface.
6. **Semantics only where geometry found something** (`semantics.py`). Geometry says "this spot differs";
   the images say what changed and in which direction (a laptop leaving a box can look like "new box surface" geometrically).
   Counter-evidence views are only those that actually looked *through* the spot, so the model never compares an
   object with whatever happened to stand in front of it. A "real change" must be confirmed twice, from two different view pairs.
   Clusters centred above 1.6 m (ceiling, skylight) are ignored: they are not obstacles and their monocular depth is unreliable.
   Duplicate clusters of one object (screen + keyboard of a laptop) are merged. For each cluster the best
   views from both videos are cropped (the 3D region is projected into each frame). A vision-language model sees
   before/after crops and confirms whether a distinct object really appeared/disappeared, and names it.
   Without an API key, an offline CLIP mode labels crops and uses before/after similarity instead.
7. **Moved = removed + added + same object** (`match.py`). A removed and an added object are paired only if
   their labels agree; identity is then confirmed (VLM, or CLIP rule offline) and the shift is checked with ICP. Displacement is reported in metres.
8. **Report** (`report/`). JSON first; the summary sentence and the viewer are generated from it, so the
   text, the 3D boxes and the evidence always agree.

Options: `--keyframes N` (per video, default 20), `--semantics auto|vlm|clip`, `--no-cache`
(keyframes and the reconstruction are cached in `<out>/work/`, so re-running with other thresholds takes ~20 s).
VLM settings come from `WC_LLM_BASE_URL / WC_LLM_API_KEY / WC_LLM_MODEL` (any OpenAI-compatible endpoint; `OPENAI_*` also works).

## Resource constraints: why CPU, and what we would do with a GPU

This was built and tested on a machine **without a GPU** (8-core Xeon, 31 GB RAM). So the default
pipeline is chosen to be accurate enough while running in about two minutes on a CPU:

| Run on `examples/room01` (CPU, fresh, no cache) | Time |
|---|---|
| Keyframe selection (ffmpeg decode + optical flow, both videos) | 2 s |
| SfM: SIFT features + exhaustive matching + mapping (39 views) | 12 s |
| Depth model load | 6 s |
| Metric depth, 39 frames (~0.3 s/frame) | 11 s |
| Visibility-aware change detection + clustering | 5 s |
| Semantic verification (VLM API calls) | 52 s |
| **Total** | **92 s** |

Peak memory: **2.0 GB**. The VLM step is network-bound (a few API calls per candidate).

**What the CPU constraint costs us**

- Monocular depth (small model, 392 px input to save time) is noisy on blank walls and at object
  edges, so the smallest reliable change is roughly a shoebox (~15–20 cm); thin or flat objects (paper, cables) are filtered out on purpose.
- SfM needs real sideways camera motion; a pure rotation sweep can fail to register.
- We use ~20 keyframes per video; more frames would help coverage but cost ~1 s each on CPU.
- No dense mask model on CPU: object extents come from the changed 3D points, so a box can cover only the part of the object whose geometry changed.

**What we would use with a GPU (better approach)**

| Stage | CPU (this repo) | With a GPU |
|---|---|---|
| Reconstruction | pycolmap SfM + Depth Anything V2 Small per frame | **VGGT / MapAnything / π³**: one feed-forward pass over all keyframes of both videos gives poses + dense, multi-view-consistent point maps in seconds. Much better depth on walls and thin objects, no SfM failure modes |
| Frames | ~20 per video | 50–100 per video for full coverage |
| Object extent | changed-point clusters | **SAM 2** masks propagated through the keyframes and lifted to 3D → full object shape and tight 3D boxes |
| Semantics | API VLM (or CLIP offline) on a few crops | local open VLM + DINOv2 features, no network needed |
| Change test | same | same, at finer tolerance (2–3 cm instead of 6 cm) because the geometry is cleaner |

The important design choice is that only the reconstruction stage changes: everything downstream
(visibility test, clustering, semantics, matching, report) works on the same `Scene` (cameras +
depth maps), so a GPU backend can be added behind the same interface without touching the change logic.
The VGGT-1B checkpoint was tried on this CPU; it fits in RAM but is far too slow for a demo, which is why it is not the default.

## Limitations

- Tested on one real room (`room01`) plus a synthetic Blender room (`tools/make_synthetic_room.py`).
  Thresholds (6 cm depth tolerance, ≥ 3 views, ≥ 4 cm height) are tuned for phone clips of a small room.
- Object boxes come from the part of the object that changed, so sizes are approximate.
- Labels and the "is this a real change" decision depend on the VLM; the offline CLIP mode is weaker.
- Run-to-run variance is real. The same clips at 4K (`examples/room01/original`, not in git) found the laptop move
  (1.26 m) but missed the cushion in one run and added two small false changes on the floor behind the bed, where the
  bed occludes most after-views. A detection is only as good as the views that saw the spot; more keyframes or a GPU
  multi-view model would reduce this.
- The "laptop removed" box is larger than the laptop (it includes the part of the box/table surface the laptop had hidden).
- Capture tips: move sideways (not just rotate), keep both videos overlapping, ~5–15 s each, avoid fast pans.

## Repository layout

```
whatchanged/   keyframes.py recon.py align.py change.py clusters.py objects.py semantics.py match.py pipeline.py report/
examples/room01/  before.mp4 after.mp4 output/
docs/PLAN.md   design notes and tradeoffs
tools/         synthetic test-room generator (Blender)
bench/         reconstruction benchmark script
```
