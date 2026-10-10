"""Semantic verification and labelling of geometric change proposals.

Geometry proposes *where* something changed; this module asks *what* it is and rejects proposals
whose before/after evidence looks the same (depth noise on blankets, walls, furniture edges).

- VLM mode: any OpenAI-compatible multimodal endpoint (WC_LLM_* or PW_LLM_* or OPENAI_* env vars).
- Offline mode: CLIP zero-shot label + before/after embedding similarity (weaker, no network).
"""
from __future__ import annotations

import base64
import json
import os

import cv2
import numpy as np

from .objects import ChangeObject, Evidence
from .recon import Scene

VOCAB = ["laptop", "cushion", "pillow", "chair", "box", "bag", "backpack", "bottle", "book", "blanket",
         "shoe", "cup", "phone", "lamp", "towel", "clothes", "toy", "plant", "monitor", "keyboard",
         "basket", "bin", "suitcase", "stool", "wall", "floor", "bed sheet", "table", "drawer", "curtain"]
BACKGROUND = {"wall", "floor", "bed sheet", "table", "drawer", "curtain"}


def context_image(scene: Scene, e: Evidence, pad: float = 0.8, size: int = 384) -> np.ndarray:
    """Crop around the evidence box with extra context and the box itself drawn in."""
    img = scene.views[e.view].image
    h, w = img.shape[:2]
    x0, y0, x1, y1 = e.box
    bw, bh = x1 - x0, y1 - y0
    X0, Y0 = int(max(0, x0 - pad * bw)), int(max(0, y0 - pad * bh))
    X1, Y1 = int(min(w, x1 + pad * bw)), int(min(h, y1 + pad * bh))
    c = img[Y0:Y1, X0:X1].copy()
    cv2.rectangle(c, (x0 - X0, y0 - Y0), (x1 - X0, y1 - Y0), (255, 230, 0), 2)
    s = size / max(c.shape[:2])
    return cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)))


def _llm_config() -> tuple[str, str, str] | None:
    for p in ("WC_LLM", "PW_LLM", "OPENAI"):
        key = os.environ.get(f"{p}_API_KEY")
        if key:
            base = os.environ.get(f"{p}_BASE_URL") or "https://api.openai.com/v1"
            model = os.environ.get(f"{p}_MODEL") or "gpt-4.1"
            return base, key, model
    return None


def _b64(img: np.ndarray) -> str:
    _ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


PROMPT = """You are verifying a 3D change detector for a room scanned twice with a phone.
The first {nb} image(s) are from the BEFORE video, the last {na} from the AFTER video. All show the
same 3D region marked by a yellow box. Camera angle, framing, blur and lighting differ, and the box
can be offset by a few centimetres, so look at the whole neighbourhood, not just the box edges.
The 3D geometry at that spot differs between the two videos. Decide what, if anything, changed.
Answer ONLY with JSON: {{"change": "disappeared"|"appeared"|"none", "object": "<short generic noun, e.g. laptop>",
"description": "<one sentence: what is at that spot before vs after>", "confidence": 0.0-1.0}}
"disappeared": a distinct, movable physical object is clearly at that spot in the BEFORE images and
clearly absent from that spot in ALL AFTER images. "appeared": the reverse. Be skeptical: prints or
patterns on fabric, wall/floor/furniture parts and shadows are not objects, and if the same thing is
visible at that spot in both videos, even partly or from another angle, answer "none"."""


DIRECTIONS = {"disappeared": "gone", "appeared": "new"}

SAME_PROMPT = """Image 1 (BEFORE video) and image 2 (AFTER video) of the same room each mark an object
with a yellow box. The first object disappeared from its spot and the second appeared somewhere else.
Is it the same physical object that was moved? Answer ONLY with JSON:
{"same_object": true|false, "object": "<short generic noun>", "reason": "<one sentence>"}"""


def showing(o: ChangeObject) -> list:
    """Evidence views in which the object itself is visible (before for removed, after for added)."""
    return o.evidence if o.kind == o.cluster.kind else o.counter_evidence


def _parse_json(txt: str) -> dict:
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1].removeprefix("json")
    return json.loads(txt[txt.find("{"): txt.rfind("}") + 1])


class Labeller:
    def __init__(self, mode: str = "auto"):
        cfg = _llm_config() if mode in ("auto", "vlm") else None
        if mode == "vlm" and cfg is None:
            raise RuntimeError("--semantics vlm needs WC_LLM_API_KEY (or OPENAI_API_KEY)")
        self.cfg = cfg
        self.mode = "vlm" if cfg else "clip"
        self._clip = None

    # --- CLIP (always used for embeddings, used for labels in offline mode) ---
    def _load_clip(self):
        if self._clip is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
            p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            with torch.no_grad():
                t = m.get_text_features(**p(text=[f"a photo of a {w}" for w in VOCAB], return_tensors="pt",
                                            padding=True))
            t = t if torch.is_tensor(t) else t.pooler_output
            self._clip = (m, p, torch.nn.functional.normalize(t, dim=-1).numpy())
        return self._clip

    def embed(self, ims: list[np.ndarray]) -> np.ndarray:
        import torch

        m, p, _ = self._load_clip()
        with torch.no_grad():
            f = m.get_image_features(**p(images=ims, return_tensors="pt"))
        f = f if torch.is_tensor(f) else f.pooler_output
        return torch.nn.functional.normalize(f, dim=-1).numpy()

    def _crop(self, scene: Scene, e: Evidence) -> np.ndarray:
        x0, y0, x1, y1 = e.box
        return scene.views[e.view].image[y0:y1, x0:x1]

    def __call__(self, scene: Scene, o: ChangeObject) -> None:
        own = self.embed([self._crop(scene, e) for e in o.evidence])
        o.embedding = own.mean(0)
        sim = float((own @ self.embed([self._crop(scene, e) for e in o.counter_evidence]).T).max()) \
            if o.counter_evidence else 0.0
        o.extra["clip_before_after_similarity"] = round(sim, 3)
        _, _, text = self._load_clip()
        clip_label = VOCAB[int(np.argmax(o.embedding @ text.T))]
        o.extra["clip_label"] = clip_label
        if self.mode == "vlm" and o.counter_evidence:
            try:
                self._vlm(scene, o)
                return
            except Exception as ex:  # noqa: BLE001 - any network/parsing problem falls back to CLIP
                o.extra["vlm_error"] = str(ex)[:200]
        o.label = clip_label
        o.verified = sim < 0.93 and clip_label not in BACKGROUND
        o.description = f"{o.label} {'removed' if o.kind == 'gone' else 'added'} (CLIP, offline)"
        o.extra["confidence"] = round(max(0.0, min(1.0, (0.97 - sim) * 8)), 2)

    def same_object(self, scene: Scene, g: ChangeObject, n: ChangeObject, sim: float) -> tuple[bool, str, str]:
        """Is the removed object g the same physical object as the added object n?"""
        if self.mode == "vlm":
            try:
                content = [{"type": "text", "text": SAME_PROMPT},
                           {"type": "image_url", "image_url": {"url": _b64(context_image(scene, showing(g)[0]))}},
                           {"type": "image_url", "image_url": {"url": _b64(context_image(scene, showing(n)[0]))}}]
                d = _parse_json(self._chat(content))
                return bool(d.get("same_object")), str(d.get("object") or n.label).lower(), str(d.get("reason", ""))
            except Exception as ex:  # noqa: BLE001 - fall back to the CLIP rule below
                n.extra["vlm_error"] = str(ex)[:200]
        ok = sim >= 0.9 and g.label == n.label
        return ok, n.label, f"CLIP similarity {sim:.2f}, labels {g.label}/{n.label}"

    def _chat(self, content: list) -> str:
        from openai import OpenAI

        base, key, model = self.cfg
        r = OpenAI(base_url=base, api_key=key, timeout=60).chat.completions.create(
            model=model, messages=[{"role": "user", "content": content}], temperature=0, max_tokens=300)
        return r.choices[0].message.content

    def _vlm(self, scene: Scene, o: ChangeObject) -> None:
        d = self._ask(scene, o, o.evidence[:2], o.counter_evidence[:2])
        change = d.get("change")
        if change in DIRECTIONS and (len(o.evidence) > 1 or len(o.counter_evidence) > 1):
            # confirm with a second, different pair of views; a real change must survive both
            d2 = self._ask(scene, o, o.evidence[1:3] or o.evidence[:1], o.counter_evidence[1:3] or o.counter_evidence[:1])
            o.extra["second_opinion"] = str(d2.get("description", ""))
            if d2.get("change") != change:
                change, d = "none", {**d, "description": d2.get("description", d.get("description", ""))}
        o.verified = change in DIRECTIONS
        if o.verified and DIRECTIONS[change] != o.kind:
            # geometry only says "this spot differs"; the images tell which way round it changed
            o.extra["direction_from_images"] = f"{o.kind}->{DIRECTIONS[change]}"
            o.kind = DIRECTIONS[change]
            o.embedding = self.embed([self._crop(scene, e) for e in showing(o)]).mean(0)
        o.label = str(d.get("object") or o.extra["clip_label"]).lower()
        o.description = str(d.get("description", ""))
        o.extra["confidence"] = float(d.get("confidence", 0.5))
        o.extra["labeller"] = self.cfg[2]

    def _ask(self, scene: Scene, o: ChangeObject, own: list, other: list) -> dict:
        before, after = (own, other) if o.cluster.kind == "gone" else (other, own)
        content = [{"type": "text", "text": PROMPT.format(nb=len(before), na=len(after))}]
        for e in before + after:
            content.append({"type": "image_url", "image_url": {"url": _b64(context_image(scene, e))}})
        return _parse_json(self._chat(content))
