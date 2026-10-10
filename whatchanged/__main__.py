import argparse

from .pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser(prog="whatchanged", description="3D change detection from two phone videos (CPU)")
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--out", default="out/run")
    ap.add_argument("--keyframes", type=int, default=20, help="keyframes per video")
    ap.add_argument("--semantics", choices=["auto", "vlm", "clip"], default="auto",
                    help="auto = VLM if an API key is set, else offline CLIP")
    ap.add_argument("--no-cache", action="store_true", help="recompute keyframes and reconstruction")
    a = ap.parse_args()
    run(a.before, a.after, a.out, a.keyframes, a.semantics, reuse=not a.no_cache)


if __name__ == "__main__":
    main()
