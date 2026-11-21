"""Ingest new media into Homeweave Studio with live Gemini calls (`studio_ingest.py`).

    python studio_ingest.py <folder>

Room photos named K*, L*, B*, Be* that are not in the catalogue are gauged by gemini-4-hydrogen
(in parallel). Home videos named Myhome* are read into the furniture inventory.
Nothing is hand-estimated here: every number comes from the model's validated answer.
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import studio_live as L

CAT = {"K": "kitchen", "L": "living", "B": "bathroom", "Be": "bedroom"}


def main(folder: Path) -> None:
    L.require_key()
    have = {p["id"] for p in json.loads(L.CATALOG_PATH.read_text(encoding="utf-8"))["photos"]}
    photos = []
    for f in sorted(folder.iterdir()):
        m = re.fullmatch(r"(Be|K|L|B)(\d+)\.(jpe?g|png|webp)", f.name, re.I)
        if m and f.stem not in have:
            photos.append((f, CAT[m.group(1)], f.stem))
    videos = [f for f in sorted(folder.iterdir()) if re.fullmatch(r"myhome\d*\.(mp4|mov)", f.name, re.I)]
    print(f"{len(photos)} new photos, {len(videos)} home videos")

    def do_photo(job):
        f, cat, pid = job
        t = time.time()
        try:
            e = L.perceive_photo(f.read_bytes(), f.name, cat, pid=pid)
            return f"  {pid:4s} ok  {e['gauged']['w']} x {e['gauged']['d']} m  conf {e['gauged']['conf']}  {len(e['items'])} items  {time.time() - t:.0f}s  {e['title']}"
        except Exception as exc:
            return f"  {pid:4s} FAILED {type(exc).__name__}: {str(exc)[:200]}"

    def do_video(f):
        t = time.time()
        try:
            items = L.perceive_video(f, f.stem.lower())
            return f"  {f.name} ok  {len(items)} pieces  {time.time() - t:.0f}s: " + ", ".join(f"{i['label']} ({i['w']}x{i['d']} m, {i['conf']})" for i in items)
        except Exception as exc:
            return f"  {f.name} FAILED {type(exc).__name__}: {str(exc)[:200]}"

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(do_photo, j) for j in photos] + [pool.submit(do_video, v) for v in videos]
        for fu in futs:
            print(fu.result(), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else ".."))
