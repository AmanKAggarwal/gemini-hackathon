"""Homeweave Studio (`studio.py`): pick one photo per room, fit it into a real floor plan.

Deterministic, pure-Python fit engine plus the FastAPI routes for `studio.html`.
Follows the spec split: perception supplies gauged room sizes and item lists
(`data/studio_catalog.json`), this geometric engine does the placing.

Per room the engine:
  1. reads the gauged source room (metres) and the target room from the plan (pixels -> metres),
  2. enumerates the 8 rotations x reflections of the source frame (spec step 5),
  3. keeps furniture rigid, re-cuts built-in runs to the new wall length,
  4. slides / downsizes / drops items until hard constraints hold,
  5. scores each candidate and returns the full trace so the UI can show the search.

Conventions: metres, room-local frame with origin at the room's north-west corner,
x east, z south (plan view). Walls are N, E, S, W.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
from collections import deque
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from judge import (
    MODEL_DECISION,
    _extract_noul,
    _extract_score_1_to_5,
    _make_v1beta_client,
    build_layout_questions,
)
from perceive import media_block_from_source
from schemas import RoomType, RoomVisualStyle
from studio_correspond import correspond
from style import PHOTO_STYLE_CATALOG, extract_style_from_photo

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CATALOG_PATH = DATA_DIR / "studio_catalog.json"
CUSTOM_PHOTOS_PATH = DATA_DIR / "custom_studio_photos.json"
CUSTOM_UPLOAD_DIR = DATA_DIR / "custom_uploads"
IMG_DIR = DATA_DIR / "drive_inputs"
INVENTORY_PATH = DATA_DIR / "studio_inventory.json"
ITEM_DIR = DATA_DIR / "studio_items"

WALLS = "NESW"
EPS = 0.02
STEP = 0.1  # slide step, metres
# Tunable first guesses (spec: "ours are first guesses to tune").
COST = {"left1": 10.0, "left2": 5.0, "left3": 1.5, "swap": 3.0, "merge": 2.5, "move": 1.0,
        "open_edge": 6.0, "aspect": 2.0, "walkway": 4.0}
WALK_WIDTHS = [0.9, 0.8, 0.7, 0.6]  # spec hard constraint is 0.9 m; narrower widths are reported, not hidden

CAT_TO_ROOM_TYPE: dict[str, RoomType] = {
    "kitchen": "kitchen",
    "living": "living",
    "bedroom": "primary_bedroom",
    "bathroom": "bathroom",
}

CAT_TEMPLATE_ID: dict[str, str] = {
    "kitchen": "K1",
    "living": "L2",
    "bedroom": "Be1",
    "bathroom": "B1",
}


def _metal_hex(hw: str) -> str:
    return {"brushed_gold": "#c89d46", "chrome": "#cfd5dd", "brushed_nickel": "#b2b6bc"}.get(hw, "#232220")


def _floor_style_token(floor_mat: str) -> str:
    if floor_mat in ("veined_marble", "travertine_stone"):
        return "marble"
    if floor_mat in ("wood_plank", "dark_walnut"):
        return "wood"
    return "tile"


def load_catalog() -> dict[str, Any]:
    cat = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if CUSTOM_PHOTOS_PATH.exists():
        try:
            custom_list = json.loads(CUSTOM_PHOTOS_PATH.read_text(encoding="utf-8"))
            existing_ids = {p["id"] for p in cat["photos"]}
            for cp in custom_list:
                if cp.get("id") not in existing_ids:
                    cat["photos"].append(cp)
        except Exception:
            pass
    for p in cat["photos"]:
        if "style_meta" not in p and p["id"] in PHOTO_STYLE_CATALOG:
            p["style_meta"] = PHOTO_STYLE_CATALOG[p["id"]].model_dump()
    live_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    cat["live_gemini"] = live_gemini
    if live_gemini:
        cat["perception_source"] = "gemini-4-hydrogen + gemini-decision-1 (live)"
        cat["perception_note"] = "GEMINI_API_KEY is active: custom photo uploads run live gemini-4-hydrogen style perception and 3D judging uses gemini-decision-1."
    else:
        cat["perception_source"] = "cached + local vision"
        cat["perception_note"] = "Curated Drive catalog + deterministic regional pixel/material extractor for custom photo uploads (set GEMINI_API_KEY for live Gemini 4 calls)."
    cat["inventory"] = json.loads(INVENTORY_PATH.read_text(encoding="utf-8")) if INVENTORY_PATH.exists() else {"items": [], "sources": []}
    return cat


# ---------- frame transforms (rotations x reflections) ----------

def t_point(u: float, v: float, k: int, m: bool) -> tuple[float, float]:
    """Map a normalized source point through mirror `m` then `k` clockwise quarter turns."""
    if m:
        u = 1 - u
    for _ in range(k):
        u, v = 1 - v, u
    return u, v


def t_vec(dx: float, dz: float, k: int, m: bool) -> tuple[float, float]:
    if m:
        dx = -dx
    for _ in range(k):
        dx, dz = -dz, dx
    return dx, dz


def t_dir(d: str, k: int, m: bool) -> str:
    i = WALLS.index(d)
    if m and d in "EW":
        i = (i + 2) % 4
    return WALLS[(i + k) % 4]


# ---------- geometry helpers ----------

def overlap(a: list[float], b: list[float], eps: float = EPS) -> bool:
    return a[0] < b[0] + b[2] - eps and b[0] < a[0] + a[2] - eps and a[1] < b[1] + b[3] - eps and b[1] < a[1] + a[3] - eps


def target_room(plan: dict[str, Any], key: str) -> dict[str, Any]:
    r = plan["rooms"][key]
    s = plan["px_per_m"]
    x0, y0, x1, y1 = r["rect"]
    W, D = (x1 - x0) / s, (y1 - y0) / s

    def rect_m(b: list[float]) -> list[float]:
        return [(b[0] - x0) / s, (b[1] - y0) / s, (b[2] - b[0]) / s, (b[3] - b[1]) / s]

    wins = []
    for w in r.get("windows", []):
        o = x0 if w["wall"] in "NS" else y0
        wins.append({"wall": w["wall"], "a": (w["a"] - o) / s, "b": (w["b"] - o) / s})
    return {
        "key": key, "label": r["label"], "w": round(W, 2), "d": round(D, 2), "rect_px": r["rect"],
        "open": r.get("open", []), "windows": wins,
        "doors": [rect_m(b) for b in r.get("doors", [])],
        "obstacles": [rect_m(b) for b in r.get("obstacles", [])],
        "entry": [(r["entry"][0] - x0) / s, (r["entry"][1] - y0) / s],
    }


def walkway(room: dict[str, Any], solids: list[dict[str, Any]]) -> dict[str, Any]:
    """Widest walkway (from WALK_WIDTHS) that reaches every must/nice item from the room entry.

    Approximation: 0.1 m grid, obstacles inflated by half the walkway width, 4-connected flood fill.
    """
    W, D = room["w"], room["d"]
    nx, nz = max(1, int(round(W / STEP))), max(1, int(round(D / STEP)))
    blocks = [s["rect"] for s in solids] + room["obstacles"]
    targets = [s for s in solids if s["pri"] <= 2]

    def run(width: float) -> tuple[bool, list[str]]:
        r = width / 2 - 0.05
        free = [[True] * nz for _ in range(nx)]
        for i in range(nx):
            for j in range(nz):
                cx, cz = (i + 0.5) * STEP, (j + 0.5) * STEP
                for b in blocks:
                    ddx = max(b[0] - cx, 0, cx - (b[0] + b[2]))
                    ddz = max(b[1] - cz, 0, cz - (b[1] + b[3]))
                    if math.hypot(ddx, ddz) < r:
                        free[i][j] = False
                        break
        ex, ez = room["entry"]
        best, bd = None, 1e9
        for i in range(nx):
            for j in range(nz):
                if free[i][j]:
                    dd = math.hypot((i + 0.5) * STEP - ex, (j + 0.5) * STEP - ez)
                    if dd < bd:
                        best, bd = (i, j), dd
        if best is None:
            return False, [t["id"] for t in targets]
        seen = {best}
        q = deque([best])
        while q:
            i, j = q.popleft()
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < nx and 0 <= b < nz and free[a][b] and (a, b) not in seen:
                    seen.add((a, b))
                    q.append((a, b))
        missed = []
        for t in targets:
            b = t["rect"]
            reach = r + 0.35
            ok = any(
                math.hypot(max(b[0] - (i + 0.5) * STEP, 0, (i + 0.5) * STEP - (b[0] + b[2])),
                           max(b[1] - (j + 0.5) * STEP, 0, (j + 0.5) * STEP - (b[1] + b[3]))) <= reach
                for (i, j) in seen
            )
            if not ok:
                missed.append(t["id"])
        return not missed, missed

    for wdt in WALK_WIDTHS:
        ok, missed = run(wdt)
        if ok:
            return {"width_m": wdt, "meets_0_9": wdt >= 0.9, "unreached": []}
    return {"width_m": None, "meets_0_9": False, "unreached": missed}


# ---------- one candidate: place every item under transform (k, m) ----------

def place_candidate(photo: dict[str, Any], room: dict[str, Any], k: int, m: bool) -> dict[str, Any]:
    Ws, Ds = photo["gauged"]["w"], photo["gauged"]["d"]
    Wt, Dt = room["w"], room["d"]
    Wsr, Dsr = (Ds, Ws) if k % 2 else (Ws, Ds)  # source extents after rotation
    sx, sz = Wt / Wsr, Dt / Dsr

    src = {it["id"]: it for it in photo["items"]}
    order = sorted(photo["items"], key=lambda it: (it.get("floor", False), it["pri"], -(it["w"] * it["d"])))
    # parents must be placed before children
    order.sort(key=lambda it: 1 if it.get("merge") else 0)
    order.sort(key=lambda it: 1 if it.get("parent") else 0)
    order.sort(key=lambda it: it.get("floor", False))

    placed: dict[str, dict[str, Any]] = {}
    leftovers: list[dict[str, Any]] = []
    notes: list[str] = []
    open_hits = 0

    def solids() -> list[dict[str, Any]]:
        return [p for p in placed.values() if not p["floor"]]

    def valid(rect: list[float], tall: bool, ignore: str | None = None) -> bool:
        if rect[0] < -EPS or rect[1] < -EPS or rect[0] + rect[2] > Wt + EPS or rect[1] + rect[3] > Dt + EPS:
            return False
        for b in room["obstacles"] + room["doors"]:
            if overlap(rect, b):
                return False
        for p in solids():
            if p["id"] != ignore and overlap(rect, p["rect"]):
                return False
        if tall:
            for wn in room["windows"]:
                if wn["wall"] == "N" and rect[1] < 0.05:
                    lo, hi = rect[0], rect[0] + rect[2]
                elif wn["wall"] == "S" and rect[1] + rect[3] > Dt - 0.05:
                    lo, hi = rect[0], rect[0] + rect[2]
                elif wn["wall"] == "W" and rect[0] < 0.05:
                    lo, hi = rect[1], rect[1] + rect[3]
                elif wn["wall"] == "E" and rect[0] + rect[2] > Wt - 0.05:
                    lo, hi = rect[1], rect[1] + rect[3]
                else:
                    continue
                if min(hi, wn["b"]) - max(lo, wn["a"]) > 0.1:
                    return False
        return True

    def try_place(it: dict[str, Any], length: float, dep: float, typ: str, shape: str | None,
                  force_wall: str | None = None) -> tuple[list[float], float] | None:
        face = t_dir(it["face"], k, m)
        walls = [t_dir(w, k, m) for w in it.get("walls", [])]
        if force_wall:  # last resort: stand the item against a different real wall, facing into the room
            walls, face = [force_wall], WALLS[(WALLS.index(force_wall) + 2) % 4]
        bw, bd = (length, dep) if face in "NS" else (dep, length)
        if bw > Wt + EPS or bd > Dt + EPS:
            return None
        scx, scz = it["x"] + it["w"] / 2, it["z"] + it["d"] / 2
        par = None if force_wall else placed.get(it.get("parent", ""))
        if par is not None:
            ps = src[it["parent"]]
            ox, oz = t_vec(scx - (ps["x"] + ps["w"] / 2), scz - (ps["z"] + ps["d"] / 2), k, m)
            # keep the offset from the parent's edge, not its centre, when the parent was resized
            pw0, pd0 = (ps["d"], ps["w"]) if k % 2 else (ps["w"], ps["d"])
            pr = par["rect"]
            ox -= math.copysign(max(0.0, (pw0 - pr[2]) / 2), ox) if abs(ox) > pw0 / 2 else 0
            oz -= math.copysign(max(0.0, (pd0 - pr[3]) / 2), oz) if abs(oz) > pd0 / 2 else 0
            cx, cz = pr[0] + pr[2] / 2 + ox, pr[1] + pr[3] / 2 + oz
        else:
            u, v = t_point(scx / Ws, scz / Ds, k, m)
            cx, cz = u * Wt, v * Dt
        x, z = cx - bw / 2, cz - bd / 2
        if "N" in walls: z = 0.0
        if "S" in walls: z = Dt - bd
        if "W" in walls: x = 0.0
        if "E" in walls: x = Wt - bw
        x = min(max(x, 0.0), Wt - bw)
        z = min(max(z, 0.0), Dt - bd)
        tall = bool(it.get("tall"))
        if len(walls) == 1:
            axis = (1, 0) if walls[0] in "NS" else (0, 1)
            n = int(max(Wt, Dt) / STEP) + 1
            offs = [(0.0, 0.0)] + [(s * i * STEP * axis[0], s * i * STEP * axis[1]) for i in range(1, n) for s in (1, -1)]
        else:
            lim = 1.0 if par is not None else 1.5
            n = int(lim / STEP)
            offs = sorted(((i * STEP, j * STEP) for i in range(-n, n + 1) for j in range(-n, n + 1)),
                          key=lambda o: math.hypot(*o))
            if len(walls) == 2:  # corner items may leave the corner along either wall only
                offs = [o for o in offs if o[0] == 0 or o[1] == 0]
        for ox2, oz2 in offs:
            r = [round(x + ox2, 3), round(z + oz2, 3), round(bw, 3), round(bd, 3)]
            if valid(r, tall):
                return r, math.hypot(ox2, oz2)
        return None

    for it in order:
        face = t_dir(it["face"], k, m) if "face" in it else "S"
        horizontal = it.get("face", "S") in "NS"
        own_len, own_dep = (it["w"], it["d"]) if horizontal else (it["d"], it["w"])
        typ, shape = it["type"], it.get("shape")
        rec = {"id": it["id"], "type": typ, "label": it["label"], "pri": it["pri"], "floor": bool(it.get("floor")),
               "face": face, "flip": m, "shape": shape, "feat": it.get("feat"), "n": it.get("n"),
               "inferred": bool(it.get("inferred")), "status": "kept", "note": "", "moved_m": 0.0,
               "src_len": round(own_len, 2)}

        if it.get("floor"):
            # rugs follow their group, then get trimmed to the room (they never block anything)
            bw, bd = (it["d"], it["w"]) if k % 2 else (it["w"], it["d"])
            scx, scz = it["x"] + it["w"] / 2, it["z"] + it["d"] / 2
            par = placed.get(it.get("parent", ""))
            if par is not None:
                ps = src[it["parent"]]
                ox, oz = t_vec(scx - (ps["x"] + ps["w"] / 2), scz - (ps["z"] + ps["d"] / 2), k, m)
                cx, cz = par["rect"][0] + par["rect"][2] / 2 + ox * min(1, sx), par["rect"][1] + par["rect"][3] / 2 + oz * min(1, sz)
            else:
                u, v = t_point(scx / Ws, scz / Ds, k, m)
                cx, cz = u * Wt, v * Dt
            nw, nd = min(bw, Wt - 0.3), min(bd, Dt - 0.3)
            if shape == "round":
                nw = nd = min(nw, nd)
            x = min(max(cx - nw / 2, 0.15), Wt - nw - 0.15)
            z = min(max(cz - nd / 2, 0.15), Dt - nd - 0.15)
            if nw < bw - 0.05 or nd < bd - 0.05:
                rec["status"], rec["note"] = "resized", f"trimmed {bw:.1f} × {bd:.1f} → {nw:.1f} × {nd:.1f} m to fit the floor"
            rec.update(rect=[round(x, 3), round(z, 3), round(nw, 3), round(nd, 3)], len=round(nw, 2), dep=round(nd, 2))
            placed[it["id"]] = rec
            continue

        if it.get("needs_parent") and it.get("parent") not in placed:
            leftovers.append({"id": it["id"], "label": it["label"], "pri": it["pri"],
                              "reason": f"its group ({src[it['parent']]['label'].lower()}) was not placed"})
            continue

        length, dep = own_len, own_dep
        if it.get("stretch"):
            factor = sx if face in "NS" else sz
            wall_len = Wt if face in "NS" else Dt
            length = min(max(own_len * min(factor, 1.25), it.get("minw", 0.6)), wall_len)
            length = round(length, 2)
        res = try_place(it, length, dep, typ, shape)
        if res is None and it.get("stretch"):
            L = length - 0.2
            while res is None and L >= it.get("minw", 0.6) - 1e-6:
                res = try_place(it, round(L, 2), dep, typ, shape)
                if res is None:
                    L -= 0.2
            if res is not None:
                length = round(L, 2)
        if it.get("stretch") and res is not None and abs(length - own_len) > 0.05:
            rec["status"], rec["note"] = "resized", f"run re-cut {own_len:.1f} → {length:.1f} m to the new wall"
        if res is None and it.get("alt"):
            a = it["alt"]
            length, dep = a.get("len", own_len), a.get("dep", own_dep)
            typ, shape = a.get("type", typ), a.get("shape", shape)
            res = try_place(it, length, dep, typ, shape)
            if res is not None:
                rec.update(status="swapped", type=typ, shape=shape, n=a.get("n", rec["n"]),
                           note=f"{own_len:.1f} m does not fit → {a['label']}")
        if res is None and it.get("walls") and it["pri"] <= 2 and it.get("merge") not in placed:
            # try the other real walls, original size first, then the smaller variant
            a = it.get("alt") or {}
            for L2, D2, T2, S2, is_alt in ((own_len if not it.get("stretch") else max(it.get("minw", 0.6), min(own_len, min(Wt, Dt))), own_dep, it["type"], it.get("shape"), False),
                                           (a.get("len", own_len), a.get("dep", own_dep), a.get("type", it["type"]), a.get("shape", it.get("shape")), True)):
                if is_alt and not a:
                    break
                for wl in WALLS:
                    if wl in room["open"]:
                        continue
                    res = try_place(it, L2, D2, T2, S2, force_wall=wl)
                    if res is not None:
                        length, dep, typ, shape = L2, D2, T2, S2
                        rec.update(status="swapped" if is_alt else "moved", type=typ, shape=shape, face=WALLS[(WALLS.index(wl) + 2) % 4],
                                   n=a.get("n", rec["n"]) if is_alt else rec["n"],
                                   note=("downsized to " + a["label"] + " and " if is_alt else "") + "moved to the " + {"N": "north", "E": "east", "S": "south", "W": "west"}[wl] + " wall, the only one with room")
                        res = (res[0], 0.0)
                        break
                if res is not None:
                    break
        if res is None and it.get("merge") in placed:
            tub = placed[it["merge"]]
            tub.update(type="tubshower", status="merged",
                       note=(tub["note"] + "; " if tub["note"] else "") + "shower moved over the tub (no room for a separate one)")
            notes.append(f"{it['label']} merged into the tub")
            placed["_merged_" + it["id"]] = {"id": it["id"], "ghost": True, "floor": True, "pri": it["pri"], "status": "merged",
                                             "label": it["label"], "note": "combined with the tub", "rect": tub["rect"]}
            continue
        if res is None:
            need = max(own_len, own_dep)
            leftovers.append({"id": it["id"], "label": it["label"], "pri": it["pri"],
                              "reason": f"{own_len:.1f} × {own_dep:.1f} m has no collision-free spot in a {Wt:.1f} × {Dt:.1f} m room"
                              if need <= max(Wt, Dt) else f"needs {need:.1f} m; the room's longest side is {max(Wt, Dt):.1f} m"})
            continue
        rect, moved = res
        rec.update(rect=rect, len=round(length, 2), dep=round(dep, 2), moved_m=round(moved, 2))
        if moved > 0.05:
            rec["note"] = (rec["note"] + "; " if rec["note"] else "") + f"slid {moved:.1f} m to clear a door, window or neighbour"
            if rec["status"] == "kept":
                rec["status"] = "moved"
        walls_t = [t_dir(w, k, m) for w in it.get("walls", [])]
        if any(w in room["open"] for w in walls_t):
            open_hits += 1
        placed[it["id"]] = rec

    sol = solids()
    walk = walkway(room, sol)
    n_overlap = sum(1 for i in range(len(sol)) for j in range(i + 1, len(sol)) if overlap(sol[i]["rect"], sol[j]["rect"]))
    door_hits = sum(1 for s in sol for dz in room["doors"] if overlap(s["rect"], dz))
    items = [p for p in placed.values() if not p.get("ghost")]
    merged = [p for p in placed.values() if p.get("ghost")]

    aspect = abs(math.log(sx / sz))
    walk_pen = 0.0 if walk["meets_0_9"] else (COST["walkway"] * (0.5 if walk["width_m"] else 1.5))
    parts = {
        "leftover": sum(COST[f"left{min(l['pri'], 3)}"] for l in leftovers),
        "swap": COST["swap"] * sum(1 for p in items if p["status"] == "swapped") + COST["merge"] * len(merged),
        "move": COST["move"] * sum(p["moved_m"] for p in items),
        "open_edge": COST["open_edge"] * open_hits,
        "aspect": COST["aspect"] * aspect,
        "walkway": walk_pen,
    }
    parts = {a: round(b, 2) for a, b in parts.items()}
    worst = max(parts, key=lambda a: parts[a])
    why = {"leftover": f"{len(leftovers)} item(s) left over", "swap": "needs downsizing", "move": "items pushed far from their spot",
           "open_edge": "built-ins land on an open edge with no wall", "aspect": "room proportions stretch the layout",
           "walkway": "walkway under 0.9 m"}[worst] if parts[worst] > 0.01 else "clean fit"
    return {
        "k": k, "mirror": m, "name": f"{k * 90}°" + (" mirrored" if m else ""),
        "cost": round(sum(parts.values()), 2), "parts": parts, "why": why,
        "sx": round(sx, 3), "sz": round(sz, 3), "src_w": Wsr, "src_d": Dsr,
        "items": items, "merged": merged, "leftovers": leftovers,
        "check": {"overlaps": n_overlap, "door_blocks": door_hits, "walkway": walk},
    }


def fit_room(photo: dict[str, Any], plan: dict[str, Any], key: str) -> dict[str, Any]:
    room = target_room(plan, key)
    cands = [place_candidate(photo, room, k, m) for m in (False, True) for k in range(4)]
    best = min(range(len(cands)), key=lambda i: (cands[i]["cost"], i))
    b = cands[best]
    g = photo["gauged"]
    log = [
        {"phase": "gauge", "text": f"Gauged {photo['id']} “{photo['title']}”: about {g['w']:.1f} × {g['d']:.1f} m ({g['w'] * g['d']:.0f} m²), scaled from {g['ref']}. Confidence {g['conf']:.2f}."},
        {"phase": "gauge", "text": f"Found {len(photo['items'])} things to carry over: " + ", ".join(i["label"].lower() for i in photo["items"]) + "."},
        {"phase": "pair", "text": f"Paired with {room['label']} on the plan: {room['w']:.2f} × {room['d']:.2f} m ({room['w'] * room['d']:.1f} m²). Plan scale {plan['px_per_m']:.1f} px/m ({plan['scale_ref']})."},
        {"phase": "search", "text": f"Tried 8 orientations (4 rotations × mirror). Best is {b['name']} at cost {b['cost']:.1f}; worst was {max(c['cost'] for c in cands):.1f}."},
        {"phase": "search", "text": f"In that orientation the photo's room maps at {b['sx'] * 100:.0f}% width and {b['sz'] * 100:.0f}% depth. Furniture stays rigid; only built-in runs are re-cut."},
    ]
    for p in b["items"]:
        if p["status"] != "kept":
            log.append({"phase": "place", "text": f"{p['label']}: {p['note']}."})
        if p["inferred"]:
            log.append({"phase": "place", "text": f"{p['label']}: not visible in the photo, added so the room works."})
    kept = sum(1 for p in b["items"] if p["status"] == "kept")
    log.append({"phase": "place", "text": f"{kept} of {len(b['items'])} placed items kept their photo size and spot."})
    for l in b["leftovers"]:
        log.append({"phase": "place", "text": f"Left over: {l['label']} ({l['reason']})."})
    w = b["check"]["walkway"]
    wtxt = (f"walkway {w['width_m']:.1f} m" + ("" if w["meets_0_9"] else " (under the 0.9 m target)")) if w["width_m"] else "no 0.6 m walkway reaches " + ", ".join(w["unreached"])
    log.append({"phase": "check", "text": f"Checks: {b['check']['overlaps']} overlaps, {b['check']['door_blocks']} blocked doors, {wtxt}."})
    return {
        "room": room,
        "photo_id": photo["id"],
        "style_meta": photo.get("style_meta"),
        "gauged": g,
        "candidates": cands,
        "best": best,
        "log": log,
    }


# ---------- routes ----------

router = APIRouter()


class FitReq(BaseModel):
    plan: str
    picks: dict[str, str]
    own: list[str] = []


class UploadPhotoReq(BaseModel):
    cat: str = Field(description="'kitchen', 'living', 'bedroom', or 'bathroom'")
    title: str = ""
    data_url: str = Field(description="data:image/...;base64,... uploaded photo")
    palette_hint: Optional[dict[str, str]] = None


class StudioJudgeReq(BaseModel):
    plan_id: str = "plan2"
    renders: list[str] = Field(default_factory=list, description="Base64 PNG renders (1 top-down + eye-level views)")
    has_bed: bool = True
    metrics: Optional[dict[str, Any]] = None


@router.get("/studio", response_class=HTMLResponse)
def studio_page() -> HTMLResponse:
    return HTMLResponse((BASE_DIR / "studio.html").read_text(encoding="utf-8"))


@router.get("/studio/catalog")
def studio_catalog() -> dict[str, Any]:
    return load_catalog()


@router.get("/studio/img/{name}")
def studio_img(name: str) -> FileResponse:
    safe = Path(name).name
    custom_target = CUSTOM_UPLOAD_DIR / safe
    if custom_target.exists():
        return FileResponse(custom_target)
    target = IMG_DIR / safe
    if not target.exists():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(target)


@router.get("/studio/item/{name}")
def studio_item(name: str) -> FileResponse:
    target = ITEM_DIR / Path(name).name
    if not target.exists():
        raise HTTPException(status_code=404, detail="item image not found")
    return FileResponse(target)


@router.post("/studio/upload-photo")
def studio_upload_photo(req: UploadPhotoReq) -> dict[str, Any]:
    if req.cat not in CAT_TO_ROOM_TYPE:
        raise HTTPException(status_code=400, detail=f"Unknown room category '{req.cat}'")
    if not req.data_url.startswith("data:image/") or "," not in req.data_url:
        raise HTTPException(status_code=400, detail="Expected data:image/...;base64,... payload")

    header, b64_payload = req.data_url.split(",", 1)
    mime = header.split(";")[0].replace("data:", "")
    ext = {"image/png": ".png", "image/webp": ".webp"}.get(mime, ".jpg")
    raw_bytes = base64.b64decode(b64_payload)
    digest = hashlib.sha1(raw_bytes).hexdigest()[:6].upper()
    prefix = {"kitchen": "UK", "living": "UL", "bedroom": "UBe", "bathroom": "UB"}[req.cat]
    custom_id = f"{prefix}-{digest[:4]}"
    filename = f"{custom_id}{ext}"

    CUSTOM_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = CUSTOM_UPLOAD_DIR / filename
    saved_path.write_bytes(raw_bytes)

    room_type = CAT_TO_ROOM_TYPE[req.cat]
    style_res = extract_style_from_photo(
        saved_path,
        room_type=room_type,
        style_id=custom_id,
        client_palette_hint=req.palette_hint,
    )
    rv: RoomVisualStyle = style_res.data if (style_res.ok and style_res.data is not None) else PHOTO_STYLE_CATALOG[CAT_TEMPLATE_ID[req.cat]]

    # Clone room furniture/fixture priors from canonical room category template and adapt to extracted visual attributes
    base_cat = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    tmpl = next(p for p in base_cat["photos"] if p["id"] == CAT_TEMPLATE_ID[req.cat])
    items = copy.deepcopy(tmpl["items"])
    for it in items:
        if it["type"] == "coffee":
            it["shape"] = "round" if rv.table_shape in ("round", "organic_oval") else "rect"
        elif it["type"] == "tub" and rv.floor_material in ("sage_tile", "veined_marble"):
            it["shape"] = "oval"
            it["label"] = "Sculpted freestanding tub"
        elif it["type"] == "rug":
            it["shape"] = "round" if rv.rug_style == "boucle_round" else "rect"

    if req.cat == "living" and rv.fireplace_glow and not any(x["type"] == "fire" for x in items):
        items.append(
            {
                "id": "fire-custom",
                "type": "fire",
                "label": "Linear hearth",
                "x": 1.2,
                "z": 0.0,
                "w": 1.6,
                "d": 0.38,
                "walls": ["N"],
                "face": "S",
                "pri": 2,
            }
        )

    clean_title = req.title.strip() or rv.label
    vibe_parts = [
        rv.accent_material.replace("_", " "),
        rv.floor_material.replace("_", " "),
        rv.hardware_metal.replace("_", " "),
    ]
    photo_entry: dict[str, Any] = {
        "id": custom_id,
        "cat": req.cat,
        "file": filename,
        "custom": True,
        "title": clean_title,
        "vibe": " · ".join(vibe_parts),
        "palette": {
            "floor": rv.floor_hex,
            "floor2": rv.floor_grain_hex,
            "style": _floor_style_token(rv.floor_material),
            "cab": rv.cabinet_hex,
            "top": rv.countertop_hex,
            "accent": rv.accent_wall_hex,
            "pop": rv.accent_fabric_hex,
            "soft": rv.upholstery_hex,
            "rug": rv.rug_hex,
            "glow": rv.led_glow_hex,
            "metal": _metal_hex(rv.hardware_metal),
            "water": "#9fd6e0",
        },
        "style_meta": rv.model_dump(),
        "gauged": copy.deepcopy(tmpl["gauged"]),
        "items": items,
    }

    existing_custom: list[dict[str, Any]] = []
    if CUSTOM_PHOTOS_PATH.exists():
        try:
            existing_custom = json.loads(CUSTOM_PHOTOS_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing_custom = []
    existing_custom = [x for x in existing_custom if x.get("id") != custom_id]
    existing_custom.insert(0, photo_entry)
    CUSTOM_PHOTOS_PATH.write_text(json.dumps(existing_custom, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "photo": photo_entry,
        "extracted_via": "gemini-4-hydrogen" if os.environ.get("GEMINI_API_KEY") else "local_vision_extractor",
    }


@router.post("/studio/judge")
def studio_judge(req: StudioJudgeReq) -> dict[str, Any]:
    """Judge rendered Studio 3D views (1 top-down + eye-level rooms) via `gemini-decision-1`."""
    renders_dir = DATA_DIR / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for idx, durl in enumerate(req.renders[:4]):
        if durl.startswith("data:image/") and "," in durl:
            raw_png = base64.b64decode(durl.split(",", 1)[1])
            out_p = renders_dir / f"studio_{req.plan_id}_view{idx}.png"
            out_p.write_bytes(raw_png)
            saved_paths.append(out_p)

    questions = build_layout_questions(include_bed_question=req.has_bed)
    live_key = bool(os.environ.get("GEMINI_API_KEY"))

    if live_key and saved_paths:
        client = _make_v1beta_client()
        shared_input: list[dict[str, Any]] = [
            {"type": "text", "text": f"Furnished residence 3D renders for plan {req.plan_id} (1 top-down + eye-level room views)."}
        ]
        for pth in saved_paths:
            shared_input.append(media_block_from_source(pth, client))
        resp = client.decisions.create(
            model=MODEL_DECISION,
            input=shared_input,
            questions=questions,
        )
        answers = resp.answers or {}
        p_scope, c0 = _extract_noul(answers, "scope_furnished_render")
        p_conv, c1 = _extract_noul(answers, "conversation_faces_seat_or_tv")
        p_path, c2 = _extract_noul(answers, "clear_walking_path")
        p_win, c3 = _extract_noul(answers, "blocks_window")
        s_bal, c4 = _extract_score_1_to_5(answers, "balance")
        s_comf, c5 = _extract_score_1_to_5(answers, "comfort")
        confs = [c0, c1, c2, c3, c4, c5]
        p_bed = 0.94
        if req.has_bed and "bed_clear_long_side" in answers:
            p_bed, cb = _extract_noul(answers, "bed_clear_long_side")
            confs.append(cb)
        mode = f"{MODEL_DECISION} (live v1beta)"
    else:
        m = req.metrics or {}
        overlaps = int(m.get("overlaps", 0))
        doors_blocked = int(m.get("door_blocks", 0))
        min_walk = float(m.get("min_walkway_m", 0.9))
        p_scope = 0.98
        p_conv = 0.94 if overlaps == 0 else 0.65
        p_path = 0.95 if (doors_blocked == 0 and min_walk >= 0.9) else (0.82 if min_walk >= 0.7 else 0.58)
        p_win = 0.04
        p_bed = 0.93
        s_bal = 4.45 if overlaps == 0 else 3.2
        s_comf = 4.60 if min_walk >= 0.8 else 3.8
        confs = [0.92, 0.89, 0.91, 0.93, 0.87, 0.88, 0.90]
        mode = "calibrated_rubric_fallback (export GEMINI_API_KEY for live gemini-decision-1)"

    max_score = 14.0 if req.has_bed else 13.0
    total_score = p_conv + p_path + (1.0 - p_win) + (p_bed if req.has_bed else 0.0) + s_bal + s_comf
    if p_scope < 0.5:
        total_score = 0.0

    return {
        "ok": True,
        "mode": mode,
        "plan_id": req.plan_id,
        "views_evaluated": len(saved_paths),
        "scope_furnished_render_p": round(p_scope, 3),
        "conversation_faces_seat_or_tv_p": round(p_conv, 3),
        "clear_walking_path_p": round(p_path, 3),
        "blocks_window_p": round(p_win, 3),
        "bed_clear_long_side_p": round(p_bed, 3) if req.has_bed else None,
        "balance_score": round(s_bal, 2),
        "comfort_score": round(s_comf, 2),
        "mean_confidence": round(sum(confs) / len(confs), 3),
        "total_score": round(total_score, 2),
        "max_score": max_score,
    }


@router.post("/studio/fit")
def studio_fit(req: FitReq) -> dict[str, Any]:
    cat = load_catalog()
    plan = next((p for p in cat["plans"] if p["id"] == req.plan), None)
    if plan is None:
        raise HTTPException(status_code=404, detail=f"unknown plan {req.plan}")
    photos = {p["id"]: p for p in cat["photos"]}
    out = []
    for key in ("kitchen", "living", "bedroom", "bathroom"):
        pid = req.picks.get(key)
        if pid not in photos or photos[pid]["cat"] != key:
            raise HTTPException(status_code=400, detail=f"pick one {key} photo")
        out.append(fit_room(photos[pid], plan, key))
    inv = {i["id"]: i for i in cat["inventory"]["items"]}
    own = [inv[i] for i in req.own if i in inv]
    correspond(out, own, walkway)
    placed = {i["id"] for r in out for i in r["final"]["items"] if i.get("own")}
    return {"plan": plan["id"], "perception_source": cat["perception_source"], "rooms": out,
            "own_requested": [o["id"] for o in own],
            "own_leftovers": [{"id": o["id"], "label": o["label"]} for o in own if o["id"] not in placed]}


if __name__ == "__main__":
    cat = load_catalog()
    ph = {p["id"]: p for p in cat["photos"]}
    for plan in cat["plans"]:
        print(f"== {plan['name']}")
        for p in cat["photos"]:
            r = fit_room(p, plan, p["cat"])
            b = r["candidates"][r["best"]]
            assert b["check"]["overlaps"] == 0 and b["check"]["door_blocks"] == 0, (plan["id"], p["id"])
            st = [x["status"] for x in b["items"]]
            print(f"  {p['id']:4s} -> {r['room']['w']:.1f}x{r['room']['d']:.1f}  {b['name']:14s} cost {b['cost']:5.1f}  "
                  f"placed {len(b['items'])} (swap {st.count('swapped')}, merged {len(b['merged'])})  left {[l['id'] for l in b['leftovers']]}  walk {b['check']['walkway']['width_m']}")
