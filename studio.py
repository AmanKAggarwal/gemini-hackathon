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

import json
import math
import os
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from studio_correspond import correspond
from studio_review import inspect, review

BASE_DIR = Path(__file__).parent
CATALOG_PATH = BASE_DIR / "data" / "studio_catalog.json"
IMG_DIR = BASE_DIR / "data" / "drive_inputs"
INVENTORY_PATH = BASE_DIR / "data" / "studio_inventory.json"
ITEM_DIR = BASE_DIR / "data" / "studio_items"

WALLS = "NESW"
EPS = 0.02
STEP = 0.1  # slide step, metres
# Tunable first guesses (spec: "ours are first guesses to tune").
COST = {"left1": 10.0, "left2": 5.0, "left3": 1.5, "swap": 3.0, "merge": 2.5, "move": 1.0,
        "open_edge": 6.0, "aspect": 2.0, "walkway": 4.0}
WALK_WIDTHS = [0.9, 0.8, 0.7, 0.6]  # spec hard constraint is 0.9 m; narrower widths are reported, not hidden


def load_catalog() -> dict[str, Any]:
    cat = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
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
    return {"room": room, "photo_id": photo["id"], "gauged": g, "candidates": cands, "best": best, "log": log}


# ---------- routes ----------

router = APIRouter()


class FitReq(BaseModel):
    plan: str
    picks: dict[str, str]
    own: list[str] = []


@router.get("/studio", response_class=HTMLResponse)
def studio_page() -> HTMLResponse:
    return HTMLResponse((BASE_DIR / "studio.html").read_text(encoding="utf-8"))


@router.get("/studio/catalog")
def studio_catalog() -> dict[str, Any]:
    return load_catalog()


@router.get("/studio/img/{name}")
def studio_img(name: str) -> FileResponse:
    target = IMG_DIR / Path(name).name
    if not target.exists():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(target)


@router.get("/studio/item/{name}")
def studio_item(name: str) -> FileResponse:
    target = ITEM_DIR / Path(name).name
    if not target.exists():
        raise HTTPException(status_code=404, detail="item image not found")
    return FileResponse(target)


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
    review(out, own, walkway, fix=True)
    placed = {i["id"] for r in out for i in r["final"]["items"] if i.get("own")}
    return {"plan": plan["id"], "perception_source": cat["perception_source"], "rooms": out,
            "own_requested": [o["id"] for o in own],
            "own_leftovers": [{"id": o["id"], "label": o["label"]} for o in own if o["id"] not in placed]}


@router.get("/studio/rank/{plan_id}")
def studio_rank(plan_id: str) -> dict[str, Any]:
    """The agent's pick: fit every photo into this plan's room of the same type and rank by fit cost."""
    cat = load_catalog()
    plan = next((p for p in cat["plans"] if p["id"] == plan_id), None)
    if plan is None:
        raise HTTPException(status_code=404, detail=f"unknown plan {plan_id}")
    out: dict[str, list[dict[str, Any]]] = {}
    for ph in cat["photos"]:
        if ph["cat"] not in plan["rooms"]:
            continue
        r = fit_room(ph, plan, ph["cat"])
        b = r["candidates"][r["best"]]
        kept = sum(1 for i in b["items"] if i["status"] == "kept")
        why = []
        if b["leftovers"]:
            why.append(f"{len(b['leftovers'])} piece(s) do not fit")
        changed = len(b["items"]) - kept
        if changed:
            why.append(f"{changed} resized or moved")
        w = b["check"]["walkway"]
        why.append(f"walkway {w['width_m']:.1f} m" if w["width_m"] else "walkway blocked")
        why.append(f"room maps at {b['sx'] * 100:.0f}% × {b['sz'] * 100:.0f}%")
        out.setdefault(ph["cat"], []).append({
            "id": ph["id"], "cost": b["cost"], "score": round(100 * math.exp(-b["cost"] / 12)), "orientation": b["name"],
            "kept": kept, "placed": len(b["items"]), "leftovers": len(b["leftovers"]), "why": ", ".join(why)})
    for v in out.values():
        v.sort(key=lambda x: x["cost"])
    return {"plan": plan_id, "rank": out}


def find_blender() -> str | None:
    """BLENDER_EXE, then PATH, then a portable copy under the user's home folder."""
    cands = [os.environ.get("BLENDER_EXE"), shutil.which("blender")]
    cands += [str(p) for p in sorted((Path.home() / "blender-portable").glob("*/blender.exe"), reverse=True)]
    cands += [str(p) for p in sorted(Path("C:/Program Files/Blender Foundation").glob("*/blender.exe"), reverse=True)]
    return next((c for c in cands if c and Path(c).exists()), None)


@router.get("/studio/blender")
def studio_blender_status() -> dict[str, Any]:
    exe = find_blender()
    return {"available": exe is not None, "exe": Path(exe).parent.name if exe else None}


@router.post("/studio/blender")
async def studio_blender(request: Request) -> FileResponse:
    """Body: the page's scene as binary glTF. Header X-Camera: JSON camera + render settings. Returns a PNG."""
    exe = find_blender()
    if exe is None:
        raise HTTPException(status_code=503, detail="Blender not found. Set BLENDER_EXE or install Blender.")
    out_dir = BASE_DIR / "data" / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")
    glb, png, cam = out_dir / f"scene_{stamp}.glb", out_dir / f"blender_{stamp}.png", out_dir / f"camera_{stamp}.json"
    glb.write_bytes(await request.body())
    cam.write_text(request.headers.get("x-camera", "{}"), encoding="utf-8")
    t0 = time.time()
    proc = subprocess.run([exe, "-b", "--factory-startup", "-P", str(BASE_DIR / "blender_render.py"), "--", str(glb), str(png), str(cam)],
                          capture_output=True, text=True, timeout=600)
    if not png.exists():
        raise HTTPException(status_code=500, detail=("Blender failed: " + (proc.stderr or proc.stdout)[-600:]))
    return FileResponse(png, media_type="image/png", headers={"X-Render-Seconds": f"{time.time() - t0:.1f}"})


class CheckReq(BaseModel):
    plan: str
    rooms: dict[str, list[dict[str, Any]]]


@router.post("/studio/check")
def studio_check(req: CheckReq) -> dict[str, Any]:
    """Live edit: the page sends the items as the user dragged them; the reviewer re-inspects, changing nothing."""
    cat = load_catalog()
    plan = next((p for p in cat["plans"] if p["id"] == req.plan), None)
    if plan is None:
        raise HTTPException(status_code=404, detail=f"unknown plan {req.plan}")
    out = {}
    for key, items in req.rooms.items():
        if key not in plan["rooms"]:
            continue
        chk, findings = inspect(key, target_room(plan, key), items, walkway)
        errors = [f for f in findings if f["level"] == "error"]
        out[key] = {"check": chk, "review": {"verdict": "needs attention" if errors else "approved", "findings": findings, "fixes": 0}}
    return out


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
