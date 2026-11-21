"""Correspondence for Homeweave Studio (`studio_correspond.py`).

Moves furniture the user already owns (`data/studio_inventory.json`, read from their videos)
into the new home. Follows the spec's "new home starts empty" adaptation: the furniture seen in
the chosen listing photo is the *staging prior*; each owned item is matched to the most analogous
staging item and takes its place as a rigid body, then hard constraints are re-checked.

Outputs per room: final item list, the correspondence pairs, leftovers (with reasons) and gaps
(staging spots nothing of the user's fills).
"""

from __future__ import annotations

import math
from typing import Any

EPS = 0.02
STEP = 0.1
WALLS = "NESW"
WALL_NAME = {"N": "north", "E": "east", "S": "south", "W": "west"}

# Staging items an owned item may replace, with a base mismatch cost (tunable first guesses).
COMPAT: dict[str, dict[str, float]] = {
    "sofa": {"sofa": 0.0, "sectional": 0.1},
    "armchair": {"armchair": 0.0, "pouf": 0.5},
    "pouf": {"pouf": 0.0, "armchair": 0.6},
    "sidetable": {"nightstand": 0.1, "coffee": 0.6, "pouf": 0.7},
    "coffee": {"coffee": 0.0},
    "rug": {"rug": 0.0},
    "bed": {"bed": 0.0},
    "bookshelf": {"tall": 0.2},
    "tv": {"tv": 0.0},
    "plant": {"plant": 0.0},
}
# Small things that sit on or against something else instead of taking floor space.
ACCESSORY = {"art", "vase", "pillow"}
HOSTS = {"pillow": ("sofa", "sectional", "bed", "armchair"),
         "vase": ("console", "coffee", "sidetable", "table", "nightstand", "counter", "island", "vanity")}
TOP_M = {"sofa": 0.5, "sectional": 0.5, "bed": 0.58, "armchair": 0.5, "console": 0.85, "coffee": 0.4, "sidetable": 0.5, "table": 0.76,
         "nightstand": 0.52, "counter": 0.9, "island": 0.91, "vanity": 0.89}
# Staging types that are furniture (replaceable). Everything else is a fixture of the new home.
MOVABLE = {"living": {"sofa", "sectional", "coffee", "armchair", "pouf", "rug", "plant", "tv"},
           "bedroom": {"bed", "nightstand", "armchair", "pouf", "rug", "plant", "tall", "sofa", "coffee", "tv"},
           "kitchen": {"rug"}, "bathroom": set()}
WALL_TYPES = {"bookshelf", "console", "table", "bench", "sofa", "bed", "tv", "fridge", "plant"}
TALL_TYPES = {"bookshelf", "fridge"}
MATCH_LIMIT = 1.0


def overlap(a: list[float], b: list[float], eps: float = EPS) -> bool:
    return a[0] < b[0] + b[2] - eps and b[0] < a[0] + a[2] - eps and a[1] < b[1] + b[3] - eps and b[1] < a[1] + a[3] - eps


def _valid(room: dict[str, Any], rect: list[float], solids: list[list[float]], tall: bool) -> bool:
    W, D = room["w"], room["d"]
    if rect[0] < -EPS or rect[1] < -EPS or rect[0] + rect[2] > W + EPS or rect[1] + rect[3] > D + EPS:
        return False
    if any(overlap(rect, b) for b in room["obstacles"] + room["doors"] + solids):
        return False
    if tall:
        for wn in room["windows"]:
            if wn["wall"] == "N" and rect[1] < 0.05 or wn["wall"] == "S" and rect[1] + rect[3] > D - 0.05:
                lo, hi = rect[0], rect[0] + rect[2]
            elif wn["wall"] == "W" and rect[0] < 0.05 or wn["wall"] == "E" and rect[0] + rect[2] > W - 0.05:
                lo, hi = rect[1], rect[1] + rect[3]
            else:
                continue
            if min(hi, wn["b"]) - max(lo, wn["a"]) > 0.1:
                return False
    return True


def _spiral(room, x, z, bw, bd, solids, tall, lim):
    n = int(lim / STEP)
    offs = sorted(((i * STEP, j * STEP) for i in range(-n, n + 1) for j in range(-n, n + 1)), key=lambda o: math.hypot(*o))
    for ox, oz in offs:
        r = [round(min(max(x + ox, 0.0), room["w"] - bw), 3), round(min(max(z + oz, 0.0), room["d"] - bd), 3), round(bw, 3), round(bd, 3)]
        if _valid(room, r, solids, tall):
            return r, math.hypot(ox, oz)
    return None


def _along_wall(room, wall, length, dep, solids, tall, start=None):
    W, D = room["w"], room["d"]
    horiz = wall in "NS"
    bw, bd = (length, dep) if horiz else (dep, length)
    L = W if horiz else D
    if length > L + EPS:
        return None
    base = (L - length) / 2 if start is None else start
    n = int(L / STEP) + 1
    for t in [0.0] + [s * i * STEP for i in range(1, n) for s in (1, -1)]:
        a = base + t
        if a < -EPS or a + length > L + EPS:
            continue
        x = a if horiz else (0.0 if wall == "W" else W - bw)
        z = (0.0 if wall == "N" else D - bd) if horiz else a
        r = [round(x, 3), round(z, 3), round(bw, 3), round(bd, 3)]
        if _valid(room, r, solids, tall):
            return r
    return None


def _own_rec(o: dict[str, Any]) -> dict[str, Any]:
    return {"id": o["id"], "type": o["type"], "label": o["label"], "pri": 1, "floor": o["type"] == "rug" or o["type"] in ACCESSORY, "face": "S",
            "flip": False, "shape": o.get("shape"), "feat": None, "n": None, "inferred": False, "status": "own",
            "note": "", "moved_m": 0.0, "own": True, "color": o["color"], "color2": o.get("color2", o["color"]),
            "h": o.get("h"), "len": o["w"], "dep": o["d"], "source": o["source"], "conf": o["conf"]}


def correspond(rooms: list[dict[str, Any]], own: list[dict[str, Any]], walkway) -> None:
    """Attach `final` to every fitted room. `rooms` come from `studio.fit_room`."""
    by_key = {r["room"]["key"]: r for r in rooms}

    def keys_for(o: dict[str, Any]) -> list[str]:
        """Room ids this piece may go to, in preference order (a 2-bedroom plan offers both bedrooms)."""
        return [k for t in o["rooms"] for k in by_key if by_key[k]["room"]["type"] == t]

    accessories = [o for o in own if o["type"] in ACCESSORY]
    own = [o for o in own if o["type"] not in ACCESSORY]
    state: dict[str, dict[str, Any]] = {}
    for key, r in by_key.items():
        best = r["candidates"][r["best"]]
        mov = MOVABLE.get(r["room"]["type"], set())
        state[key] = {"fixed": [dict(i) for i in best["items"] if i["type"] not in mov],
                      "staging": [dict(i) for i in best["items"] if i["type"] in mov],
                      "own": [], "pairs": [], "dropped": [], "photo_leftovers": best["leftovers"]}
    leftovers: list[dict[str, Any]] = []

    # ---- 1. match owned items to staging items (greedy over all compatible pairs, cheapest first) ----
    cands = []
    for o in own:
        for rank, key in enumerate(keys_for(o)):
            for s in state[key]["staging"]:
                base = COMPAT.get(o["type"], {}).get(s["type"])
                if base is None:
                    continue
                cost = base + 0.15 * abs(o["w"] - s["len"]) + 0.3 * min(rank, 2)
                if cost < MATCH_LIMIT:
                    cands.append((cost, o["id"], key, s["id"]))
    cands.sort()
    matched: dict[str, tuple[str, str, float]] = {}
    taken: set[tuple[str, str]] = set()
    for cost, oid, key, sid in cands:
        if oid in matched or (key, sid) in taken:
            continue
        matched[oid] = (key, sid, cost)
        taken.add((key, sid))

    def solid_rects(key: str) -> list[list[float]]:
        st = state[key]
        return [i["rect"] for i in st["fixed"] if not i["floor"]] + [i["rect"] for i in st["own"] if not i["floor"]]

    # ---- 2. matched items take the staging item's pose, rigidly, with their own size ----
    for o in sorted(own, key=lambda o: -(o["w"] * o["d"])):
        if o["id"] not in matched:
            continue
        key, sid, cost = matched[o["id"]]
        st, room = state[key], by_key[key]["room"]
        s = next(i for i in st["staging"] if i["id"] == sid)
        rec = _own_rec(o)
        rec["face"] = s["face"]
        bw, bd = (o["w"], o["d"]) if s["face"] in "NS" else (o["d"], o["w"])
        sx, sz, sw, sd = s["rect"]
        if rec["floor"]:
            bw, bd = min(bw, room["w"] - 0.3), min(bd, room["d"] - 0.3)
            x = min(max(sx + sw / 2 - bw / 2, 0.15), room["w"] - bw - 0.15)
            z = min(max(sz + sd / 2 - bd / 2, 0.15), room["d"] - bd - 0.15)
            res = ([round(x, 3), round(z, 3), round(bw, 3), round(bd, 3)], 0.0)
        else:
            x, z = sx + sw / 2 - bw / 2, sz + sd / 2 - bd / 2
            if sx < 0.05: x = 0.0
            if sz < 0.05: z = 0.0
            if sx + sw > room["w"] - 0.05: x = room["w"] - bw
            if sz + sd > room["d"] - 0.05: z = room["d"] - bd
            res = _spiral(room, x, z, bw, bd, solid_rects(key), o["type"] in TALL_TYPES, 1.5) if bw <= room["w"] and bd <= room["d"] else None
        if res is None:
            matched.pop(o["id"])
            continue
        rec["rect"], rec["moved_m"] = res[0], round(res[1], 2)
        rec["match"] = {"id": s["id"], "label": s["label"], "cost": round(cost, 2)}
        rec["note"] = f"takes the spot of the photo's {s['label'].lower()}" + (f", nudged {res[1]:.1f} m to clear" if res[1] > 0.05 else "")
        st["staging"].remove(s)
        st["own"].append(rec)
        st["pairs"].append({"own": o["id"], "own_label": o["label"], "staging": s["label"], "cost": round(cost, 2),
                            "size_change_m": round(o["w"] - s["len"], 2), "moved_m": rec["moved_m"]})

    # ---- 3. unmatched items: nothing analogous in the photo, so find them a legal spot ----
    for o in sorted(own, key=lambda o: -(o["w"] * o["d"])):
        if o["id"] in matched:
            continue
        done, why = False, ""
        for key in keys_for(o):
            st, room = state[key], by_key[key]["room"]
            rec = _own_rec(o)
            tall = o["type"] in TALL_TYPES
            if rec["floor"]:
                bw, bd = min(o["w"], room["w"] - 0.3), min(o["d"], room["d"] - 0.3)
                rec["rect"] = [round((room["w"] - bw) / 2, 3), round((room["d"] - bd) / 2, 3), round(bw, 3), round(bd, 3)]
                rec["note"] = "no rug in the photo; centred in the room"
            elif o["type"] in WALL_TYPES:
                rect = None
                for wl in sorted((w for w in WALLS if w not in room["open"]), key=lambda w: -(room["w"] if w in "NS" else room["d"])):
                    rect = _along_wall(room, wl, o["w"], o["d"], solid_rects(key), tall)
                    if rect:
                        rec["face"] = WALLS[(WALLS.index(wl) + 2) % 4]
                        rec["note"] = f"nothing like it in the photo; stood against the {WALL_NAME[wl]} wall"
                        break
                if rect is None:
                    why = f"no free wall run of {o['w']:.1f} m in the {room['label'].lower()}"
                    continue
                rec["rect"] = rect
            else:
                res = _spiral(room, room["w"] / 2 - o["w"] / 2, room["d"] / 2 - o["d"] / 2, o["w"], o["d"], solid_rects(key), False, 2.5)
                if res is None:
                    why = f"no free floor spot in the {room['label'].lower()}"
                    continue
                rec["rect"] = res[0]
                rec["note"] = "nothing like it in the photo; placed in open floor"
            st["own"].append(rec)
            done = True
            break
        if not done:
            leftovers.append({"id": o["id"], "label": o["label"], "pri": 1, "own": True,
                              "reason": why or "no room of the right type on this plan"})

    # ---- 4. staging the user did not replace stays as a gap, unless one of their pieces now sits there ----
    for key, st in state.items():
        own_solid = [i["rect"] for i in st["own"] if not i["floor"]]
        keep = []
        for s in st["staging"]:
            if not s["floor"] and any(overlap(s["rect"], r) for r in own_solid):
                st["dropped"].append(s["label"])
            else:
                s["gap"] = bool(own)
                keep.append(s)
        st["staging"] = keep

    # ---- 5. accessories: pillows go on a sofa or bed, vases on a surface, pictures on a free wall ----
    load: dict[str, int] = {}
    art_n: dict[str, int] = {}
    for o in sorted(accessories, key=lambda o: -o["conf"]):
        done = False
        for key in keys_for(o):
            st, room = state[key], by_key[key]["room"]
            rec = _own_rec(o)
            if o["type"] == "art":
                sides = [w for w in WALLS if w not in room["open"] and not any(wn["wall"] == w for wn in room["windows"])]
                if not sides:
                    continue
                wl = max(sides, key=lambda w: (room["w"] if w in "NS" else room["d"]))
                L = room["w"] if wl in "NS" else room["d"]
                n = art_n.get(key + wl, 0)
                if n >= max(1, int(L // 1.1)):
                    continue
                art_n[key + wl] = n + 1
                w = min(o["w"], L - 0.4)
                a = min(max(L * (n + 1) / (max(1, int(L // 1.1)) + 1) - w / 2, 0.2), L - w - 0.2)
                rec["rect"] = [round(a, 3), 0.0 if wl == "N" else round(room["d"] - 0.05, 3), round(w, 3), 0.05] if wl in "NS" else \
                              [0.0 if wl == "W" else round(room["w"] - 0.05, 3), round(a, 3), 0.05, round(w, 3)]
                rec.update(face=WALLS[(WALLS.index(wl) + 2) % 4], wall=wl, at_h=1.55, note=f"hung on the {WALL_NAME[wl]} wall")
            else:
                pool = [i for i in st["own"] + st["staging"] + st["fixed"] if i["type"] in HOSTS[o["type"]] and not i["floor"]]
                pool = [i for i in pool if load.get(i["id"], 0) < (3 if o["type"] == "pillow" else 1)]
                if not pool:
                    continue
                host = min(pool, key=lambda i: (0 if i.get("own") else 1, load.get(i["id"], 0)))
                k = load.get(host["id"], 0)
                load[host["id"]] = k + 1
                hx, hz, hw, hd = host["rect"]
                s_ = min(o["w"], 0.4)
                along = (k - 1) * 0.5 if o["type"] == "pillow" else 0.0
                cx = hx + hw / 2 + (along if hw >= hd else 0.0)
                cz = hz + hd / 2 + (along if hd > hw else 0.0)
                cx, cz = min(max(cx, hx + s_ / 2), hx + hw - s_ / 2), min(max(cz, hz + s_ / 2), hz + hd - s_ / 2)
                rec["rect"] = [round(cx - s_ / 2, 3), round(cz - s_ / 2, 3), round(s_, 3), round(s_, 3)]
                rec.update(on=host["id"], at_h=host.get("h") if host.get("own") and host.get("h") and host["type"] not in ("sofa", "bed", "armchair") else TOP_M.get(host["type"], 0.5),
                           note=f"set on {'your' if host.get('own') else 'the'} {host['label'].lower()}")
            st["own"].append(rec)
            done = True
            break
        if not done:
            leftovers.append({"id": o["id"], "label": o["label"], "pri": 3, "own": True,
                              "reason": {"pillow": "no sofa, bed or armchair left to put it on", "vase": "no free table or counter top for it", "art": "no free wall without a window"}[o["type"]]})
    own = own + accessories

    for key, st in state.items():
        room = by_key[key]["room"]
        keep = st["staging"]
        items = st["fixed"] + keep + st["own"]
        sol = [i for i in items if not i["floor"]]
        n_ov = sum(1 for a in range(len(sol)) for b in range(a + 1, len(sol)) if overlap(sol[a]["rect"], sol[b]["rect"]))
        door = sum(1 for s in sol for dz in room["doors"] if overlap(s["rect"], dz))
        outside = sum(1 for s in sol if s["rect"][0] < -EPS or s["rect"][1] < -EPS or s["rect"][0] + s["rect"][2] > room["w"] + EPS or s["rect"][1] + s["rect"][3] > room["d"] + EPS)
        log = [{"phase": "match", "text": f"Your {p['own_label'].lower()} ↔ the photo's {p['staging'].lower()} (mismatch {p['cost']:.2f}, {p['size_change_m']:+.1f} m longer)."} for p in st["pairs"]]
        log += [{"phase": "match", "text": f"Your {i['label'].lower()}: {i['note']}."} for i in st["own"] if "match" not in i]
        log += [{"phase": "match", "text": f"Dropped the photo's {d.lower()} to make room for your furniture."} for d in st["dropped"]]
        by_key[key]["final"] = {
            "items": items, "pairs": st["pairs"], "dropped": st["dropped"],
            "gaps": [s["label"] for s in keep if s.get("gap") and not s["floor"]],
            "leftovers": st["photo_leftovers"] + [l for l in leftovers if key in keys_for(next(o for o in own if o["id"] == l["id"]))[:1]],
            "check": {"overlaps": n_ov, "door_blocks": door, "outside_room": outside, "walkway": walkway(room, sol)},
            "log": log,
        }
