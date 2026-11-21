"""Reviewer agent for Homeweave Studio (`studio_review.py`).

A second, independent pass over the placer's output. The placer (`studio.py` +
`studio_correspond.py`) only knows geometry; the reviewer knows what a room is *for*.
It reads the finished layout, raises findings, and (when `fix=True`) corrects them:
moves a piece to another suitable room, or sends it back to the truck with a reason.

Rule-based and deterministic, so its verdicts are reproducible in front of judges.
With GEMINI_API_KEY set, `judge.py`'s Decisions-API rubric can score the same layouts.
"""

from __future__ import annotations

from typing import Any

from studio_correspond import ACCESSORY, EPS, TALL_TYPES, WALLS, WALL_NAME, WALL_TYPES, _along_wall, _spiral, overlap

FAMILY = {"sectional": "sofa"}
NAMES = {"tv": "TV", "plant": "plant", "fridge": "fridge", "sofa": "sofa", "bed": "bed", "coffee": "coffee table", "rug": "rug", "armchair": "armchair", "table": "table",
         "bookshelf": "bookshelf", "console": "console"}
# How many of each furniture family a room type can sensibly hold (tunable house rules).
CAPS: dict[str, dict[str, int]] = {
    "bedroom": {"sofa": 1, "bed": 1, "coffee": 1, "rug": 1, "armchair": 2, "table": 0, "bookshelf": 1, "console": 1, "tv": 1, "plant": 3, "fridge": 0},
    "living": {"sofa": 2, "bed": 0, "coffee": 1, "rug": 1, "armchair": 2, "table": 1, "bookshelf": 1, "console": 2, "tv": 1, "plant": 4, "fridge": 0},
    "kitchen": {"sofa": 0, "bed": 0, "coffee": 0, "rug": 1, "armchair": 0, "table": 1, "fridge": 1, "plant": 2},
    "bathroom": {"sofa": 0, "bed": 0, "coffee": 0, "armchair": 0, "table": 0},
}
BEDROOM_SOFA_MIN_AREA = 12.0   # m2: below this a bedroom gets no sofa at all
WALK_OK = 0.8                  # reviewer accepts 0.8 m; the 0.9 m target is still reported
ESSENTIAL = {"bed", "counter", "island", "tub", "tubshower", "shower", "vanity", "toilet", "fire", "tall", "seat", "closet", "washer"}


def fam(it: dict[str, Any]) -> str:
    return FAMILY.get(it["type"], it["type"])


def cap_for(key: str, room: dict[str, Any], family: str) -> int | None:
    rtype = room.get("type", key)
    cap = CAPS.get(rtype, {}).get(family)
    if rtype == "bedroom" and family == "sofa" and room["w"] * room["d"] < BEDROOM_SOFA_MIN_AREA:
        return 0
    return cap


def compute_check(room: dict[str, Any], items: list[dict[str, Any]], walkway) -> dict[str, Any]:
    sol = [i for i in items if not i["floor"]]
    # the plan's own fixtures are ground truth: never flag them against each other, a door, or the room edge
    pairs = [(sol[a], sol[b]) for a in range(len(sol)) for b in range(a + 1, len(sol))
             if not (sol[a].get("plan") and sol[b].get("plan")) and overlap(sol[a]["rect"], sol[b]["rect"])]
    loose = [s for s in sol if not s.get("plan")]
    doors = [s for s in loose if any(overlap(s["rect"], dz) for dz in room["doors"])]
    outside = [s for s in loose if s["rect"][0] < -EPS or s["rect"][1] < -EPS or s["rect"][0] + s["rect"][2] > room["w"] + EPS
               or s["rect"][1] + s["rect"][3] > room["d"] + EPS or any(overlap(s["rect"], o) for o in room["obstacles"])]
    return {"overlaps": len(pairs), "door_blocks": len(doors), "outside_room": len(outside), "walkway": walkway(room, sol),
            "overlap_pairs": [[a["label"], b["label"]] for a, b in pairs], "door_items": [s["label"] for s in doors],
            "outside_items": [s["label"] for s in outside],
            "bad_ids": sorted({x["id"] for a, b in pairs for x in (a, b)} | {s["id"] for s in doors} | {s["id"] for s in outside})}


def walk_score(w: dict[str, Any]) -> float:
    return w["width_m"] if w["width_m"] else -0.1 * len(w["unreached"])


def inspect(key: str, room: dict[str, Any], items: list[dict[str, Any]], walkway) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Flag problems without changing anything (used for live edits)."""
    chk = compute_check(room, items, walkway)
    out: list[dict[str, Any]] = []
    for a, b in chk["overlap_pairs"]:
        out.append({"rule": "overlap", "level": "error", "text": f"{a} overlaps {b}."})
    for n in chk["door_items"]:
        out.append({"rule": "door", "level": "error", "text": f"{n} blocks a door swing."})
    for n in chk["outside_items"]:
        out.append({"rule": "bounds", "level": "error", "text": f"{n} is outside the room or inside a closet."})
    counts: dict[str, list[str]] = {}
    for i in items:
        if i["type"] not in ACCESSORY:
            counts.setdefault(fam(i), []).append(i["label"])
    for f, labels in counts.items():
        cap = cap_for(key, room, f)
        if cap is not None and len(labels) > cap:
            out.append({"rule": "count", "level": "error",
                        "text": f"{len(labels)} {NAMES.get(f, f)}s in the {room['label'].lower()} ({', '.join(labels)}); this room takes at most {cap}."})
    w = chk["walkway"]
    if not w["width_m"]:
        out.append({"rule": "walkway", "level": "error", "text": "No 0.6 m walkway reaches every piece from the door."})
    elif w["width_m"] < WALK_OK:
        out.append({"rule": "walkway", "level": "warn", "text": f"Walkway is {w['width_m']:.1f} m; the reviewer wants {WALK_OK:.1f} m and the target is 0.9 m."})
    return chk, out


def _relocate(o: dict[str, Any], it: dict[str, Any], from_key: str, by_key: dict[str, Any], walkway) -> str | None:
    """Try the piece's other preferred rooms. Returns the room key it moved to, or None."""
    if it["type"] in ACCESSORY:
        return None
    for key in [k for t in o["rooms"] for k in by_key if by_key[k]["room"]["type"] == t]:
        if key == from_key:
            continue
        room, F = by_key[key]["room"], by_key[key]["final"]
        cap = cap_for(key, room, fam(it))
        if cap is not None and sum(1 for i in F["items"] if fam(i) == fam(it)) + 1 > cap:
            continue
        solids = [i["rect"] for i in F["items"] if not i["floor"]]
        rect, face, note = None, it["face"], ""
        if it["floor"]:
            bw, bd = min(o["w"], room["w"] - 0.3), min(o["d"], room["d"] - 0.3)
            rect = [round((room["w"] - bw) / 2, 3), round((room["d"] - bd) / 2, 3), round(bw, 3), round(bd, 3)]
        elif o["type"] in WALL_TYPES:
            for wl in sorted((w for w in WALLS if w not in room["open"]), key=lambda w: -(room["w"] if w in "NS" else room["d"])):
                rect = _along_wall(room, wl, o["w"], o["d"], solids, o["type"] in TALL_TYPES)
                if rect:
                    face, note = WALLS[(WALLS.index(wl) + 2) % 4], f"against the {WALL_NAME[wl]} wall"
                    break
        else:
            res = _spiral(room, room["w"] / 2 - o["w"] / 2, room["d"] / 2 - o["d"] / 2, o["w"], o["d"], solids, False, 2.5)
            rect = res[0] if res else None
            note = "in open floor"
        if rect is None:
            continue
        moved = {**it, "rect": rect, "face": face, "len": o["w"], "dep": o["d"], "moved_m": 0.0}
        moved.pop("match", None)
        before = walk_score(F["check"]["walkway"])
        after = walkway(room, [i for i in F["items"] + [moved] if not i["floor"]])
        if walk_score(after) < min(before, WALK_OK):
            continue
        moved["note"] = f"moved here by the reviewer, {note}".rstrip(", ")
        F["items"].append(moved)
        F["check"] = compute_check(room, F["items"], walkway)
        return key
    return None


def review(rooms: list[dict[str, Any]], own: list[dict[str, Any]], walkway, fix: bool = True) -> list[dict[str, Any]]:
    """Attach `review` to every room; with `fix`, correct what it finds. Returns pieces sent back to the truck."""
    by_key = {r["room"]["key"]: r for r in rooms}
    own_by_id = {o["id"]: o for o in own}
    truck: list[dict[str, Any]] = []
    notes: dict[str, list[dict[str, Any]]] = {k: [] for k in by_key}

    def remove(key: str, it: dict[str, Any], why: str, rule: str) -> None:
        r = by_key[key]
        F, room = r["final"], r["room"]
        riders = [i for i in F["items"] if i.get("on") == it["id"]]
        F["items"] = [i for i in F["items"] if i is not it and i.get("on") != it["id"]]
        F["check"] = compute_check(room, F["items"], walkway)
        for rd in riders:
            if rd.get("own"):
                truck.append({"id": rd["id"], "label": rd["label"], "pri": 3, "own": True, "reason": f"reviewer: it sat on the {it['label'].lower()}, which was moved out"})
        if not it.get("own"):
            notes[key].append({"rule": rule, "level": "fixed", "text": f"{why} Removed the photo's {it['label'].lower()}."})
            return
        dest = _relocate(own_by_id[it["id"]], it, key, by_key, walkway)
        if dest:
            notes[key].append({"rule": rule, "level": "fixed", "text": f"{why} Moved your {it['label'].lower()} to the {by_key[dest]['room']['label'].lower()}."})
            notes[dest].append({"rule": rule, "level": "info", "text": f"Took your {it['label'].lower()} from the {room['label'].lower()}."})
        else:
            truck.append({"id": it["id"], "label": it["label"], "pri": 1, "own": True, "reason": f"reviewer: {why.rstrip('.').lower()}, and no other room has space"})
            notes[key].append({"rule": rule, "level": "fixed", "text": f"{why} Your {it['label'].lower()} goes back on the truck; no other room has space."})

    if fix:
        # Rule 1: a room only takes so many of each kind (for example one sofa in a bedroom).
        for key, r in by_key.items():
            room = r["room"]
            fams = sorted({fam(i) for i in r["final"]["items"]})
            for f in fams:
                cap = cap_for(key, room, f)
                group = [i for i in r["final"]["items"] if fam(i) == f and i["type"] not in ACCESSORY]
                if cap is None or len(group) <= cap:
                    continue
                # keep the user's best-matched pieces; photo pieces go first, then unmatched, then worst match
                group = [i for i in group if not i.get("plan")]   # a plan fixture is never the one to go
                over = len([i for i in r["final"]["items"] if fam(i) == f and i["type"] not in ACCESSORY]) - cap
                group.sort(key=lambda i: (0 if not i.get("own") else 1 if "match" not in i else 2, -(i.get("match") or {}).get("cost", 0)))
                group = group[:max(0, over)] + [None] * 0
                for it in group:
                    remove(key, it, f"{over + cap} {NAMES.get(f, f)}s in the {room['label'].lower()} is too many (at most {cap}).", "count")
        # Rule 2: every piece must be reachable along a walkway; clear the least important pieces until it is.
        for key, r in by_key.items():
            room = r["room"]
            for _ in range(8):
                F = r["final"]
                cur = F["check"]["walkway"]
                if cur["width_m"] and cur["width_m"] >= WALK_OK:
                    break
                def tier(i: dict[str, Any]) -> int:
                    if not i.get("own"):
                        return 0 if i["pri"] >= 3 else 2
                    if "match" not in i:
                        return 1
                    return 4 if fam(i) == "sofa" else 3

                cands = [i for i in F["items"] if not i["floor"] and i["type"] not in ESSENTIAL]
                best, gain = None, 0.0
                for t in range(5):  # least important tier first; only reach for a main sofa as a last resort
                    for it in sorted((c for c in cands if tier(c) == t), key=lambda i: i["rect"][2] * i["rect"][3]):
                        trial = walkway(room, [i for i in F["items"] if i is not it and not i["floor"]])
                        g = walk_score(trial) - walk_score(cur)
                        if g > gain + 1e-6:
                            best, gain = it, g
                    if best is not None:
                        break
                if best is None:
                    break
                was = f"{cur['width_m']:.1f} m" if cur["width_m"] else "blocked"
                remove(key, best, f"The walkway in the {room['label'].lower()} was {was}.", "walkway")

    for key, r in by_key.items():
        room, F = r["room"], r["final"]
        F["check"], open_issues = inspect(key, room, F["items"], walkway)
        fixed = [n for n in notes[key] if n["level"] == "fixed"]
        errors = [n for n in open_issues if n["level"] == "error"]
        verdict = "needs attention" if errors else ("corrected" if fixed else "approved")
        r["review"] = {"verdict": verdict, "findings": notes[key] + open_issues, "fixes": len(fixed)}
        first = {}
        for k2, r2 in by_key.items():
            first.setdefault(r2["room"]["type"], k2)
        F["leftovers"] = F["leftovers"] + [t for t in truck if first.get(own_by_id[t["id"]]["rooms"][0]) == key]
    return truck
