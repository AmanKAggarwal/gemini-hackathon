"""Natural-language layout edits for Homeweave Studio (`studio_prompt.py`).

"Put the TV in the bedroom", "remove the coffee table", "NO plants in the living room",
"move the console to the north wall".

Same idea and directive shape as `layout_prompt.py` on feat/photo-to-blender-3d (one directive
per piece: target room, target wall, visible), but pointed at the Studio's own data: the items
the user actually has in each room of the plan they picked. gemini-4-neon turns the sentence
into directives; the deterministic engine applies them and the reviewer re-inspects.
Plan fixtures (tub, counters, closets) and doors can never be moved by a prompt.
"""

from __future__ import annotations

import os
import time
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

import studio_live
from studio_correspond import ACCESSORY, HOSTS, TALL_TYPES, TOP_M, WALLS, WALL_NAME, WALL_TYPES, _along_wall, _own_rec, _spiral

Wall = Literal["N", "E", "S", "W"]


class ItemDirective(BaseModel):
    item_id: str = Field(description="id of one listed piece, copied exactly")
    remove: bool = Field(default=False, description="true to take the piece out of the home (back on the truck)")
    target_room: Optional[str] = Field(default=None, description="room key to move it to; null to keep its room")
    target_wall: Optional[Wall] = Field(default=None, description="wall to stand it against: N, E, S or W; null for no preference")
    reason: str = ""


class SceneDirectives(BaseModel):
    directives: list[ItemDirective] = Field(default_factory=list)
    summary: str = Field(description="one plain sentence saying what will change, or why nothing can")


PROMPT = """You turn one instruction about a furnished apartment into directives. Return one JSON object matching the schema. No prose.

Rooms (key: label, type):
{rooms}

Pieces that can be moved or removed (id | what | room key):
{items}

Pieces waiting on the truck, which can be brought in by giving a target_room (id | what):
{truck}

Fixed by the floor plan and never movable: {fixed}. If the instruction asks to move or remove one of these, emit no directive for it and say so in the summary.

Rules:
- Use only ids from the lists above, copied exactly. One directive per piece affected.
- "No X in room R" or "remove X" means remove = true for every matching piece (in room R, if a room is named).
- "Move / put X in room R" means target_room = R's key. If several pieces match and the instruction is singular, pick the most likely one (the biggest, or the only one of that kind).
- A wall is N (top of the plan), S (bottom), W (left), E (right). Set target_wall only when the instruction names a wall or side.
- If the instruction names nothing you can act on, return an empty directives list and explain in the summary.

Instruction: "{text}"
"""


def parse(text: str, rooms: dict[str, dict[str, Any]], items: dict[str, list[dict[str, Any]]], truck: list[dict[str, Any]], producer=None) -> tuple[SceneDirectives, float]:
    if producer is None:
        studio_live.require_key()
    movable = [(k, i) for k, lst in items.items() for i in lst if not i.get("plan")]
    fixed = sorted({i["label"].lower() for lst in items.values() for i in lst if i.get("plan")})
    prompt = PROMPT.format(
        rooms="\n".join(f"- {k}: {r['label']}, {r['type']}" for k, r in rooms.items()),
        items="\n".join(f"- {i['id']} | {'your ' if i.get('own') else 'from the photo: '}{i['label']} ({i['type']}) | {k}" for k, i in movable) or "- none",
        truck="\n".join(f"- {t['id']} | {t['label']}" for t in truck) or "- none",
        fixed=", ".join(fixed) or "nothing", text=text.replace('"', "'")[:600])
    t0 = time.time()
    if producer:
        res = producer(prompt)
    else:
        from perceive import _call_structured_interaction
        res = _call_structured_interaction(client=studio_live._client(), model=studio_live.MODEL_PLAN, content_blocks=[], prompt_text=prompt,
                                           schema_cls=SceneDirectives, thinking_level=os.environ.get("STUDIO_PLAN_THINKING", "low"))
    if not res.ok or res.data is None:
        raise RuntimeError("Gemini's answer failed schema validation twice: " + "; ".join(res.errors)[-300:])
    return res.data, round(time.time() - t0, 1)


def _place(room: dict[str, Any], here: list[dict[str, Any]], rec: dict[str, Any], wall: Optional[str]) -> Optional[str]:
    """Find `rec` a legal spot in `room`. Mutates rec; returns a note, or None if nothing fits."""
    solids = [i["rect"] for i in here if not i["floor"]]
    ln, dp = rec["len"], rec["dep"]
    for k in ("on", "wall", "at_h", "match"):
        rec.pop(k, None)
    if rec["type"] == "rug":
        bw, bd = min(ln, room["w"] - 0.3), min(dp, room["d"] - 0.3)
        rec["rect"] = [round((room["w"] - bw) / 2, 3), round((room["d"] - bd) / 2, 3), round(bw, 3), round(bd, 3)]
        return "centred on the floor"
    if rec["type"] in ("pillow", "vase"):
        hosts = [i for i in here if i["type"] in HOSTS[rec["type"]] and not i["floor"]]
        if not hosts:
            return None
        h = min(hosts, key=lambda i: (0 if i.get("own") else 1, sum(1 for j in here if j.get("on") == i["id"])))
        n, s_ = sum(1 for j in here if j.get("on") == h["id"]), min(ln, 0.4)
        hx, hz, hw, hd = h["rect"]
        cx = min(max(hx + hw / 2 + ((n - 1) * 0.5 if hw >= hd else 0), hx + s_ / 2), hx + hw - s_ / 2)
        cz = min(max(hz + hd / 2 + ((n - 1) * 0.5 if hd > hw else 0), hz + s_ / 2), hz + hd - s_ / 2)
        rec.update(rect=[round(cx - s_ / 2, 3), round(cz - s_ / 2, 3), round(s_, 3), round(s_, 3)], on=h["id"], at_h=TOP_M.get(h["type"], 0.5))
        return f"set on the {h['label'].lower()}"
    closed = [w for w in WALLS if w not in room["open"]]
    if rec["type"] == "art":
        wl = wall if wall in closed else max(closed, key=lambda w: (room["w"] if w in "NS" else room["d"]), default=None)
        if wl is None:
            return None
        L = room["w"] if wl in "NS" else room["d"]
        w = min(ln, L - 0.4)
        a = (L - w) / 2 + 0.7 * sum(1 for j in here if j.get("wall") == wl)
        a = min(max(a, 0.2), L - w - 0.2)
        rec["rect"] = [round(a, 3), 0.0 if wl == "N" else round(room["d"] - 0.05, 3), round(w, 3), 0.05] if wl in "NS" else \
                      [0.0 if wl == "W" else round(room["w"] - 0.05, 3), round(a, 3), 0.05, round(w, 3)]
        rec.update(face=WALLS[(WALLS.index(wl) + 2) % 4], wall=wl, at_h=1.55)
        return f"hung on the {WALL_NAME[wl]} wall"
    tall = rec["type"] in TALL_TYPES
    order = [wall] if wall else (sorted(closed, key=lambda w: -(room["w"] if w in "NS" else room["d"])) if rec["type"] in WALL_TYPES else [])
    for wl in order:
        rect = _along_wall(room, wl, ln, dp, solids, tall)
        if rect:
            rec.update(rect=rect, face=WALLS[(WALLS.index(wl) + 2) % 4])
            return f"against the {WALL_NAME[wl]} wall"
    if wall:
        return None
    res = _spiral(room, room["w"] / 2 - ln / 2, room["d"] / 2 - dp / 2, ln, dp, solids, tall, 2.5)
    if res:
        rec.update(rect=res[0], face="S")
        return "in open floor"
    return None


def apply(directives: SceneDirectives, rooms: dict[str, dict[str, Any]], items: dict[str, list[dict[str, Any]]],
          truck: list[dict[str, Any]], inventory: dict[str, dict[str, Any]]) -> list[str]:
    """Apply directives in place. Returns one plain note per directive."""
    notes: list[str] = []
    where = {i["id"]: k for k, lst in items.items() for i in lst}
    for d in directives.directives:
        key = where.get(d.item_id)
        on_truck = next((t for t in truck if t["id"] == d.item_id), None)
        if key is None and on_truck is None:
            notes.append(f"Skipped an unknown piece id ({d.item_id}).")
            continue
        if key is not None:
            it = next(i for i in items[key] if i["id"] == d.item_id)
            if it.get("plan"):
                notes.append(f"The {it['label'].lower()} is part of the floor plan and stays where it is drawn.")
                continue
        if d.remove:
            if key is None:
                continue
            riders = [i for i in items[key] if i.get("on") == it["id"]]
            items[key] = [i for i in items[key] if i is not it and i.get("on") != it["id"]]
            for gone in [it] + riders:
                if gone.get("own"):
                    truck.append({"id": gone["id"], "label": gone["label"]})
                where.pop(gone["id"], None)
            notes.append(f"Took the {it['label'].lower()} out of the {rooms[key]['label'].lower()}" + (f", along with {len(riders)} thing(s) that sat on it." if riders else "."))
            continue
        target = d.target_room if d.target_room in rooms else key
        if target is None:
            notes.append(f"No room given for {on_truck['label'].lower()}; it stays on the truck.")
            continue
        if key is None:
            src = inventory.get(d.item_id)
            if src is None:
                notes.append(f"Could not find {on_truck['label'].lower()} in the inventory.")
                continue
            rec = _own_rec(src)
        else:
            rec = dict(it)
        others = [i for i in items[target] if i["id"] != rec["id"] and i.get("on") != rec["id"]]
        note = _place(rooms[target], others, rec, d.target_wall)
        if note is None:
            side = f" on the {WALL_NAME[d.target_wall]} wall" if d.target_wall else ""
            notes.append(f"No room for the {rec['label'].lower()}{side} in the {rooms[target]['label'].lower()}; left as it was.")
            continue
        rec["note"] = f"placed by your instruction, {note}"
        if key is not None:
            riders = [i for i in items[key] if i.get("on") == it["id"]]
            items[key] = [i for i in items[key] if i is not it and i.get("on") != it["id"]]
            for r in riders:   # what sat on it goes back on the truck rather than floating
                if r.get("own"):
                    truck.append({"id": r["id"], "label": r["label"]})
        else:
            truck[:] = [t for t in truck if t["id"] != rec["id"]]
        items[target] = [i for i in items[target] if i["id"] != rec["id"]] + [rec]
        where[rec["id"]] = target
        moved = "" if key == target else (f" from the {rooms[key]['label'].lower()}" if key else " from the truck")
        notes.append(f"Moved the {rec['label'].lower()}{moved} to the {rooms[target]['label'].lower()}, {note}.")
    return notes
