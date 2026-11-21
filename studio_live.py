"""Live Gemini calls for Homeweave Studio (`studio_live.py`).

Three models, each with one job:
  - gemini-4-hydrogen : gauge a room photo the studio has never seen (size, items, materials)
  - gemini-4-neon     : read a new floor-plan image into room rectangles, doors and scale
  - gemini-decision-1 : second opinion on a finished layout, from renders (Decisions API)

Gemini perceives and judges; the deterministic engine in `studio.py` still does the placing.
Every response is validated against a Pydantic schema, retried once, then flagged.
The key is read from `.env` (GEMINI_API_KEY=...) or the environment; it is never logged.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
CATALOG_PATH = BASE_DIR / "data" / "studio_catalog.json"
IMG_DIR = BASE_DIR / "data" / "drive_inputs"
MODEL_PHOTO = "gemini-4-hydrogen"
MODEL_PLAN = "gemini-4-neon"
MODEL_DECISION = "gemini-decision-1"
PREFIX = {"kitchen": "K", "bathroom": "B", "living": "L", "bedroom": "Be"}
ROOM_KEYS = ("kitchen", "living", "bedroom", "bathroom")


def load_env() -> None:
    """Minimal .env reader (no extra dependency). Existing environment variables win."""
    env = BASE_DIR / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8-sig").splitlines():
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$", line)
        if m and m.group(2).strip().strip("\"'"):
            os.environ.setdefault(m.group(1), m.group(2).strip().strip("\"'"))


def live_available() -> bool:
    load_env()
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def require_key() -> None:
    if not live_available():
        raise RuntimeError("No Gemini key found. Put GEMINI_API_KEY=... in the .env file next to app.py and try again.")


# ---------------- schemas the models must return ----------------

ITEM_TYPES = Literal["counter", "island", "fridge", "tall", "sofa", "sectional", "coffee", "tv", "fire", "rug", "plant", "pouf",
                     "armchair", "bed", "nightstand", "seat", "vanity", "toilet", "tub", "shower"]
Wall = Literal["N", "E", "S", "W"]


class LiveItem(BaseModel):
    id: str
    type: ITEM_TYPES
    label: str
    x: float = Field(description="left edge in metres from the room's west wall")
    z: float = Field(description="top edge in metres from the room's north (far) wall")
    w: float = Field(description="extent along x in metres")
    d: float = Field(description="extent along z in metres")
    walls: list[Wall] = Field(default_factory=list, description="walls the item stands against")
    face: Wall = Field(description="direction the item's front points")
    built_in: bool = Field(default=False, description="true for counter runs, media walls, fireplaces: things re-cut to a new wall")
    feature: Optional[Literal["hob", "sink"]] = None
    stools: Optional[int] = None
    shape: Optional[Literal["round", "oval", "built"]] = None
    priority: int = Field(ge=1, le=3, description="1 = the room does not work without it, 3 = decor")
    parent: Optional[str] = Field(default=None, description="id of the item this one is arranged around, if any")


class LivePhoto(BaseModel):
    title: str = Field(description="two or three word name for the look")
    vibe: str = Field(description="one line on materials and lighting")
    room_w_m: float
    room_d_m: float
    confidence: float = Field(ge=0, le=1)
    scale_ref: str = Field(description="the object of known size the estimate was scaled from")
    floor_style: Literal["wood", "marble", "tile", "carpet"]
    cabinet_color: str
    top_color: str
    accent_color: str
    upholstery_color: str
    rug_color: str
    metal_color: str
    items: list[LiveItem]


class LivePlanWindow(BaseModel):
    wall: Wall
    a_px: float
    b_px: float


class LivePlanRoom(BaseModel):
    key: Literal["kitchen", "living", "bedroom", "bathroom"]
    label: str
    rect_px: list[float] = Field(min_length=4, max_length=4, description="interior [x0, y0, x1, y1] in image pixels")
    open_sides: list[Wall] = Field(default_factory=list, description="sides with no wall (open plan)")
    doors_px: list[list[float]] = Field(default_factory=list, description="door swing areas inside the room, each [x0, y0, x1, y1]")
    windows: list[LivePlanWindow] = Field(default_factory=list)
    entry_px: list[float] = Field(min_length=2, max_length=2, description="a point just inside the room's doorway")
    printed_dims: Optional[str] = None


class LivePlan(BaseModel):
    name: str
    subtitle: str
    px_per_m: float
    scale_ref: str
    confidence: float = Field(ge=0, le=1)
    entrance_px: list[float] = Field(min_length=2, max_length=2)
    rooms: list[LivePlanRoom]


ALLOWED = {
    "kitchen": "counter (a straight run of base units against one wall; use one item per wall run), island (include bar stools in its depth and set stools), fridge, tall (tall unit or lit shelf tower), rug, plant",
    "bathroom": "vanity, toilet, tub (shape built or oval), shower, rug (bath mat), plant",
    "living": "sofa, sectional (L-shaped; w and d are its bounding box), coffee, tv (media wall or console), fire, rug, plant, pouf, armchair",
    "bedroom": "bed (shape round if round), nightstand, armchair, tv, seat (window seat), tall (shelf or wardrobe), rug, plant, pouf, sofa, coffee",
}

PROMPT_PHOTO = """You are gauging one interior photo of a {cat} so its layout can be rebuilt in a different room.

Return one JSON object matching the schema. No prose.

Frame: imagine the room as a rectangle seen from above. North (z = 0) is the far wall the camera faces, south is where the camera stands, west is the left wall, east the right wall. x runs west to east, z runs north to south, both in metres from the north-west corner.

1. Estimate the room's width (room_w_m, west to east) and depth (room_d_m, north to south). Scale from one object of known size you can actually see (counter depth 0.62 m, door 0.81 m wide, queen bed 1.52 m wide, toilet 0.70 m deep, a TV's diagonal) and name it in scale_ref. Set confidence lower when most of the room is out of frame.
2. List the furniture and fixtures as axis-aligned rectangles (x, z, w, d). Allowed types for a {cat}: {allowed}.
   - walls: the walls the item touches. face: the way its front points (a counter on the north wall faces S).
   - built_in = true for counter runs, media walls and fireplaces.
   - priority 1 for what the room needs, 2 for nice to have, 3 for decor. At most 8 items; skip anything under 0.3 m.
   - parent: for a coffee table or rug, the id of the sofa or bed it belongs to.
   - Items must not overlap each other (rugs may sit under things) and must lie inside the room.
3. Colours as #rrggbb hex, read from the photo: cabinet_color (cabinetry or main wood), top_color (counter or table tops), upholstery_color, rug_color, accent_color, metal_color. floor_style is the floor material.

Only report what is visible. If a needed fixture is out of frame (for example the toilet), leave it out; do not invent it.
"""

PROMPT_PLAN = """You are reading an apartment floor plan image that is {w} x {h} pixels. Pixel (0, 0) is the top-left corner; x grows right, y grows down.

Return one JSON object matching the schema. No prose.

1. Scale: px_per_m is how many image pixels equal one metre. If the plan prints room dimensions, derive it from them and say which in scale_ref. Otherwise assume an interior door is 0.81 m wide, measure one, and say so.
2. Give exactly one room for each of these keys: kitchen, living, bedroom (the primary or master one), bathroom (the one nearest that bedroom). For each:
   - rect_px: the largest axis-aligned rectangle of clear interior floor, inside the wall lines, as [x0, y0, x1, y1]. In an open plan, give each zone its own rectangle and list the side with no wall in open_sides.
   - doors_px: for every door that swings into the room, the floor rectangle its swing covers.
   - windows: the wall (N top, S bottom, W left, E right) and the pixel range along it (x for N and S, y for E and W).
   - entry_px: a point just inside the doorway people walk in through.
   - printed_dims: the dimension text printed for that room, if any.
3. entrance_px: a point just outside the apartment's front door.

Report what the drawing shows. If a required room is missing from the plan, leave it out rather than inventing one.
"""


def _client():
    load_env()
    from google import genai  # imported lazily so the studio still runs offline without the SDK configured
    return genai.Client()


def _structured(model: str, media: dict[str, Any], prompt: str, schema: type[BaseModel], thinking: Optional[str] = None):
    from perceive import _call_structured_interaction
    return _call_structured_interaction(client=_client(), model=model, content_blocks=[media], prompt_text=prompt,
                                        schema_cls=schema, thinking_level=thinking)


def _hex(c: str, fallback: str) -> str:
    return c.lower() if isinstance(c, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", c) else fallback


def _mix(a: str, b: str, t: float) -> str:
    A, B = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(A, B))


def sample_palette(path: Path) -> dict[str, Any]:
    """Floor, wall, glow and swatches read from the photo's pixels (not from the model)."""
    try:
        import colorsys
        from PIL import Image
    except ImportError:
        return {}
    im = Image.open(path).convert("RGB")
    W, H = im.size
    hexs = lambda rgb: "#%02x%02x%02x" % tuple(int(v) for v in rgb)

    def median(box):
        px = sorted(im.crop(box).resize((48, 48)).get_flattened_data() if hasattr(im, "get_flattened_data") else im.crop(box).resize((48, 48)).getdata(), key=sum)
        mid = px[len(px) // 4: 3 * len(px) // 4]
        return tuple(sum(p[i] for p in mid) / len(mid) for i in range(3))

    q = im.resize((96, 96)).quantize(6, method=Image.Quantize.MEDIANCUT, kmeans=3)
    pal = q.getpalette()[:18]
    sw = [tuple(pal[i * 3:i * 3 + 3]) for _, i in sorted(q.getcolors(), reverse=True)]
    warm = max(sw, key=lambda r: (r[0] - r[2]) + 0.5 * sum(r) / 3)
    h, l, s = colorsys.rgb_to_hls(*[v / 255 for v in warm])
    glow = [v * 255 for v in colorsys.hls_to_rgb(h, max(l, 0.68), max(s, 0.85))]
    return {"floor": hexs(median((int(W * .15), int(H * .86), int(W * .85), H))), "wall": hexs(median((int(W * .1), int(H * .12), int(W * .9), int(H * .42)))),
            "glow": hexs(glow), "swatches": [hexs(x) for x in sw], "size": [W, H]}


ALT = {"sectional": {"type": "sofa", "len": 2.2, "dep": 0.95, "label": "2.2 m three-seat sofa"}, "island": {"len": 1.7, "n": 2, "label": "island cut to 1.7 m with 2 stools"},
       "vanity": {"len": 0.7, "dep": 0.48, "label": "0.7 m vanity"}, "shower": {"len": 0.9, "dep": 0.9, "label": "0.9 m shower"},
       "bed": {"len": 1.4, "dep": 2.0, "label": "1.4 m double bed"}, "sofa": {"len": 1.8, "dep": 0.9, "label": "1.8 m sofa"}, "coffee": {"len": 0.8, "dep": 0.5, "label": "0.8 m coffee table"}}


def photo_to_entry(data: LivePhoto, cat: str, pid: str, filename: str, sampled: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    W, D = max(1.5, min(12.0, data.room_w_m)), max(1.5, min(12.0, data.room_d_m))
    items, seen = [], set()
    tub = next((i.id for i in data.items if i.type == "tub"), None)
    for i in data.items[:10]:
        if i.id in seen:
            continue
        seen.add(i.id)
        w, d = max(0.2, min(i.w, W)), max(0.2, min(i.d, D))
        it: dict[str, Any] = {"id": i.id, "type": i.type, "label": i.label, "x": round(max(0.0, min(i.x, W - w)), 2), "z": round(max(0.0, min(i.z, D - d)), 2),
                              "w": round(w, 2), "d": round(d, 2), "walls": list(dict.fromkeys(i.walls)), "face": i.face, "pri": i.priority}
        if i.type == "rug":
            it["floor"] = True
            it.pop("walls")
        if i.built_in or i.type in ("counter", "tv", "fire"):
            it["stretch"], it["minw"] = True, round(max(0.6, min(max(w, d) * 0.5, 1.6)), 2)
        if i.type in ("fridge", "tall", "shower"):
            it["tall"] = True
        if i.feature:
            it["feat"] = i.feature
        if i.stools:
            it["n"] = int(max(1, min(i.stools, 5)))
        if i.shape:
            it["shape"] = i.shape
        if i.parent and i.parent != i.id and any(o.id == i.parent for o in data.items):
            it["parent"] = i.parent
        if i.type == "shower" and tub:
            it["merge"] = tub
        if i.type in ALT and i.type != ("bed" if i.shape == "round" else ""):
            it["alt"] = dict(ALT[i.type])
        if i.type == "tub":
            it["alt"] = {"len": 1.5, "dep": 0.75, "shape": "built", "label": "built-in 1.5 m tub"}
        items.append(it)
    floor_model = {"wood": "#a57c55", "marble": "#d9cbb5", "tile": "#9a968e", "carpet": "#b9ab98"}[data.floor_style]
    floor = _mix(floor_model, sampled["floor"], 0.55) if sampled.get("floor") else floor_model
    palette = {"floor": floor, "floor2": _mix(floor, "#000000", 0.2), "style": data.floor_style,
               "cab": _hex(data.cabinet_color, "#6b4a33"), "top": _hex(data.top_color, "#e6dccb"), "accent": _hex(data.accent_color, "#222222"),
               "soft": _hex(data.upholstery_color, "#e2d8c8"), "rug": _hex(data.rug_color, "#cbbda4"), "metal": _hex(data.metal_color, "#333333"),
               "glow": sampled.get("glow", "#ffc46b")}
    if sampled.get("wall"):
        palette["wall"] = sampled["wall"]
    return {"id": pid, "cat": cat, "file": filename, "title": data.title, "vibe": data.vibe, "palette": palette,
            "gauged": {"w": round(W, 2), "d": round(D, 2), "conf": round(data.confidence, 2), "ref": data.scale_ref},
            "items": items, "swatches": sampled.get("swatches", []), "live": meta}


def _save_catalog(mutate) -> dict[str, Any]:
    cat = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    out = mutate(cat)
    CATALOG_PATH.write_text(json.dumps(cat, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def _media(path: Path) -> dict[str, Any]:
    from perceive import media_block_from_source
    return media_block_from_source(path)


def perceive_photo(image: bytes, original_name: str, cat: str, producer=None) -> dict[str, Any]:
    """Gauge a new room photo with gemini-4-hydrogen and add it to the catalogue. `producer` is a test hook."""
    if cat not in PREFIX:
        raise ValueError(f"unknown room type {cat}")
    if producer is None:
        require_key()
    ext = (Path(original_name).suffix or ".jpg").lower()
    existing = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["photos"]
    nums = [int(m.group(1)) for p in existing if p["cat"] == cat and (m := re.fullmatch(PREFIX[cat] + r"(\d+)", p["id"]))]
    pid = f"{PREFIX[cat]}{max(nums, default=0) + 1}"
    path = IMG_DIR / f"{pid}{ext}"
    path.write_bytes(image)
    t0 = time.time()
    try:
        prompt = PROMPT_PHOTO.format(cat=cat, allowed=ALLOWED[cat])
        res = producer(prompt) if producer else _structured(MODEL_PHOTO, _media(path), prompt, LivePhoto)
        if not res.ok or res.data is None:
            raise RuntimeError("Gemini's answer failed schema validation twice: " + "; ".join(res.errors)[-400:])
        if not res.data.items:
            raise RuntimeError("Gemini found no furniture or fixtures in this photo.")
        meta = {"model": MODEL_PHOTO, "seconds": round(time.time() - t0, 1), "attempts": res.attempts, "source_file": original_name}
        entry = photo_to_entry(res.data, cat, pid, path.name, sample_palette(path), meta)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    _save_catalog(lambda c: c["photos"].append(entry))
    return entry


def plan_to_entry(data: LivePlan, pid: str, filename: str, size: list[int], meta: dict[str, Any]) -> dict[str, Any]:
    rooms: dict[str, Any] = {}
    for r in data.rooms:
        if r.key in rooms:
            continue
        x0, y0, x1, y1 = r.rect_px
        x0, x1, y0, y1 = max(0, min(x0, x1)), min(size[0], max(x0, x1)), max(0, min(y0, y1)), min(size[1], max(y0, y1))
        if x1 - x0 < 10 or y1 - y0 < 10:
            continue
        rooms[r.key] = {"label": r.label, "rect": [round(v) for v in (x0, y0, x1, y1)], "open": list(dict.fromkeys(r.open_sides)),
                        "windows": [{"wall": w.wall, "a": round(min(w.a_px, w.b_px)), "b": round(max(w.a_px, w.b_px))} for w in r.windows],
                        "doors": [[round(max(x0, min(b[0], b[2]))), round(max(y0, min(b[1], b[3]))), round(min(x1, max(b[0], b[2]))), round(min(y1, max(b[1], b[3])))]
                                  for b in r.doors_px if len(b) == 4],
                        "obstacles": [], "entry": [round(min(max(r.entry_px[0], x0), x1)), round(min(max(r.entry_px[1], y0), y1))]}
        if r.printed_dims:
            rooms[r.key]["printed_text"] = r.printed_dims
    missing = [k for k in ROOM_KEYS if k not in rooms]
    if missing:
        raise RuntimeError("Gemini could not find these rooms on the plan: " + ", ".join(missing) + ". The studio needs a kitchen, living room, bedroom and bathroom.")
    if not 8 <= data.px_per_m <= 400:
        raise RuntimeError(f"Implausible scale from Gemini: {data.px_per_m:.1f} px per metre.")
    xs = [v for r in rooms.values() for v in (r["rect"][0], r["rect"][2])]
    ys = [v for r in rooms.values() for v in (r["rect"][1], r["rect"][3])]
    pad = 0.6 * data.px_per_m
    view = [round(max(0, min(xs) - pad)), round(max(0, min(ys) - pad)), round(min(size[0], max(xs) + pad)), round(min(size[1], max(ys) + pad))]
    return {"id": pid, "name": data.name, "subtitle": data.subtitle, "image": filename, "size": size, "view": view, "px_per_m": round(data.px_per_m, 2),
            "scale_ref": data.scale_ref + f" (read by {MODEL_PLAN}, confidence {data.confidence:.2f})", "rooms": rooms,
            "entrance": [round(data.entrance_px[0]), round(data.entrance_px[1])], "live": meta}


def perceive_plan(image: bytes, original_name: str, producer=None) -> dict[str, Any]:
    """Read a new floor plan with gemini-4-neon and add it to the catalogue. `producer` is a test hook."""
    if producer is None:
        require_key()
    from PIL import Image
    ext = (Path(original_name).suffix or ".png").lower()
    existing = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["plans"]
    nums = [int(m.group(1)) for p in existing if (m := re.fullmatch(r"plan(\d+)", p["id"]))]
    pid = f"plan{max(nums, default=0) + 1}"
    path = IMG_DIR / f"{pid}{ext}"
    path.write_bytes(image)
    t0 = time.time()
    try:
        size = list(Image.open(path).size)
        prompt = PROMPT_PLAN.format(w=size[0], h=size[1])
        res = producer(prompt) if producer else _structured(MODEL_PLAN, _media(path), prompt, LivePlan, thinking=os.environ.get("STUDIO_PLAN_THINKING", "low"))
        if not res.ok or res.data is None:
            raise RuntimeError("Gemini's answer failed schema validation twice: " + "; ".join(res.errors)[-400:])
        meta = {"model": MODEL_PLAN, "seconds": round(time.time() - t0, 1), "attempts": res.attempts, "source_file": original_name}
        entry = plan_to_entry(res.data, pid, path.name, size, meta)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    _save_catalog(lambda c: c["plans"].append(entry))
    return entry


def second_opinion(rooms: list[dict[str, Any]], runner=None) -> list[dict[str, Any]]:
    """Decisions API verdict per room from a render of the finished layout. `runner` is a test hook."""
    if runner is None:
        require_key()
    from judge import _extract_noul, _extract_score_1_to_5, _make_v1beta_client, build_layout_questions
    client = None if runner else _make_v1beta_client()
    out = []
    for r in rooms:
        questions = build_layout_questions(include_bed_question=bool(r.get("has_bed")))
        header, b64 = r["image"].split(",", 1)
        shared = [{"type": "text", "text": f"Top-down render of a furnished {r['label']} in a planned apartment. Furniture inside a blue ring belongs to the resident."},
                  {"type": "image", "data": b64, "mime_type": header.split(";")[0].replace("data:", "") or "image/jpeg"}]
        t0 = time.time()
        answers = runner(shared, questions) if runner else (client.decisions.create(model=MODEL_DECISION, input=shared, questions=questions).answers or {})
        res: dict[str, Any] = {}
        for k, q in questions.items():
            if k not in answers:
                continue
            v, c = (_extract_noul if q["type"] == "noul" else _extract_score_1_to_5)(answers, k)
            res[k] = {"type": q["type"], "value": round(v, 3), "confidence": round(c, 3), "question": q["instructions"]}
        path_p = res.get("clear_walking_path", {}).get("value")
        out.append({"key": r["key"], "label": r["label"], "answers": res, "seconds": round(time.time() - t0, 1), "model": MODEL_DECISION,
                    "says_walkable": None if path_p is None else path_p >= 0.5})
    return out
