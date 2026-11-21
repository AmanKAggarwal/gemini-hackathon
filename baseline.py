"""Top-down PNG renderer and Gemini-only spatial placement baseline (`baseline.py`, Prompt 7).

Provides:
- `render_room_topdown_png`: dependency-free (`numpy` + `zlib` + `struct`) top-down RGB PNG
  renderer for rooms and furniture (with front-facing strips), used by both the `gemini-4-neon`
  baseline and server-side layout judging in `eval.py`.
- `run_gemini_baseline`: sends top-down images of `src_room` and empty `new_room` + room JSON +
  item dimensions to `gemini-4-neon` with Prompt 7 and validates against `BaselineLayoutResponse`.
"""

from __future__ import annotations

import base64
import json
import math
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from google import genai
from shapely.geometry import Polygon

from check import door_geometry, facing_unit_vector, item_footprint, wall_segments
from schemas import (
    BaselineLayoutResponse,
    FurnitureItem,
    LeftoverItem,
    Placement,
    Pose2D,
    Provenance,
    Room,
    ValidatedStageResult,
    to_gemini_response_format,
    validate_with_retry,
)

MODEL_NEON = "gemini-4-neon"

PROMPT_BASELINE = """You are arranging furniture. You get a top-down image of the person's current room, a top-down image of an empty new room, the new room's JSON (polygon, doors, windows, fixtures, in metres), and the list of furniture with dimensions.

Place every item in the new room so the arrangement feels like the old room, adapted. Keep items that were used together close together and facing the same way relative to each other.

Rules: no two items may overlap; nothing may block a door or its swing; leave walkways at least 0.9 m wide; every item must be fully inside the room. If an item cannot fit, leave it out and list it under leftovers with the reason.

Return JSON only: {placements: [{item_id, pose: {x, z, rot_deg}}], leftovers: [{item_id, reason}]}. Coordinates are metres in the new room's frame, angles in degrees clockwise from +z seen from above.
"""

GROUP_PALETTE = [
    (59, 130, 246),
    (16, 185, 129),
    (245, 158, 11),
    (139, 92, 246),
    (236, 72, 153),
    (20, 184, 166),
]


def _encode_rgb_png(img: np.ndarray) -> bytes:
    """Encode `(H, W, 3)` uint8 RGB array into valid PNG bytes using stdlib `zlib`."""
    h, w, _ = img.shape
    raw_rows = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", crc)

    ihdr = struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(raw_rows, level=6)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _draw_line(
    canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], thickness: int = 2
) -> None:
    h, w, _ = canvas.shape
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps + 1).astype(int)
    ys = np.linspace(y0, y1, steps + 1).astype(int)
    r = max(0, thickness // 2)
    for px, py in zip(xs, ys):
        y_lo, y_hi = max(0, py - r), min(h, py + r + 1)
        x_lo, x_hi = max(0, px - r), min(w, px + r + 1)
        canvas[y_lo:y_hi, x_lo:x_hi] = color


def _fill_polygon(canvas: np.ndarray, pts_px: Sequence[tuple[float, float]], color: tuple[int, int, int]) -> None:
    """Scanline-fill a 2D polygon onto `canvas`."""
    if len(pts_px) < 3:
        return
    h, w, _ = canvas.shape
    arr = np.asarray(pts_px, dtype=float)
    min_y = max(0, int(math.floor(arr[:, 1].min())))
    max_y = min(h - 1, int(math.ceil(arr[:, 1].max())))
    n = len(arr)
    for y in range(min_y, max_y + 1):
        scan_y = y + 0.5
        x_hits: list[float] = []
        for i in range(n):
            x1, y1 = arr[i]
            x2, y2 = arr[(i + 1) % n]
            if (y1 <= scan_y < y2) or (y2 <= scan_y < y1):
                t = (scan_y - y1) / (y2 - y1)
                x_hits.append(x1 + t * (x2 - x1))
        x_hits.sort()
        for k in range(0, len(x_hits) - 1, 2):
            xa = max(0, int(math.ceil(x_hits[k])))
            xb = min(w - 1, int(math.floor(x_hits[k + 1])))
            if xb >= xa:
                canvas[y, xa : xb + 1] = color


def render_room_topdown_png(
    room: Room | dict[str, object],
    items_or_placements: Sequence[FurnitureItem | Placement | dict[str, object]] = (),
    *,
    size_px: int = 512,
) -> bytes:
    """Render a room and optional furniture poses to a crisp top-down PNG (+x right, +z down)."""
    room_obj = room if isinstance(room, Room) else Room.model_validate(room)
    canvas = np.full((size_px, size_px, 3), 243, dtype=np.uint8)
    if not room_obj.polygon or len(room_obj.polygon) < 3:
        return _encode_rgb_png(canvas)

    room_poly = Polygon(room_obj.polygon)
    minx, minz, maxx, maxz = room_poly.bounds
    pad = 0.6
    span = max(maxx - minx + 2 * pad, maxz - minz + 2 * pad, 1.0)
    scale = (size_px - 24) / span
    ox = minx - pad
    oz = minz - pad

    def _to_px(wx: float, wz: float) -> tuple[int, int]:
        return int(round(12 + (wx - ox) * scale)), int(round(12 + (wz - oz) * scale))

    # Floor fill & walls
    room_px = [_to_px(pt[0], pt[1]) for pt in room_obj.polygon]
    _fill_polygon(canvas, room_px, (255, 255, 255))
    segs = wall_segments(room_obj.polygon)
    for p0, p1 in segs:
        x0, y0 = _to_px(float(p0[0]), float(p0[1]))
        x1, y1 = _to_px(float(p1[0]), float(p1[1]))
        _draw_line(canvas, x0, y0, x1, y1, (20, 35, 58), thickness=5)

    # Fixtures
    for fx in room_obj.fixtures:
        if fx.footprint and len(fx.footprint) >= 3:
            _fill_polygon(canvas, [_to_px(p[0], p[1]) for p in fx.footprint], (185, 194, 206))

    # Windows (blue bars) & Doors (amber swing arcs)
    for win in room_obj.windows:
        if segs:
            p0, p1 = segs[win.wall_index % len(segs)]
            vec = p1 - p0
            wlen = max(float(np.linalg.norm(vec)), 1e-6)
            t = vec / wlen
            wa = p0 + min(win.offset_m, wlen) * t
            wb = p0 + min(win.offset_m + win.width_m, wlen) * t
            _draw_line(canvas, *_to_px(float(wa[0]), float(wa[1])), *_to_px(float(wb[0]), float(wb[1])), (59, 130, 246), thickness=6)

    for door in room_obj.doors:
        d_line, d_swing, _, _ = door_geometry(door, segs, room_poly)
        if d_swing is not None:
            _fill_polygon(
                canvas,
                [_to_px(cx, cz) for cx, cz in d_swing.exterior.coords[:-1]],
                (251, 239, 210),
            )
        coords = list(d_line.coords)
        _draw_line(
            canvas,
            *_to_px(coords[0][0], coords[0][1]),
            *_to_px(coords[1][0], coords[1][1]),
            (217, 119, 6),
            thickness=5,
        )

    # Furniture boxes + darker front strip
    group_color_map: dict[str, tuple[int, int, int]] = {}
    for entry in items_or_placements:
        if isinstance(entry, dict):
            dims = entry.get("dims_m")
            pose = entry.get("pose")
            gid = str(entry.get("group_id") or entry.get("category") or "default")
            if dims is None or pose is None:
                continue
            from schemas import DimensionsM
            dims_obj = dims if isinstance(dims, DimensionsM) else DimensionsM.model_validate(dims)
            pose_obj = pose if isinstance(pose, Pose2D) else Pose2D.model_validate(pose)
        else:
            if entry.dims_m is None or entry.pose is None:
                continue
            dims_obj = entry.dims_m
            pose_obj = entry.pose
            gid = str(entry.group_id or entry.category or "default")

        if gid not in group_color_map:
            group_color_map[gid] = GROUP_PALETTE[len(group_color_map) % len(GROUP_PALETTE)]
        base_col = group_color_map[gid]
        fp = item_footprint(dims_obj, pose_obj)
        fp_coords = list(fp.exterior.coords[:-1])
        _fill_polygon(canvas, [_to_px(cx, cz) for cx, cz in fp_coords], base_col)

        # Front face connects corners 2 and 3 (+d/2 in local space)
        c2 = _to_px(fp_coords[2][0], fp_coords[2][1])
        c3 = _to_px(fp_coords[3][0], fp_coords[3][1])
        dark_col = (max(0, base_col[0] - 55), max(0, base_col[1] - 55), max(0, base_col[2] - 55))
        _draw_line(canvas, c2[0], c2[1], c3[0], c3[1], dark_col, thickness=5)

    return _encode_rgb_png(canvas)


def render_room_topdown_data_url(
    room: Room | dict[str, object],
    items_or_placements: Sequence[FurnitureItem | Placement | dict[str, object]] = (),
    *,
    size_px: int = 512,
) -> str:
    png_bytes = render_room_topdown_png(room, items_or_placements, size_px=size_px)
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def run_gemini_baseline(
    src_room: Room | dict[str, object],
    src_items: Sequence[FurnitureItem | dict[str, object]],
    new_room: Room | dict[str, object],
    *,
    client: Optional[genai.Client] = None,
    transport_override: Optional[Callable[[Optional[str]], str | dict[str, object]]] = None,
) -> tuple[ValidatedStageResult[BaselineLayoutResponse], list[Placement], list[LeftoverItem]]:
    """Call `gemini-4-neon` with Prompt 7 and top-down renders; convert to `Placement`s for `check_layout`."""
    src_room_obj = src_room if isinstance(src_room, Room) else Room.model_validate(src_room)
    new_room_obj = new_room if isinstance(new_room, Room) else Room.model_validate(new_room)
    item_objs = [
        it if isinstance(it, FurnitureItem) else FurnitureItem.model_validate(it)
        for it in src_items
    ]
    items_by_id = {it.id: it for it in item_objs}

    if transport_override is not None:
        raw_res = validate_with_retry(transport_override, BaselineLayoutResponse)
    else:
        active_client = client or genai.Client()
        src_png_b64 = base64.b64encode(
            render_room_topdown_png(src_room_obj, item_objs)
        ).decode("ascii")
        new_png_b64 = base64.b64encode(render_room_topdown_png(new_room_obj, [])).decode("ascii")

        items_summary = [
            {
                "item_id": it.id,
                "category": it.category,
                "group_id": it.group_id,
                "dims_m": it.dims_m.model_dump() if it.dims_m else None,
            }
            for it in item_objs
            if it.dims_m is not None
        ]
        response_format = to_gemini_response_format(BaselineLayoutResponse)

        def _producer(prev_error: Optional[str]) -> str:
            text = (
                f"{PROMPT_BASELINE}\n\n"
                f"NEW ROOM JSON:\n{new_room_obj.model_dump_json(indent=2)}\n\n"
                f"FURNITURE LIST:\n{json.dumps(items_summary, indent=2)}"
            )
            if prev_error:
                text += f"\n\nPREVIOUS JSON VALIDATION ERROR:\n{prev_error}"
            interaction = active_client.interactions.create(
                model=MODEL_NEON,
                input=[
                    {"type": "text", "text": "[Current room top-down]"},
                    {"type": "image", "data": src_png_b64, "mime_type": "image/png"},
                    {"type": "text", "text": "[Empty new room top-down]"},
                    {"type": "image", "data": new_png_b64, "mime_type": "image/png"},
                    {"type": "text", "text": text},
                ],
                response_format=response_format,
                generation_config={"max_output_tokens": 32768},
                store=False,
            )
            if not interaction.output_text:
                raise ValueError("Empty baseline response from gemini-4-neon.")
            return interaction.output_text

        raw_res = validate_with_retry(_producer, BaselineLayoutResponse)

    if not raw_res.ok or raw_res.data is None:
        return raw_res, [], []

    placements: list[Placement] = []
    for bp in raw_res.data.placements:
        src_it = items_by_id.get(bp.item_id)
        if src_it is None or src_it.dims_m is None:
            continue
        placements.append(
            Placement(
                item_id=src_it.id,
                category=src_it.category,
                group_id=src_it.group_id,
                dims_m=src_it.dims_m,
                room_id=new_room_obj.id,
                room_pair=[src_room_obj.id, new_room_obj.id],
                pose=bp.pose,
                confidence=src_it.confidence,
                provenance=Provenance(source="gemini_4_neon_baseline", from_room=src_room_obj.id),
            )
        )

    leftovers = [
        LeftoverItem(
            item_id=bl.item_id if bl.item_id in items_by_id else f"src-unknown-{i}",
            reason=bl.reason,
            provenance=Provenance(source="gemini_4_neon_baseline"),
        )
        for i, bl in enumerate(raw_res.data.leftovers)
        if bl.item_id in items_by_id
    ]
    return raw_res, placements, leftovers


if __name__ == "__main__":
    sample_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent / "samples" / "living_pair.json"
    )
    case = json.loads(sample_path.read_text(encoding="utf-8"))
    png_bytes = render_room_topdown_png(case["src_room"], case["src_items"])
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    mock_resp = {
        "placements": [
            {"item_id": "src-sofa-01", "pose": {"x": 2.6, "z": 3.8, "rot_deg": 180.0}},
            {"item_id": "src-coffee-01", "pose": {"x": 2.6, "z": 2.5, "rot_deg": 180.0}},
        ],
        "leftovers": [{"item_id": "src-sectional-oversize", "reason": "wider than room"}],
    }
    res, pls, los = run_gemini_baseline(
        case["src_room"],
        case["src_items"],
        case["new_room"],
        transport_override=lambda _err: mock_resp,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "topdown_png_bytes": len(png_bytes),
                "validated": res.ok,
                "placements": len(pls),
                "leftovers": len(los),
            },
            indent=2,
        )
    )
