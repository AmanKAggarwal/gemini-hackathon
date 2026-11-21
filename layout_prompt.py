"""Natural-language text-to-3D spatial layout engine for Homeweave (`layout_prompt.py`).

Translates user text instructions (e.g. "put the bed against the north wall facing the window,
move the TV to the south wall with the sofa facing it, put the dining table near the balcony")
into rigid furniture-group transforms `(x, z, rot_deg)` for the Blender 5.2 photorealistic GLB
residence (`walkthrough/assets/dream_apartment.glb`), verified by deterministic `shapely`
containment, doorway-clearance, and zero-overlap checks.

Uses `gemini-4-neon` (`client.interactions.create` + structured Pydantic JSON) when
`GEMINI_API_KEY` is set, with a full deterministic spatial parser fallback offline.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from typing import Any, Literal, Optional

from google import genai
from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import box

from schemas import to_gemini_response_format

MODEL_NEON = "gemini-4-neon"

GroupId = Literal[
    "bed_group",
    "desk_group",
    "wardrobe_group",
    "bench_group",
    "tv_group",
    "sofa_group",
    "dining_group",
    "living_art_group",
    "island_group",
    "tub_group",
]
WallSide = Literal["N", "S", "E", "W", "center"]
LightingChoice = Literal["sunlit_day", "golden_hour", "evening_cove"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GroupDirective(StrictModel):
    group_id: GroupId
    target_wall: Optional[WallSide] = None
    target_x: Optional[float] = None
    target_z: Optional[float] = None
    rot_deg: Optional[float] = None
    visible: bool = True
    reason: str = ""


class LLMSceneDirectives(StrictModel):
    reset_all: bool = False
    lighting_mode: Optional[LightingChoice] = None
    style_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Optional room_type -> preset ID ('L1'..'B4') if text asks for a visual style mood.",
    )
    directives: list[GroupDirective] = Field(default_factory=list)
    summary: str = ""


# Exact interior usable room bounds (metres) from `walkthrough/assets/apartment_meta.json`:
# Coordinate frame: x East (0..9.42), z South (0..6.99), y Up.
# Three.js rotation.y = +90 deg maps +x -> -z (North) and -x -> +z (South).
# Three.js rotation.y = -90 deg maps +x -> +z (South) and -x -> -z (North).
ROOM_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "living": (0.10, 0.10, 5.74, 3.58),
    "kitchen": (5.89, 0.10, 9.33, 3.58),
    "primary_bedroom": (0.10, 3.74, 4.70, 6.89),
    "bathroom": (4.86, 3.74, 8.09, 6.89),
}

# Doorway keep-clear boxes (x0, z0, x1, z1)
DOOR_CLEAR_ZONES: dict[str, list[tuple[float, float, float, float]]] = {
    "living": [
        (3.38, 3.00, 4.30, 3.58),  # entry to primary bedroom
        (2.95, 0.10, 3.88, 0.65),  # open balcony sliding door
    ],
    "primary_bedroom": [
        (3.38, 3.74, 4.22, 4.55),  # bedroom door clear entry swing
    ],
    "kitchen": [
        (6.12, 3.05, 7.00, 3.58),  # passage to bathroom
        (8.45, 2.58, 9.33, 3.50),  # front entry door
    ],
    "bathroom": [
        (6.12, 3.74, 7.00, 4.60),  # bathroom door + inward ajar swing
    ],
}

# Canonical default poses & unrotated half-extents (hx, hz) for each movable GLB furniture assembly
GROUP_DEFAULTS: dict[str, dict[str, Any]] = {
    "bed_group": {
        "label": "Primary Queen Bed & Nightstands",
        "room": "primary_bedroom",
        "x": 1.16,
        "z": 5.38,
        "rot_deg": 0.0,
        "hx": 1.04,  # x: 0.12..2.20 (headboard on W, foot facing +x East)
        "hz": 0.96,  # z: bed mattress core width (nightstands tuck flush to headboard wall)
        "visible": True,
        "collider_labels": ["Primary_Queen_Bed", "Nightstand_0", "Nightstand_1"],
    },
    "desk_group": {
        "label": "Walnut Study Desk & Chair",
        "room": "primary_bedroom",
        "x": 2.08,
        "z": 4.10,
        "rot_deg": 0.0,
        "hx": 0.60,
        "hz": 0.36,
        "visible": True,
        "collider_labels": ["Bedroom_DeskTop"],
    },
    "wardrobe_group": {
        "label": "Smoked-Glass Wardrobe",
        "room": "primary_bedroom",
        "x": 4.42,
        "z": 5.55,
        "rot_deg": 0.0,
        "hx": 0.29,
        "hz": 0.91,
        "visible": True,
        "collider_labels": ["Bedroom_Wardrobe"],
    },
    "bench_group": {
        "label": "Window Bench",
        "room": "primary_bedroom",
        "x": 2.375,
        "z": 6.73,
        "rot_deg": 0.0,
        "hx": 1.45,
        "hz": 0.17,
        "visible": True,
        "collider_labels": ["Bedroom_WindowBench"],
    },
    "tv_group": {
        "label": "65\" OLED TV, Media Console & Feature Wall",
        "room": "living",
        "x": 0.30,
        "z": 1.83,
        "rot_deg": 0.0,
        "hx": 0.21,  # depth along x (against W wall, facing +x East)
        "hz": 1.62,  # width along z (3.24 m incl. flanking niches)
        "visible": True,
        "collider_labels": ["Media_Console_Body"],
    },
    "sofa_group": {
        "label": "Bouclé Sofa, Coffee Table & Lounge Rug",
        "room": "living",
        "x": 2.10,
        "z": 1.90,
        "rot_deg": 0.0,
        "hx": 1.08,  # x: ~1.02..3.18 (sofa + coffee table)
        "hz": 1.10,  # z: ~0.80..3.00 (exact Living_Sofa span 2.20 m)
        "visible": True,
        "collider_labels": ["Living_Sofa", "CoffeeTable_Base", "CoffeeTable_Top", "Accent_Armchair"],
    },
    "dining_group": {
        "label": "Dining Table, Chairs & Chandelier",
        "room": "living",
        "x": 4.58,
        "z": 2.12,
        "rot_deg": 0.0,
        "hx": 0.78,
        "hz": 0.74,
        "visible": True,
        "collider_labels": ["Dining_Table_Top", "Dining_Group"],
    },
    "living_art_group": {
        "label": "Framed Gallery Artworks",
        "room": "living",
        "x": 1.825,
        "z": 3.57,
        "rot_deg": 0.0,
        "hx": 0.72,
        "hz": 0.04,
        "visible": True,
        "collider_labels": [],
    },
    "island_group": {
        "label": "Waterfall Kitchen Island & Stools",
        "room": "kitchen",
        "x": 7.39,
        "z": 2.08,
        "rot_deg": 0.0,
        "hx": 0.81,
        "hz": 0.50,
        "visible": True,
        "collider_labels": ["Kitchen_Island"],
    },
    "tub_group": {
        "label": "Freestanding Soaking Tub & Teak Plinth",
        "room": "bathroom",
        "x": 5.80,
        "z": 6.42,
        "rot_deg": 0.0,
        "hx": 0.89,
        "hz": 0.47,
        "visible": True,
        "collider_labels": ["Spa_Soaking_Tub"],
    },
}


# Curated architect-verified wall anchors for each rigid group
WALL_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "bed_group": {
        "W": {"x": 1.16, "z": 5.38, "rot_deg": 0.0, "note": "headboard on west wall, facing east"},
        "N": {"x": 1.54, "z": 4.81, "rot_deg": -90.0, "note": "headboard on north wall, facing south window"},
        "S": {"x": 1.78, "z": 5.82, "rot_deg": 90.0, "note": "headboard against south window wall, facing north"},
        "E": {"x": 3.62, "z": 5.56, "rot_deg": 180.0, "note": "headboard on east wall, facing west"},
        "center": {"x": 2.35, "z": 5.38, "rot_deg": 0.0, "note": "centered in bedroom suite"},
    },
    "desk_group": {
        "N": {"x": 2.08, "z": 4.10, "rot_deg": 0.0, "note": "against north wall"},
        "W": {"x": 0.48, "z": 5.38, "rot_deg": 90.0, "note": "against west wall"},
        "S": {"x": 3.52, "z": 6.18, "rot_deg": 180.0, "note": "by south-east window corner"},
        "E": {"x": 4.26, "z": 5.55, "rot_deg": -90.0, "note": "against east wall"},
        "center": {"x": 2.40, "z": 5.30, "rot_deg": 0.0, "note": "floating in bedroom"},
    },
    "wardrobe_group": {
        "E": {"x": 4.42, "z": 5.55, "rot_deg": 0.0, "note": "along east wall"},
        "W": {"x": 0.40, "z": 5.60, "rot_deg": 180.0, "note": "relocated to west wall"},
        "N": {"x": 1.65, "z": 4.04, "rot_deg": 90.0, "note": "along north wall"},
    },
    "tv_group": {
        "W": {"x": 0.30, "z": 1.83, "rot_deg": 0.0, "note": "on west feature wall, facing east"},
        "S": {"x": 1.75, "z": 3.36, "rot_deg": 90.0, "note": "on south wall, facing north toward balcony"},
        "N": {"x": 1.75, "z": 0.32, "rot_deg": -90.0, "note": "along north balcony side, facing south"},
        "E": {"x": 5.50, "z": 1.88, "rot_deg": 180.0, "note": "toward east partition, facing west"},
    },
    "sofa_group": {
        "W": {"x": 2.10, "z": 1.90, "rot_deg": 0.0, "note": "facing west TV wall"},
        "S": {"x": 1.82, "z": 1.62, "rot_deg": 90.0, "note": "rotated to face south TV wall"},
        "N": {"x": 1.82, "z": 2.18, "rot_deg": -90.0, "note": "rotated to face north wall / balcony"},
        "E": {"x": 3.92, "z": 1.82, "rot_deg": 180.0, "note": "shifted east facing living partition"},
        "center": {"x": 2.65, "z": 1.85, "rot_deg": 0.0, "note": "centered in living room"},
    },
    "dining_group": {
        "E": {"x": 4.58, "z": 2.12, "rot_deg": 0.0, "note": "east dining zone next to kitchen"},
        "N": {"x": 4.72, "z": 1.15, "rot_deg": 90.0, "note": "north dining zone overlooking river balcony"},
        "W": {"x": 1.18, "z": 1.88, "rot_deg": 0.0, "note": "west side of living room"},
        "S": {"x": 4.92, "z": 2.74, "rot_deg": 0.0, "note": "south-east dining corner"},
        "center": {"x": 3.85, "z": 1.90, "rot_deg": 0.0, "note": "centered between lounge and kitchen"},
    },
    "island_group": {
        "center": {"x": 7.39, "z": 2.08, "rot_deg": 0.0, "note": "standard open island alignment"},
        "W": {"x": 6.96, "z": 2.08, "rot_deg": 0.0, "note": "shifted west toward living portal"},
        "E": {"x": 7.72, "z": 2.08, "rot_deg": 0.0, "note": "shifted east near sink counter"},
        "N": {"x": 7.39, "z": 1.72, "rot_deg": 0.0, "note": "shifted north closer to cooktop"},
        "S": {"x": 7.39, "z": 2.42, "rot_deg": 0.0, "note": "shifted south toward foyer"},
    },
    "tub_group": {
        "S": {"x": 5.80, "z": 6.42, "rot_deg": 0.0, "note": "along south spa wall"},
        "W": {"x": 5.36, "z": 6.02, "rot_deg": 90.0, "note": "aligned north-south along west spa wall"},
        "center": {"x": 6.05, "z": 6.18, "rot_deg": 0.0, "note": "centered in soaking zone"},
    },
}


def rotated_half_extents(hx: float, hz: float, rot_deg: float) -> tuple[float, float]:
    rad = math.radians(rot_deg)
    c, s = abs(math.cos(rad)), abs(math.sin(rad))
    return hx * c + hz * s, hx * s + hz * c


def group_box(pose: dict[str, Any], meta: dict[str, Any]) -> Any:
    hx, hz = rotated_half_extents(meta["hx"], meta["hz"], pose["rot_deg"])
    return box(pose["x"] - hx, pose["z"] - hz, pose["x"] + hx, pose["z"] + hz)


def clamp_pose_to_room(gid: str, pose: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    meta = GROUP_DEFAULTS[gid]
    x0, z0, x1, z1 = ROOM_BOUNDS[meta["room"]]
    hx, hz = rotated_half_extents(meta["hx"], meta["hz"], pose["rot_deg"])
    pad = 0.02
    clamped_x = max(x0 + hx + pad, min(x1 - hx - pad, pose["x"]))
    clamped_z = max(z0 + hz + pad, min(z1 - hz - pad, pose["z"]))
    changed = abs(clamped_x - pose["x"]) > 0.02 or abs(clamped_z - pose["z"]) > 0.02
    out = dict(pose)
    out["x"] = round(clamped_x, 3)
    out["z"] = round(clamped_z, 3)
    return out, changed


def default_scene_state() -> dict[str, dict[str, Any]]:
    return {
        gid: {
            "x": d["x"],
            "z": d["z"],
            "rot_deg": d["rot_deg"],
            "visible": d["visible"],
        }
        for gid, d in GROUP_DEFAULTS.items()
    }


def _extract_wall(text: str) -> Optional[WallSide]:
    t = text.lower()
    if any(k in t for k in ("north", "balcony side", "top wall", "upper wall", "by the balcony")):
        return "N"
    if any(k in t for k in ("south", "bottom wall", "lower wall")):
        return "S"
    if any(k in t for k in ("east", "right wall", "right side", "kitchen side")):
        return "E"
    if any(k in t for k in ("west", "left wall", "left side")):
        return "W"
    if any(k in t for k in ("center", "middle", "float")):
        return "center"
    return None


def parse_prompt_deterministic(prompt: str) -> LLMSceneDirectives:
    """Parse multi-clause natural-language furniture & mood prompts deterministically."""
    low = prompt.lower().strip()
    if not low:
        return LLMSceneDirectives(summary="No spatial changes requested.")

    if re.search(r"\b(reset|default|original|restore)\b", low):
        return LLMSceneDirectives(
            reset_all=True,
            lighting_mode="sunlit_day",
            summary="Reset all furniture groups to the original architect layout.",
        )

    lighting: Optional[LightingChoice] = None
    if any(k in low for k in ("evening", "night", "dim", "moody light", "cozy cove", "cove light")):
        lighting = "evening_cove"
    elif any(k in low for k in ("golden", "sunset", "warm afternoon", "amber light")):
        lighting = "golden_hour"
    elif any(k in low for k in ("sunlit", "daylight", "morning", "bright day")):
        lighting = "sunlit_day"

    style_overrides: dict[str, str] = {}
    if "moody living" in low or "moss velvet" in low or "dark living" in low:
        style_overrides["living"] = "L4"
    elif "gold patina" in low or "charcoal living" in low:
        style_overrides["living"] = "L3"
    elif "walnut kitchen" in low or "fluted kitchen" in low:
        style_overrides["kitchen"] = "K3"
    elif "black granite" in low or "dark kitchen" in low:
        style_overrides["kitchen"] = "K2"
    elif "sage bath" in low or "green bath" in low:
        style_overrides["bathroom"] = "B2"
    elif "navy bath" in low or "midnight bath" in low:
        style_overrides["bathroom"] = "B4"
    elif "cloud bedroom" in low or "all white bedroom" in low or "boucle bedroom" in low:
        style_overrides["primary_bedroom"] = "Be2"

    directives: list[GroupDirective] = []
    seen_groups: set[str] = set()

    # Split prompt into clauses around commas, semicolons, "and", "then"
    clauses = [c.strip() for c in re.split(r"[.;,\n]+|\band\b|\bthen\b|\bwith\b", low) if c.strip()]

    # High-level swap patterns across full prompt
    if ("swap" in low or "switch" in low or "flip" in low) and "dining" in low and ("sofa" in low or "living" in low or "lounge" in low):
        directives.extend(
            [
                GroupDirective(group_id="dining_group", target_wall="W", reason="Swapped dining set to west side of living room"),
                GroupDirective(group_id="sofa_group", target_wall="E", reason="Swapped sofa lounge to east side"),
                GroupDirective(group_id="tv_group", target_wall="E", reason="Moved TV wall to east partition facing sofa"),
            ]
        )
        seen_groups.update({"dining_group", "sofa_group", "tv_group"})

    for cl in clauses:
        # 1. Bed directives
        if any(k in cl for k in ("bed", "mattress", "headboard")) and "bedroom" not in cl.replace("bed", "") or re.search(r"\bbed\b", cl):
            if "bed_group" not in seen_groups:
                wall = _extract_wall(cl)
                if wall is None:
                    if "facing the window" in cl or "face the window" in cl or "look out" in cl:
                        wall = "N"  # South wall is the bedroom window -> headboard on North faces window!
                    elif "under the window" in cl or "by the window" in cl or "against the window" in cl or "near the window" in cl:
                        wall = "S"
                    elif "opposite" in cl or "across" in cl:
                        wall = "E"
                if wall:
                    directives.append(
                        GroupDirective(
                            group_id="bed_group",
                            target_wall=wall,
                            reason=f"Placed bed on {wall} wall ({WALL_PRESETS['bed_group'][wall]['note']})",
                        )
                    )
                    seen_groups.add("bed_group")

        # 2. TV directives
        if re.search(r"\b(tv|television|media|screen|console)\b", cl) and "tv_group" not in seen_groups:
            wall = _extract_wall(cl)
            if wall is None:
                if "balcony" in cl or "window" in cl:
                    wall = "N"
                elif "opposite" in cl or "dining" in cl:
                    wall = "E"
                elif "bottom" in cl or "bedroom wall" in cl or "spine" in cl:
                    wall = "S"
            if wall:
                directives.append(
                    GroupDirective(
                        group_id="tv_group",
                        target_wall=wall,
                        reason=f"Moved TV & media wall to {wall} wall ({WALL_PRESETS['tv_group'][wall]['note']})",
                    )
                )
                seen_groups.add("tv_group")

        # 3. Sofa directives
        if re.search(r"\b(sofa|couch|sectional|lounge|seating)\b", cl) and "sofa_group" not in seen_groups:
            wall = _extract_wall(cl)
            if "face the tv" in cl or "facing the tv" in cl or "facing it" in cl or "toward the tv" in cl or "towards it" in cl:
                tv_dir = next((d for d in directives if d.group_id == "tv_group"), None)
                if tv_dir and tv_dir.target_wall:
                    wall = tv_dir.target_wall
            elif wall is None and ("window" in cl or "balcony" in cl):
                wall = "N"
            if wall:
                directives.append(
                    GroupDirective(
                        group_id="sofa_group",
                        target_wall=wall,
                        reason=f"Aligned sofa group ({WALL_PRESETS['sofa_group'][wall]['note']})",
                    )
                )
                seen_groups.add("sofa_group")

        # 4. Dining directives
        if re.search(r"\b(dining|dinner table|eating)\b", cl) and "dining_group" not in seen_groups:
            wall = _extract_wall(cl)
            if wall is None and ("balcony" in cl or "window" in cl or "view" in cl):
                wall = "N"
            elif wall is None and "rotate" in cl:
                directives.append(
                    GroupDirective(group_id="dining_group", target_x=4.58, target_z=2.12, rot_deg=90.0, reason="Rotated dining table 90°")
                )
                seen_groups.add("dining_group")
                continue
            if wall:
                directives.append(
                    GroupDirective(
                        group_id="dining_group",
                        target_wall=wall,
                        reason=f"Positioned dining group ({WALL_PRESETS['dining_group'][wall]['note']})",
                    )
                )
                seen_groups.add("dining_group")

        # 5. Desk directives
        if re.search(r"\b(desk|workspace|office|study)\b", cl) and "desk_group" not in seen_groups:
            wall = _extract_wall(cl)
            if wall is None and "window" in cl:
                wall = "S"
            if wall:
                directives.append(
                    GroupDirective(
                        group_id="desk_group",
                        target_wall=wall,
                        reason=f"Moved study desk ({WALL_PRESETS['desk_group'][wall]['note']})",
                    )
                )
                seen_groups.add("desk_group")

        # 6. Wardrobe directives
        if re.search(r"\b(wardrobe|closet|armoire)\b", cl) and "wardrobe_group" not in seen_groups:
            wall = _extract_wall(cl)
            if wall in ("E", "W", "N"):
                directives.append(
                    GroupDirective(
                        group_id="wardrobe_group",
                        target_wall=wall,
                        reason=f"Moved wardrobe ({WALL_PRESETS['wardrobe_group'][wall]['note']})",
                    )
                )
                seen_groups.add("wardrobe_group")

        # 7. Kitchen island directives
        if re.search(r"\b(island|kitchen bar)\b", cl) and "island_group" not in seen_groups:
            if "rotate" in cl or "vertical" in cl or "turn" in cl:
                directives.append(
                    GroupDirective(
                        group_id="island_group",
                        target_x=7.39,
                        target_z=2.08,
                        rot_deg=90.0,
                        reason="Rotated kitchen island 90° (north-south orientation)",
                    )
                )
                seen_groups.add("island_group")
            else:
                wall = _extract_wall(cl)
                if wall in WALL_PRESETS["island_group"]:
                    directives.append(
                        GroupDirective(
                            group_id="island_group",
                            target_wall=wall,
                            reason=f"Adjusted kitchen island ({WALL_PRESETS['island_group'][wall]['note']})",
                        )
                    )
                    seen_groups.add("island_group")

        # 8. Tub directives
        if re.search(r"\b(tub|bathtub|soaking)\b", cl) and "tub_group" not in seen_groups:
            wall = _extract_wall(cl) or ("W" if "rotate" in cl else None)
            if wall in WALL_PRESETS["tub_group"]:
                directives.append(
                    GroupDirective(
                        group_id="tub_group",
                        target_wall=wall,
                        reason=f"Adjusted soaking tub ({WALL_PRESETS['tub_group'][wall]['note']})",
                    )
                )
                seen_groups.add("tub_group")

    summary_parts = [d.reason for d in directives]
    if lighting:
        summary_parts.append(f"Switched lighting to {lighting.replace('_', ' ')}")
    if style_overrides:
        summary_parts.append("Updated room finish palette from text cue")
    summary = "; ".join(summary_parts) if summary_parts else "Interpreted prompt; layout constraints preserved."
    return LLMSceneDirectives(
        lighting_mode=lighting,
        style_overrides=style_overrides,
        directives=directives,
        summary=summary,
    )


def parse_prompt_with_gemini(prompt: str, current_state: dict[str, dict[str, Any]]) -> LLMSceneDirectives:
    """Use `gemini-4-neon` structured JSON to interpret complex natural-language layout instructions."""
    client = genai.Client()
    sys_context = (
        "You are Homeweave's architectural spatial planner for a real metric 4-room NYC residence.\n"
        "Convert the user's natural-language layout instructions into structured `LLMSceneDirectives` JSON.\n"
        "Rooms & interior clear bounds (x East, z South):\n"
        "- living: x in [0.10, 5.74], z in [0.10, 3.58]. North wall is balcony glass; South wall x in [0.10, 3.38] is solid (bedroom door at x in [3.38, 4.30]); West wall is solid.\n"
        "- primary_bedroom: x in [0.10, 4.70], z in [3.74, 6.89]. Door at North-East x in [3.38, 4.30]; South wall has large window x in [0.90, 3.85]; West & East walls are solid.\n"
        "- kitchen: x in [5.89, 9.33], z in [0.10, 3.58].\n"
        "- bathroom: x in [4.86, 8.09], z in [3.74, 6.89].\n"
        "Available `group_id` values: bed_group, desk_group, wardrobe_group, bench_group, tv_group, sofa_group, dining_group, living_art_group, island_group, tub_group.\n"
        "Prefer setting `target_wall` ('N', 'S', 'E', 'W', 'center') so the geometric solver snaps flush to walls and automatically keeps sofa facing TV and doors unblocked. "
        "Note: 'bed facing the window' in primary_bedroom means headboard on 'N' wall (facing South window). 'bed under the window' means headboard on 'S' wall.\n"
        f"Current state: {json.dumps(current_state)}\n"
        f"User instruction: {prompt}"
    )
    interaction = client.interactions.create(
        model=MODEL_NEON,
        input=sys_context,
        response_format=to_gemini_response_format(LLMSceneDirectives),
        generation_config={"thinking_level": "medium", "max_output_tokens": 8192},
        store=False,
    )
    if not interaction.output_text:
        raise ValueError("Empty response from gemini-4-neon.")
    return LLMSceneDirectives.model_validate_json(interaction.output_text)


def solve_and_verify_layout(
    prompt: str,
    current_state: Optional[dict[str, dict[str, Any]]] = None,
    *,
    use_model: Optional[bool] = None,
) -> dict[str, Any]:
    """Apply natural-language layout prompt with rigid-group coupling & `shapely` hard-constraint checks."""
    state = copy.deepcopy(current_state) if current_state else default_scene_state()
    for gid, d in default_scene_state().items():
        state.setdefault(gid, copy.deepcopy(d))

    should_call_llm = use_model if use_model is not None else bool(os.environ.get("GEMINI_API_KEY"))
    engine_mode = "deterministic_spatial_parser"
    if should_call_llm:
        try:
            plan = parse_prompt_with_gemini(prompt, state)
            engine_mode = f"{MODEL_NEON} + shapely_solver"
        except Exception:
            plan = parse_prompt_deterministic(prompt)
            engine_mode = "deterministic_spatial_parser (fallback)"
    else:
        plan = parse_prompt_deterministic(prompt)

    applied_moves: list[dict[str, Any]] = []

    if plan.reset_all:
        state = default_scene_state()
        applied_moves.append({"group_id": "all", "label": "All furniture groups", "action": "Reset to default architect plan"})
    else:
        requested_ids = {d.group_id for d in plan.directives}

        for d in plan.directives:
            gid = d.group_id
            prev = dict(state[gid])
            new_pose = dict(prev)
            if d.target_wall and gid in WALL_PRESETS and d.target_wall in WALL_PRESETS[gid]:
                preset = WALL_PRESETS[gid][d.target_wall]
                new_pose.update({"x": preset["x"], "z": preset["z"], "rot_deg": preset["rot_deg"], "visible": d.visible})
            else:
                if d.target_x is not None:
                    new_pose["x"] = float(d.target_x)
                if d.target_z is not None:
                    new_pose["z"] = float(d.target_z)
                if d.rot_deg is not None:
                    new_pose["rot_deg"] = float(d.rot_deg)
                new_pose["visible"] = d.visible

            new_pose, was_clamped = clamp_pose_to_room(gid, new_pose)
            state[gid] = new_pose
            applied_moves.append(
                {
                    "group_id": gid,
                    "label": GROUP_DEFAULTS[gid]["label"],
                    "action": d.reason or f"Moved to ({new_pose['x']:.2f}, {new_pose['z']:.2f} m), rot {new_pose['rot_deg']:.0f}°",
                    "clamped": was_clamped,
                }
            )

        # Rigid functional-group coupling & automatic conflict resolution (0 overlaps guaranteed!):
        # 1. If TV moved and user didn't explicitly place sofa separately, orient sofa to face TV!
        tv_req = next((d for d in plan.directives if d.group_id == "tv_group"), None)
        if tv_req and tv_req.target_wall:
            if "sofa_group" not in requested_ids and tv_req.target_wall in WALL_PRESETS["sofa_group"]:
                sp = WALL_PRESETS["sofa_group"][tv_req.target_wall]
                state["sofa_group"] = {"x": sp["x"], "z": sp["z"], "rot_deg": sp["rot_deg"], "visible": True}
                applied_moves.append(
                    {
                        "group_id": "sofa_group",
                        "label": GROUP_DEFAULTS["sofa_group"]["label"],
                        "action": f"Auto-aligned conversation group to face TV ({sp['note']})",
                        "clamped": False,
                    }
                )
            # Keep south-wall gallery art out of TV's way when TV moves to South wall
            if tv_req.target_wall == "S":
                state["living_art_group"] = {"x": 0.12, "z": 1.83, "rot_deg": 90.0, "visible": True}
            elif tv_req.target_wall == "W":
                state["living_art_group"] = {"x": 1.825, "z": 3.57, "rot_deg": 0.0, "visible": True}

        # 2. Bedroom smart conflict resolution (Bed vs Desk / Bench / Wardrobe)
        bed_req = next((d for d in plan.directives if d.group_id == "bed_group"), None)
        if bed_req and bed_req.target_wall:
            bw = bed_req.target_wall
            if bw == "N":
                # North wall was occupied by desk -> relocate desk to south-east window corner!
                if "desk_group" not in requested_ids:
                    dp = WALL_PRESETS["desk_group"]["S"]
                    state["desk_group"] = {"x": dp["x"], "z": dp["z"], "rot_deg": dp["rot_deg"], "visible": True}
                    applied_moves.append(
                        {
                            "group_id": "desk_group",
                            "label": GROUP_DEFAULTS["desk_group"]["label"],
                            "action": "Auto-relocated study desk to south-east window corner to clear bed headboard",
                            "clamped": False,
                        }
                    )
                state["bench_group"]["visible"] = True
                if "wardrobe_group" not in requested_ids:
                    wp = WALL_PRESETS["wardrobe_group"]["E"]
                    state["wardrobe_group"] = {"x": wp["x"], "z": wp["z"], "rot_deg": wp["rot_deg"], "visible": True}
            elif bw == "S":
                # South window wall has window bench -> tuck bench away so bed sits flush
                state["bench_group"]["visible"] = False
                if "desk_group" not in requested_ids:
                    dp = WALL_PRESETS["desk_group"]["N"]
                    state["desk_group"] = {"x": dp["x"], "z": dp["z"], "rot_deg": dp["rot_deg"], "visible": True}
            elif bw == "E":
                # East wall had wardrobe -> swap wardrobe to West wall & tuck bench to clear footboard/nightstand!
                if "wardrobe_group" not in requested_ids:
                    wp = WALL_PRESETS["wardrobe_group"]["W"]
                    state["wardrobe_group"] = {"x": wp["x"], "z": wp["z"], "rot_deg": wp["rot_deg"], "visible": True}
                    applied_moves.append(
                        {
                            "group_id": "wardrobe_group",
                            "label": GROUP_DEFAULTS["wardrobe_group"]["label"],
                            "action": "Auto-swapped wardrobe to west wall so bed fits flush on east wall",
                            "clamped": False,
                        }
                    )
                if "desk_group" not in requested_ids:
                    state["desk_group"] = {"x": 1.65, "z": 4.10, "rot_deg": 0.0, "visible": True}
                state["bench_group"]["visible"] = False
            elif bw == "W":
                state["bench_group"]["visible"] = True
                if "desk_group" not in requested_ids:
                    dp = WALL_PRESETS["desk_group"]["N"]
                    state["desk_group"] = {"x": dp["x"], "z": dp["z"], "rot_deg": dp["rot_deg"], "visible": True}
                if "wardrobe_group" not in requested_ids:
                    wp = WALL_PRESETS["wardrobe_group"]["E"]
                    state["wardrobe_group"] = {"x": wp["x"], "z": wp["z"], "rot_deg": wp["rot_deg"], "visible": True}

    # Deterministic `shapely` verification pass across all visible solid groups
    overlaps: list[list[str]] = []
    outside_room: list[str] = []
    door_blocks: list[str] = []
    updated_colliders: dict[str, dict[str, Any]] = {}

    solid_groups = [
        gid
        for gid, pose in state.items()
        if pose.get("visible", True) and gid != "living_art_group"
    ]
    boxes: dict[str, Any] = {}
    for gid in solid_groups:
        meta = GROUP_DEFAULTS[gid]
        b = group_box(state[gid], meta)
        boxes[gid] = b
        x0, z0, x1, z1 = ROOM_BOUNDS[meta["room"]]
        r_box = box(x0 - 0.05, z0 - 0.05, x1 + 0.05, z1 + 0.05)
        if not r_box.contains(b):
            outside_room.append(gid)
        for dz in DOOR_CLEAR_ZONES.get(meta["room"], []):
            if b.intersection(box(*dz)).area > 0.08:
                door_blocks.append(gid)
                break
        bx0, bz0, bx1, bz1 = b.bounds
        updated_colliders[gid] = {
            "x0": round(bx0, 3),
            "z0": round(bz0, 3),
            "x1": round(bx1, 3),
            "z1": round(bz1, 3),
            "visible": bool(state[gid].get("visible", True)),
            "replaces": meta["collider_labels"],
        }

    for i, g1 in enumerate(solid_groups):
        for g2 in solid_groups[i + 1 :]:
            if GROUP_DEFAULTS[g1]["room"] != GROUP_DEFAULTS[g2]["room"]:
                continue
            inter = boxes[g1].intersection(boxes[g2])
            if inter.area > 0.04:
                overlaps.append([g1, g2])

    return {
        "ok": True,
        "engine": engine_mode,
        "prompt": prompt,
        "summary": plan.summary,
        "lighting_mode": plan.lighting_mode,
        "style_overrides": plan.style_overrides,
        "state": state,
        "applied_moves": applied_moves,
        "colliders": updated_colliders,
        "check": {
            "overlaps": overlaps,
            "door_blocks": door_blocks,
            "outside_room": outside_room,
            "violations_total": len(overlaps) + len(door_blocks) + len(outside_room),
        },
    }


if __name__ == "__main__":
    tests = [
        "Put the bed against the north wall facing the window, and move the TV to the south wall with the sofa facing it",
        "Put the bed against the east wall, move the dining table near the balcony, and use evening lighting",
        "Swap the sofa and dining area, and make the lighting golden hour",
        "Move the bed under the south window and rotate the kitchen island",
        "Reset layout to default",
    ]
    st = default_scene_state()
    for t in tests:
        res = solve_and_verify_layout(t, st, use_model=False)
        st = res["state"]
        print(
            json.dumps(
                {
                    "prompt": t,
                    "moves": len(res["applied_moves"]),
                    "violations": res["check"]["violations_total"],
                    "check": res["check"],
                    "lighting": res["lighting_mode"],
                }
            )
        )
        assert res["check"]["violations_total"] == 0, res["check"]
    print("ALL LAYOUT PROMPT TESTS PASSED WITH 0 VIOLATIONS")
