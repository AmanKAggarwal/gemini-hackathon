"""Natural-language text-to-3D spatial layout engine for Homeweave (`layout_prompt.py`).

Translates user text instructions—including cross-room moves ("put the TV in the bedroom and
in the living room, NO TV"), wall placements ("put the bed on the north wall facing the window"),
removals ("remove the desk"), swaps ("swap the sofa and dining area"), and mood/lighting cues—into
rigid furniture-group transforms `(x, z, rot_deg, visible, room)` for the Blender 5.2 photorealistic
GLB residence (`walkthrough/assets/dream_apartment.glb`), verified by deterministic `shapely`
containment, doorway-clearance, and zero-overlap checks.

Uses `gemini-4-neon` (`client.interactions.create` + structured Pydantic JSON) when
`GEMINI_API_KEY` is set, with a comprehensive multi-clause spatial parser fallback offline.
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
    "tv_niches_group",
    "sofa_group",
    "dining_group",
    "living_art_group",
    "island_group",
    "tub_group",
]
RoomKey = Literal["living", "kitchen", "primary_bedroom", "bathroom", "balcony"]
WallSide = Literal["N", "S", "E", "W", "center"]
LightingChoice = Literal["sunlit_day", "golden_hour", "evening_cove"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GroupDirective(StrictModel):
    group_id: GroupId
    target_room: Optional[RoomKey] = None
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
ROOM_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "living": (0.10, 0.10, 5.74, 3.58),
    "kitchen": (5.89, 0.10, 9.33, 3.58),
    "primary_bedroom": (0.10, 3.74, 4.70, 6.89),
    "bathroom": (4.86, 3.74, 8.09, 6.89),
    "balcony": (0.42, -1.58, 5.38, -0.10),
}

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
    "balcony": [],
}

# Canonical default poses & unrotated half-extents (hx, hz) for each movable GLB furniture assembly
GROUP_DEFAULTS: dict[str, dict[str, Any]] = {
    "bed_group": {
        "label": "Primary Queen Bed & Nightstands",
        "room": "primary_bedroom",
        "x": 1.16,
        "z": 5.38,
        "rot_deg": 0.0,
        "hx": 1.04,
        "hz": 0.96,
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
        "hx": 0.20,  # depth along x (faces +x East at rot=0)
        "hz": 1.15,  # 2.30 m clean TV slab + console width (fits bedroom & living walls cleanly!)
        "visible": True,
        "collider_labels": ["Media_Console_Body"],
    },
    "tv_niches_group": {
        "label": "Living Wall Walnut Display Niches",
        "room": "living",
        "x": 0.30,
        "z": 1.83,
        "rot_deg": 0.0,
        "hx": 0.14,
        "hz": 1.62,
        "visible": True,
        "collider_labels": [],
    },
    "sofa_group": {
        "label": "Bouclé Sofa, Coffee Table & Lounge Rug",
        "room": "living",
        "x": 2.10,
        "z": 1.90,
        "rot_deg": 0.0,
        "hx": 1.08,
        "hz": 1.10,
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

# Same-room wall presets
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

# Cross-room placement anchors: CROSS_ROOM_PRESETS[group_id][target_room] -> wall_map or default pose
CROSS_ROOM_PRESETS: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
    "tv_group": {
        "primary_bedroom": {
            "E": {"x": 4.46, "z": 5.70, "rot_deg": 180.0, "note": "mounted on bedroom east wall facing the bed"},
            "W": {"x": 0.32, "z": 5.38, "rot_deg": 0.0, "note": "mounted on bedroom west wall facing east"},
            "N": {"x": 1.70, "z": 3.96, "rot_deg": -90.0, "note": "mounted on bedroom north wall facing south"},
            "S": {"x": 1.75, "z": 6.66, "rot_deg": 90.0, "note": "placed along bedroom south wall facing north"},
        },
        "kitchen": {
            "S": {"x": 8.05, "z": 3.36, "rot_deg": 90.0, "note": "mounted on kitchen/foyer south wall"},
        },
    },
    "desk_group": {
        "living": {
            "W": {"x": 0.52, "z": 1.85, "rot_deg": 90.0, "note": "placed along living room west wall"},
            "N": {"x": 4.45, "z": 0.55, "rot_deg": 0.0, "note": "placed by the living room balcony glass"},
            "S": {"x": 2.40, "z": 3.16, "rot_deg": 180.0, "note": "placed along living room south wall"},
            "E": {"x": 4.85, "z": 2.10, "rot_deg": -90.0, "note": "placed in east living corner"},
        },
        "balcony": {
            "center": {"x": 2.80, "z": -0.82, "rot_deg": 0.0, "note": "placed outdoors on the riverfront balcony"},
        },
    },
    "bed_group": {
        "living": {
            "W": {"x": 1.18, "z": 1.85, "rot_deg": 0.0, "note": "moved bed into living room against west wall"},
            "S": {"x": 1.75, "z": 2.48, "rot_deg": 90.0, "note": "moved bed into living room against south wall facing balcony"},
            "N": {"x": 1.75, "z": 1.20, "rot_deg": -90.0, "note": "moved bed into living room near balcony glass"},
        },
    },
    "sofa_group": {
        "primary_bedroom": {
            "E": {"x": 3.55, "z": 5.55, "rot_deg": 0.0, "note": "moved sofa lounge into bedroom"},
            "W": {"x": 1.25, "z": 5.40, "rot_deg": 180.0, "note": "moved sofa lounge to bedroom west side"},
        },
    },
    "dining_group": {
        "primary_bedroom": {
            "center": {"x": 3.25, "z": 5.45, "rot_deg": 0.0, "note": "moved dining table into bedroom"},
        },
        "balcony": {
            "center": {"x": 2.85, "z": -0.86, "rot_deg": 90.0, "note": "moved dining set out onto riverfront balcony"},
        },
    },
    "wardrobe_group": {
        "living": {
            "W": {"x": 0.40, "z": 1.85, "rot_deg": 180.0, "note": "moved wardrobe to living room west wall"},
            "S": {"x": 1.80, "z": 3.26, "rot_deg": -90.0, "note": "moved wardrobe to living room south wall"},
        },
    },
    "tub_group": {
        "balcony": {
            "center": {"x": 2.85, "z": -0.88, "rot_deg": 0.0, "note": "moved soaking tub onto the riverfront balcony"},
        },
        "primary_bedroom": {
            "S": {"x": 2.35, "z": 6.15, "rot_deg": 0.0, "note": "moved soaking tub into primary bedroom suite"},
        },
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
    room = pose.get("room") or meta["room"]
    x0, z0, x1, z1 = ROOM_BOUNDS[room]
    hx, hz = rotated_half_extents(meta["hx"], meta["hz"], pose["rot_deg"])
    pad = 0.02
    lo_x, hi_x = x0 + hx + pad, x1 - hx - pad
    lo_z, hi_z = z0 + hz + pad, z1 - hz - pad
    clamped_x = 0.5 * (x0 + x1) if lo_x > hi_x else max(lo_x, min(hi_x, pose["x"]))
    clamped_z = 0.5 * (z0 + z1) if lo_z > hi_z else max(lo_z, min(hi_z, pose["z"]))
    changed = abs(clamped_x - pose["x"]) > 0.02 or abs(clamped_z - pose["z"]) > 0.02
    out = dict(pose)
    out["room"] = room
    out["x"] = round(clamped_x, 3)
    out["z"] = round(clamped_z, 3)
    return out, changed


def default_scene_state() -> dict[str, dict[str, Any]]:
    return {
        gid: {
            "room": d["room"],
            "x": d["x"],
            "z": d["z"],
            "rot_deg": d["rot_deg"],
            "visible": d["visible"],
        }
        for gid, d in GROUP_DEFAULTS.items()
    }


def _extract_wall(text: str) -> Optional[WallSide]:
    t = text.lower()
    if any(k in t for k in ("north", "top wall", "upper wall")):
        return "N"
    if any(k in t for k in ("south", "bottom wall", "lower wall")):
        return "S"
    if any(k in t for k in ("east", "right wall", "right side")):
        return "E"
    if any(k in t for k in ("west", "left wall", "left side")):
        return "W"
    if any(k in t for k in ("center", "middle", "float")):
        return "center"
    return None


def _extract_room(text: str) -> Optional[RoomKey]:
    t = text.lower()
    if any(k in t for k in ("bedroom", "master bed", "primary bed", "sleeping room")):
        return "primary_bedroom"
    if any(k in t for k in ("living room", "living area", "lounge room", "living")):
        return "living"
    if any(k in t for k in ("kitchen", "foyer")):
        return "kitchen"
    if any(k in t for k in ("bathroom", "spa bath", "bath room")):
        return "bathroom"
    if any(k in t for k in ("on the balcony", "to the balcony", "onto the balcony", "outside on balcony")):
        return "balcony"
    return None


def _is_removal_clause(clause: str, item_regex: str) -> bool:
    t = clause.lower()
    if re.search(rf"\b(no|remove|delete|hide|drop|without|get rid of|take out|clear)\b[^.;]*\b({item_regex})\b", t):
        return True
    if re.search(rf"\b({item_regex})\b[^.;]*\b(gone|removed|hidden|deleted|off)\b", t):
        return True
    return False


def _is_restore_clause(clause: str, item_regex: str) -> bool:
    t = clause.lower()
    return bool(re.search(rf"\b(bring back|restore|add back|show|put back)\b[^.;]*\b({item_regex})\b", t))


ITEM_SPECS: list[tuple[GroupId, str]] = [
    ("tv_group", r"tv|television|media console|media wall|screen"),
    ("bed_group", r"bed(?!room)|mattress|headboard"),
    ("sofa_group", r"sofa|couch|sectional|lounge|seating"),
    ("dining_group", r"dining table|dining set|dining area|dining|dinner table"),
    ("desk_group", r"desk|workspace|study table|office"),
    ("wardrobe_group", r"wardrobe|closet|armoire"),
    ("bench_group", r"window bench|bench"),
    ("island_group", r"kitchen island|island|bar stools"),
    ("tub_group", r"soaking tub|bathtub|tub"),
]


def parse_prompt_deterministic(prompt: str, current_state: dict[str, dict[str, Any]]) -> LLMSceneDirectives:
    """Parse multi-clause natural-language furniture, cross-room, removal & mood prompts."""
    low = prompt.lower().strip()
    if not low:
        return LLMSceneDirectives(summary="Enter a spatial instruction to move or restyle furniture.")

    if re.search(r"\b(reset|default|original)\b", low) and not any(
        re.search(rf"\b({pat})\b", low) for _, pat in ITEM_SPECS
    ):
        return LLMSceneDirectives(
            reset_all=True,
            lighting_mode="sunlit_day",
            summary="Reset all furniture groups to the original architect layout.",
        )

    lighting: Optional[LightingChoice] = None
    if any(k in low for k in ("evening", "night", "dim light", "moody light", "cozy cove", "cove light")):
        lighting = "evening_cove"
    elif any(k in low for k in ("golden", "sunset", "warm afternoon", "amber light")):
        lighting = "golden_hour"
    elif any(k in low for k in ("sunlit", "daylight", "morning light", "bright day")):
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
    moved_groups: set[str] = set()
    removed_groups: set[str] = set()

    # High-level swap patterns across full prompt
    if ("swap" in low or "switch" in low or "flip" in low) and "dining" in low and ("sofa" in low or "living" in low or "lounge" in low):
        directives.extend(
            [
                GroupDirective(group_id="dining_group", target_room="living", target_wall="W", reason="Swapped dining set to west side of living room"),
                GroupDirective(group_id="sofa_group", target_room="living", target_wall="E", reason="Swapped sofa lounge to east side"),
                GroupDirective(group_id="tv_group", target_room="living", target_wall="E", reason="Moved TV wall to east partition facing sofa"),
            ]
        )
        moved_groups.update({"dining_group", "sofa_group", "tv_group"})

    # Split into clauses on sentence boundaries, semicolons, "and", "then"
    # Keep commas inside a clause when the comma only separates room prefix + "NO TV" (e.g. "in the living room, NO TV")
    raw_clauses = [c.strip() for c in re.split(r"[.;\n]+|\band\b|\bthen\b|\bbut\b", low) if c.strip()]
    clauses: list[str] = []
    for rc in raw_clauses:
        # Split on comma ONLY if both sides mention an item or a verb
        parts = [p.strip() for p in rc.split(",") if p.strip()]
        if len(parts) > 1 and all(
            any(re.search(rf"\b({pat})\b", p) for _, pat in ITEM_SPECS) or any(w in p for w in ("put", "move", "place", "rotate", "use", "make", "swap"))
            for p in parts
        ):
            clauses.extend(parts)
        else:
            clauses.append(rc)

    # First pass: positive placements, moves, rotations, restores
    for cl in clauses:
        for gid, pat in ITEM_SPECS:
            if not re.search(rf"\b({pat})\b", cl):
                continue
            if _is_removal_clause(cl, pat):
                continue  # handled in pass 2 so moves win over "in room B, no X"
            if gid in moved_groups:
                continue

            if _is_restore_clause(cl, pat):
                dflt = GROUP_DEFAULTS[gid]
                directives.append(
                    GroupDirective(
                        group_id=gid,
                        target_room=dflt["room"],
                        target_x=dflt["x"],
                        target_z=dflt["z"],
                        rot_deg=dflt["rot_deg"],
                        visible=True,
                        reason=f"Restored {dflt['label']}",
                    )
                )
                moved_groups.add(gid)
                continue

            cur_room = current_state.get(gid, {}).get("room", GROUP_DEFAULTS[gid]["room"])
            target_room = _extract_room(cl)
            wall = _extract_wall(cl)

            # Smart inference when user says "facing the window", "near the balcony", "rotate", etc.
            if gid == "bed_group":
                if wall is None:
                    if "facing the window" in cl or "face the window" in cl or "look out" in cl:
                        wall = "N"
                    elif "under the window" in cl or "by the window" in cl or "against the window" in cl or "near the window" in cl:
                        wall = "S"
                    elif "opposite" in cl or "across" in cl:
                        wall = "E"
            elif gid == "tv_group":
                if wall is None and target_room is None:
                    if "balcony" in cl or "window" in cl:
                        wall = "N"
                    elif "opposite" in cl or "dining" in cl:
                        wall = "E"
                    elif "spine" in cl or "bedroom wall" in cl:
                        wall = "S"
            elif gid == "sofa_group":
                if "face the tv" in cl or "facing the tv" in cl or "facing it" in cl or "toward the tv" in cl or "towards it" in cl:
                    tv_dir = next((d for d in directives if d.group_id == "tv_group"), None)
                    if tv_dir and tv_dir.target_wall:
                        wall = tv_dir.target_wall
                        target_room = tv_dir.target_room or target_room
                elif wall is None and ("window" in cl or "balcony" in cl):
                    wall = "N"
            elif gid == "dining_group":
                if wall is None and target_room is None:
                    if "balcony" in cl or "window" in cl or "view" in cl:
                        wall = "N"
                    elif "rotate" in cl or "turn" in cl:
                        directives.append(
                            GroupDirective(group_id=gid, target_room="living", target_x=4.58, target_z=2.12, rot_deg=90.0, visible=True, reason="Rotated dining table 90°")
                        )
                        moved_groups.add(gid)
                        continue
            elif gid == "desk_group":
                if wall is None and "window" in cl:
                    wall = "S" if (target_room in (None, "primary_bedroom")) else "N"
            elif gid == "island_group":
                if "rotate" in cl or "vertical" in cl or "turn" in cl:
                    directives.append(
                        GroupDirective(group_id=gid, target_room="kitchen", target_x=7.39, target_z=2.08, rot_deg=90.0, visible=True, reason="Rotated kitchen island 90°")
                    )
                    moved_groups.add(gid)
                    continue
            elif gid == "tub_group":
                if wall is None and "rotate" in cl:
                    wall = "W"

            # Apply cross-room or same-room directive!
            if target_room and target_room != GROUP_DEFAULTS[gid]["room"]:
                directives.append(
                    GroupDirective(
                        group_id=gid,
                        target_room=target_room,
                        target_wall=wall,
                        visible=True,
                        reason=f"Moved {GROUP_DEFAULTS[gid]['label']} into {target_room.replace('_', ' ')}",
                    )
                )
                moved_groups.add(gid)
            elif target_room and target_room == GROUP_DEFAULTS[gid]["room"] and (target_room != cur_room or wall is not None or "put" in cl or "move" in cl):
                directives.append(
                    GroupDirective(
                        group_id=gid,
                        target_room=target_room,
                        target_wall=wall,
                        visible=True,
                        reason=f"Placed {GROUP_DEFAULTS[gid]['label']} in {target_room.replace('_', ' ')}",
                    )
                )
                moved_groups.add(gid)
            elif wall is not None:
                directives.append(
                    GroupDirective(
                        group_id=gid,
                        target_room=cur_room,
                        target_wall=wall,
                        visible=True,
                        reason=f"Moved {GROUP_DEFAULTS[gid]['label']} to {wall} wall",
                    )
                )
                moved_groups.add(gid)

    # Second pass: removal / "NO X" clauses!
    for cl in clauses:
        for gid, pat in ITEM_SPECS:
            if not re.search(rf"\b({pat})\b", cl):
                continue
            if not _is_removal_clause(cl, pat):
                continue
            rm_room = _extract_room(cl)
            # If user ALSO moved this exact item to ANOTHER room in the same prompt
            # (e.g., "put the TV in the bedroom and in the living room, NO TV"),
            # keep the move in the destination room and just note it's cleared from the old room!
            if gid in moved_groups:
                moved_dir = next((d for d in directives if d.group_id == gid), None)
                if moved_dir and (rm_room is None or moved_dir.target_room != rm_room):
                    moved_dir.reason += f" (removed from {(rm_room or GROUP_DEFAULTS[gid]['room']).replace('_', ' ')})"
                    continue
            if gid not in removed_groups:
                directives.append(
                    GroupDirective(
                        group_id=gid,
                        visible=False,
                        reason=f"Removed {GROUP_DEFAULTS[gid]['label']}" + (f" from {rm_room.replace('_', ' ')}" if rm_room else ""),
                    )
                )
                removed_groups.add(gid)

    summary_parts = [d.reason for d in directives if d.reason]
    if lighting:
        summary_parts.append(f"Switched lighting to {lighting.replace('_', ' ')}")
    if style_overrides:
        summary_parts.append("Updated room finish palette from text cue")
    summary = (
        "; ".join(summary_parts)
        if summary_parts
        else "Could not match a furniture move — try e.g. 'Put the TV in the bedroom', 'No TV in living room', 'Put bed on north wall', or 'Swap sofa and dining'."
    )
    return LLMSceneDirectives(
        lighting_mode=lighting,
        style_overrides=style_overrides,
        directives=directives,
        summary=summary,
    )


def parse_prompt_with_gemini(prompt: str, current_state: dict[str, dict[str, Any]]) -> LLMSceneDirectives:
    """Use `gemini-4-neon` structured JSON to interpret arbitrary natural-language layout instructions."""
    client = genai.Client()
    sys_context = (
        "You are Homeweave's architectural spatial planner for a real metric 4-room + balcony NYC residence.\n"
        "Convert the user's natural-language layout instructions into structured `LLMSceneDirectives` JSON.\n"
        "Rooms & interior clear bounds (x East, z South):\n"
        "- living: x in [0.10, 5.74], z in [0.10, 3.58]. North wall is balcony glass; South wall x in [0.10, 3.38] is solid; West wall is solid.\n"
        "- primary_bedroom: x in [0.10, 4.70], z in [3.74, 6.89]. Door at North-East x in [3.38, 4.22]; South wall has large window; West & East walls are solid.\n"
        "- kitchen: x in [5.89, 9.33], z in [0.10, 3.58].\n"
        "- bathroom: x in [4.86, 8.09], z in [3.74, 6.89].\n"
        "- balcony: x in [0.42, 5.38], z in [-1.58, -0.10].\n"
        "Available `group_id` values: bed_group, desk_group, wardrobe_group, bench_group, tv_group, sofa_group, dining_group, living_art_group, island_group, tub_group.\n"
        "Rules:\n"
        "1. Cross-room moves ARE supported: set `target_room` (e.g., put `tv_group` in `primary_bedroom` and `visible=True`).\n"
        "2. Removals ARE supported: set `visible=False` if the user wants an item removed completely. If the user says 'put TV in bedroom and no TV in living room', emit ONE directive for `tv_group` with `target_room='primary_bedroom'` and `visible=True` (moving it clears the living room!).\n"
        "3. Prefer setting `target_room` + `target_wall` ('N','S','E','W','center') so our deterministic `shapely` solver snaps cleanly with 0 overlaps.\n"
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


def _pick_cross_room_pose(gid: str, target_room: str, wall: Optional[str], state: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Choose the smartest collision-free wall anchor when moving `gid` into `target_room`."""
    # 1. If returning to default room, use WALL_PRESETS
    if target_room == GROUP_DEFAULTS[gid]["room"]:
        wmap = WALL_PRESETS.get(gid, {})
        if wall and wall in wmap:
            p = wmap[wall]
            return {"room": target_room, "x": p["x"], "z": p["z"], "rot_deg": p["rot_deg"], "visible": True}
        d = GROUP_DEFAULTS[gid]
        return {"room": target_room, "x": d["x"], "z": d["z"], "rot_deg": d["rot_deg"], "visible": True}

    # 2. Smart contextual wall choice for TV moved into Primary Bedroom: mount OPPOSITE the bed!
    if gid == "tv_group" and target_room == "primary_bedroom" and wall is None:
        bed_rot = float(state.get("bed_group", {}).get("rot_deg", 0.0)) % 360.0
        # Bed at rot 0 -> headboard W, facing E -> put TV on E wall!
        # Bed at rot 180 -> headboard E, facing W -> put TV on W wall!
        # Bed at rot 270 (-90) -> headboard N, facing S -> put TV on S wall!
        # Bed at rot 90 -> headboard S, facing N -> put TV on N wall!
        if abs(bed_rot - 180.0) < 20:
            wall = "W"
        elif abs(bed_rot - 270.0) < 20:
            wall = "S"
        elif abs(bed_rot - 90.0) < 20:
            wall = "N"
        else:
            wall = "E"

    rmap = CROSS_ROOM_PRESETS.get(gid, {}).get(target_room, {})
    if rmap:
        if wall and wall in rmap:
            p = rmap[wall]
            return {"room": target_room, "x": p["x"], "z": p["z"], "rot_deg": p["rot_deg"], "visible": True}
        first = next(iter(rmap.values()))
        return {"room": target_room, "x": first["x"], "z": first["z"], "rot_deg": first["rot_deg"], "visible": True}

    # Generic room center fallback
    x0, z0, x1, z1 = ROOM_BOUNDS[target_room]
    return {"room": target_room, "x": round(0.5 * (x0 + x1), 2), "z": round(0.5 * (z0 + z1), 2), "rot_deg": 0.0, "visible": True}


def solve_and_verify_layout(
    prompt: str,
    current_state: Optional[dict[str, dict[str, Any]]] = None,
    *,
    use_model: Optional[bool] = None,
) -> dict[str, Any]:
    """Apply natural-language layout prompt with cross-room support, rigid coupling & `shapely` checks."""
    state = copy.deepcopy(current_state) if current_state else default_scene_state()
    for gid, d in default_scene_state().items():
        if gid not in state:
            state[gid] = copy.deepcopy(d)
        else:
            state[gid].setdefault("room", d["room"])

    should_call_llm = use_model if use_model is not None else bool(os.environ.get("GEMINI_API_KEY"))
    engine_mode = "deterministic_spatial_parser"
    if should_call_llm:
        try:
            plan = parse_prompt_with_gemini(prompt, state)
            engine_mode = f"{MODEL_NEON} + shapely_solver"
        except Exception:
            plan = parse_prompt_deterministic(prompt, state)
            engine_mode = "deterministic_spatial_parser (fallback)"
    else:
        plan = parse_prompt_deterministic(prompt, state)

    applied_moves: list[dict[str, Any]] = []

    if plan.reset_all:
        state = default_scene_state()
        applied_moves.append({"group_id": "all", "label": "All furniture groups", "action": "Reset to default architect plan"})
    else:
        requested_ids = {d.group_id for d in plan.directives}

        for d in plan.directives:
            gid = d.group_id
            prev = dict(state[gid])
            if not d.visible:
                prev["visible"] = False
                state[gid] = prev
                applied_moves.append(
                    {
                        "group_id": gid,
                        "label": GROUP_DEFAULTS[gid]["label"],
                        "action": d.reason or f"Removed {GROUP_DEFAULTS[gid]['label']}",
                        "clamped": False,
                    }
                )
                continue

            target_room = d.target_room or prev.get("room") or GROUP_DEFAULTS[gid]["room"]
            if d.target_x is not None and d.target_z is not None:
                new_pose = {
                    "room": target_room,
                    "x": float(d.target_x),
                    "z": float(d.target_z),
                    "rot_deg": float(d.rot_deg if d.rot_deg is not None else prev["rot_deg"]),
                    "visible": True,
                }
            else:
                new_pose = _pick_cross_room_pose(gid, target_room, d.target_wall, state)
                if d.rot_deg is not None:
                    new_pose["rot_deg"] = float(d.rot_deg)

            new_pose, was_clamped = clamp_pose_to_room(gid, new_pose)
            state[gid] = new_pose
            applied_moves.append(
                {
                    "group_id": gid,
                    "label": GROUP_DEFAULTS[gid]["label"],
                    "action": d.reason or f"Moved to {target_room} ({new_pose['x']:.2f}, {new_pose['z']:.2f} m)",
                    "clamped": was_clamped,
                }
            )

        # ---- Rigid coupling & automatic cross-room conflict resolution ----
        # A. Keep `tv_niches_group` synced: visible ONLY when `tv_group` is visible on Living West wall!
        tv_st = state["tv_group"]
        tv_on_living_west = (
            tv_st.get("visible", True)
            and tv_st.get("room") == "living"
            and abs(float(tv_st["x"]) - 0.30) < 0.25
            and abs(float(tv_st["rot_deg"]) % 360.0) < 15.0
        )
        state["tv_niches_group"]["visible"] = tv_on_living_west

        # B. If TV moved into Primary Bedroom -> clear any wall conflict with Wardrobe / Bench / Desk!
        if tv_st.get("visible", True) and tv_st.get("room") == "primary_bedroom":
            tv_x, tv_z = float(tv_st["x"]), float(tv_st["z"])
            # If TV is on East wall (x > 4.0) and wardrobe is on East wall -> relocate wardrobe to North wall (or hide if desk occupies it)!
            if tv_x > 3.9 and state["wardrobe_group"].get("visible", True) and state["wardrobe_group"].get("room") == "primary_bedroom" and float(state["wardrobe_group"]["x"]) > 3.8:
                if "wardrobe_group" not in requested_ids:
                    # Place wardrobe cleanly against bedroom north wall if bed is on west wall, or tuck away cleanly
                    if abs(float(state["bed_group"]["rot_deg"]) % 360.0) < 15.0:
                        state["wardrobe_group"] = {"room": "primary_bedroom", "x": 4.42, "z": 5.55, "rot_deg": 0.0, "visible": False}
                        applied_moves.append(
                            {
                                "group_id": "wardrobe_group",
                                "label": GROUP_DEFAULTS["wardrobe_group"]["label"],
                                "action": "Cleared bedroom east wardrobe wall to mount the 65\" TV facing the bed",
                                "clamped": False,
                            }
                        )
            if tv_z > 6.3 and state["bench_group"].get("visible", True):
                state["bench_group"]["visible"] = False

        # C. If TV was moved inside Living, auto-align Sofa & Living Art
        tv_req = next((d for d in plan.directives if d.group_id == "tv_group"), None)
        if tv_req and tv_req.visible and state["tv_group"].get("room") == "living" and tv_req.target_wall:
            if "sofa_group" not in requested_ids and tv_req.target_wall in WALL_PRESETS["sofa_group"]:
                sp = WALL_PRESETS["sofa_group"][tv_req.target_wall]
                state["sofa_group"] = {"room": "living", "x": sp["x"], "z": sp["z"], "rot_deg": sp["rot_deg"], "visible": True}
                applied_moves.append(
                    {
                        "group_id": "sofa_group",
                        "label": GROUP_DEFAULTS["sofa_group"]["label"],
                        "action": f"Auto-aligned conversation group to face TV ({sp['note']})",
                        "clamped": False,
                    }
                )
            if tv_req.target_wall == "S":
                state["living_art_group"] = {"room": "living", "x": 0.12, "z": 1.83, "rot_deg": 90.0, "visible": True}
            elif tv_req.target_wall == "W":
                state["living_art_group"] = {"room": "living", "x": 1.825, "z": 3.57, "rot_deg": 0.0, "visible": True}

        # D. If Bed moved into Living room -> shift Sofa out of its way!
        bed_st = state["bed_group"]
        if bed_st.get("visible", True) and bed_st.get("room") == "living" and "sofa_group" not in requested_ids:
            state["sofa_group"] = {"room": "living", "x": 3.92, "z": 1.82, "rot_deg": 180.0, "visible": True}
            state["dining_group"]["visible"] = False

        # E. Bedroom wall conflict resolution (Bed vs Desk / Bench / Wardrobe)
        bed_req = next((d for d in plan.directives if d.group_id == "bed_group"), None)
        if bed_req and bed_req.visible and state["bed_group"].get("room") == "primary_bedroom" and bed_req.target_wall:
            bw = bed_req.target_wall
            if bw == "N":
                if "desk_group" not in requested_ids:
                    dp = WALL_PRESETS["desk_group"]["S"]
                    state["desk_group"] = {"room": "primary_bedroom", "x": dp["x"], "z": dp["z"], "rot_deg": dp["rot_deg"], "visible": True}
                    applied_moves.append(
                        {
                            "group_id": "desk_group",
                            "label": GROUP_DEFAULTS["desk_group"]["label"],
                            "action": "Auto-relocated study desk to south-east window corner to clear bed headboard",
                            "clamped": False,
                        }
                    )
                state["bench_group"]["visible"] = True
                if "wardrobe_group" not in requested_ids and not (tv_st.get("visible") and tv_st.get("room") == "primary_bedroom"):
                    wp = WALL_PRESETS["wardrobe_group"]["E"]
                    state["wardrobe_group"] = {"room": "primary_bedroom", "x": wp["x"], "z": wp["z"], "rot_deg": wp["rot_deg"], "visible": True}
            elif bw == "S":
                state["bench_group"]["visible"] = False
                if "desk_group" not in requested_ids:
                    dp = WALL_PRESETS["desk_group"]["N"]
                    state["desk_group"] = {"room": "primary_bedroom", "x": dp["x"], "z": dp["z"], "rot_deg": dp["rot_deg"], "visible": True}
            elif bw == "E":
                if "wardrobe_group" not in requested_ids:
                    wp = WALL_PRESETS["wardrobe_group"]["W"]
                    state["wardrobe_group"] = {"room": "primary_bedroom", "x": wp["x"], "z": wp["z"], "rot_deg": wp["rot_deg"], "visible": True}
                    applied_moves.append(
                        {
                            "group_id": "wardrobe_group",
                            "label": GROUP_DEFAULTS["wardrobe_group"]["label"],
                            "action": "Auto-swapped wardrobe to west wall so bed fits flush on east wall",
                            "clamped": False,
                        }
                    )
                if "desk_group" not in requested_ids:
                    state["desk_group"] = {"room": "primary_bedroom", "x": 1.65, "z": 4.10, "rot_deg": 0.0, "visible": True}
                state["bench_group"]["visible"] = False
            elif bw == "W":
                state["bench_group"]["visible"] = True
                if "desk_group" not in requested_ids:
                    dp = WALL_PRESETS["desk_group"]["N"]
                    state["desk_group"] = {"room": "primary_bedroom", "x": dp["x"], "z": dp["z"], "rot_deg": dp["rot_deg"], "visible": True}
                if "wardrobe_group" not in requested_ids and not (tv_st.get("visible") and tv_st.get("room") == "primary_bedroom"):
                    wp = WALL_PRESETS["wardrobe_group"]["E"]
                    state["wardrobe_group"] = {"room": "primary_bedroom", "x": wp["x"], "z": wp["z"], "rot_deg": wp["rot_deg"], "visible": True}

    # Deterministic `shapely` verification pass across all visible solid groups
    overlaps: list[list[str]] = []
    outside_room: list[str] = []
    door_blocks: list[str] = []
    updated_colliders: dict[str, dict[str, Any]] = {}

    for gid, pose in state.items():
        meta = GROUP_DEFAULTS[gid]
        if gid in ("living_art_group", "tv_niches_group"):
            continue
        if not pose.get("visible", True):
            updated_colliders[gid] = {
                "x0": 0.0,
                "z0": 0.0,
                "x1": 0.0,
                "z1": 0.0,
                "visible": False,
                "replaces": meta["collider_labels"],
            }
            continue

        b = group_box(pose, meta)
        room = pose.get("room") or meta["room"]
        x0, z0, x1, z1 = ROOM_BOUNDS[room]
        r_box = box(x0 - 0.06, z0 - 0.06, x1 + 0.06, z1 + 0.06)
        if not r_box.contains(b):
            outside_room.append(gid)
        for dz in DOOR_CLEAR_ZONES.get(room, []):
            if b.intersection(box(*dz)).area > 0.08:
                door_blocks.append(gid)
                break
        bx0, bz0, bx1, bz1 = b.bounds
        updated_colliders[gid] = {
            "x0": round(bx0, 3),
            "z0": round(bz0, 3),
            "x1": round(bx1, 3),
            "z1": round(bz1, 3),
            "visible": True,
            "replaces": meta["collider_labels"],
        }

    solid_visible = [
        gid
        for gid, pose in state.items()
        if pose.get("visible", True) and gid not in ("living_art_group", "tv_niches_group")
    ]
    boxes = {gid: group_box(state[gid], GROUP_DEFAULTS[gid]) for gid in solid_visible}
    for i, g1 in enumerate(solid_visible):
        for g2 in solid_visible[i + 1 :]:
            if state[g1].get("room") != state[g2].get("room"):
                continue
            if boxes[g1].intersection(boxes[g2]).area > 0.04:
                overlaps.append([g1, g2])

    return {
        "ok": True,
        "matched": bool(applied_moves or plan.lighting_mode or plan.style_overrides),
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
        "put the TV in the bedroom and in the living room, NO TV",
        "put the bed against the north wall facing the window",
        "remove the study desk and bring the TV back to the living room south wall",
        "put the bed against the east wall, move the dining table near the balcony, and use evening lighting",
        "swap the sofa and dining area, and make the lighting golden hour",
        "put the soaking tub on the balcony",
        "reset layout to default",
    ]
    st = default_scene_state()
    for t in tests:
        res = solve_and_verify_layout(t, st, use_model=False)
        st = res["state"]
        print(
            json.dumps(
                {
                    "prompt": t,
                    "matched": res["matched"],
                    "summary": res["summary"],
                    "tv": (st["tv_group"]["room"], st["tv_group"]["x"], st["tv_group"]["z"], st["tv_group"]["visible"]),
                    "niches_vis": st["tv_niches_group"]["visible"],
                    "violations": res["check"]["violations_total"],
                    "check": res["check"],
                }
            )
        )
        assert res["matched"] is True, t
        assert res["check"]["violations_total"] == 0, (t, res["check"])
    print("ALL CROSS-ROOM + REMOVAL + SPATIAL TESTS PASSED WITH 0 VIOLATIONS")
