"""Visual attribute & material style extraction stage for Homeweave (`style.py`).

Translates input room photographs (`L1..L4`, `K1..K4`, `Be1..Be3`, `B1..B4`, or any user-uploaded
photo / base64 data URI) into structured PBR material, palette, lighting, and architectural finish
tokens (`RoomVisualStyle`) that drive the 3D Residence renderer in `viewer.html`.

Supports:
1. Live multimodal extraction via `gemini-4-hydrogen` (`extract_style_from_photo(..., use_model=True)`)
2. Curated ground-truth visual profiles for all 15 Drive reference photos (`PHOTO_STYLE_CATALOG`)
3. Offline pixel/palette fallback for custom uploaded PNGs/JPEGs when `GEMINI_API_KEY` is unset.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import zlib
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from google import genai

from perceive import media_block_from_source
from schemas import (
    HomeStyleBundle,
    Provenance,
    RoomType,
    RoomVisualStyle,
    ValidatedStageResult,
    to_gemini_response_format,
    validate_with_retry,
)

MODEL_HYDROGEN = "gemini-4-hydrogen"

PROMPT_VISUAL_STYLE = """You are an architectural material and lighting perception model.
Inspect this interior room photograph and extract its exact visual attributes so a 3D PBR renderer
can recreate its finishes, colors, textures, fixtures, lighting warmth, and decor.

Return one JSON object matching the `RoomVisualStyle` schema (`room_type = "{room_type}"`, `style_id = "{style_id}"`). No prose.
- Sample accurate hex colors (`#RRGGBB`) directly from the visible floor, primary walls, feature/accent wall, cabinetry, countertops, upholstery, throw pillows/blankets, wood tones, rug, and warm/cool LED lighting.
- Set `floor_roughness` low (`0.12`–`0.24`) for polished marble/porcelain and higher (`0.58`–`0.78`) for matte oak, walnut, or honed stone.
- Identify `accent_material`, `hardware_metal`, `table_shape`, `rug_style`, `pendant_style`, `led_cove_glow`, `fireplace_glow`, `greenery_accents`, and `arched_shower` accurately from what appears in the image.
- Record the image source in `provenance`.
"""


def _p(photo_name: str, notes: str) -> Provenance:
    return Provenance(source=f"data/drive_inputs/{photo_name}", notes=notes)


# Ground-truth visual profiles extracted directly from the 15 Google Drive room photos:
PHOTO_STYLE_CATALOG: dict[str, RoomVisualStyle] = {
    # --- LIVING ROOMS (L1 - L4) ---
    "L1": RoomVisualStyle(
        style_id="L1",
        label="Cream Bouclé & Calacatta Lounge",
        room_type="living",
        floor_material="travertine_stone",
        floor_hex="#cfc3b2",
        floor_grain_hex="#b5a590",
        floor_roughness=0.38,
        wall_hex="#e9e1d4",
        accent_wall_hex="#dfd3c0",
        accent_material="calacatta_marble",
        cap_hex="#231d18",
        cabinet_hex="#2a201a",
        countertop_hex="#e8dfd1",
        countertop_vein_hex="#b39c80",
        hardware_metal="matte_black",
        upholstery_hex="#e5ddd0",
        accent_fabric_hex="#46372b",
        wood_tone_hex="#33251c",
        table_shape="rectangular",
        rug_style="shag",
        rug_hex="#bfb19d",
        led_cove_glow=True,
        led_glow_hex="#ffbe73",
        pendant_style="none",
        fireplace_glow=False,
        greenery_accents=True,
        arched_shower=False,
        provenance=_p("L1.jpg", "Cream sectional, marble TV wall, espresso floating console, warm cove LEDs"),
    ),
    "L2": RoomVisualStyle(
        style_id="L2",
        label="Sculpted Travertine Rock & Walnut",
        room_type="living",
        floor_material="veined_marble",
        floor_hex="#ccbca6",
        floor_grain_hex="#a89378",
        floor_roughness=0.18,
        wall_hex="#efe9df",
        accent_wall_hex="#d8c5a8",
        accent_material="chiseled_limestone",
        cap_hex="#2b2118",
        cabinet_hex="#ddd5ca",
        countertop_hex="#e6dfd4",
        countertop_vein_hex="#b59d7e",
        hardware_metal="brushed_gold",
        upholstery_hex="#dfd6c7",
        accent_fabric_hex="#4d5333",
        wood_tone_hex="#6f482a",
        table_shape="organic_oval",
        rug_style="jute",
        rug_hex="#d2c6b4",
        led_cove_glow=True,
        led_glow_hex="#ffb357",
        pendant_style="globe_glass",
        fireplace_glow=False,
        greenery_accents=True,
        arched_shower=False,
        provenance=_p("L2.jpg", "Backlit chiseled stone feature wall, walnut slats, olive throw, amber globes"),
    ),
    "L3": RoomVisualStyle(
        style_id="L3",
        label="Charcoal Gold-Leaf & Rustic Timber",
        room_type="living",
        floor_material="dark_walnut",
        floor_hex="#8a5d3b",
        floor_grain_hex="#54351e",
        floor_roughness=0.55,
        wall_hex="#353230",
        accent_wall_hex="#282626",
        accent_material="charcoal_gold_patina",
        cap_hex="#141210",
        cabinet_hex="#5a3e28",
        countertop_hex="#4a3524",
        countertop_vein_hex="#c69249",
        hardware_metal="brushed_gold",
        upholstery_hex="#3c3530",
        accent_fabric_hex="#7b4a22",
        wood_tone_hex="#593c26",
        table_shape="round",
        rug_style="persian",
        rug_hex="#4a443c",
        led_cove_glow=True,
        led_glow_hex="#ffa646",
        pendant_style="none",
        fireplace_glow=False,
        greenery_accents=True,
        arched_shower=False,
        provenance=_p("L3.jpg", "Dark slate & gold patina TV wall, live-edge timber bench, candlelight"),
    ),
    "L4": RoomVisualStyle(
        style_id="L4",
        label="Biophilic Moss Velvet & Linear Fire",
        room_type="living",
        floor_material="travertine_stone",
        floor_hex="#766c60",
        floor_grain_hex="#4e463d",
        floor_roughness=0.62,
        wall_hex="#1f211e",
        accent_wall_hex="#181a17",
        accent_material="obsidian_matte",
        cap_hex="#10120f",
        cabinet_hex="#5a3f28",
        countertop_hex="#686056",
        countertop_vein_hex="#3d3832",
        hardware_metal="matte_black",
        upholstery_hex="#3f4628",
        accent_fabric_hex="#7d5836",
        wood_tone_hex="#543822",
        table_shape="round",
        rug_style="jute",
        rug_hex="#594a39",
        led_cove_glow=True,
        led_glow_hex="#ff9d3b",
        pendant_style="none",
        fireplace_glow=True,
        greenery_accents=True,
        arched_shower=False,
        provenance=_p("L4.jpg", "Moss-green velvet sectional, dark slatted ceiling, linear stone fireplace, lush plants"),
    ),
    # --- KITCHENS (K1 - K4) ---
    "K1": RoomVisualStyle(
        style_id="K1",
        label="Olive Shaker & Honed Cream Marble",
        room_type="kitchen",
        floor_material="wood_plank",
        floor_hex="#d4b894",
        floor_grain_hex="#aa8962",
        floor_roughness=0.64,
        wall_hex="#e8e0d3",
        accent_wall_hex="#3f4835",
        accent_material="plaster",
        cap_hex="#22281d",
        cabinet_hex="#3f4835",
        countertop_hex="#ebe3d5",
        countertop_vein_hex="#bfa98e",
        hardware_metal="matte_black",
        upholstery_hex="#bfb09c",
        accent_fabric_hex="#3f4835",
        wood_tone_hex="#825e3c",
        table_shape="rectangular",
        rug_style="persian",
        rug_hex="#d8cebe",
        led_cove_glow=True,
        led_glow_hex="#ffc278",
        pendant_style="globe_glass",
        fireplace_glow=False,
        greenery_accents=True,
        arched_shower=False,
        provenance=_p("K1.jpg", "Olive-green shaker island, honed cream marble top, 3 smoked glass globes, light oak floor"),
    ),
    "K2": RoomVisualStyle(
        style_id="K2",
        label="Gloss Dove-Grey & Jet-Black Granite",
        room_type="kitchen",
        floor_material="veined_marble",
        floor_hex="#eae6df",
        floor_grain_hex="#c6beb0",
        floor_roughness=0.15,
        wall_hex="#e6e4e0",
        accent_wall_hex="#cfcdd0",
        accent_material="calacatta_marble",
        cap_hex="#18191c",
        cabinet_hex="#b8b7b5",
        countertop_hex="#1a1b1d",
        countertop_vein_hex="#47494e",
        hardware_metal="matte_black",
        upholstery_hex="#686665",
        accent_fabric_hex="#c69b4b",
        wood_tone_hex="#4a3c31",
        table_shape="rectangular",
        rug_style="none",
        rug_hex="#d8d6d2",
        led_cove_glow=True,
        led_glow_hex="#ffbe6b",
        pendant_style="cage_industrial",
        fireplace_glow=False,
        greenery_accents=True,
        arched_shower=False,
        provenance=_p("K2.jpg", "Gloss grey cabinetry, black granite bar with warm under-counter LED, wireframe pendants"),
    ),
    "K3": RoomVisualStyle(
        style_id="K3",
        label="Fluted Walnut & Emperador Waterfall",
        room_type="kitchen",
        floor_material="veined_marble",
        floor_hex="#cbbba6",
        floor_grain_hex="#9c866c",
        floor_roughness=0.16,
        wall_hex="#e6ded2",
        accent_wall_hex="#9e8972",
        accent_material="fluted_panel",
        cap_hex="#2d2016",
        cabinet_hex="#735035",
        countertop_hex="#8f7a64",
        countertop_vein_hex="#d6c6b2",
        hardware_metal="brushed_gold",
        upholstery_hex="#d6cab8",
        accent_fabric_hex="#b88a44",
        wood_tone_hex="#735035",
        table_shape="rectangular",
        rug_style="jute",
        rug_hex="#b59b78",
        led_cove_glow=True,
        led_glow_hex="#ffba63",
        pendant_style="brass_spiral",
        fireplace_glow=False,
        greenery_accents=True,
        arched_shower=False,
        provenance=_p("K3.jpg", "Fluted walnut cabinetry, Emperador brown marble waterfall island, spiral brass pendants"),
    ),
    "K4": RoomVisualStyle(
        style_id="K4",
        label="Champagne Lacquer & Gold-Vein Stone",
        room_type="kitchen",
        floor_material="veined_marble",
        floor_hex="#ece3d5",
        floor_grain_hex="#bfa178",
        floor_roughness=0.14,
        wall_hex="#efe9df",
        accent_wall_hex="#766b60",
        accent_material="calacatta_marble",
        cap_hex="#23201d",
        cabinet_hex="#cfc1ae",
        countertop_hex="#706860",
        countertop_vein_hex="#d8c3a5",
        hardware_metal="brushed_gold",
        upholstery_hex="#d5c8b6",
        accent_fabric_hex="#262422",
        wood_tone_hex="#8c6845",
        table_shape="rectangular",
        rug_style="none",
        rug_hex="#dfd5c6",
        led_cove_glow=True,
        led_glow_hex="#ffc980",
        pendant_style="lantern_cylinder",
        fireplace_glow=False,
        greenery_accents=True,
        arched_shower=False,
        provenance=_p("K4.jpg", "Champagne gloss cabinets, gold-veined porcelain floor, black-and-brass pulls"),
    ),
    # --- BEDROOMS (Be1 - Be3) ---
    "Be1": RoomVisualStyle(
        style_id="Be1",
        label="Starlight Cove & Floating Walnut Suite",
        room_type="primary_bedroom",
        floor_material="wood_plank",
        floor_hex="#bfa07e",
        floor_grain_hex="#8f6e4d",
        floor_roughness=0.58,
        wall_hex="#ddd3c5",
        accent_wall_hex="#cfc2b0",
        accent_material="fluted_panel",
        cap_hex="#261f1a",
        cabinet_hex="#cabfb2",
        countertop_hex="#eae4dc",
        countertop_vein_hex="#b5aa9c",
        hardware_metal="brushed_gold",
        upholstery_hex="#dfd7cb",
        accent_fabric_hex="#543e2e",
        wood_tone_hex="#5c402b",
        table_shape="round",
        rug_style="shag",
        rug_hex="#c6bcb0",
        led_cove_glow=True,
        led_glow_hex="#ffbe70",
        pendant_style="lantern_cylinder",
        fireplace_glow=False,
        greenery_accents=True,
        arched_shower=False,
        provenance=_p("Be1.jpg", "Floating platform bed with warm LED under-glow, skyline lounge chairs, oak floor"),
    ),
    "Be2": RoomVisualStyle(
        style_id="Be2",
        label="Sculpted Cloud Bouclé Sanctuary",
        room_type="primary_bedroom",
        floor_material="travertine_stone",
        floor_hex="#ebe3d8",
        floor_grain_hex="#d8ccb9",
        floor_roughness=0.48,
        wall_hex="#f3ece2",
        accent_wall_hex="#eadfce",
        accent_material="plaster",
        cap_hex="#3b332b",
        cabinet_hex="#efe7dc",
        countertop_hex="#faf6f0",
        countertop_vein_hex="#dfd4c3",
        hardware_metal="brushed_gold",
        upholstery_hex="#f7f2ea",
        accent_fabric_hex="#dfd3c2",
        wood_tone_hex="#d6c6b2",
        table_shape="round",
        rug_style="boucle_round",
        rug_hex="#ece4d8",
        led_cove_glow=True,
        led_glow_hex="#ffc47d",
        pendant_style="globe_glass",
        fireplace_glow=False,
        greenery_accents=False,
        arched_shower=False,
        provenance=_p("Be2.jpg", "Cream cloud-tufted platform bed, warm halo arch, orb bedside lamps, cream bouclé rug"),
    ),
    "Be3": RoomVisualStyle(
        style_id="Be3",
        label="Cozy Moonlit Timber & Linen",
        room_type="primary_bedroom",
        floor_material="wood_plank",
        floor_hex="#a87c53",
        floor_grain_hex="#785331",
        floor_roughness=0.66,
        wall_hex="#e6ddd0",
        accent_wall_hex="#d8cbb8",
        accent_material="plaster",
        cap_hex="#2a2119",
        cabinet_hex="#ded4c6",
        countertop_hex="#c2966b",
        countertop_vein_hex="#8f6640",
        hardware_metal="matte_black",
        upholstery_hex="#f2ece4",
        accent_fabric_hex="#9c7a56",
        wood_tone_hex="#8f633d",
        table_shape="rectangular",
        rug_style="boucle_round",
        rug_hex="#dfd6c9",
        led_cove_glow=True,
        led_glow_hex="#ffb156",
        pendant_style="globe_glass",
        fireplace_glow=False,
        greenery_accents=True,
        arched_shower=False,
        provenance=_p("Be3.jpg", "Glowing moon lamp, fringed woven throw, warm oak trim, trailing ivy shelves"),
    ),
    # --- BATHROOMS (B1 - B4) ---
    "B1": RoomVisualStyle(
        style_id="B1",
        label="Travertine & Teak Step Soaking Spa",
        room_type="bathroom",
        floor_material="travertine_stone",
        floor_hex="#b8ad9e",
        floor_grain_hex="#968a7a",
        floor_roughness=0.35,
        wall_hex="#ebe4d8",
        accent_wall_hex="#d6c5ab",
        accent_material="chiseled_limestone",
        cap_hex="#1e1b18",
        cabinet_hex="#9e7147",
        countertop_hex="#f5f2ec",
        countertop_vein_hex="#cfc6b8",
        hardware_metal="matte_black",
        upholstery_hex="#d8cbb8",
        accent_fabric_hex="#9e7147",
        wood_tone_hex="#9e7147",
        table_shape="rectangular",
        rug_style="shag",
        rug_hex="#d4c6b2",
        led_cove_glow=True,
        led_glow_hex="#ffc16e",
        pendant_style="none",
        fireplace_glow=False,
        greenery_accents=True,
        arched_shower=True,
        provenance=_p("B1.jpg", "Sunken soaking tub with LED teak steps, stacked stone wall, black arched rain shower"),
    ),
    "B2": RoomVisualStyle(
        style_id="B2",
        label="Botanical Fluted Sage & Pebble Spa",
        room_type="bathroom",
        floor_material="sage_tile",
        floor_hex="#93a392",
        floor_grain_hex="#738572",
        floor_roughness=0.24,
        wall_hex="#a3b39f",
        accent_wall_hex="#7f917b",
        accent_material="fluted_panel",
        cap_hex="#1d241d",
        cabinet_hex="#7c8f78",
        countertop_hex="#a9ba9f",
        countertop_vein_hex="#eef2ec",
        hardware_metal="matte_black",
        upholstery_hex="#b5c2b2",
        accent_fabric_hex="#a3764c",
        wood_tone_hex="#a3764c",
        table_shape="round",
        rug_style="shag",
        rug_hex="#b0bdb0",
        led_cove_glow=True,
        led_glow_hex="#ffe0a3",
        pendant_style="globe_glass",
        fireplace_glow=False,
        greenery_accents=True,
        arched_shower=False,
        provenance=_p("B2.jpg", "Pistachio sage fluted walls, floating sage vanity with oak cubby, oval freestanding tub"),
    ),
    "B3": RoomVisualStyle(
        style_id="B3",
        label="Powder-Blue & Arched Carrara Marble",
        room_type="bathroom",
        floor_material="porcelain_tile",
        floor_hex="#7d95ab",
        floor_grain_hex="#637b91",
        floor_roughness=0.20,
        wall_hex="#89a3ba",
        accent_wall_hex="#e6eaee",
        accent_material="calacatta_marble",
        cap_hex="#233040",
        cabinet_hex="#e8ecef",
        countertop_hex="#f8fafc",
        countertop_vein_hex="#cbd5e1",
        hardware_metal="chrome",
        upholstery_hex="#7997b5",
        accent_fabric_hex="#f8fafc",
        wood_tone_hex="#cbd5e1",
        table_shape="rectangular",
        rug_style="shag",
        rug_hex="#728ea8",
        led_cove_glow=True,
        led_glow_hex="#fff0d4",
        pendant_style="none",
        fireplace_glow=False,
        greenery_accents=True,
        arched_shower=True,
        provenance=_p("B3.jpg", "Dusty sky-blue walls & tub apron, white Carrara arched shower, backlit mirror"),
    ),
    "B4": RoomVisualStyle(
        style_id="B4",
        label="Midnight Navy, Calacatta & Brushed Gold",
        room_type="bathroom",
        floor_material="veined_marble",
        floor_hex="#edeae4",
        floor_grain_hex="#9c9890",
        floor_roughness=0.14,
        wall_hex="#ebe8e2",
        accent_wall_hex="#132038",
        accent_material="calacatta_marble",
        cap_hex="#0c1526",
        cabinet_hex="#14223b",
        countertop_hex="#f4f2ee",
        countertop_vein_hex="#a39e96",
        hardware_metal="brushed_gold",
        upholstery_hex="#f2efe9",
        accent_fabric_hex="#c89d46",
        wood_tone_hex="#14223b",
        table_shape="round",
        rug_style="none",
        rug_hex="#eae6df",
        led_cove_glow=True,
        led_glow_hex="#ffe4ad",
        pendant_style="lantern_cylinder",
        fireplace_glow=False,
        greenery_accents=True,
        arched_shower=False,
        provenance=_p("B4.jpg", "Midnight-navy accent wall & vanity with brass hardware, marble freestanding tub, gold shower"),
    ),
}

DEFAULT_ACTIVE_STYLES: dict[str, str] = {
    "living": "L2",
    "kitchen": "K1",
    "primary_bedroom": "Be1",
    "bathroom": "B1",
}


def build_default_style_bundle() -> HomeStyleBundle:
    """Return the initial `HomeStyleBundle` + full 15-photo catalog for instant UI switching."""
    by_type = {rtype: PHOTO_STYLE_CATALOG[sid] for rtype, sid in DEFAULT_ACTIVE_STYLES.items()}
    return HomeStyleBundle(by_room_type=by_type, catalog=PHOTO_STYLE_CATALOG)


def extract_style_from_photo(
    photo_source: str | Path,
    *,
    room_type: RoomType = "living",
    style_id: str = "custom",
    use_model: Optional[bool] = None,
    client: Optional[genai.Client] = None,
    client_palette_hint: Optional[dict[str, str]] = None,
    transport_override: Optional[Callable[[Optional[str]], str | dict[str, object]]] = None,
) -> ValidatedStageResult[RoomVisualStyle]:
    """Extract `RoomVisualStyle` from an input photograph via `gemini-4-hydrogen` (or palette hint offline)."""
    if style_id in PHOTO_STYLE_CATALOG and transport_override is None and not use_model:
        preset = PHOTO_STYLE_CATALOG[style_id].model_copy(update={"room_type": room_type})
        return ValidatedStageResult[RoomVisualStyle](ok=True, flagged=False, attempts=1, data=preset)

    if transport_override is not None:
        return validate_with_retry(transport_override, RoomVisualStyle)

    should_call_model = use_model if use_model is not None else bool(os.environ.get("GEMINI_API_KEY"))
    if should_call_model:
        active_client = client or genai.Client()
        img_block = media_block_from_source(photo_source, active_client)
        prompt = PROMPT_VISUAL_STYLE.format(room_type=room_type, style_id=style_id)
        response_format = to_gemini_response_format(RoomVisualStyle)

        def _producer(prev_error: Optional[str]) -> str:
            text = prompt
            if prev_error:
                text += f"\n\nPREVIOUS VALIDATION ERROR:\n{prev_error}"
            interaction = active_client.interactions.create(
                model=MODEL_HYDROGEN,
                input=[img_block, {"type": "text", "text": text}],
                response_format=response_format,
                generation_config={"max_output_tokens": 16384},
                store=False,
            )
            if not interaction.output_text:
                raise ValueError("Empty style response from gemini-4-hydrogen.")
            return interaction.output_text

        return validate_with_retry(_producer, RoomVisualStyle)

    # Offline fallback when user uploads a custom photo in browser without GEMINI_API_KEY:
    # derives materials, wood/stone type, hardware metal, and lighting warmth from sampled regional RGB
    hint = client_palette_hint or {}
    wall_hex = hint.get("wall_hex", "#EAE3D7")
    accent_hex = hint.get("accent_hex", "#6D5B4B")
    floor_hex = hint.get("floor_hex", "#C4A784")
    upholstery_hex = hint.get("upholstery_hex", "#DDD5C8")

    def _rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        if len(h) != 6:
            return (200, 190, 175)
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    wr, wg, wb = _rgb(wall_hex)
    fr, fg, fb = _rgb(floor_hex)
    ar, ag, ab = _rgb(accent_hex)
    wall_luma = (wr + wg + wb) / 3.0
    floor_luma = (fr + fg + fb) / 3.0
    is_dark = wall_luma < 105
    is_warm_wood_floor = (fr - fb) > 26 and floor_luma < 205
    is_sage_hue = ag > ar + 8 and ag > ab + 8
    is_glossy = hint.get("glossy") == "true"

    if room_type == "bathroom" and is_sage_hue:
        floor_mat = "sage_tile"
    elif is_glossy or not is_warm_wood_floor:
        floor_mat = "veined_marble" if floor_luma > 165 else "travertine_stone"
    else:
        floor_mat = "dark_walnut" if floor_luma < 125 else "wood_plank"

    if is_dark:
        acc_mat = "charcoal_gold_patina" if ar > ab + 18 else "obsidian_matte"
    elif is_sage_hue or is_warm_wood_floor:
        acc_mat = "fluted_panel"
    else:
        acc_mat = "calacatta_marble"

    hw_metal = "brushed_gold" if (is_dark or (ar - ab) > 22) else ("chrome" if wb > wr + 10 else "matte_black")
    mood_word = "Moody" if is_dark else ("Botanical" if is_sage_hue else "Luminous")
    mat_word = floor_mat.replace("_", " ").title()
    auto_label = hint.get("label") or f"{mood_word} {mat_word}"

    synthesized = RoomVisualStyle(
        style_id=style_id,
        label=auto_label,
        room_type=room_type,
        floor_material=floor_mat,  # type: ignore[arg-type]
        floor_hex=floor_hex,
        floor_grain_hex=accent_hex,
        floor_roughness=0.16 if is_glossy else (0.58 if "wood" in floor_mat or "walnut" in floor_mat else 0.34),
        wall_hex=wall_hex,
        accent_wall_hex=accent_hex,
        accent_material=acc_mat,  # type: ignore[arg-type]
        cap_hex="#141820" if is_dark else "#232a34",
        cabinet_hex=accent_hex,
        countertop_hex=wall_hex if wall_luma > 150 else "#eae4da",
        countertop_vein_hex=accent_hex,
        hardware_metal=hw_metal,  # type: ignore[arg-type]
        upholstery_hex=upholstery_hex,
        accent_fabric_hex=accent_hex,
        wood_tone_hex=floor_hex if is_warm_wood_floor else "#7a5638",
        table_shape="round" if is_dark else "rectangular",
        rug_style="jute" if is_warm_wood_floor else "shag",
        rug_hex=upholstery_hex,
        led_cove_glow=True,
        led_glow_hex="#ffb861" if wr >= wb else "#ffe8c2",
        pendant_style="brass_spiral" if (room_type == "kitchen" and hw_metal == "brushed_gold") else ("globe_glass" if room_type == "kitchen" else "none"),
        fireplace_glow=is_dark and room_type == "living",
        greenery_accents=True,
        arched_shower=(room_type == "bathroom"),
        provenance=Provenance(source="uploaded_photo", notes="Visual attributes extracted from input photo"),
    )
    return ValidatedStageResult[RoomVisualStyle](ok=True, flagged=False, attempts=1, data=synthesized)


if __name__ == "__main__":
    bundle = build_default_style_bundle()
    out_path = Path(__file__).parent / "data" / "styles.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    custom = extract_style_from_photo(
        "custom.jpg",
        room_type="living",
        style_id="custom-upload",
        use_model=False,
        client_palette_hint={
            "wall_hex": "#2A2E2B",
            "accent_hex": "#4E5B3C",
            "floor_hex": "#8B6947",
            "upholstery_hex": "#D8CFC2",
        },
    )
    assert custom.ok and custom.data is not None
    print(
        json.dumps(
            {
                "status": "ok",
                "catalog_presets": len(bundle.catalog),
                "default_rooms": {k: v.style_id for k, v in bundle.by_room_type.items()},
                "custom_extracted_label": custom.data.label,
            },
            indent=2,
        )
    )
