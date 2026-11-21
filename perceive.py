"""Media perception stage for Homeweave (`perceive.py`, Prompts 1, 2, and 3).

Pure JSON-producing functions backed by `google-genai >= 3.2.0` Interactions API:
- `perceive_listing` (`gemini-4-hydrogen`, Prompt 1)
- `perceive_home_video` (`gemini-4-hydrogen`, Prompt 2)
- `perceive_hard_plan` (`gemini-4-neon` with `thinking_level="high"`, Prompt 3)
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from google import genai

from schemas import (
    HardPlanResult,
    Home,
    HomeVideoInventory,
    ValidatedStageResult,
    to_gemini_response_format,
    validate_with_retry,
)

MODEL_HYDROGEN = "gemini-4-hydrogen"
MODEL_NEON = "gemini-4-neon"
INLINE_VIDEO_MAX_BYTES = 14 * 1024 * 1024  # Use Files API above ~14 MB


PROMPT_LISTING_TEMPLATE = """You are reading a real estate listing (prefix `{prefix}-`). You get its photos and, if present, its floor plan.

Return one JSON object matching the Home schema (`id = "{prefix}-home"`, `source = "listing"`). No prose.

For each room: id (`{prefix}-kitchen` style), type, polygon in metres as [x,z] corners in order, ceiling_height_m, doors [{{id, wall_index, offset_m, width_m, swing, null_reason, provenance}}], windows [{{id, wall_index, offset_m, width_m, sill_height_m, height_m, provenance}}], fixtures [{{id, room_id, type, footprint, null_reason, provenance}}] for built-ins such as counters, islands, fireplaces, closets, tv_wall.

Also return staging[]: furniture visible in the photos, each with id (`{prefix}-stage-...`), room_id, category, approximate dims_m, pose, group_kind, replaceable=true, and provenance. These are placeholders, not the user's furniture.

Also return features[]: up to 8 things a buyer would like about this home, each {{id, home_id, room_id, label, evidence: [photo index strings], provenance}}.

Scale rules:
- If the plan prints dimensions, use them and set scale_check.method = "plan_labels".
- Otherwise assume an interior door is 0.81 m wide, set scale_check.method = "door_width", and name that door in scale_check.reference.
- If a photo and the plan disagree on a room's size by more than 10 percent, keep the plan value and add the room id to scale_check.conflicts.

Conventions:
- Units are metres. Right-handed axes, y up, floor at y = 0. Angles are degrees clockwise from +z seen from above.
- If you cannot see something, set it to null and provide a non-empty `null_reason`. Never guess a default.
- Do not invent rooms that appear in neither the photos nor the plan. Every object needs `provenance` naming the photo index (`photo:0`) or `"plan"`.
"""

PROMPT_HOME_VIDEO = """You are watching a handheld phone video of someone's home. Build an inventory of their furniture and the rooms it sits in.

Return one JSON object matching the HomeVideoInventory schema (`home_id = "src-home"`). No prose.

rooms: as in the Home schema, ids like `src-living`. Estimate each polygon in metres as [x,z] corners from what the camera shows.

items: every movable piece of furniture, each once, even if it appears at several moments. For each: id (`src-sofa-01`), room_id, category, dims_m {w,d,h}, pose {x,z,rot_deg} in its room, front_dir, against_wall (true or false), nearest_fixture (`window`, `door`, `tv_wall`, `fireplace`, `closet`, `island`, `other`, or null), group_id, must_keep=true, confidence 0 to 1, null_reason (required whenever dims_m or pose is null), provenance {source: "video", seen_at: [timestamps], scale_ref: "..."}.

groups: furniture used together, each {id (`src-grp-...`), room_id, kind, item_ids, provenance}. Kinds: `conversation`, `sleep`, `dining`, `work`, `storage`, `other`.

Scale: find a door in each room and assume it is 0.81 m wide and 2.03 m tall, then size everything else from it. Name that door in `provenance.scale_ref` and `scale_check.reference`. If no door is visible, set dims_m to null with `null_reason = "no scale reference"`.

Counting rules: if you are not sure two sightings are the same object, list it once and add a note to `uncertain[]`. Put any item with confidence under 0.6 in `uncertain[]` with a one-line question for the user, for example "Is the grey chair at 0:42 the same one seen at 1:15?". Ignore decor smaller than 0.3 m, people and pets.
"""

PROMPT_HARD_PLAN_TEMPLATE = """This floor plan (prefix `{prefix}-`) is low quality: it may be a scan, a phone photo of a hand sketch, or use mixed units. Read it carefully.

Work in this order and show the result of each step in the `working` object:
1. `raw_dimension_labels`: List every dimension label you can read, with its unit as written.
2. `converted_metres`: Convert all of them to metres.
3. `polygon_closure_checks`: Check that the walls of each room close into a polygon. Where they do not, say which label you distrust and why.
4. `area_check`: Check total area against any stated area.

Then return `rooms[]` in the Room schema (ids prefixed `{prefix}-`), `plan_confidence` 0 to 1, `unreadable[]` listing anything you could not read, `scale_check`, and `provenance`. Use null with a non-empty `null_reason` for anything unreadable. Do not fill gaps with typical room sizes.
"""


def media_block_from_source(src: str | Path, client: Optional[genai.Client] = None) -> dict[str, Any]:
    """Convert a local file path or `data:...;base64,...` URI into an Interactions content block."""
    if isinstance(src, str) and src.startswith("data:"):
        header, b64_data = src.split(",", 1)
        mime = header.split(";")[0].replace("data:", "") or "image/png"
        kind = "document" if mime == "application/pdf" else ("video" if mime.startswith("video/") else "image")
        return {"type": kind, "data": b64_data, "mime_type": mime}

    path = Path(src)
    mime_guess, _ = mimetypes.guess_type(path.name)
    mime = mime_guess or "application/octet-stream"

    if mime.startswith("video/") or path.suffix.lower() in {".mp4", ".mov", ".webm", ".m4v"}:
        if mime == "application/octet-stream":
            mime = "video/mp4"
        size = path.stat().st_size
        if size > INLINE_VIDEO_MAX_BYTES:
            if client is None:
                client = genai.Client()
            uploaded = client.files.upload(file=str(path))
            while not uploaded.state or str(uploaded.state).upper().endswith("PROCESSING"):
                time.sleep(3)
                uploaded = client.files.get(name=uploaded.name)
            state_str = str(uploaded.state).upper()
            if "FAILED" in state_str:
                raise RuntimeError(f"Gemini Files API processing failed for {path.name}: {state_str}")
            return {"type": "video", "uri": uploaded.uri, "mime_type": uploaded.mime_type or mime}
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"type": "video", "data": b64, "mime_type": mime}

    if mime == "application/pdf" or path.suffix.lower() == ".pdf":
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"type": "document", "data": b64, "mime_type": "application/pdf"}

    if mime == "application/octet-stream":
        mime = "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image", "data": b64, "mime_type": mime}


def _call_structured_interaction(
    *,
    client: genai.Client,
    model: str,
    content_blocks: list[dict[str, Any]],
    prompt_text: str,
    schema_cls: type[Any],
    thinking_level: Optional[str] = None,
) -> ValidatedStageResult[Any]:
    """Run `client.interactions.create` with structured JSON output and 1x validation retry."""
    response_format = to_gemini_response_format(schema_cls)
    gen_cfg: dict[str, Any] = {"max_output_tokens": 32768}
    if thinking_level:
        gen_cfg["thinking_level"] = thinking_level

    def _producer(prev_error: Optional[str]) -> str:
        text = prompt_text
        if prev_error:
            text = (
                f"{prompt_text}\n\nPREVIOUS ATTEMPT FAILED SCHEMA VALIDATION:\n{prev_error}\n"
                "Fix every validation error (including ID prefixes, required provenance, and null_reason) and return valid JSON."
            )
        interaction = client.interactions.create(
            model=model,
            input=[*content_blocks, {"type": "text", "text": text}],
            response_format=response_format,
            generation_config=gen_cfg,
            store=False,
        )
        if not interaction.output_text:
            raise ValueError(f"Model {model} returned empty output_text (status={interaction.status}).")
        return interaction.output_text

    return validate_with_retry(_producer, schema_cls)


def perceive_hard_plan(
    plan_source: str | Path,
    *,
    prefix: str = "lst1",
    client: Optional[genai.Client] = None,
    transport_override: Optional[Callable[[Optional[str]], str | dict[str, object]]] = None,
) -> ValidatedStageResult[HardPlanResult]:
    """Read a scanned, hand-drawn, or mixed-unit floor plan with `gemini-4-neon` (`high` thinking)."""
    if transport_override is not None:
        return validate_with_retry(transport_override, HardPlanResult)

    active_client = client or genai.Client()
    plan_block = media_block_from_source(plan_source, active_client)
    prompt = PROMPT_HARD_PLAN_TEMPLATE.format(prefix=prefix)
    return _call_structured_interaction(
        client=active_client,
        model=MODEL_NEON,
        content_blocks=[plan_block],
        prompt_text=prompt,
        schema_cls=HardPlanResult,
        thinking_level="high",
    )


def perceive_listing(
    photo_sources: Sequence[str | Path],
    *,
    plan_source: Optional[str | Path] = None,
    listing_index: int = 1,
    hard_plan: bool = False,
    client: Optional[genai.Client] = None,
    transport_override: Optional[Callable[[Optional[str]], str | dict[str, object]]] = None,
) -> ValidatedStageResult[Home]:
    """Read all photos and optional floor plan of one listing into a validated `Home` (`gemini-4-hydrogen`)."""
    if transport_override is not None:
        return validate_with_retry(transport_override, Home)

    prefix = f"lst{listing_index}"
    active_client = client or genai.Client()
    blocks: list[dict[str, Any]] = []
    for idx, p_src in enumerate(photo_sources):
        blocks.append({"type": "text", "text": f"[Listing photo index {idx}]"})
        blocks.append(media_block_from_source(p_src, active_client))

    extra_context = ""
    if plan_source is not None:
        blocks.append({"type": "text", "text": "[Listing floor plan]"})
        blocks.append(media_block_from_source(plan_source, active_client))
        if hard_plan:
            hp_res = perceive_hard_plan(plan_source, prefix=prefix, client=active_client)
            if hp_res.ok and hp_res.data is not None:
                extra_context = (
                    "\n\nPRE-PARSED FLOOR PLAN GEOMETRY FROM NEON (use these plan dimensions as authoritative plan labels):\n"
                    + hp_res.data.model_dump_json(indent=2)
                )

    prompt = PROMPT_LISTING_TEMPLATE.format(prefix=prefix) + extra_context
    return _call_structured_interaction(
        client=active_client,
        model=MODEL_HYDROGEN,
        content_blocks=blocks,
        prompt_text=prompt,
        schema_cls=Home,
    )


def perceive_home_video(
    video_source: str | Path,
    *,
    client: Optional[genai.Client] = None,
    transport_override: Optional[Callable[[Optional[str]], str | dict[str, object]]] = None,
) -> ValidatedStageResult[HomeVideoInventory]:
    """Read a walkthrough phone video of the user's current home (`gemini-4-hydrogen`, Prompt 2)."""
    if transport_override is not None:
        return validate_with_retry(transport_override, HomeVideoInventory)

    active_client = client or genai.Client()
    v_block = media_block_from_source(video_source, active_client)
    return _call_structured_interaction(
        client=active_client,
        model=MODEL_HYDROGEN,
        content_blocks=[v_block],
        prompt_text=PROMPT_HOME_VIDEO,
        schema_cls=HomeVideoInventory,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Homeweave media perception stage (Prompts 1-3).")
    parser.add_argument("--listing-photos", nargs="*", help="Paths to listing photos.")
    parser.add_argument("--plan", type=str, default=None, help="Path to listing floor plan image/PDF.")
    parser.add_argument("--hard-plan", action="store_true", help="Use gemini-4-neon high thinking on plan.")
    parser.add_argument("--listing-index", type=int, default=1)
    parser.add_argument("--video", type=str, default=None, help="Path to current-home walkthrough video.")
    parser.add_argument("--out", type=str, default=None, help="Optional output JSON path in ./data.")
    args = parser.parse_args()

    if args.video:
        res = perceive_home_video(args.video)
        out_text = res.model_dump_json(indent=2)
        if args.out:
            Path(args.out).write_text(out_text, encoding="utf-8")
        print(out_text)
    elif args.listing_photos or args.plan:
        if args.hard_plan and not args.listing_photos and args.plan:
            res_hp = perceive_hard_plan(args.plan, prefix=f"lst{args.listing_index}")
            out_text = res_hp.model_dump_json(indent=2)
        else:
            res_l = perceive_listing(
                args.listing_photos or [],
                plan_source=args.plan,
                listing_index=args.listing_index,
                hard_plan=args.hard_plan,
            )
            out_text = res_l.model_dump_json(indent=2)
        if args.out:
            Path(args.out).write_text(out_text, encoding="utf-8")
        print(out_text)
    else:
        # Offline smoke test verifying prompt assembly, data-URI media blocks, and schema validation
        sample_file = Path(__file__).parent / "samples" / "sample_schemas.json"
        sample_data = json.loads(sample_file.read_text(encoding="utf-8"))
        tiny_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        blk = media_block_from_source(tiny_png)
        assert blk["type"] == "image" and blk["mime_type"] == "image/png"

        lst_home = json.loads(json.dumps(sample_data["home"]))
        lst_home["id"] = "lst1-home"
        lst_home["source"] = "listing"
        mock_listing = perceive_listing([], transport_override=lambda _err: lst_home)
        mock_video = perceive_home_video("mock.mp4", transport_override=lambda _err: sample_data["inventory"])
        assert mock_listing.ok and mock_video.ok
        print(
            json.dumps(
                {
                    "status": "ok",
                    "mode": "offline_self_test",
                    "models_configured": {"perception": MODEL_HYDROGEN, "hard_plans": MODEL_NEON},
                    "listing_home_id": mock_listing.data.id if mock_listing.data else None,
                    "video_items": len(mock_video.data.items) if mock_video.data else 0,
                },
                indent=2,
            )
        )
