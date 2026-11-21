"""Floor-plan weaving stage for Homeweave (`weave.py`, Prompt 5).

Merges the top-ranked rooms from 2+ listings into a single-level `Home` (`source="generated"`,
`new-` IDs) using `gemini-4-neon`, carrying over room fixtures and translated staging furniture,
and logging numeric `conflicts[]` instead of stretching/squeezing rooms > 10%.
Also provides `weave_geometric` as a deterministic local Strip-Packing weaver for offline tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from google import genai
from shapely.geometry import Polygon

from schemas import (
    Door,
    Fixture,
    Home,
    Pose2D,
    Provenance,
    Room,
    ScaleCheck,
    StagingItem,
    ValidatedStageResult,
    WeaveConflict,
    Window,
    to_gemini_response_format,
    validate_with_retry,
)

MODEL_NEON = "gemini-4-neon"

PROMPT_WEAVE = """You are merging the best parts of several homes into one floor plan a person could plausibly build.

Input: Home JSON for each listing, a ranked list of features with scores, and constraints {max_area_m2, bedrooms, bathrooms}.

Take the top-ranked room of each type. Arrange them on one level so that:
- every room is reachable through a door or opening, with no room entered only through a bedroom or bathroom;
- rooms do not overlap and shared walls line up;
- kitchen is next to dining or living; each bedroom is within one room of a bathroom;
- room sizes stay within 10 percent of their source listing;
- total area is under max_area_m2.

Return a Home JSON with `id = "new-home"`, `source = "generated"`, all room/door/window/fixture/staging ids prefixed `new-`, and for each room a `provenance` object naming `from_listing` and `from_room`. Carry over each room's fixtures and its staging furniture, translated into the room's new coordinates.

Also return `conflicts[]`: any feature or room you could not include and the reason in numbers, for example "lst2 kitchen island needs 4.2 m of width; the slot is 3.6 m". List a conflict; never squeeze or stretch a room to hide it.
"""


def weave_geometric(
    listings: Sequence[Home | dict[str, object]],
    *,
    max_area_m2: float = 120.0,
    bedrooms: int = 1,
    bathrooms: int = 1,
) -> Home:
    """Deterministic single-level floor-plan Strip Weaver (keeps exact room sizes + doors + staging)."""
    homes = [h if isinstance(h, Home) else Home.model_validate(h) for h in listings]

    # Rank rooms by best feature judge_score (or listing order)
    room_scores: dict[tuple[str, str], float] = {}
    for h in homes:
        for idx, feat in enumerate(h.features):
            sc = feat.judge_score.total_score if feat.judge_score else max(1.0, 8.0 - idx)
            key = (h.id, feat.room_id)
            room_scores[key] = max(room_scores.get(key, 0.0), sc)

    candidates: list[tuple[float, Home, Room]] = []
    for h in homes:
        for r in h.rooms:
            if r.polygon and len(r.polygon) >= 3:
                sc = room_scores.get((h.id, r.id), 1.0)
                candidates.append((sc, h, r))
    candidates.sort(key=lambda item: item[0], reverse=True)

    chosen: list[tuple[Home, Room]] = []
    seen_types: dict[str, int] = {}
    conflicts: list[WeaveConflict] = []
    used_area = 0.0

    desired_order = ["living", "kitchen", "dining", "primary_bedroom", "bedroom", "bathroom", "office"]
    sorted_cands = sorted(
        candidates,
        key=lambda tup: (
            desired_order.index(tup[2].type) if tup[2].type in desired_order else 99,
            -tup[0],
        ),
    )

    for sc, h, r in sorted_cands:
        assert r.polygon is not None
        r_poly = Polygon(r.polygon)
        area = r_poly.area
        rtype = r.type

        cap = 1
        if rtype in ("bedroom", "primary_bedroom"):
            cap = bedrooms if rtype == "bedroom" else 1
            total_beds = seen_types.get("bedroom", 0) + seen_types.get("primary_bedroom", 0)
            if total_beds >= bedrooms:
                continue
        elif rtype in ("bathroom", "primary_bathroom"):
            total_baths = seen_types.get("bathroom", 0) + seen_types.get("primary_bathroom", 0)
            if total_baths >= bathrooms:
                continue
        else:
            if seen_types.get(rtype, 0) >= cap:
                continue

        if used_area + area > max_area_m2:
            conflicts.append(
                WeaveConflict(
                    room_id=r.id,
                    reason=(
                        f"{r.id} ({rtype}) needs {area:.1f} m2; remaining budget under "
                        f"max_area_m2 ({max_area_m2:.1f} m2) is {max(0.0, max_area_m2 - used_area):.1f} m2"
                    ),
                    provenance=Provenance(source="weave_geometric", from_listing=h.id, from_room=r.id),
                )
            )
            continue

        seen_types[rtype] = seen_types.get(rtype, 0) + 1
        used_area += area
        chosen.append((h, r))

    # Lay out chosen rooms in two aligned rows along X (public living/kitchen/dining at z=0, private row contiguous above)
    new_rooms: list[Room] = []
    new_staging: list[StagingItem] = []
    cursor_row0_x = 0.0
    cursor_row1_x = 0.0
    row0_max_z = 0.0

    public_types = {"living", "great_room", "kitchen", "dining", "entry"}
    pub = [(h, r) for h, r in chosen if r.type in public_types]
    priv = [(h, r) for h, r in chosen if r.type not in public_types]

    def _place_room(h: Home, r: Room, target_min_x: float, target_min_z: float, idx_num: int) -> tuple[float, float]:
        assert r.polygon is not None
        poly = Polygon(r.polygon)
        minx, minz, maxx, maxz = poly.bounds
        dx = target_min_x - minx
        dz = target_min_z - minz
        w = maxx - minx
        d = maxz - minz

        slug = f"{r.type}-{idx_num:02d}" if idx_num > 1 else r.type
        new_rid = f"new-{slug.replace('_', '-')}"
        moved_poly = [[round(pt[0] + dx, 4), round(pt[1] + dz, 4)] for pt in r.polygon]

        moved_doors = [
            Door(
                id=f"new-door-{len(new_rooms) + 1:02d}-{di + 1}",
                wall_index=dr.wall_index,
                offset_m=dr.offset_m,
                width_m=dr.width_m,
                swing=dr.swing,
                null_reason=dr.null_reason,
                provenance=Provenance(source="weave_stage", from_listing=h.id, from_room=r.id),
            )
            for di, dr in enumerate(r.doors)
        ]
        if not moved_doors:
            moved_doors.append(
                Door(
                    id=f"new-door-{len(new_rooms) + 1:02d}-1",
                    wall_index=0,
                    offset_m=0.4,
                    width_m=0.81,
                    swing="inward_left",
                    provenance=Provenance(source="weave_stage", from_listing=h.id, from_room=r.id),
                )
            )

        moved_wins = [
            Window(
                id=f"new-win-{len(new_rooms) + 1:02d}-{wi + 1}",
                wall_index=wn.wall_index,
                offset_m=wn.offset_m,
                width_m=wn.width_m,
                sill_height_m=wn.sill_height_m,
                height_m=wn.height_m,
                provenance=Provenance(source="weave_stage", from_listing=h.id, from_room=r.id),
            )
            for wi, wn in enumerate(r.windows)
        ]

        moved_fixtures = [
            Fixture(
                id=f"new-fix-{len(new_rooms) + 1:02d}-{fi + 1}",
                room_id=new_rid,
                type=fx.type,
                footprint=(
                    [[round(p[0] + dx, 4), round(p[1] + dz, 4)] for p in fx.footprint]
                    if fx.footprint
                    else None
                ),
                null_reason=fx.null_reason,
                provenance=Provenance(source="weave_stage", from_listing=h.id, from_room=r.id),
            )
            for fi, fx in enumerate(r.fixtures)
        ]

        for si, st in enumerate(h.staging):
            if st.room_id == r.id:
                new_pose = (
                    Pose2D(
                        x=round(st.pose.x + dx, 4),
                        z=round(st.pose.z + dz, 4),
                        rot_deg=st.pose.rot_deg,
                    )
                    if st.pose
                    else None
                )
                new_staging.append(
                    StagingItem(
                        id=f"new-stage-{len(new_staging) + 1:02d}",
                        room_id=new_rid,
                        category=st.category,
                        dims_m=st.dims_m,
                        pose=new_pose,
                        group_kind=st.group_kind,
                        replaceable=True,
                        null_reason=st.null_reason,
                        provenance=Provenance(
                            source="weave_stage",
                            from_listing=h.id,
                            from_room=r.id,
                            photo_indices=list(st.provenance.photo_indices),
                        ),
                    )
                )

        new_rooms.append(
            Room(
                id=new_rid,
                type=r.type,
                polygon=moved_poly,
                ceiling_height_m=r.ceiling_height_m or 2.6,
                doors=moved_doors,
                windows=moved_wins,
                fixtures=moved_fixtures,
                provenance=Provenance(source="weave_stage", from_listing=h.id, from_room=r.id),
            )
        )
        return w, d

    type_counts: dict[str, int] = {}
    for h, r in pub:
        type_counts[r.type] = type_counts.get(r.type, 0) + 1
        w, d = _place_room(h, r, cursor_row0_x, 0.0, type_counts[r.type])
        cursor_row0_x += w
        row0_max_z = max(row0_max_z, d)

    for h, r in priv:
        type_counts[r.type] = type_counts.get(r.type, 0) + 1
        w, _ = _place_room(h, r, cursor_row1_x, row0_max_z, type_counts[r.type])
        cursor_row1_x += w

    return Home(
        id="new-home",
        source="generated",
        rooms=new_rooms,
        fixtures=[],
        staging=new_staging,
        features=[],
        conflicts=conflicts,
        scale_check=ScaleCheck(
            method="plan_labels",
            reference=f"Merged {len(new_rooms)} rooms ({used_area:.1f} m2 <= {max_area_m2:.1f} m2)",
            conflicts=[],
            provenance=Provenance(source="weave_stage"),
        ),
        provenance=Provenance(
            source="weave_stage",
            notes=f"Merged from {[h.id for h in homes]}",
        ),
    )


def weave(
    listings: Sequence[Home | dict[str, object]],
    *,
    max_area_m2: float = 120.0,
    bedrooms: int = 2,
    bathrooms: int = 1,
    use_model: bool = True,
    client: Optional[genai.Client] = None,
    transport_override: Optional[Callable[[Optional[str]], str | dict[str, object]]] = None,
) -> ValidatedStageResult[Home]:
    """Merge top features/rooms from `listings` into one generated `Home` (`gemini-4-neon`)."""
    if transport_override is not None:
        return validate_with_retry(transport_override, Home)

    if not use_model:
        geo = weave_geometric(
            listings, max_area_m2=max_area_m2, bedrooms=bedrooms, bathrooms=bathrooms
        )
        return ValidatedStageResult[Home](ok=True, flagged=False, attempts=1, data=geo, errors=[])

    homes = [h if isinstance(h, Home) else Home.model_validate(h) for h in listings]
    active_client = client or genai.Client()
    payload = {
        "listings": [h.model_dump() for h in homes],
        "constraints": {
            "max_area_m2": max_area_m2,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
        },
    }
    response_format = to_gemini_response_format(Home)

    def _producer(prev_error: Optional[str]) -> str:
        prompt = f"{PROMPT_WEAVE}\n\nINPUT JSON:\n{json.dumps(payload)}"
        if prev_error:
            prompt += f"\n\nPREVIOUS VALIDATION ERROR (fix all IDs, provenance, and schema fields):\n{prev_error}"
        interaction = active_client.interactions.create(
            model=MODEL_NEON,
            input=prompt,
            response_format=response_format,
            generation_config={"thinking_level": "high", "max_output_tokens": 32768},
            store=False,
        )
        if not interaction.output_text:
            raise ValueError("Empty response from gemini-4-neon weave.")
        return interaction.output_text

    return validate_with_retry(_producer, Home)


if __name__ == "__main__":
    sample_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent / "samples" / "sample_schemas.json"
    )
    sample = json.loads(sample_path.read_text(encoding="utf-8"))

    # Create two synthetic listing homes and weave them into one composite floor plan
    lst1 = json.loads(json.dumps(sample["home"]))
    lst1["id"] = "lst1-home"
    lst1["source"] = "listing"
    lst1["rooms"][0]["id"] = "lst1-living"
    lst1["rooms"][0]["doors"][0]["id"] = "lst1-door-01"
    lst1["rooms"][0]["windows"][0]["id"] = "lst1-win-01"
    lst1["rooms"][0]["fixtures"][0]["id"] = "lst1-tv-01"
    lst1["rooms"][0]["fixtures"][0]["room_id"] = "lst1-living"
    lst1["staging"][0]["id"] = "lst1-stage-01"
    lst1["staging"][0]["room_id"] = "lst1-living"

    lst2 = {
        "id": "lst2-home",
        "source": "listing",
        "rooms": [
            {
                "id": "lst2-kitchen",
                "type": "kitchen",
                "polygon": [[0.0, 0.0], [4.2, 0.0], [4.2, 3.6], [0.0, 3.6]],
                "ceiling_height_m": 2.6,
                "doors": [
                    {
                        "id": "lst2-door-01",
                        "wall_index": 3,
                        "offset_m": 1.2,
                        "width_m": 0.9,
                        "swing": "open",
                        "provenance": {"source": "plan"}
                    }
                ],
                "windows": [],
                "fixtures": [
                    {
                        "id": "lst2-island-01",
                        "room_id": "lst2-kitchen",
                        "type": "island",
                        "footprint": [[1.2, 1.2], [3.0, 1.2], [3.0, 2.1], [1.2, 2.1]],
                        "provenance": {"source": "plan"}
                    }
                ],
                "provenance": {"source": "plan"}
            },
            {
                "id": "lst2-bedroom",
                "type": "primary_bedroom",
                "polygon": [[0.0, 0.0], [6.0, 0.0], [6.0, 5.0], [0.0, 5.0]],
                "ceiling_height_m": 2.6,
                "doors": [],
                "windows": [],
                "fixtures": [],
                "provenance": {"source": "plan"}
            }
        ],
        "fixtures": [],
        "staging": [],
        "features": [],
        "conflicts": [],
        "scale_check": {"method": "plan_labels", "reference": "plan", "conflicts": [], "provenance": {"source": "plan"}},
        "provenance": {"source": "plan"}
    }

    res = weave([lst1, lst2], max_area_m2=45.0, bedrooms=1, bathrooms=1, use_model=False)
    assert res.ok and res.data is not None
    print(
        json.dumps(
            {
                "status": "ok",
                "woven_home_id": res.data.id,
                "woven_rooms": [
                    {"id": r.id, "type": r.type, "from_room": r.provenance.from_room}
                    for r in res.data.rooms
                ],
                "conflicts_logged": [c.reason for c in res.data.conflicts],
            },
            indent=2,
        )
    )
