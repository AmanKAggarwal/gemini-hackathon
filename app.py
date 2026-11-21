"""FastAPI server for Homeweave (`app.py`).

Wires all pure JSON stages (`perceive_listing`, `perceive_home_video`, `perceive_hard_plan`,
`judge_features`, `weave`, `arrange`, `judge_layouts`, `run_gemini_baseline`) to `./data`
and serves the architectural 3D Residence viewer (`viewer.html`) at `/`.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from shapely.geometry import Polygon

from arrange import arrange
from check import check_layout
from judge import judge_features, judge_layouts
from perceive import perceive_home_video, perceive_listing
from schemas import (
    CheckResult,
    CostBreakdown,
    Home,
    HomeStyleBundle,
    HomeVideoInventory,
    Layout,
    LeftoverItem,
    Placement,
    Provenance,
    Room,
    RoomType,
)
from style import build_default_style_bundle, extract_style_from_photo
from weave import weave

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = BASE_DIR / "samples"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def pair_rooms(src_rooms: list[Room], new_rooms: list[Room]) -> list[tuple[Room, Room]]:
    """Bipartite room-type + area matching between `src_rooms` and `new_rooms`."""
    candidates: list[tuple[float, int, int]] = []
    living_family = {"living", "great_room", "den"}
    bed_family = {"primary_bedroom", "bedroom"}
    bath_family = {"primary_bathroom", "bathroom"}

    for i, sr in enumerate(src_rooms):
        sa = Polygon(sr.polygon).area if sr.polygon and len(sr.polygon) >= 3 else 15.0
        for j, nr in enumerate(new_rooms):
            na = Polygon(nr.polygon).area if nr.polygon and len(nr.polygon) >= 3 else 15.0
            if sr.type == nr.type:
                type_cost = 0.0
            elif {sr.type, nr.type} <= living_family or {sr.type, nr.type} <= bed_family or {sr.type, nr.type} <= bath_family:
                type_cost = 0.35
            else:
                type_cost = 5.0
            area_cost = abs(sa - na) / max(sa, na, 1.0)
            candidates.append((type_cost + area_cost, i, j))
    candidates.sort(key=lambda t: t[0])

    used_s: set[int] = set()
    used_n: set[int] = set()
    pairs: list[tuple[Room, Room]] = []
    for cost, i, j in candidates:
        if i not in used_s and j not in used_n and cost < 3.5:
            used_s.add(i)
            used_n.add(j)
            pairs.append((src_rooms[i], new_rooms[j]))
    return pairs


def arrange_whole_home(inv: HomeVideoInventory, home: Home) -> tuple[list[Layout], list[LeftoverItem]]:
    """Arrange every paired room in `home` and combine into up to 5 whole-residence `Layout`s."""
    pairs = pair_rooms(inv.rooms, home.rooms)
    paired_src_ids = {sr.id for sr, _ in pairs}

    unmatched_leftovers: list[LeftoverItem] = [
        LeftoverItem(
            item_id=it.id,
            reason=f"unmatched_room: no matching room in {home.id} for source room {it.room_id}",
            provenance=it.provenance,
        )
        for it in inv.items
        if it.room_id not in paired_src_ids
    ]

    per_pair_layouts: list[list[Layout]] = []
    for sr, nr in pairs:
        r_items = [it for it in inv.items if it.room_id == sr.id]
        if not r_items:
            continue
        r_groups = [g for g in inv.groups if g.room_id in (None, sr.id)]
        r_stage = [st for st in home.staging if st.room_id == nr.id]
        r_layouts, _ = arrange(sr, r_items, r_groups, nr, staging=r_stage)
        if r_layouts:
            per_pair_layouts.append(r_layouts)

    if not per_pair_layouts:
        return [], unmatched_leftovers

    max_k = max(len(ls) for ls in per_pair_layouts)
    combined_layouts: list[Layout] = []

    for k in range(min(5, max_k)):
        merged_pls: list[Placement] = []
        merged_los: list[LeftoverItem] = list(unmatched_leftovers)
        merged_warnings: list[str] = []
        overlaps_all: list[list[str]] = []
        door_blocks_all: list[str] = []
        outside_all: list[str] = []
        flagged_all: list[str] = []
        clear_fails_sum = 0
        clear_paths_all: list[str] = []
        rel_scores: list[float] = []
        cost_sum = CostBreakdown()

        for ls in per_pair_layouts:
            chosen = ls[min(k, len(ls) - 1)]
            merged_pls.extend(chosen.placements)
            merged_los.extend(chosen.leftovers)
            merged_warnings.extend(chosen.warnings)
            cost_sum = CostBreakdown(
                fixture_mismatch=round(cost_sum.fixture_mismatch + chosen.costs.fixture_mismatch, 4),
                wall_mismatch=round(cost_sum.wall_mismatch + chosen.costs.wall_mismatch, 4),
                clearance_penalty=round(cost_sum.clearance_penalty + chosen.costs.clearance_penalty, 4),
                staging_distance=round(cost_sum.staging_distance + chosen.costs.staging_distance, 4),
                group_overlap=round(cost_sum.group_overlap + chosen.costs.group_overlap, 4),
                group_distance_delta=round(cost_sum.group_distance_delta + chosen.costs.group_distance_delta, 4),
                total=round(cost_sum.total + chosen.costs.total, 4),
            )
            if chosen.check:
                overlaps_all.extend(chosen.check.overlaps)
                door_blocks_all.extend(chosen.check.door_blocks)
                outside_all.extend(chosen.check.outside_room)
                flagged_all.extend(chosen.check.flagged_items)
                clear_fails_sum += chosen.check.clearance_fails
                clear_paths_all.extend(chosen.check.clearance_fail_paths)
                rel_scores.append(chosen.check.relations_kept)

        mean_rel = round(sum(rel_scores) / len(rel_scores), 4) if rel_scores else 1.0
        v_total = len(overlaps_all) + len(door_blocks_all) + clear_fails_sum + len(outside_all)
        merged_check = CheckResult(
            overlaps=overlaps_all,
            door_blocks=sorted(set(door_blocks_all)),
            clearance_fails=clear_fails_sum,
            clearance_fail_paths=clear_paths_all,
            outside_room=sorted(set(outside_all)),
            relations_kept=mean_rel,
            violations_total=v_total,
            flagged_items=sorted(set(flagged_all)),
        )
        combined_layouts.append(
            Layout(
                id=f"new-layout-{k + 1:02d}",
                rank=k + 1,
                room_pair=[f"{sr.id}->{nr.id}" for sr, nr in pairs],
                placements=merged_pls,
                costs=cost_sum,
                check=merged_check,
                warnings=merged_warnings,
                leftovers=merged_los,
                provenance=Provenance(
                    source="arrange_engine",
                    notes=f"Whole-residence layout {k + 1} across {len(pairs)} rooms",
                ),
            )
        )

    top_lo = combined_layouts[0].leftovers if combined_layouts else unmatched_leftovers
    return combined_layouts, top_lo


def ensure_seed_data(force: bool = False) -> None:
    """Seed `./data/home.json`, `./data/inventory.json`, `./data/layouts.json`, and `./data/styles.json`."""
    styles_file = DATA_DIR / "styles.json"
    if force or not styles_file.exists():
        styles_file.write_text(build_default_style_bundle().model_dump_json(indent=2), encoding="utf-8")

    home_file = DATA_DIR / "home.json"
    layouts_file = DATA_DIR / "layouts.json"
    inv_file = DATA_DIR / "inventory.json"
    if not force and home_file.exists() and layouts_file.exists() and inv_file.exists():
        return

    res_bundle = json.loads((SAMPLES_DIR / "residence_1br.json").read_text(encoding="utf-8"))
    seeded_home = Home.model_validate(res_bundle["home"])
    seeded_inv = HomeVideoInventory.model_validate(res_bundle["inventory"])
    layouts, _ = arrange_whole_home(seeded_inv, seeded_home)

    home_file.write_text(seeded_home.model_dump_json(indent=2), encoding="utf-8")
    inv_file.write_text(seeded_inv.model_dump_json(indent=2), encoding="utf-8")
    layouts_file.write_text(
        json.dumps([lay.model_dump() for lay in layouts], indent=2),
        encoding="utf-8",
    )


ensure_seed_data(force=False)
app = FastAPI(title="Homeweave", version="1.2.0")


class PerceiveListingReq(BaseModel):
    photo_sources: list[str] = Field(default_factory=list)
    plan_source: Optional[str] = None
    listing_index: int = 1
    hard_plan: bool = False


class PerceiveVideoReq(BaseModel):
    video_source: str


class JudgeFeaturesReq(BaseModel):
    home: dict[str, Any]
    photo_sources: list[str] = Field(default_factory=list)
    rubric_text: str = "Bright open living space with room for our existing furniture"


class WeaveReq(BaseModel):
    listings: list[dict[str, Any]]
    max_area_m2: float = 120.0
    bedrooms: int = 2
    bathrooms: int = 1
    use_model: bool = True


class ArrangeReq(BaseModel):
    src_room: Optional[dict[str, Any]] = None
    src_items: Optional[list[dict[str, Any]]] = None
    groups: list[dict[str, Any]] = Field(default_factory=list)
    new_room: Optional[dict[str, Any]] = None
    staging: Optional[list[dict[str, Any]]] = None


class JudgeRendersReq(BaseModel):
    layout_id: Optional[str] = None
    renders: list[str] = Field(default_factory=list)
    renders_by_layout: dict[str, list[str]] = Field(default_factory=dict)
    has_bed: bool = False


class ExtractStyleReq(BaseModel):
    photo_source: str = Field(description="Preset ID ('L2', 'K3'), path, or data:image/... URI.")
    room_type: RoomType = "living"
    style_id: str = "custom"
    palette_hint: Optional[dict[str, str]] = None


class SpaceInputEntry(BaseModel):
    room_type: RoomType
    photo_source: str = Field(description="Preset ID ('L1'..'B4') or uploaded data:image/... URI.")
    style_id: str = "custom"
    palette_hint: Optional[dict[str, str]] = None


class BuildStyledResidenceReq(BaseModel):
    spaces: list[SpaceInputEntry]


@app.get("/", response_class=HTMLResponse)
@app.get("/viewer.html", response_class=HTMLResponse)
def serve_viewer() -> HTMLResponse:
    return HTMLResponse((BASE_DIR / "viewer.html").read_text(encoding="utf-8"))


@app.get("/home.json")
def serve_home_root() -> FileResponse:
    ensure_seed_data()
    return FileResponse(DATA_DIR / "home.json", media_type="application/json")


@app.get("/layouts.json")
def serve_layouts_root() -> FileResponse:
    ensure_seed_data()
    return FileResponse(DATA_DIR / "layouts.json", media_type="application/json")


@app.get("/styles.json")
def serve_styles_root() -> FileResponse:
    ensure_seed_data()
    return FileResponse(DATA_DIR / "styles.json", media_type="application/json")


@app.get("/drive_inputs/{filename}")
def serve_drive_input(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    target = DATA_DIR / "drive_inputs" / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"{safe_name} not found in drive_inputs")
    return FileResponse(target)


@app.post("/style/extract")
def api_extract_style(req: ExtractStyleReq) -> JSONResponse:
    res = extract_style_from_photo(
        req.photo_source,
        room_type=req.room_type,
        style_id=req.style_id,
        client_palette_hint=req.palette_hint,
    )
    if res.ok and res.data is not None:
        styles_path = DATA_DIR / "styles.json"
        bundle = (
            HomeStyleBundle.model_validate_json(styles_path.read_text(encoding="utf-8"))
            if styles_path.exists()
            else build_default_style_bundle()
        )
        bundle.by_room_type[req.room_type] = res.data
        bundle.catalog[res.data.style_id] = res.data
        styles_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    return JSONResponse(res.model_dump())


@app.post("/style/build-residence")
def api_build_styled_residence(req: BuildStyledResidenceReq) -> JSONResponse:
    """Process input pictures for each space (living, kitchen, bedroom, bathroom) and return full 3D payload."""
    ensure_seed_data()
    styles_path = DATA_DIR / "styles.json"
    bundle = (
        HomeStyleBundle.model_validate_json(styles_path.read_text(encoding="utf-8"))
        if styles_path.exists()
        else build_default_style_bundle()
    )

    extracted_summary: dict[str, Any] = {}
    for sp in req.spaces:
        res = extract_style_from_photo(
            sp.photo_source,
            room_type=sp.room_type,
            style_id=sp.style_id,
            client_palette_hint=sp.palette_hint,
        )
        if res.ok and res.data is not None:
            bundle.by_room_type[sp.room_type] = res.data
            bundle.catalog[res.data.style_id] = res.data
            extracted_summary[sp.room_type] = res.data.model_dump()

    styles_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    home_json = json.loads((DATA_DIR / "home.json").read_text(encoding="utf-8"))
    layouts_json = json.loads((DATA_DIR / "layouts.json").read_text(encoding="utf-8"))
    return JSONResponse(
        {
            "ok": True,
            "styles": bundle.model_dump(),
            "extracted": extracted_summary,
            "home": home_json,
            "layouts": layouts_json,
        }
    )


@app.get("/data/{filename}")
def serve_data_file(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    target = DATA_DIR / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"{safe_name} not found in ./data")
    return FileResponse(target, media_type="application/json")


@app.post("/perceive/listing")
def api_perceive_listing(req: PerceiveListingReq) -> JSONResponse:
    res = perceive_listing(
        req.photo_sources,
        plan_source=req.plan_source,
        listing_index=req.listing_index,
        hard_plan=req.hard_plan,
    )
    if res.ok and res.data is not None:
        (DATA_DIR / f"lst{req.listing_index}_home.json").write_text(
            res.data.model_dump_json(indent=2), encoding="utf-8"
        )
    return JSONResponse(res.model_dump())


@app.post("/perceive/video")
def api_perceive_video(req: PerceiveVideoReq) -> JSONResponse:
    res = perceive_home_video(req.video_source)
    if res.ok and res.data is not None:
        (DATA_DIR / "inventory.json").write_text(res.data.model_dump_json(indent=2), encoding="utf-8")
    return JSONResponse(res.model_dump())


@app.post("/judge/features")
def api_judge_features(req: JudgeFeaturesReq) -> JSONResponse:
    scored = judge_features(req.home, photo_sources=req.photo_sources, rubric_text=req.rubric_text)
    return JSONResponse(scored.model_dump())


@app.post("/weave")
def api_weave(req: WeaveReq) -> JSONResponse:
    res = weave(
        req.listings,
        max_area_m2=req.max_area_m2,
        bedrooms=req.bedrooms,
        bathrooms=req.bathrooms,
        use_model=req.use_model,
    )
    if res.ok and res.data is not None:
        (DATA_DIR / "home.json").write_text(res.data.model_dump_json(indent=2), encoding="utf-8")
    return JSONResponse(res.model_dump())


@app.post("/arrange")
def api_arrange(req: ArrangeReq) -> JSONResponse:
    if req.src_room and req.new_room and req.src_items is not None:
        layouts, leftovers = arrange(
            req.src_room, req.src_items, req.groups, req.new_room, staging=req.staging
        )
    else:
        inv_path = DATA_DIR / "inventory.json"
        home_path = DATA_DIR / "home.json"
        if not inv_path.exists() or not home_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Provide room payload or save ./data/inventory.json and ./data/home.json first.",
            )
        inv = HomeVideoInventory.model_validate_json(inv_path.read_text(encoding="utf-8"))
        home = Home.model_validate_json(home_path.read_text(encoding="utf-8"))
        layouts, leftovers = arrange_whole_home(inv, home)

    (DATA_DIR / "layouts.json").write_text(
        json.dumps([lay.model_dump() for lay in layouts], indent=2),
        encoding="utf-8",
    )
    return JSONResponse(
        {
            "layouts": [lay.model_dump() for lay in layouts],
            "leftovers": [lo.model_dump() for lo in leftovers],
        }
    )


@app.post("/judge")
def api_judge_layouts(req: JudgeRendersReq) -> JSONResponse:
    layouts_path = DATA_DIR / "layouts.json"
    if not layouts_path.exists():
        raise HTTPException(status_code=404, detail="layouts.json not found in ./data")

    saved_layouts = [
        Layout.model_validate(x)
        for x in json.loads(layouts_path.read_text(encoding="utf-8"))
    ]

    renders_map = dict(req.renders_by_layout)
    if req.layout_id and req.renders:
        renders_map[req.layout_id] = req.renders

    renders_dir = DATA_DIR / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    for lid, rlist in renders_map.items():
        for idx, durl in enumerate(rlist):
            if durl.startswith("data:image/png;base64,"):
                raw_png = base64.b64decode(durl.split(",", 1)[1])
                (renders_dir / f"{lid}_view{idx}.png").write_bytes(raw_png)

    target_layouts = (
        [l for l in saved_layouts if l.id in renders_map]
        if renders_map
        else saved_layouts
    )

    if not os.environ.get("GEMINI_API_KEY"):
        def _fallback_runner(_inp: list[dict[str, Any]], qs: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
            return {
                k: ({"noul": 0.06 if k == "blocks_window" else 0.92, "confidence": 0.88} if q["type"] == "noul" else {"score": 3.3, "confidence": 0.85})
                for k, q in qs.items()
            }
        judged = judge_layouts(
            target_layouts, renders_map, has_bed=req.has_bed, decision_runner=_fallback_runner
        )
        judged["note"] = "GEMINI_API_KEY not set; used offline mock rubric runner."
    else:
        judged = judge_layouts(target_layouts, renders_map, has_bed=req.has_bed)

    scored_by_id = {l["id"]: Layout.model_validate(l) for l in judged["layouts"]}
    merged = [scored_by_id.get(l.id, l) for l in saved_layouts]
    merged.sort(
        key=lambda x: (x.judge_score.total_score if x.judge_score else -x.costs.total),
        reverse=True,
    )
    final_layouts = [m.model_copy(update={"rank": i + 1}) for i, m in enumerate(merged)]
    layouts_path.write_text(
        json.dumps([fl.model_dump() for fl in final_layouts], indent=2),
        encoding="utf-8",
    )
    judged["layouts"] = [fl.model_dump() for fl in final_layouts]
    return JSONResponse(judged)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
