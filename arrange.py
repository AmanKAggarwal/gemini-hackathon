"""2D geometric correspondence & arrangement engine for Homeweave (`arrange.py`, Prompt 6).

Takes a source room, source furniture inventory, functional groups, target room, and optional
staging items; returns up to 5 ranked collision-free `Layout`s plus `leftovers` in < 2 s.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

from check import (
    AREA_EPS,
    WALL_CONTAIN_TOL,
    check_layout,
    door_geometry,
    facing_unit_vector,
    inward_wall_normal,
    item_footprint,
    wall_segments,
)
from schemas import (
    CostBreakdown,
    FunctionalGroup,
    FurnitureItem,
    Layout,
    LeftoverItem,
    Placement,
    Pose2D,
    Provenance,
    Room,
    StagingItem,
)

# All soft-cost weights live in one dict at the top of the file (Prompt 6).
WEIGHTS: dict[str, float] = {
    "fixture_mismatch": 1.0,
    "wall_mismatch": 0.5,
    "clearance_penalty": 1.0,
    "staging_distance": 0.1,
    "group_overlap": 1.0,
    "group_distance_delta": 1.0,
    "dropped_item_penalty": 5.0,
}

SLIDE_STEP_M = 0.25
REFINE_STEP_M = 0.10
WALL_INSET_GAP_M = 0.03
BEAM_WIDTH = 5
CANDIDATES_PER_GROUP = 5


@dataclass
class _LocalMember:
    item: FurnitureItem
    lx: float  # along anchor wall tangent
    lz: float  # along anchor wall inward normal (depth into room)
    rel_rot_deg: float  # item rot_deg minus anchor frame rot_deg


@dataclass
class _GroupSpec:
    id: str
    kind: str
    members: list[_LocalMember]
    bbox_w: float  # span along local X
    bbox_d: float  # span along local Z
    area: float
    src_centroid: tuple[float, float]
    against_wall: bool
    nearest_fixture: Optional[str]
    primary_rel_rot_deg: float


@dataclass
class _GroupCandidate:
    group_id: str
    centroid: tuple[float, float]
    rot_deg: float
    mirrored: bool
    wall_index: Optional[int]
    against_wall: bool
    placements: list[Placement]
    footprints: dict[str, Polygon]
    union_poly: Polygon
    fixture_mismatch: float
    wall_mismatch: float
    clearance_penalty: float
    staging_distance: float
    score: float


def _frame_rot_from_inward_normal(n_in: np.ndarray) -> float:
    """Clockwise angle from +z so local +z points along `n_in`."""
    # facing_unit_vector(theta) = (-sin(theta), cos(theta)) = (nx, nz)
    return math.degrees(math.atan2(-float(n_in[0]), float(n_in[1]))) % 360.0


def _world_to_local(
    wx: float, wz: float, origin_x: float, origin_z: float, frame_rot_deg: float
) -> tuple[float, float]:
    dx = wx - origin_x
    dz = wz - origin_z
    theta = math.radians(frame_rot_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    lx = dx * c + dz * s
    lz = -dx * s + dz * c
    return lx, lz


def _local_to_world(
    lx: float, lz: float, origin_x: float, origin_z: float, frame_rot_deg: float
) -> tuple[float, float]:
    theta = math.radians(frame_rot_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    wx = origin_x + lx * c - lz * s
    wz = origin_z + lx * s + lz * c
    return wx, wz


def _room_fixture_points(room: Room, segs: Sequence[tuple[np.ndarray, np.ndarray]], room_poly: Polygon) -> list[tuple[str, np.ndarray]]:
    """Collect `(fixture_kind, centroid_xz)` for windows, doors, and built-in fixtures."""
    pts: list[tuple[str, np.ndarray]] = []
    for win in room.windows:
        if segs:
            p0, p1 = segs[win.wall_index % len(segs)]
            t, _, wlen = inward_wall_normal(p0, p1, room_poly)
            off = min(win.offset_m + 0.5 * win.width_m, wlen)
            pts.append(("window", p0 + off * t))
    for door in room.doors:
        _, _, d_mid, _ = door_geometry(door, segs, room_poly)
        pts.append(("door", d_mid))
    for fix in room.fixtures:
        if fix.footprint and len(fix.footprint) >= 3:
            fp = Polygon(fix.footprint)
            pts.append((fix.type, np.array([fp.centroid.x, fp.centroid.y])))
    return pts


def _build_group_specs(
    src_room: Room,
    src_items: Sequence[FurnitureItem],
    groups: Sequence[FunctionalGroup],
    max_new_wall_m: float,
    new_room_bounds_span: tuple[float, float],
) -> tuple[list[_GroupSpec], list[LeftoverItem]]:
    """Describe source groups in local wall-aligned coordinates; filter unplaceable items."""
    leftovers: list[LeftoverItem] = []
    valid_items: dict[str, FurnitureItem] = {}
    max_room_dim = max(max_new_wall_m, max(new_room_bounds_span))

    for it in src_items:
        if it.dims_m is None or it.pose is None:
            reason = it.null_reason or "missing dimensions or pose in source video"
            leftovers.append(
                LeftoverItem(
                    item_id=it.id,
                    reason=f"missing_geometry: {reason}",
                    provenance=it.provenance,
                )
            )
            continue
        min_span = min(it.dims_m.w, it.dims_m.d)
        max_span = max(it.dims_m.w, it.dims_m.d)
        if max_span > max_new_wall_m + 1e-3 or min_span > min(new_room_bounds_span) + 1e-3:
            leftovers.append(
                LeftoverItem(
                    item_id=it.id,
                    reason=f"needs {max_span:.2f} m; longest free wall is {max_new_wall_m:.2f} m",
                    suggestions=["move to a larger room or omit"],
                    provenance=it.provenance,
                )
            )
            continue
        valid_items[it.id] = it

    # Build group membership; unassigned valid items become singleton groups
    assigned: set[str] = set()
    raw_groups: list[tuple[str, str, list[FurnitureItem]]] = []
    for grp in groups:
        members = [valid_items[iid] for iid in grp.item_ids if iid in valid_items]
        if members:
            raw_groups.append((grp.id, grp.kind, members))
            assigned.update(m.id for m in members)
    for iid, it in valid_items.items():
        if iid not in assigned:
            gid = it.group_id or f"src-grp-single-{iid.split('-', 1)[-1]}"
            raw_groups.append((gid, "other", [it]))

    src_poly = Polygon(src_room.polygon) if src_room.polygon and len(src_room.polygon) >= 3 else None
    src_segs = wall_segments(src_room.polygon) if src_room.polygon else []
    src_fix_pts = _room_fixture_points(src_room, src_segs, src_poly) if src_poly else []

    specs: list[_GroupSpec] = []
    for gid, kind, members in raw_groups:
        primary = max(members, key=lambda m: m.dims_m.w * m.dims_m.d if m.dims_m else 0.0)
        against_wall = any(m.against_wall for m in members)

        # Anchor frame: align local +z with nearest source wall's inward normal when against_wall
        frame_rot = 0.0
        if src_poly and src_segs and primary.pose:
            p_pt = np.array([primary.pose.x, primary.pose.z])
            best_dist = float("inf")
            best_n_in = np.array([0.0, 1.0])
            for p0, p1 in src_segs:
                line = LineString([(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))])
                d = line.distance(Polygon(item_footprint(primary.dims_m, primary.pose))) if primary.dims_m else float("inf")
                if d < best_dist:
                    best_dist = d
                    _, best_n_in, _ = inward_wall_normal(p0, p1, src_poly)
            if best_dist <= 0.35:
                against_wall = True
            if against_wall:
                frame_rot = _frame_rot_from_inward_normal(best_n_in)

        cx = float(np.mean([m.pose.x for m in members if m.pose]))
        cz = float(np.mean([m.pose.z for m in members if m.pose]))

        # Compute bounding box in group's local frame and recenter on local bbox center
        temp_coords: list[tuple[FurnitureItem, float, float, float]] = []
        min_lx, max_lx = float("inf"), float("-inf")
        min_lz, max_lz = float("inf"), float("-inf")
        total_area = 0.0

        for m in members:
            assert m.pose is not None and m.dims_m is not None
            fp = item_footprint(m.dims_m, m.pose)
            total_area += m.dims_m.w * m.dims_m.d
            for vx, vz in fp.exterior.coords[:-1]:
                lx, lz = _world_to_local(vx, vz, cx, cz, frame_rot)
                min_lx = min(min_lx, lx)
                max_lx = max(max_lx, lx)
                min_lz = min(min_lz, lz)
                max_lz = max(max_lz, lz)
            mlx, mlz = _world_to_local(m.pose.x, m.pose.z, cx, cz, frame_rot)
            rel_rot = (m.pose.rot_deg - frame_rot) % 360.0
            temp_coords.append((m, mlx, mlz, rel_rot))

        mid_lx = 0.5 * (min_lx + max_lx)
        mid_lz = 0.5 * (min_lz + max_lz)
        adj_cx, adj_cz = _local_to_world(mid_lx, mid_lz, cx, cz, frame_rot)

        local_members = [
            _LocalMember(item=m, lx=mlx - mid_lx, lz=mlz - mid_lz, rel_rot_deg=rel_rot)
            for m, mlx, mlz, rel_rot in temp_coords
        ]
        bbox_w = max(0.05, max_lx - min_lx)
        bbox_d = max(0.05, max_lz - min_lz)

        # Nearest fixture type
        nearest_fix: Optional[str] = next(
            (m.nearest_fixture for m in members if m.nearest_fixture), None
        )
        if nearest_fix is None and src_fix_pts:
            g_vec = np.array([adj_cx, adj_cz])
            dists = [(ftype, float(np.linalg.norm(fpt - g_vec))) for ftype, fpt in src_fix_pts]
            ftype, fdist = min(dists, key=lambda kv: kv[1])
            if fdist <= 2.2:
                nearest_fix = ftype

        primary_rel_rot = (primary.pose.rot_deg - frame_rot) % 360.0 if primary.pose else 0.0
        specs.append(
            _GroupSpec(
                id=gid,
                kind=kind,
                members=local_members,
                bbox_w=bbox_w,
                bbox_d=bbox_d,
                area=total_area,
                src_centroid=(adj_cx, adj_cz),
                against_wall=against_wall,
                nearest_fixture=nearest_fix,
                primary_rel_rot_deg=primary_rel_rot,
            )
        )

    specs.sort(key=lambda g: g.area, reverse=True)
    return specs, leftovers


def _instantiate_group(
    spec: _GroupSpec,
    cx: float,
    cz: float,
    group_rot_deg: float,
    mirrored: bool,
    src_room_id: str,
    new_room_id: str,
    matched_neighbors: list[str],
) -> tuple[list[Placement], dict[str, Polygon]]:
    """Place all members of `spec` rigidly at `(cx, cz)` with `group_rot_deg` and optional mirror."""
    placements: list[Placement] = []
    footprints: dict[str, Polygon] = {}
    for lm in spec.members:
        lx = -lm.lx if mirrored else lm.lx
        lz = lm.lz
        rel_rot = (-lm.rel_rot_deg) % 360.0 if mirrored else lm.rel_rot_deg
        rot_deg = (group_rot_deg + rel_rot) % 360.0
        wx, wz = _local_to_world(lx, lz, cx, cz, group_rot_deg)
        pose = Pose2D(x=round(wx, 4), z=round(wz, 4), rot_deg=round(rot_deg, 2))
        assert lm.item.dims_m is not None
        fp = item_footprint(lm.item.dims_m, pose)
        footprints[lm.item.id] = fp
        placements.append(
            Placement(
                item_id=lm.item.id,
                category=lm.item.category,
                group_id=spec.id,
                dims_m=lm.item.dims_m,
                room_id=new_room_id,
                room_pair=[src_room_id, new_room_id],
                pose=pose,
                confidence=lm.item.confidence,
                matched_neighbors=list(matched_neighbors),
                provenance=Provenance(
                    source="arrange_engine",
                    from_room=src_room_id,
                    seen_at=list(lm.item.provenance.seen_at),
                    scale_ref=lm.item.provenance.scale_ref,
                ),
            )
        )
    return placements, footprints


def _generate_candidates_for_group(
    spec: _GroupSpec,
    src_room: Room,
    new_room: Room,
    room_poly: Polygon,
    room_padded_prep: object,
    segs: Sequence[tuple[np.ndarray, np.ndarray]],
    door_swings: Sequence[tuple[str, LineString, Optional[Polygon]]],
    win_lines: Sequence[tuple[str, LineString, float]],
    fixture_polys: Sequence[tuple[str, str, Polygon]],
    new_fix_pts: Sequence[tuple[str, np.ndarray]],
    staging_items: Sequence[StagingItem],
) -> list[_GroupCandidate]:
    """Slide rigid group along every viable wall in 0.25 m steps, 90 deg rotations, optional mirror."""
    raw_candidates: list[_GroupCandidate] = []

    # Matching staging anchor if provided
    staging_pts: list[np.ndarray] = []
    primary_cat = spec.members[0].item.category if spec.members else ""
    for st in staging_items:
        if st.pose is not None and (st.group_kind == spec.kind or st.category == primary_cat):
            staging_pts.append(np.array([st.pose.x, st.pose.z]))

    for w_idx, (p0, p1) in enumerate(segs):
        t, n_in, wall_len = inward_wall_normal(p0, p1, room_poly)
        base_wall_rot = _frame_rot_from_inward_normal(n_in)

        for rot_step in (0.0, 90.0, 180.0, 270.0):
            span_along = spec.bbox_w if rot_step in (0.0, 180.0) else spec.bbox_d
            span_into = spec.bbox_d if rot_step in (0.0, 180.0) else spec.bbox_w
            if wall_len + 1e-3 < span_along:
                continue

            start_s = 0.5 * span_along + WALL_INSET_GAP_M
            end_s = wall_len - 0.5 * span_along - WALL_INSET_GAP_M
            if end_s < start_s - 1e-3:
                s_positions = [0.5 * wall_len]
            else:
                num_steps = max(1, int(math.floor((end_s - start_s) / SLIDE_STEP_M)) + 1)
                s_positions = [start_s + k * SLIDE_STEP_M for k in range(num_steps)]
                if end_s - s_positions[-1] > 0.05:
                    s_positions.append(end_s)

            insets = [0.5 * span_into + WALL_INSET_GAP_M]
            if not spec.against_wall:
                insets.append(0.5 * span_into + 0.95)

            for inset in insets:
                cand_against_wall = inset <= 0.5 * span_into + 0.15
                for s_off in s_positions:
                    center_xz = p0 + s_off * t + inset * n_in
                    cx, cz = float(center_xz[0]), float(center_xz[1])

                    for mirrored in (False, True):
                        group_rot = (base_wall_rot + rot_step) % 360.0
                        placements, fps = _instantiate_group(
                            spec,
                            cx,
                            cz,
                            group_rot,
                            mirrored,
                            src_room.id,
                            new_room.id,
                            matched_neighbors=[],
                        )
                        union_fp = unary_union(list(fps.values()))
                        if not room_padded_prep.contains(union_fp):
                            continue

                        # Built-in floor fixture hard collision check
                        hits_builtin = False
                        for _, ftype, fpoly in fixture_polys:
                            if ftype not in ("tv_wall", "window") and union_fp.intersects(fpoly):
                                if union_fp.intersection(fpoly).area > AREA_EPS:
                                    hits_builtin = True
                                    break
                        if hits_builtin:
                            continue

                        # 1. Fixture mismatch (weight 1.0)
                        matched_neighbors: list[str] = []
                        fix_mismatch = 0.0
                        if spec.nearest_fixture:
                            same_type = [
                                fpt
                                for ftype, fpt in new_fix_pts
                                if ftype == spec.nearest_fixture
                            ]
                            if same_type:
                                min_d = min(float(np.linalg.norm(fpt - center_xz)) for fpt in same_type)
                                if spec.nearest_fixture == "tv_wall":
                                    # Conversation sets often face tv_wall from across the room; media stands sit near it
                                    if spec.kind == "conversation":
                                        primary_pose = placements[0].pose
                                        fv = facing_unit_vector(primary_pose.rot_deg)
                                        to_tv = same_type[0] - np.array([primary_pose.x, primary_pose.z])
                                        norm_tv = float(np.linalg.norm(to_tv))
                                        cos_a = (
                                            (fv[0] * to_tv[0] + fv[1] * to_tv[1]) / norm_tv
                                            if norm_tv > 0.05
                                            else 1.0
                                        )
                                        fix_mismatch = 0.0 if cos_a > 0.65 else 1.0
                                    else:
                                        fix_mismatch = 0.0 if min_d <= 1.2 else min(1.0, (min_d - 1.2) / 2.5)
                                else:
                                    fix_mismatch = 0.0 if min_d <= 2.2 else min(1.0, (min_d - 2.2) / 2.5)
                                if fix_mismatch < 0.4:
                                    matched_neighbors.append(spec.nearest_fixture)
                            else:
                                fix_mismatch = 0.5

                        # 2. Wall mismatch (weight 0.5): against-wall parity + orientation relative to wall
                        wall_mismatch = 0.0
                        if spec.against_wall != cand_against_wall:
                            wall_mismatch += 1.0
                        if spec.against_wall and rot_step != 0.0:
                            wall_mismatch += 0.5 if rot_step == 180.0 else 0.35

                        # 3. Clearance to doors and windows (weight 1.0)
                        clear_pen = 0.0
                        for did, d_line, d_swing in door_swings:
                            if d_swing is not None and union_fp.intersects(d_swing):
                                if union_fp.intersection(d_swing).area > AREA_EPS:
                                    clear_pen += 4.0
                            dist_d = union_fp.distance(d_line)
                            if dist_d < 0.90:
                                clear_pen += (0.90 - dist_d) * 1.5
                        for wid, w_line, sill_h in win_lines:
                            for pl in placements:
                                assert pl.dims_m is not None
                                if pl.dims_m.h > sill_h + 0.15:
                                    d_w = fps[pl.item_id].distance(w_line)
                                    if d_w < 0.20:
                                        clear_pen += 0.6
                                else:
                                    if fps[pl.item_id].distance(w_line) < 0.60 and wid not in matched_neighbors:
                                        matched_neighbors.append(wid)

                        # 4. Staging distance (weight 0.1)
                        stage_dist = (
                            min(float(np.linalg.norm(spt - center_xz)) for spt in staging_pts)
                            if staging_pts
                            else 0.0
                        )

                        if matched_neighbors:
                            placements = [
                                pl.model_copy(update={"matched_neighbors": matched_neighbors})
                                for pl in placements
                            ]

                        score = (
                            WEIGHTS["fixture_mismatch"] * fix_mismatch
                            + WEIGHTS["wall_mismatch"] * wall_mismatch
                            + WEIGHTS["clearance_penalty"] * clear_pen
                            + WEIGHTS["staging_distance"] * stage_dist
                        )
                        raw_candidates.append(
                            _GroupCandidate(
                                group_id=spec.id,
                                centroid=(cx, cz),
                                rot_deg=group_rot,
                                mirrored=mirrored,
                                wall_index=w_idx,
                                against_wall=cand_against_wall,
                                placements=placements,
                                footprints=fps,
                                union_poly=union_fp,
                                fixture_mismatch=fix_mismatch,
                                wall_mismatch=wall_mismatch,
                                clearance_penalty=clear_pen,
                                staging_distance=stage_dist,
                                score=score,
                            )
                        )

    raw_candidates.sort(key=lambda c: c.score)

    # Keep top 5 diverse candidates (distinct wall/rotation or >= 0.45 m apart)
    diverse: list[_GroupCandidate] = []
    for cand in raw_candidates:
        too_close = False
        for kept in diverse:
            dist = math.hypot(cand.centroid[0] - kept.centroid[0], cand.centroid[1] - kept.centroid[1])
            same_rot = abs(((cand.rot_deg - kept.rot_deg + 180.0) % 360.0) - 180.0) < 15.0
            if dist < 0.45 and same_rot and cand.mirrored == kept.mirrored:
                too_close = True
                break
        if not too_close:
            diverse.append(cand)
            if len(diverse) >= CANDIDATES_PER_GROUP:
                break

    if len(diverse) < CANDIDATES_PER_GROUP:
        for cand in raw_candidates:
            if cand not in diverse:
                diverse.append(cand)
                if len(diverse) >= CANDIDATES_PER_GROUP:
                    break
    return diverse


def _assembly_cost(
    chosen: Sequence[_GroupCandidate],
    specs_by_id: dict[str, _GroupSpec],
    scale_ratio: float,
    door_swings: Sequence[tuple[str, LineString, Optional[Polygon]]],
) -> CostBreakdown:
    fix_m = sum(c.fixture_mismatch for c in chosen)
    wall_m = sum(c.wall_mismatch for c in chosen)
    clear_p = sum(c.clearance_penalty for c in chosen)
    stage_d = sum(c.staging_distance for c in chosen)

    overlap_pen = 0.0
    dist_delta = 0.0
    for i in range(len(chosen)):
        ci = chosen[i]
        si = specs_by_id[ci.group_id]
        for j in range(i + 1, len(chosen)):
            cj = chosen[j]
            sj = specs_by_id[cj.group_id]
            if ci.union_poly.intersects(cj.union_poly):
                inter_area = ci.union_poly.intersection(cj.union_poly).area
                if inter_area > AREA_EPS:
                    overlap_pen += 5.0 + 4.0 * inter_area
            else:
                gap = ci.union_poly.distance(cj.union_poly)
                if 0.0 < gap < 0.55:
                    overlap_pen += 0.5 * (0.55 - gap)

            d_src = math.hypot(
                si.src_centroid[0] - sj.src_centroid[0],
                si.src_centroid[1] - sj.src_centroid[1],
            )
            d_new = math.hypot(
                ci.centroid[0] - cj.centroid[0],
                ci.centroid[1] - cj.centroid[1],
            )
            dist_delta += abs(d_new - d_src * scale_ratio)

    total = (
        WEIGHTS["fixture_mismatch"] * fix_m
        + WEIGHTS["wall_mismatch"] * wall_m
        + WEIGHTS["clearance_penalty"] * clear_p
        + WEIGHTS["staging_distance"] * stage_d
        + WEIGHTS["group_overlap"] * overlap_pen
        + WEIGHTS["group_distance_delta"] * dist_delta
    )
    return CostBreakdown(
        fixture_mismatch=round(WEIGHTS["fixture_mismatch"] * fix_m, 4),
        wall_mismatch=round(WEIGHTS["wall_mismatch"] * wall_m, 4),
        clearance_penalty=round(WEIGHTS["clearance_penalty"] * clear_p, 4),
        staging_distance=round(WEIGHTS["staging_distance"] * stage_d, 4),
        group_overlap=round(WEIGHTS["group_overlap"] * overlap_pen, 4),
        group_distance_delta=round(WEIGHTS["group_distance_delta"] * dist_delta, 4),
        total=round(total, 4),
    )


def _refine_assembly(
    chosen: list[_GroupCandidate],
    specs_by_id: dict[str, _GroupSpec],
    src_room: Room,
    new_room: Room,
    room_poly: Polygon,
    room_padded_prep: object,
    segs: Sequence[tuple[np.ndarray, np.ndarray]],
    door_swings: Sequence[tuple[str, LineString, Optional[Polygon]]],
    win_lines: Sequence[tuple[str, LineString, float]],
    scale_ratio: float,
) -> tuple[list[_GroupCandidate], CostBreakdown]:
    """Try 0.1 m nudges and 10-deg wall-direction snaps; keep strictly improving moves."""
    current = list(chosen)
    best_cost = _assembly_cost(current, specs_by_id, scale_ratio, door_swings)

    wall_angles: list[float] = []
    for p0, p1 in segs:
        _, n_in, _ = inward_wall_normal(p0, p1, room_poly)
        base = _frame_rot_from_inward_normal(n_in)
        for step in (0.0, 90.0, 180.0, 270.0):
            wall_angles.append((base + step) % 360.0)

    deltas = (
        (-REFINE_STEP_M, 0.0),
        (REFINE_STEP_M, 0.0),
        (0.0, -REFINE_STEP_M),
        (0.0, REFINE_STEP_M),
    )

    for _pass in range(2):
        improved = False
        for idx, cand in enumerate(current):
            spec = specs_by_id[cand.group_id]
            snapped_rot = cand.rot_deg
            for wa in wall_angles:
                diff = abs(((cand.rot_deg - wa + 180.0) % 360.0) - 180.0)
                if 0.01 < diff <= 10.0:
                    snapped_rot = wa
                    break

            for dx, dz in (*deltas, (0.0, 0.0)):
                if dx == 0.0 and dz == 0.0 and snapped_rot == cand.rot_deg:
                    continue
                nx = cand.centroid[0] + dx
                nz = cand.centroid[1] + dz
                pls, fps = _instantiate_group(
                    spec,
                    nx,
                    nz,
                    snapped_rot,
                    cand.mirrored,
                    src_room.id,
                    new_room.id,
                    matched_neighbors=(
                        cand.placements[0].matched_neighbors if cand.placements else []
                    ),
                )
                ufp = unary_union(list(fps.values()))
                if not room_padded_prep.contains(ufp):
                    continue

                clear_pen = 0.0
                for _, d_line, d_swing in door_swings:
                    if d_swing is not None and ufp.intersects(d_swing):
                        if ufp.intersection(d_swing).area > AREA_EPS:
                            clear_pen += 4.0
                    dist_d = ufp.distance(d_line)
                    if dist_d < 0.90:
                        clear_pen += (0.90 - dist_d) * 1.5
                for _, w_line, sill_h in win_lines:
                    for pl in pls:
                        assert pl.dims_m is not None
                        if pl.dims_m.h > sill_h + 0.15 and fps[pl.item_id].distance(w_line) < 0.20:
                            clear_pen += 0.6

                trial_cand = _GroupCandidate(
                    group_id=cand.group_id,
                    centroid=(nx, nz),
                    rot_deg=snapped_rot,
                    mirrored=cand.mirrored,
                    wall_index=cand.wall_index,
                    against_wall=cand.against_wall,
                    placements=pls,
                    footprints=fps,
                    union_poly=ufp,
                    fixture_mismatch=cand.fixture_mismatch,
                    wall_mismatch=cand.wall_mismatch,
                    clearance_penalty=clear_pen,
                    staging_distance=cand.staging_distance,
                    score=cand.score,
                )
                trial_list = list(current)
                trial_list[idx] = trial_cand
                trial_cost = _assembly_cost(trial_list, specs_by_id, scale_ratio, door_swings)
                if trial_cost.total + 1e-5 < best_cost.total:
                    current = trial_list
                    best_cost = trial_cost
                    cand = trial_cand
                    improved = True
        if not improved:
            break

    return current, best_cost


def _prune_hard_violations(
    new_room: Room,
    placements: list[Placement],
    base_leftovers: Sequence[LeftoverItem],
    src_items: Sequence[FurnitureItem],
    groups: Sequence[FunctionalGroup],
) -> tuple[list[Placement], list[LeftoverItem], list[str]]:
    """Iteratively remove violating items until `check_layout` reports 0 violations."""
    kept = list(placements)
    leftovers = list(base_leftovers)
    warnings: list[str] = []
    items_by_id = {it.id: it for it in src_items}

    def _drop_priority(iid: str) -> tuple[int, float, float]:
        it = items_by_id.get(iid)
        if it is None or it.dims_m is None:
            return (0, 0.0, 0.0)
        # Drop non-must_keep first, lower confidence first, larger blocking footprint first
        return (1 if it.must_keep else 0, it.confidence, -(it.dims_m.w * it.dims_m.d))

    while kept:
        chk = check_layout(new_room, kept, src_items=src_items, groups=groups)
        if chk.violations_total == 0:
            return kept, leftovers, warnings

        victim_id: Optional[str] = None
        reason_str = ""

        if chk.outside_room:
            victim_id = sorted(chk.outside_room, key=_drop_priority)[0]
            it = items_by_id.get(victim_id)
            dims_txt = f"{it.dims_m.w:.2f} m x {it.dims_m.d:.2f} m" if it and it.dims_m else "item"
            reason_str = f"outside_room: footprint ({dims_txt}) extends outside {new_room.id} polygon"
        elif chk.door_blocks:
            victim_id = sorted(chk.door_blocks, key=_drop_priority)[0]
            door_w = new_room.doors[0].width_m if new_room.doors else 0.81
            reason_str = f"door_swing_blocked: intersects {door_w:.2f} m door swing arc in {new_room.id}"
        elif chk.overlaps:
            pair = chk.overlaps[0]
            candidates = [pid for pid in pair if any(p.item_id == pid for p in kept)]
            victim_id = sorted(candidates, key=_drop_priority)[0]
            other = pair[1] if pair[0] == victim_id else pair[0]
            reason_str = f"footprint_overlap: collides with {other} in {new_room.id}"
        elif chk.clearance_fails > 0:
            # Find single item whose removal reduces clearance_fails most
            best_cf = chk.clearance_fails
            for cand_p in sorted(kept, key=lambda p: _drop_priority(p.item_id)):
                sub = [p for p in kept if p.item_id != cand_p.item_id]
                sub_chk = check_layout(new_room, sub, src_items=src_items, groups=groups)
                if sub_chk.clearance_fails < best_cf:
                    best_cf = sub_chk.clearance_fails
                    victim_id = cand_p.item_id
                    if best_cf == 0:
                        break
            if victim_id is None:
                victim_id = sorted([p.item_id for p in kept], key=_drop_priority)[0]
            paths_txt = ", ".join(chk.clearance_fail_paths[:2])
            reason_str = f"walkway_under_0.90m: blocks 0.90 m circulation ({paths_txt})"

        assert victim_id is not None
        dropped = next(p for p in kept if p.item_id == victim_id)
        kept = [p for p in kept if p.item_id != victim_id]
        leftovers.append(
            LeftoverItem(
                item_id=victim_id,
                reason=reason_str,
                suggestions=["try an adjacent room or smaller piece"],
                provenance=dropped.provenance,
            )
        )
        warnings.append(f"Moved {victim_id} to leftovers ({reason_str}).")

    return kept, leftovers, warnings


def arrange(
    src_room: Room | dict[str, object],
    src_items: Sequence[FurnitureItem | dict[str, object]],
    groups: Sequence[FunctionalGroup | dict[str, object]],
    new_room: Room | dict[str, object],
    staging: Optional[Sequence[StagingItem | dict[str, object]]] = None,
) -> tuple[list[Layout], list[LeftoverItem]]:
    """Place `src_items` into `new_room`, returning `(top_5_layouts, top1_leftovers)`."""
    src_room_obj = src_room if isinstance(src_room, Room) else Room.model_validate(src_room)
    new_room_obj = new_room if isinstance(new_room, Room) else Room.model_validate(new_room)
    src_item_objs = [
        it if isinstance(it, FurnitureItem) else FurnitureItem.model_validate(it)
        for it in src_items
    ]
    group_objs = [
        g if isinstance(g, FunctionalGroup) else FunctionalGroup.model_validate(g)
        for g in groups
    ]
    staging_objs = (
        [s if isinstance(s, StagingItem) else StagingItem.model_validate(s) for s in staging]
        if staging
        else []
    )

    if not new_room_obj.polygon or len(new_room_obj.polygon) < 3:
        empty_lo = [
            LeftoverItem(
                item_id=it.id,
                reason=f"target room {new_room_obj.id} has no readable polygon",
                provenance=it.provenance,
            )
            for it in src_item_objs
        ]
        return [], empty_lo

    room_poly = Polygon(new_room_obj.polygon)
    room_padded_prep = prep(room_poly.buffer(WALL_CONTAIN_TOL))
    segs = wall_segments(new_room_obj.polygon)
    wall_lengths = [float(np.linalg.norm(p1 - p0)) for p0, p1 in segs]
    max_wall_len = max(wall_lengths) if wall_lengths else 0.0
    minx, minz, maxx, maxz = room_poly.bounds

    specs, base_leftovers = _build_group_specs(
        src_room_obj,
        src_item_objs,
        group_objs,
        max_new_wall_m=max_wall_len,
        new_room_bounds_span=(maxx - minx, maxz - minz),
    )

    door_swings: list[tuple[str, LineString, Optional[Polygon]]] = []
    for d in new_room_obj.doors:
        d_line, d_swing, _, _ = door_geometry(d, segs, room_poly)
        door_swings.append((d.id, d_line, d_swing))

    win_lines: list[tuple[str, LineString, float]] = []
    for w in new_room_obj.windows:
        if segs:
            p0, p1 = segs[w.wall_index % len(segs)]
            t, _, wlen = inward_wall_normal(p0, p1, room_poly)
            s0 = min(w.offset_m, wlen)
            s1 = min(w.offset_m + w.width_m, wlen)
            pt0, pt1 = p0 + s0 * t, p0 + s1 * t
            win_lines.append(
                (
                    w.id,
                    LineString([(float(pt0[0]), float(pt0[1])), (float(pt1[0]), float(pt1[1]))]),
                    w.sill_height_m if w.sill_height_m is not None else 0.9,
                )
            )

    fixture_polys: list[tuple[str, str, Polygon]] = []
    for fix in new_room_obj.fixtures:
        if fix.footprint and len(fix.footprint) >= 3:
            fixture_polys.append((fix.id, fix.type, Polygon(fix.footprint)))

    new_fix_pts = _room_fixture_points(new_room_obj, segs, room_poly)

    src_area = (
        Polygon(src_room_obj.polygon).area
        if src_room_obj.polygon and len(src_room_obj.polygon) >= 3
        else room_poly.area
    )
    scale_ratio = math.sqrt(room_poly.area / max(0.5, src_area))

    # Candidate generation per group (peeling oversized multi-item groups into individual items if needed)
    active_specs: list[_GroupSpec] = []
    cands_by_gid: dict[str, list[_GroupCandidate]] = {}
    for spec in specs:
        cands = _generate_candidates_for_group(
            spec,
            src_room_obj,
            new_room_obj,
            room_poly,
            room_padded_prep,
            segs,
            door_swings,
            win_lines,
            fixture_polys,
            new_fix_pts,
            staging_objs,
        )
        if cands:
            active_specs.append(spec)
            cands_by_gid[spec.id] = cands
        elif len(spec.members) > 1:
            # Fall back to singleton sub-groups when combined group bbox exceeds room geometry
            for lm in spec.members:
                assert lm.item.dims_m is not None
                sub_spec = _GroupSpec(
                    id=f"{spec.id}-{lm.item.id.split('-', 1)[-1]}",
                    kind=spec.kind,
                    members=[_LocalMember(item=lm.item, lx=0.0, lz=0.0, rel_rot_deg=0.0)],
                    bbox_w=lm.item.dims_m.w,
                    bbox_d=lm.item.dims_m.d,
                    area=lm.item.dims_m.w * lm.item.dims_m.d,
                    src_centroid=(
                        (lm.item.pose.x, lm.item.pose.z) if lm.item.pose else spec.src_centroid
                    ),
                    against_wall=lm.item.against_wall,
                    nearest_fixture=lm.item.nearest_fixture,
                    primary_rel_rot_deg=0.0,
                )
                sub_cands = _generate_candidates_for_group(
                    sub_spec,
                    src_room_obj,
                    new_room_obj,
                    room_poly,
                    room_padded_prep,
                    segs,
                    door_swings,
                    win_lines,
                    fixture_polys,
                    new_fix_pts,
                    staging_objs,
                )
                if sub_cands:
                    active_specs.append(sub_spec)
                    cands_by_gid[sub_spec.id] = sub_cands
                else:
                    max_d = max(lm.item.dims_m.w, lm.item.dims_m.d)
                    base_leftovers.append(
                        LeftoverItem(
                            item_id=lm.item.id,
                            reason=f"needs {max_d:.2f} m; no wall/floor slot fits in {new_room_obj.id} (max wall {max_wall_len:.2f} m)",
                            provenance=lm.item.provenance,
                        )
                    )
        else:
            lm = spec.members[0]
            max_d = max(lm.item.dims_m.w, lm.item.dims_m.d) if lm.item.dims_m else 0.0
            base_leftovers.append(
                LeftoverItem(
                    item_id=lm.item.id,
                    reason=f"needs {max_d:.2f} m; longest free wall is {max_wall_len:.2f} m",
                    provenance=lm.item.provenance,
                )
            )

    active_specs.sort(key=lambda g: g.area, reverse=True)
    specs_by_id = {s.id: s for s in active_specs}

    if not active_specs:
        empty_layout = Layout(
            id="new-layout-01",
            rank=1,
            room_pair=[src_room_obj.id, new_room_obj.id],
            placements=[],
            costs=CostBreakdown(),
            check=check_layout(new_room_obj, [], src_items=src_item_objs, groups=group_objs),
            warnings=["No items fit in target room."],
            leftovers=base_leftovers,
            provenance=Provenance(source="arrange_engine", from_room=src_room_obj.id),
        )
        return [empty_layout], base_leftovers

    # Step 4: Beam search (width = 5), largest group first
    beam: list[tuple[float, list[_GroupCandidate]]] = [(0.0, [])]
    for spec in active_specs:
        next_beam: list[tuple[float, list[_GroupCandidate]]] = []
        for _, partial in beam:
            for cand in cands_by_gid[spec.id]:
                extended = [*partial, cand]
                cb = _assembly_cost(extended, specs_by_id, scale_ratio, door_swings)
                next_beam.append((cb.total, extended))
        next_beam.sort(key=lambda pair: pair[0])

        # Deduplicate near-identical partial assemblies in beam
        pruned: list[tuple[float, list[_GroupCandidate]]] = []
        for score_val, seq in next_beam:
            is_dup = False
            for _, kept_seq in pruned:
                max_shift = max(
                    math.hypot(a.centroid[0] - b.centroid[0], a.centroid[1] - b.centroid[1])
                    for a, b in zip(seq, kept_seq)
                )
                if max_shift < 0.30:
                    is_dup = True
                    break
            if not is_dup:
                pruned.append((score_val, seq))
                if len(pruned) >= BEAM_WIDTH * 2:
                    break
        beam = pruned[:BEAM_WIDTH]

    # Step 5 & 6: Refine each beam assembly + hard-constraint pruning via check_layout
    scored_layouts: list[tuple[float, Layout]] = []
    for b_idx, (_, seq) in enumerate(beam):
        refined_seq, cost_bd = _refine_assembly(
            seq,
            specs_by_id,
            src_room_obj,
            new_room_obj,
            room_poly,
            room_padded_prep,
            segs,
            door_swings,
            win_lines,
            scale_ratio,
        )
        flat_placements = [pl for grp_c in refined_seq for pl in grp_c.placements]
        kept_pls, final_leftovers, warnings = _prune_hard_violations(
            new_room_obj, flat_placements, base_leftovers, src_item_objs, group_objs
        )
        final_check = check_layout(
            new_room_obj, kept_pls, src_items=src_item_objs, groups=group_objs
        )
        dropped_extra = len(flat_placements) - len(kept_pls)
        rank_key = cost_bd.total + WEIGHTS["dropped_item_penalty"] * dropped_extra

        layout = Layout(
            id=f"new-layout-{b_idx + 1:02d}",
            rank=b_idx + 1,
            room_pair=[src_room_obj.id, new_room_obj.id],
            placements=kept_pls,
            costs=cost_bd,
            check=final_check,
            warnings=warnings,
            leftovers=final_leftovers,
            provenance=Provenance(
                source="arrange_engine",
                from_room=src_room_obj.id,
                notes=f"Beam candidate {b_idx + 1}, {len(kept_pls)} items placed",
            ),
        )
        scored_layouts.append((rank_key, layout))

    scored_layouts.sort(key=lambda pair: pair[0])

    # Keep up to 5 distinct final layouts
    distinct_layouts: list[Layout] = []
    for _, lay in scored_layouts:
        pose_map = {p.item_id: p.pose for p in lay.placements}
        dup = False
        for existing in distinct_layouts:
            ex_map = {p.item_id: p.pose for p in existing.placements}
            if set(pose_map.keys()) == set(ex_map.keys()) and pose_map:
                max_d = max(
                    math.hypot(pose_map[k].x - ex_map[k].x, pose_map[k].z - ex_map[k].z)
                    for k in pose_map
                )
                if max_d < 0.20:
                    dup = True
                    break
        if not dup:
            distinct_layouts.append(lay)
            if len(distinct_layouts) >= 5:
                break

    final_ranked = [
        lay.model_copy(update={"id": f"new-layout-{idx + 1:02d}", "rank": idx + 1})
        for idx, lay in enumerate(distinct_layouts)
    ]
    top_leftovers = final_ranked[0].leftovers if final_ranked else base_leftovers
    return final_ranked, top_leftovers


def arrange_stage(payload: dict[str, object]) -> dict[str, object]:
    """Pure JSON-in / JSON-out wrapper for `arrange`."""
    layouts, leftovers = arrange(
        src_room=payload["src_room"],  # type: ignore[arg-type]
        src_items=payload["src_items"],  # type: ignore[arg-type]
        groups=payload.get("groups", []),  # type: ignore[arg-type]
        new_room=payload["new_room"],  # type: ignore[arg-type]
        staging=payload.get("staging"),  # type: ignore[arg-type]
    )
    return {
        "layouts": [lay.model_dump() for lay in layouts],
        "leftovers": [lo.model_dump() for lo in leftovers],
    }


if __name__ == "__main__":
    sample_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent / "samples" / "living_pair.json"
    )
    case = json.loads(sample_path.read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    layouts_out, leftovers_out = arrange(
        case["src_room"],
        case["src_items"],
        case["groups"],
        case["new_room"],
        staging=case.get("staging"),
    )
    elapsed = time.perf_counter() - t0

    summary = {
        "runtime_sec": round(elapsed, 4),
        "under_2s_target": elapsed < 2.0,
        "layouts_returned": len(layouts_out),
        "layouts": [
            {
                "id": lay.id,
                "rank": lay.rank,
                "placed_count": len(lay.placements),
                "total_cost": lay.costs.total,
                "check": lay.check.model_dump() if lay.check else None,
            }
            for lay in layouts_out
        ],
        "leftovers": [lo.model_dump() for lo in leftovers_out],
    }
    print(json.dumps(summary, indent=2))
