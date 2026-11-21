"""Deterministic 2D geometry layout checker for Homeweave (`check.py`, Prompt 6).

Scores any furniture layout from any source (both `arrange.py` and the Gemini-only baseline):
- `overlaps`: pairs of intersecting item footprints
- `door_blocks`: items intersecting a door's quarter-circle swing
- `clearance_fails`: 0.1 m raster grid + 0.45 m obstacle inflation (= 0.9 m walkway) A* checks
  from every door to every other door and to the room centroid
- `outside_room`: items not fully inside the room polygon
- `relations_kept`: share of same-group pairs within 0.3 m distance delta and 30 deg relative facing
- `violations_total`: sum of hard-constraint violations
"""

from __future__ import annotations

import heapq
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

from schemas import (
    CheckResult,
    DimensionsM,
    Door,
    FunctionalGroup,
    FurnitureItem,
    Placement,
    Pose2D,
    Room,
)

AREA_EPS = 1e-4  # 1 cm^2 tolerance to ignore floating-point edge grazing
WALL_CONTAIN_TOL = 2e-3  # 2 mm tolerance for items snapped flush against walls
GRID_RES_M = 0.10
WALKWAY_HALF_WIDTH_M = 0.45  # 2 * 0.45 m = 0.90 m minimum walkway width


def item_footprint(dims: DimensionsM, pose: Pose2D) -> Polygon:
    """Build rotated rectangular footprint in metres.

    Coordinate convention:
    - Floor plane (x, z), +y up.
    - At rot_deg = 0, width `w` spans [-w/2, +w/2] along +x and depth `d` spans [-d/2, +d/2] along +z
      (front face at +d/2 pointing toward +z).
    - `rot_deg` rotates clockwise around +y when viewed from above:
      world_x = x + lx * cos(theta) - lz * sin(theta)
      world_z = z + lx * sin(theta) + lz * cos(theta)
    """
    hw = dims.w * 0.5
    hd = dims.d * 0.5
    theta = math.radians(pose.rot_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    local_corners = ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd))
    pts = [(pose.x + lx * c - lz * s, pose.z + lx * s + lz * c) for lx, lz in local_corners]
    return Polygon(pts)


def facing_unit_vector(rot_deg: float) -> tuple[float, float]:
    """Unit front-facing vector in (x, z) for clockwise angle `rot_deg` from +z."""
    theta = math.radians(rot_deg)
    return (-math.sin(theta), math.cos(theta))


def wall_segments(room_poly_coords: Sequence[Sequence[float]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Ordered [(p0, p1), ...] wall segments around a room polygon."""
    n = len(room_poly_coords)
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n):
        p0 = np.asarray(room_poly_coords[i], dtype=float)
        p1 = np.asarray(room_poly_coords[(i + 1) % n], dtype=float)
        segs.append((p0, p1))
    return segs


def inward_wall_normal(
    p0: np.ndarray, p1: np.ndarray, room_poly: Polygon
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return `(unit_tangent, unit_inward_normal, wall_length)` for a room wall edge."""
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length < 1e-9:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0
    t = vec / length
    n1 = np.array([-t[1], t[0]])
    n2 = np.array([t[1], -t[0]])
    mid = 0.5 * (p0 + p1)
    probe1 = Point(float(mid[0] + 0.05 * n1[0]), float(mid[1] + 0.05 * n1[1]))
    n_in = n1 if room_poly.contains(probe1) else n2
    return t, n_in, length


def door_geometry(
    door: Door, segs: Sequence[tuple[np.ndarray, np.ndarray]], room_poly: Polygon
) -> tuple[LineString, Optional[Polygon], np.ndarray, np.ndarray]:
    """Return `(opening_line, swing_quarter_circle_or_None, midpoint_xz, inward_normal)`."""
    if not segs:
        origin = np.zeros(2)
        return LineString([(0.0, 0.0), (0.81, 0.0)]), None, origin, np.array([0.0, 1.0])

    idx = door.wall_index % len(segs)
    p0, p1 = segs[idx]
    t, n_in, wall_len = inward_wall_normal(p0, p1, room_poly)
    offset = min(max(0.0, door.offset_m), max(0.0, wall_len - 0.05))
    width = min(door.width_m, max(0.1, wall_len - offset)) if wall_len > 0 else door.width_m

    d0 = p0 + offset * t
    d1 = d0 + width * t
    mid = 0.5 * (d0 + d1)
    opening = LineString([(float(d0[0]), float(d0[1])), (float(d1[0]), float(d1[1]))])

    swing = door.swing or "inward_left"
    if swing in ("sliding", "open"):
        # Sliding/open doors have no rotating leaf, only a narrow 0.15 m threshold strip
        strip = Polygon(
            [
                (float(d0[0]), float(d0[1])),
                (float(d1[0]), float(d1[1])),
                (float(d1[0] + 0.15 * n_in[0]), float(d1[1] + 0.15 * n_in[1])),
                (float(d0[0] + 0.15 * n_in[0]), float(d0[1] + 0.15 * n_in[1])),
            ]
        )
        return opening, strip, mid, n_in

    is_inward = swing.startswith("inward")
    is_left = swing.endswith("left")
    hinge = d0 if is_left else d1
    along_dir = t if is_left else -t
    sweep_normal = n_in if is_inward else -n_in

    arc_pts = [(float(hinge[0]), float(hinge[1]))]
    for step in range(17):
        ang = 0.5 * math.pi * (step / 16.0)
        pt = hinge + width * (math.cos(ang) * along_dir + math.sin(ang) * sweep_normal)
        arc_pts.append((float(pt[0]), float(pt[1])))
    sector = Polygon(arc_pts)
    return opening, sector, mid, n_in


def _astar_reachable(
    free_grid: np.ndarray,
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
) -> bool:
    """8-connected A* search over boolean `free_grid[row, col]`."""
    rows, cols = free_grid.shape
    sr, sc = start_rc
    gr, gc = goal_rc
    if not (0 <= sr < rows and 0 <= sc < cols and 0 <= gr < rows and 0 <= gc < cols):
        return False
    if not free_grid[sr, sc] or not free_grid[gr, gc]:
        return False
    if start_rc == goal_rc:
        return True

    sqrt2 = math.sqrt(2.0)
    neighbors = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, sqrt2),
        (-1, 1, sqrt2),
        (1, -1, sqrt2),
        (1, 1, sqrt2),
    )

    def _heuristic(r: int, c: int) -> float:
        dr = abs(r - gr)
        dc = abs(c - gc)
        return (dr + dc) + (sqrt2 - 2.0) * min(dr, dc)

    g_cost = np.full((rows, cols), np.inf, dtype=float)
    g_cost[sr, sc] = 0.0
    heap: list[tuple[float, int, int]] = [(_heuristic(sr, sc), sr, sc)]
    closed = np.zeros((rows, cols), dtype=bool)

    while heap:
        _, r, c = heapq.heappop(heap)
        if (r, c) == (gr, gc):
            return True
        if closed[r, c]:
            continue
        closed[r, c] = True
        base_g = g_cost[r, c]
        for dr, dc, step_cost in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and free_grid[nr, nc] and not closed[nr, nc]:
                # Prevent diagonal corner cutting through blocked orthogonal cells
                if dr != 0 and dc != 0:
                    if not (free_grid[r + dr, c] and free_grid[r, c + dc]):
                        continue
                cand_g = base_g + step_cost
                if cand_g < g_cost[nr, nc]:
                    g_cost[nr, nc] = cand_g
                    heapq.heappush(heap, (cand_g + _heuristic(nr, nc), nr, nc))
    return False


def _nearest_free_cell(
    free_grid: np.ndarray,
    xs: np.ndarray,
    zs: np.ndarray,
    target_xz: tuple[float, float],
    max_radius_m: float,
) -> Optional[tuple[int, int]]:
    """Find the closest navigable grid cell `(row, col)` within `max_radius_m` of `target_xz`."""
    free_rc = np.argwhere(free_grid)
    if free_rc.size == 0:
        return None
    tx, tz = target_xz
    fx = xs[free_rc[:, 1]]
    fz = zs[free_rc[:, 0]]
    d2 = (fx - tx) ** 2 + (fz - tz) ** 2
    best_idx = int(np.argmin(d2))
    if float(d2[best_idx]) > max_radius_m * max_radius_m:
        return None
    return int(free_rc[best_idx, 0]), int(free_rc[best_idx, 1])


def _angle_diff_deg(a: float, b: float) -> float:
    """Signed shortest angle `a - b` in [-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def _compute_relations_kept(
    placements_by_id: dict[str, Placement],
    src_items: Optional[Sequence[FurnitureItem]],
    groups: Optional[Sequence[FunctionalGroup]],
) -> float:
    """Share of same-group item pairs within 0.3 m distance delta and 30 deg relative facing."""
    if not src_items:
        return 1.0

    src_by_id = {it.id: it for it in src_items if it.pose is not None}
    pair_list: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    if groups:
        for grp in groups:
            valid = [iid for iid in grp.item_ids if iid in src_by_id]
            for i in range(len(valid)):
                for j in range(i + 1, len(valid)):
                    key = (min(valid[i], valid[j]), max(valid[i], valid[j]))
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        pair_list.append(key)
    else:
        by_grp: dict[str, list[str]] = {}
        for it in src_items:
            if it.group_id and it.pose is not None:
                by_grp.setdefault(it.group_id, []).append(it.id)
        for members in by_grp.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    key = (min(members[i], members[j]), max(members[i], members[j]))
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        pair_list.append(key)

    if not pair_list:
        return 1.0

    kept = 0
    for id_a, id_b in pair_list:
        if id_a not in placements_by_id or id_b not in placements_by_id:
            continue
        sa = src_by_id[id_a].pose
        sb = src_by_id[id_b].pose
        na = placements_by_id[id_a].pose
        nb = placements_by_id[id_b].pose
        assert sa is not None and sb is not None

        d_src = math.hypot(sb.x - sa.x, sb.z - sa.z)
        d_new = math.hypot(nb.x - na.x, nb.z - na.z)
        if abs(d_new - d_src) >= 0.30:
            continue

        # Relative facing magnitude (preserved under rigid rotation and mirror reflection)
        rel_src = abs(_angle_diff_deg(sa.rot_deg, sb.rot_deg))
        rel_new = abs(_angle_diff_deg(na.rot_deg, nb.rot_deg))
        if abs(rel_new - rel_src) > 30.0:
            continue

        # Also check unsigned angle between item A's facing and the vector A -> B
        if d_src > 0.05 and d_new > 0.05:
            fa_src = facing_unit_vector(sa.rot_deg)
            fa_new = facing_unit_vector(na.rot_deg)
            v_src = ((sb.x - sa.x) / d_src, (sb.z - sa.z) / d_src)
            v_new = ((nb.x - na.x) / d_new, (nb.z - na.z) / d_new)
            ang_src = math.degrees(
                math.acos(max(-1.0, min(1.0, fa_src[0] * v_src[0] + fa_src[1] * v_src[1])))
            )
            ang_new = math.degrees(
                math.acos(max(-1.0, min(1.0, fa_new[0] * v_new[0] + fa_new[1] * v_new[1])))
            )
            if abs(ang_new - ang_src) > 30.0:
                continue

        kept += 1

    return round(kept / len(pair_list), 4)


def check_layout(
    room: Room | dict[str, object],
    placements: Sequence[Placement | dict[str, object]],
    *,
    src_items: Optional[Sequence[FurnitureItem | dict[str, object]]] = None,
    groups: Optional[Sequence[FunctionalGroup | dict[str, object]]] = None,
) -> CheckResult:
    """Evaluate hard geometric constraints and group-relation preservation on any layout."""
    room_obj = room if isinstance(room, Room) else Room.model_validate(room)
    src_item_objs = (
        [it if isinstance(it, FurnitureItem) else FurnitureItem.model_validate(it) for it in src_items]
        if src_items is not None
        else None
    )
    group_objs = (
        [g if isinstance(g, FunctionalGroup) else FunctionalGroup.model_validate(g) for g in groups]
        if groups is not None
        else None
    )
    dims_lookup: dict[str, DimensionsM] = {}
    if src_item_objs:
        for it in src_item_objs:
            if it.dims_m is not None:
                dims_lookup[it.id] = it.dims_m

    placement_objs: list[Placement] = []
    for raw_p in placements:
        if isinstance(raw_p, Placement):
            p = raw_p
        else:
            d = dict(raw_p)
            if "provenance" not in d:
                d["provenance"] = {"source": "external_layout"}
            p = Placement.model_validate(d)
        if p.dims_m is None and p.item_id in dims_lookup:
            p = p.model_copy(update={"dims_m": dims_lookup[p.item_id]})
        placement_objs.append(p)

    if not room_obj.polygon or len(room_obj.polygon) < 3:
        return CheckResult(
            violations_total=0,
            relations_kept=1.0,
        )

    room_poly = Polygon(room_obj.polygon)
    room_padded = room_poly.buffer(WALL_CONTAIN_TOL)
    segs = wall_segments(room_obj.polygon)

    footprints: dict[str, Polygon] = {}
    outside_room: list[str] = []
    flagged: set[str] = set()

    for p in placement_objs:
        if p.dims_m is None:
            continue
        fp = item_footprint(p.dims_m, p.pose)
        footprints[p.item_id] = fp
        if fp.difference(room_padded).area > AREA_EPS:
            outside_room.append(p.item_id)
            flagged.add(p.item_id)

    # Pairwise overlaps among placed items (plus floor-standing built-in fixtures)
    overlaps: list[list[str]] = []
    ids = list(footprints.keys())
    for i in range(len(ids)):
        fp_i = footprints[ids[i]]
        for j in range(i + 1, len(ids)):
            fp_j = footprints[ids[j]]
            if fp_i.intersects(fp_j) and fp_i.intersection(fp_j).area > AREA_EPS:
                overlaps.append([ids[i], ids[j]])
                flagged.add(ids[i])
                flagged.add(ids[j])

    fixture_polys: list[Polygon] = []
    for fix in room_obj.fixtures:
        if fix.footprint and len(fix.footprint) >= 3 and fix.type not in ("tv_wall", "window"):
            fpoly = Polygon(fix.footprint)
            if fpoly.area > 0.04:
                fixture_polys.append(fpoly)
                for iid, fp in footprints.items():
                    if fp.intersects(fpoly) and fp.intersection(fpoly).area > AREA_EPS:
                        overlaps.append([iid, fix.id])
                        flagged.add(iid)

    # Door swing blocks
    door_blocks_set: set[str] = set()
    door_infos: list[tuple[Door, LineString, np.ndarray, np.ndarray]] = []
    for door in room_obj.doors:
        opening_line, swing_poly, d_mid, n_in = door_geometry(door, segs, room_poly)
        door_infos.append((door, opening_line, d_mid, n_in))
        threshold = opening_line.buffer(0.08)
        for iid, fp in footprints.items():
            if swing_poly is not None and fp.intersects(swing_poly):
                if fp.intersection(swing_poly).area > AREA_EPS:
                    door_blocks_set.add(iid)
                    flagged.add(iid)
            elif fp.intersects(threshold) and fp.intersection(threshold).area > AREA_EPS:
                door_blocks_set.add(iid)
                flagged.add(iid)
    door_blocks = sorted(door_blocks_set)

    # 0.9 m walkway clearance via 0.1 m grid + 0.45 m obstacle inflation + A*
    clearance_fails = 0
    clearance_fail_paths: list[str] = []
    if door_infos:
        all_obstacles = list(footprints.values()) + fixture_polys
        obs_inflated = (
            unary_union(all_obstacles).buffer(WALKWAY_HALF_WIDTH_M)
            if all_obstacles
            else Polygon()
        )

        # Navigable interior shrinks walls by 0.45 m so wall-to-item paths also >= 0.90 m,
        # while opening 0.55 m portals at doors so 0.81 m doorways remain enterable when clear.
        nav_poly = room_poly.buffer(-WALKWAY_HALF_WIDTH_M)
        portals = [
            opening.buffer(0.55).intersection(room_poly.buffer(-0.04))
            for _, opening, _, _ in door_infos
        ]
        if portals:
            nav_poly = unary_union([nav_poly, *portals])
        if not obs_inflated.is_empty:
            nav_poly = nav_poly.difference(obs_inflated)

        minx, minz, maxx, maxz = room_poly.bounds
        xs = np.arange(minx + 0.5 * GRID_RES_M, maxx, GRID_RES_M)
        zs = np.arange(minz + 0.5 * GRID_RES_M, maxz, GRID_RES_M)

        if xs.size > 0 and zs.size > 0 and not nav_poly.is_empty:
            free_grid = np.zeros((zs.size, xs.size), dtype=bool)
            prepared_nav = prep(nav_poly)
            for r_idx, zv in enumerate(zs):
                for c_idx, xv in enumerate(xs):
                    if prepared_nav.contains(Point(float(xv), float(zv))):
                        free_grid[r_idx, c_idx] = True

            door_cells: list[tuple[str, Optional[tuple[int, int]]]] = []
            for door, _, d_mid, n_in in door_infos:
                probe_xz = (float(d_mid[0] + 0.25 * n_in[0]), float(d_mid[1] + 0.25 * n_in[1]))
                cell = _nearest_free_cell(free_grid, xs, zs, probe_xz, max_radius_m=0.45)
                door_cells.append((door.id, cell))

            cx, cz = room_poly.centroid.x, room_poly.centroid.y
            centroid_cell = _nearest_free_cell(
                free_grid, xs, zs, (float(cx), float(cz)), max_radius_m=1.25
            )

            # Door-to-door paths
            for i in range(len(door_cells)):
                id_i, cell_i = door_cells[i]
                for j in range(i + 1, len(door_cells)):
                    id_j, cell_j = door_cells[j]
                    if cell_i is None or cell_j is None or not _astar_reachable(free_grid, cell_i, cell_j):
                        clearance_fails += 1
                        clearance_fail_paths.append(f"{id_i}->{id_j}")

            # Door-to-centroid paths
            for id_i, cell_i in door_cells:
                if (
                    cell_i is None
                    or centroid_cell is None
                    or not _astar_reachable(free_grid, cell_i, centroid_cell)
                ):
                    clearance_fails += 1
                    clearance_fail_paths.append(f"{id_i}->centroid")
        else:
            # Entire room is blocked at 0.9 m clearance
            n_doors = len(door_infos)
            for i in range(n_doors):
                for j in range(i + 1, n_doors):
                    clearance_fails += 1
                    clearance_fail_paths.append(
                        f"{door_infos[i][0].id}->{door_infos[j][0].id}"
                    )
                clearance_fails += 1
                clearance_fail_paths.append(f"{door_infos[i][0].id}->centroid")

    placements_by_id = {p.item_id: p for p in placement_objs}
    relations_kept = _compute_relations_kept(placements_by_id, src_item_objs, group_objs)
    violations_total = len(overlaps) + len(door_blocks) + clearance_fails + len(outside_room)

    return CheckResult(
        overlaps=overlaps,
        door_blocks=door_blocks,
        clearance_fails=clearance_fails,
        clearance_fail_paths=clearance_fail_paths,
        outside_room=outside_room,
        relations_kept=relations_kept,
        violations_total=violations_total,
        flagged_items=sorted(flagged),
    )


if __name__ == "__main__":
    sample_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent / "samples" / "sample_schemas.json"
    )
    raw = json.loads(sample_path.read_text(encoding="utf-8"))
    room_data = raw["home"]["rooms"][0]
    placements_data = raw["layout"]["placements"]
    src_items_data = raw["inventory"]["items"]
    groups_data = raw["inventory"]["groups"]

    clean_res = check_layout(
        room_data, placements_data, src_items=src_items_data, groups=groups_data
    )

    # Also verify violation detection when sofa is shoved into the door swing:
    bad_placements = json.loads(json.dumps(placements_data))
    bad_placements[0]["pose"] = {"x": 0.4, "z": 1.0, "rot_deg": 90.0}
    bad_res = check_layout(
        room_data, bad_placements, src_items=src_items_data, groups=groups_data
    )

    print(
        json.dumps(
            {
                "clean_layout": clean_res.model_dump(),
                "violating_layout": bad_res.model_dump(),
            },
            indent=2,
        )
    )
