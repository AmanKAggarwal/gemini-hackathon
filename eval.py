"""Evaluation harness and stress-test ladder for Homeweave (`eval.py`, Prompt 9).

Runs three methods on a folder of room-pair cases (no expected answers hardcoded):
1. `engine_top1`: `arrange.py` rank-1 layout
2. `engine_judge_top5`: `arrange.py` top 5 layouts ranked by `gemini-decision-1`
3. `gemini_baseline`: `gemini-4-neon` spatial placement baseline (Prompt 7)

Scores every output with `check_layout`, computes 95% bootstrap CIs + paired bootstrap
(`engine_top1 - gemini_baseline`), writes `results.csv` and `results.md`, and in `--stress` mode
raises 4 stress dimensions until zero-violation rate falls below 50%, writing `breaking_points.md`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from shapely.affinity import scale as shapely_scale
from shapely.geometry import Polygon

from arrange import arrange
from baseline import render_room_topdown_data_url, run_gemini_baseline
from check import check_layout
from judge import judge_layouts
from schemas import DimensionsM, Door, Fixture, FunctionalGroup, FurnitureItem, Pose2D, Room, Window


@dataclass
class CaseMetrics:
    case_name: str
    method: str
    zero_violation: float
    violations_total: float
    placement_rate: float
    relations_kept: float
    runtime_sec: float
    json_failed: bool = False
    first_failure_mode: str = "none"


def _dominant_failure_mode(chk_dict: dict[str, Any], json_failed: bool) -> str:
    if json_failed:
        return "json_validation_failure"
    if chk_dict.get("outside_room"):
        return "outside_room"
    if chk_dict.get("overlaps"):
        return "footprint_overlap"
    if chk_dict.get("door_blocks"):
        return "door_swing_block"
    if chk_dict.get("clearance_fails", 0) > 0:
        return "walkway_under_0.9m"
    return "none"


def _bootstrap_ci(values: Sequence[float], n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(np.mean(arr))
    if arr.size == 1:
        return mean_val, mean_val, mean_val
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = np.mean(arr[idx], axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return mean_val, float(lo), float(hi)


def _paired_bootstrap_diff(
    eng_vals: Sequence[float], base_vals: Sequence[float], n_boot: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    diffs = np.asarray(eng_vals, dtype=float) - np.asarray(base_vals, dtype=float)
    return _bootstrap_ci(diffs.tolist(), n_boot=n_boot, seed=seed)


def _offline_mock_baseline(
    src_room: Room, src_items: Sequence[FurnitureItem], new_room: Room
) -> dict[str, Any]:
    """Deterministic naive proportional-centroid mapper used only when `--offline` is requested."""
    src_p = Polygon(src_room.polygon) if src_room.polygon else Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    new_p = Polygon(new_room.polygon) if new_room.polygon else Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    sminx, sminz, smaxx, smaxz = src_p.bounds
    nminx, nminz, nmaxx, nmaxz = new_p.bounds
    sw = max(smaxx - sminx, 0.1)
    sz = max(smaxz - sminz, 0.1)
    nw = max(nmaxx - nminx, 0.1)
    nz = max(nmaxz - nminz, 0.1)

    pls = []
    for it in src_items:
        if it.pose is None or it.dims_m is None:
            continue
        nx = nminx + ((it.pose.x - sminx) / sw) * nw
        nz_coord = nminz + ((it.pose.z - sminz) / sz) * nz
        pls.append(
            {
                "item_id": it.id,
                "pose": {"x": round(nx, 3), "z": round(nz_coord, 3), "rot_deg": it.pose.rot_deg},
            }
        )
    return {"placements": pls, "leftovers": []}


def evaluate_case(
    case_name: str,
    case_data: dict[str, Any],
    *,
    offline: bool = False,
    skip_judge: bool = False,
) -> list[CaseMetrics]:
    """Run all three methods on a single room-pair case and score with `check_layout`."""
    src_room = Room.model_validate(case_data["src_room"])
    new_room = Room.model_validate(case_data["new_room"])
    src_items = [FurnitureItem.model_validate(x) for x in case_data["src_items"]]
    groups = [FunctionalGroup.model_validate(g) for g in case_data.get("groups", [])]
    staging = case_data.get("staging")
    owned_count = max(1, sum(1 for it in src_items if it.dims_m is not None and it.pose is not None))

    out: list[CaseMetrics] = []

    # 1. Arrange Engine (top-1)
    t0 = time.perf_counter()
    layouts, _ = arrange(src_room, src_items, groups, new_room, staging=staging)
    t_eng = time.perf_counter() - t0
    top1_pls = layouts[0].placements if layouts else []
    chk1 = check_layout(new_room, top1_pls, src_items=src_items, groups=groups)
    out.append(
        CaseMetrics(
            case_name=case_name,
            method="engine_top1",
            zero_violation=1.0 if chk1.violations_total == 0 else 0.0,
            violations_total=float(chk1.violations_total),
            placement_rate=len(top1_pls) / owned_count,
            relations_kept=chk1.relations_kept,
            runtime_sec=t_eng,
            first_failure_mode=_dominant_failure_mode(chk1.model_dump(), False),
        )
    )

    # 2. Arrange Engine + Judge pick from top 5
    if layouts and not skip_judge:
        t1 = time.perf_counter()
        renders_map = {
            lay.id: [render_room_topdown_data_url(new_room, lay.placements)]
            for lay in layouts
        }
        has_bed = any(it.category == "bed" for it in src_items)
        if offline or not os.environ.get("GEMINI_API_KEY"):
            # Offline heuristic mock for decision_runner when no API key is configured
            def _offline_judge_runner(_inp: list[dict[str, Any]], qs: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
                return {
                    k: ({"noul": 0.05 if k == "blocks_window" else 0.9, "confidence": 0.9} if q["type"] == "noul" else {"score": 3.2, "confidence": 0.85})
                    for k, q in qs.items()
                }
            judged = judge_layouts(layouts, renders_map, has_bed=has_bed, decision_runner=_offline_judge_runner)
        else:
            judged = judge_layouts(layouts, renders_map, has_bed=has_bed)
        t_judge = t_eng + (time.perf_counter() - t1)
        picked_pls = judged["layouts"][0]["placements"] if judged["layouts"] else []
        chk_j = check_layout(new_room, picked_pls, src_items=src_items, groups=groups)
        out.append(
            CaseMetrics(
                case_name=case_name,
                method="engine_judge_top5",
                zero_violation=1.0 if chk_j.violations_total == 0 else 0.0,
                violations_total=float(chk_j.violations_total),
                placement_rate=len(picked_pls) / owned_count,
                relations_kept=chk_j.relations_kept,
                runtime_sec=t_judge,
                first_failure_mode=_dominant_failure_mode(chk_j.model_dump(), False),
            )
        )

    # 3. Gemini-only placement baseline (`gemini-4-neon`)
    t2 = time.perf_counter()
    if offline or not os.environ.get("GEMINI_API_KEY"):
        mock_payload = _offline_mock_baseline(src_room, src_items, new_room)
        b_res, b_pls, _ = run_gemini_baseline(
            src_room, src_items, new_room, transport_override=lambda _e: mock_payload
        )
    else:
        b_res, b_pls, _ = run_gemini_baseline(src_room, src_items, new_room)
    t_base = time.perf_counter() - t2

    if not b_res.ok:
        # Prompt 9: Log every baseline response that fails JSON validation as a failure, not a skip.
        out.append(
            CaseMetrics(
                case_name=case_name,
                method="gemini_baseline",
                zero_violation=0.0,
                violations_total=float(owned_count),
                placement_rate=0.0,
                relations_kept=0.0,
                runtime_sec=t_base,
                json_failed=True,
                first_failure_mode="json_validation_failure",
            )
        )
    else:
        chk_b = check_layout(new_room, b_pls, src_items=src_items, groups=groups)
        out.append(
            CaseMetrics(
                case_name=case_name,
                method="gemini_baseline",
                zero_violation=1.0 if chk_b.violations_total == 0 else 0.0,
                violations_total=float(chk_b.violations_total),
                placement_rate=len(b_pls) / owned_count,
                relations_kept=chk_b.relations_kept,
                runtime_sec=t_base,
                json_failed=False,
                first_failure_mode=_dominant_failure_mode(chk_b.model_dump(), False),
            )
        )

    return out


# --- Stress-ladder mutators (Prompt 9) ---

def _mutate_furniture_count(case_data: dict[str, Any], target_count: int) -> dict[str, Any]:
    mut = json.loads(json.dumps(case_data))
    base_items = [it for it in mut["src_items"] if it.get("dims_m") and it.get("pose")]
    if not base_items:
        return mut
    out_items = list(base_items)
    idx = 0
    while len(out_items) < target_count:
        template = json.loads(json.dumps(base_items[idx % len(base_items)]))
        clone_num = len(out_items) + 1
        template["id"] = f"src-dup-{clone_num:03d}"
        template["group_id"] = f"src-grp-dup-{clone_num:03d}"
        out_items.append(template)
        idx += 1
    mut["src_items"] = out_items[:target_count]
    return mut


def _mutate_scale_gap(case_data: dict[str, Any], area_factor: float) -> dict[str, Any]:
    mut = json.loads(json.dumps(case_data))
    poly_pts = mut["new_room"]["polygon"]
    lin_scale = math.sqrt(area_factor)
    scaled_poly = shapely_scale(Polygon(poly_pts), xfact=lin_scale, yfact=lin_scale, origin=(0.0, 0.0))
    mut["new_room"]["polygon"] = [
        [round(x, 4), round(z, 4)] for x, z in list(scaled_poly.exterior.coords)[:-1]
    ]
    for d in mut["new_room"].get("doors", []):
        d["offset_m"] = round(d["offset_m"] * lin_scale, 4)
    for w in mut["new_room"].get("windows", []):
        w["offset_m"] = round(w["offset_m"] * lin_scale, 4)
        w["width_m"] = round(w["width_m"] * lin_scale, 4)
    for fx in mut["new_room"].get("fixtures", []):
        if fx.get("footprint"):
            fx["footprint"] = [
                [round(pt[0] * lin_scale, 4), round(pt[1] * lin_scale, 4)]
                for pt in fx["footprint"]
            ]
    return mut


def _mutate_mirror(case_data: dict[str, Any]) -> dict[str, Any]:
    mut = json.loads(json.dumps(case_data))
    poly_pts = mut["new_room"]["polygon"]
    max_x = max(pt[0] for pt in poly_pts)
    # Mirror across X and reverse vertex order to preserve polygon winding
    flipped = [[round(max_x - pt[0], 4), round(pt[1], 4)] for pt in reversed(poly_pts)]
    mut["new_room"]["polygon"] = flipped
    return mut


def _mutate_noise(case_data: dict[str, Any], pct: float, seed: int = 119) -> dict[str, Any]:
    mut = json.loads(json.dumps(case_data))
    rng = np.random.default_rng(seed)
    for it in mut["src_items"]:
        if it.get("dims_m"):
            for k in ("w", "d", "h"):
                jitter = float(rng.uniform(1.0 - pct, 1.0 + pct))
                it["dims_m"][k] = round(max(0.15, it["dims_m"][k] * jitter), 4)
    return mut


def run_stress_ladder(
    cases: Sequence[tuple[str, dict[str, Any]]],
    *,
    offline: bool = False,
    out_md: Path,
) -> list[dict[str, str]]:
    """Raise one dimension at a time until clean rate (zero violations & >= 50% items placed) < 50%."""
    ladders: list[tuple[str, list[tuple[str, Any]]]] = [
        ("furniture_count", [("15 items", lambda c: _mutate_furniture_count(c, 15)), ("30 items", lambda c: _mutate_furniture_count(c, 30)), ("60 items", lambda c: _mutate_furniture_count(c, 60))]),
        ("scale_gap", [("0.8x area", lambda c: _mutate_scale_gap(c, 0.8)), ("0.6x area", lambda c: _mutate_scale_gap(c, 0.6)), ("0.4x area", lambda c: _mutate_scale_gap(c, 0.4))]),
        ("mirror", [("mirrored_x", _mutate_mirror)]),
        ("dimension_noise", [("5% jitter", lambda c: _mutate_noise(c, 0.05)), ("10% jitter", lambda c: _mutate_noise(c, 0.10)), ("20% jitter", lambda c: _mutate_noise(c, 0.20))]),
    ]

    rows: list[dict[str, str]] = []
    for dim_name, levels in ladders:
        for method in ("engine_top1", "gemini_baseline"):
            broke_level = "held through max level"
            failed_first = "none"
            for level_label, mut_fn in levels:
                clean_flags: list[float] = []
                failures: list[str] = []
                for cname, cdata in cases:
                    mut_case = mut_fn(cdata)
                    metrics = evaluate_case(cname, mut_case, offline=offline, skip_judge=True)
                    m = next(x for x in metrics if x.method == method)
                    # A layout counts as clean if 0 violations AND at least half of items placed
                    is_clean = (m.zero_violation == 1.0) and (m.placement_rate >= 0.50)
                    clean_flags.append(1.0 if is_clean else 0.0)
                    if not is_clean:
                        mode = m.first_failure_mode if m.first_failure_mode != "none" else "capacity_exhausted (<50% placed)"
                        failures.append(mode)
                clean_rate = float(np.mean(clean_flags)) if clean_flags else 0.0
                if clean_rate < 0.50:
                    broke_level = f"{level_label} (clean rate {clean_rate:.0%})"
                    failed_first = max(set(failures), key=failures.count) if failures else "violations"
                    break
            rows.append(
                {
                    "dimension": dim_name,
                    "method": method,
                    "breaking_point": broke_level,
                    "first_failure": failed_first,
                }
            )

    md_lines = [
        "# Breaking Points Ladder",
        "",
        "| Dimension | Method | Level Where It Broke | What Failed First |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        md_lines.append(
            f"| `{r['dimension']}` | `{r['method']}` | {r['breaking_point']} | `{r['first_failure']}` |"
        )
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return rows


def write_eval_reports(all_metrics: Sequence[CaseMetrics], out_dir: Path) -> dict[str, Any]:
    """Write `results.csv` and `results.md` with 95% bootstrap CIs and paired bootstrap diff."""
    methods = ["engine_top1", "engine_judge_top5", "gemini_baseline"]
    by_method: dict[str, list[CaseMetrics]] = {m: [] for m in methods}
    for cm in all_metrics:
        if cm.method in by_method:
            by_method[cm.method].append(cm)

    csv_rows: list[dict[str, Any]] = []
    md_lines = [
        "# Homeweave Evaluation Results",
        "",
        "| Method | Zero-Violation Rate (95% CI) | Mean Violations (95% CI) | Placement Rate (95% CI) | Relations Kept (95% CI) | Runtime s (95% CI) | JSON Failures |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for m in methods:
        ms = by_method[m]
        if not ms:
            continue
        zv = _bootstrap_ci([x.zero_violation for x in ms])
        vt = _bootstrap_ci([x.violations_total for x in ms])
        pr = _bootstrap_ci([x.placement_rate for x in ms])
        rk = _bootstrap_ci([x.relations_kept for x in ms])
        rt = _bootstrap_ci([x.runtime_sec for x in ms])
        jf = sum(1 for x in ms if x.json_failed)

        row = {
            "method": m,
            "n_cases": len(ms),
            "zero_violation_mean": round(zv[0], 4),
            "zero_violation_ci_lo": round(zv[1], 4),
            "zero_violation_ci_hi": round(zv[2], 4),
            "mean_violations": round(vt[0], 4),
            "violations_ci_lo": round(vt[1], 4),
            "violations_ci_hi": round(vt[2], 4),
            "placement_rate_mean": round(pr[0], 4),
            "placement_rate_ci_lo": round(pr[1], 4),
            "placement_rate_ci_hi": round(pr[2], 4),
            "relations_kept_mean": round(rk[0], 4),
            "relations_kept_ci_lo": round(rk[1], 4),
            "relations_kept_ci_hi": round(rk[2], 4),
            "runtime_sec_mean": round(rt[0], 4),
            "runtime_sec_ci_lo": round(rt[1], 4),
            "runtime_sec_ci_hi": round(rt[2], 4),
            "json_failures": jf,
        }
        csv_rows.append(row)
        md_lines.append(
            f"| `{m}` | {zv[0]:.2f} [{zv[1]:.2f}, {zv[2]:.2f}] | "
            f"{vt[0]:.2f} [{vt[1]:.2f}, {vt[2]:.2f}] | "
            f"{pr[0]:.2f} [{pr[1]:.2f}, {pr[2]:.2f}] | "
            f"{rk[0]:.2f} [{rk[1]:.2f}, {rk[2]:.2f}] | "
            f"{rt[0]:.3f} [{rt[1]:.3f}, {rt[2]:.3f}] | {jf} |"
        )

    paired_summary: dict[str, Any] = {}
    eng_ms = by_method["engine_top1"]
    base_ms = by_method["gemini_baseline"]
    if eng_ms and base_ms and len(eng_ms) == len(base_ms):
        dzv = _paired_bootstrap_diff([x.zero_violation for x in eng_ms], [x.zero_violation for x in base_ms])
        dvt = _paired_bootstrap_diff([x.violations_total for x in eng_ms], [x.violations_total for x in base_ms])
        drk = _paired_bootstrap_diff([x.relations_kept for x in eng_ms], [x.relations_kept for x in base_ms])
        paired_summary = {
            "zero_violation_diff": [round(v, 4) for v in dzv],
            "violations_diff": [round(v, 4) for v in dvt],
            "relations_kept_diff": [round(v, 4) for v in drk],
        }
        md_lines.extend(
            [
                "",
                "### Paired Bootstrap (`engine_top1 - gemini_baseline`)",
                f"- **Zero-violation delta:** `{dzv[0]:+.2f}` (95% CI `[{dzv[1]:+.2f}, {dzv[2]:+.2f}]`)",
                f"- **Mean violations delta:** `{dvt[0]:+.2f}` (95% CI `[{dvt[1]:+.2f}, {dvt[2]:+.2f}]`)",
                f"- **Relations kept delta:** `{drk[0]:+.2f}` (95% CI `[{drk[1]:+.2f}, {drk[2]:+.2f}]`)",
            ]
        )

    csv_path = out_dir / "results.csv"
    if csv_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    md_path = out_dir / "results.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {"rows": csv_rows, "paired_diff": paired_summary, "csv": str(csv_path), "md": str(md_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Homeweave evaluation & stress harness (Prompt 9).")
    parser.add_argument("--cases", type=str, default=str(Path(__file__).parent / "samples"), help="Folder of case JSONs.")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "data"), help="Directory for results.csv / results.md.")
    parser.add_argument("--stress", action="store_true", help="Run 4-dimension breaking-point stress ladder.")
    parser.add_argument("--offline", action="store_true", help="Use deterministic local mock for model calls when testing without API credits.")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded_cases: list[tuple[str, dict[str, Any]]] = []
    for jf in sorted(cases_path.glob("*.json")):
        raw = json.loads(jf.read_text(encoding="utf-8"))
        if "src_room" in raw and "new_room" in raw and "src_items" in raw:
            loaded_cases.append((jf.stem, raw))

    all_case_metrics: list[CaseMetrics] = []
    for cname, cdata in loaded_cases:
        all_case_metrics.extend(evaluate_case(cname, cdata, offline=args.offline))

    summary = write_eval_reports(all_case_metrics, out_dir)
    if args.stress and loaded_cases:
        bp_path = out_dir / "breaking_points.md"
        summary["breaking_points"] = run_stress_ladder(
            loaded_cases, offline=args.offline, out_md=bp_path
        )
        summary["breaking_points_md"] = str(bp_path)

    print(json.dumps(summary, indent=2))
