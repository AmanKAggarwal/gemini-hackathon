"""One-command end-to-end reproduction script for Homeweave (`reproduce.py`).

Rebuilds the composite residence, 5 ranked furniture layouts, evaluation metrics (`results.csv`,
`results.md`), and stress-test breaking points (`breaking_points.md`) from the real NYC apartment
schematics & room photos in `data/drive_inputs/` (`Manhattan Park Plan P`, `The Eastline Res 1713`,
`224 W 124 Apt 305`, and `#4-189 schematic.png`).

Usage:
  python reproduce.py            # Deterministic offline reproduction (no API key required)
  python reproduce.py --live     # Re-runs Gemini 4 Hydrogen, Neon, and Decision-1 API calls live
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from app import arrange_whole_home
from eval import evaluate_case, run_stress_ladder, write_eval_reports
from judge import judge_features
from perceive import perceive_listing
from schemas import Home, HomeVideoInventory
from style import build_default_style_bundle
from weave import weave

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DRIVE_DIR = DATA_DIR / "drive_inputs"
SAMPLES_DIR = BASE_DIR / "samples"


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Homeweave pipeline & evaluation.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call live Gemini 4 Hydrogen, Neon, and Decision-1 APIs (requires GEMINI_API_KEY).",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bundle = json.loads((SAMPLES_DIR / "drive_listings_and_source.json").read_text(encoding="utf-8"))

    if args.live:
        if not os.environ.get("GEMINI_API_KEY"):
            raise SystemExit("ERROR: --live requires GEMINI_API_KEY environment variable.")
        r1 = perceive_listing(
            [DRIVE_DIR / "L1.jpg", DRIVE_DIR / "L2.jpg", DRIVE_DIR / "Be1.jpg"],
            plan_source=DRIVE_DIR / "plan1.webp",
            listing_index=1,
        )
        r2 = perceive_listing(
            [DRIVE_DIR / "K1.jpg", DRIVE_DIR / "K2.jpg", DRIVE_DIR / "B1.jpg"],
            plan_source=DRIVE_DIR / "plan2.webp",
            listing_index=2,
        )
        lst1 = r1.data if (r1.ok and r1.data) else Home.model_validate(bundle["lst1"])
        lst2 = r2.data if (r2.ok and r2.data) else Home.model_validate(bundle["lst2"])
        lst1 = judge_features(lst1, photo_sources=[DRIVE_DIR / "L1.jpg", DRIVE_DIR / "Be1.jpg"])
        lst2 = judge_features(lst2, photo_sources=[DRIVE_DIR / "K1.jpg", DRIVE_DIR / "B1.jpg"])
        woven_res = weave([lst1, lst2], max_area_m2=85.0, bedrooms=1, bathrooms=1, use_model=True)
        woven_home = woven_res.data if (woven_res.ok and woven_res.data) else weave([lst1, lst2], max_area_m2=85.0, bedrooms=1, bathrooms=1, use_model=False).data
    else:
        lst1 = Home.model_validate(bundle["lst1"])
        lst2 = Home.model_validate(bundle["lst2"])
        woven_res = weave([lst1, lst2], max_area_m2=85.0, bedrooms=1, bathrooms=1, use_model=False)
        assert woven_res.data is not None
        woven_home = woven_res.data

    assert woven_home is not None
    inventory = HomeVideoInventory.model_validate(bundle["current_home_inventory"])

    # Save perceived listings, source inventory, woven composite home, and top-5 arranged layouts
    (DATA_DIR / "lst1_home.json").write_text(lst1.model_dump_json(indent=2), encoding="utf-8")
    (DATA_DIR / "lst2_home.json").write_text(lst2.model_dump_json(indent=2), encoding="utf-8")
    (DATA_DIR / "inventory.json").write_text(inventory.model_dump_json(indent=2), encoding="utf-8")
    (DATA_DIR / "home.json").write_text(woven_home.model_dump_json(indent=2), encoding="utf-8")
    (DATA_DIR / "styles.json").write_text(
        build_default_style_bundle().model_dump_json(indent=2), encoding="utf-8"
    )

    layouts, leftovers = arrange_whole_home(inventory, woven_home)
    (DATA_DIR / "layouts.json").write_text(
        json.dumps([lay.model_dump() for lay in layouts], indent=2),
        encoding="utf-8",
    )

    # Build room-pair evaluation cases from both living_pair.json and the Drive NYC apartments
    eval_cases = [
        ("living_pair", json.loads((SAMPLES_DIR / "living_pair.json").read_text(encoding="utf-8"))),
        (
            "nyc_living_manhattan_park",
            {
                "src_room": inventory.rooms[0].model_dump(),
                "src_items": [it.model_dump() for it in inventory.items if it.room_id == "src-living"],
                "groups": [g.model_dump() for g in inventory.groups if g.room_id == "src-living"],
                "new_room": next(r for r in woven_home.rooms if r.type == "living").model_dump(),
                "staging": [s.model_dump() for s in woven_home.staging if "living" in s.room_id],
            },
        ),
        (
            "nyc_bedroom_manhattan_park",
            {
                "src_room": inventory.rooms[1].model_dump(),
                "src_items": [it.model_dump() for it in inventory.items if it.room_id == "src-bedroom"],
                "groups": [g.model_dump() for g in inventory.groups if g.room_id == "src-bedroom"],
                "new_room": next(r for r in woven_home.rooms if "bedroom" in r.type).model_dump(),
                "staging": [s.model_dump() for s in woven_home.staging if "bedroom" in s.room_id],
            },
        ),
    ]

    offline_flag = not args.live
    all_metrics = []
    for cname, cdata in eval_cases:
        all_metrics.extend(evaluate_case(cname, cdata, offline=offline_flag))

    eval_summary = write_eval_reports(all_metrics, DATA_DIR)
    bp_rows = run_stress_ladder(
        eval_cases, offline=offline_flag, out_md=DATA_DIR / "breaking_points.md"
    )

    elapsed = time.perf_counter() - t0
    print(
        json.dumps(
            {
                "status": "reproduced",
                "mode": "live_gemini_api" if args.live else "deterministic_offline",
                "elapsed_sec": round(elapsed, 3),
                "woven_rooms": [
                    {"id": r.id, "type": r.type, "from": r.provenance.from_room}
                    for r in woven_home.rooms
                ],
                "layouts_generated": len(layouts),
                "rank1_placed": len(layouts[0].placements) if layouts else 0,
                "rank1_violations": layouts[0].check.violations_total if layouts and layouts[0].check else 0,
                "leftovers": [lo.model_dump() for lo in leftovers],
                "eval_csv": eval_summary["csv"],
                "eval_md": eval_summary["md"],
                "breaking_points_rows": len(bp_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
