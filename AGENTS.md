# Homeweave — Agent & Workspace Guide

One-day hackathon implementation of **Homeweave** (Gemini 4 NYC Hackathon · Element 119 · Sat Nov 21, 2026).
Source of truth documents:
- `/Users/kyuseongchoi/Downloads/Homeweave — Project Spec.md`
- `/Users/kyuseongchoi/Downloads/Homeweave Build Kit.html`

## Non-Negotiable Conventions
- **Units & Axes:** Metres (`m`), right-handed axes, `+y` up, floor at `y = 0`. Floor polygons are `[x, z]`. Angles (`rot_deg`) are degrees **clockwise from `+z`** viewed from above (`Three.js rotation.y = -radians(rot_deg)`).
- **IDs:** Prefixed `src-` (current home), `new-` (generated home), `lst1-`, `lst2-` (listings).
- **Provenance & Unknowns:** Every entity carries a `provenance` object. Unknown measurements are `null` paired with a non-empty `null_reason`—never guessed defaults.
- **Validation:** Every Gemini response validates against Pydantic v2 models in `schemas.py` via `validate_with_retry` (retries once on failure, then flags).
- **Model Split (`google-genai >= 3.2.0`):**
  - `gemini-4-hydrogen`: `perceive_listing` & `perceive_home_video` (`client.interactions.create`)
  - `gemini-4-neon`: `perceive_hard_plan` (`thinking_level="high"`), `weave`, and `run_gemini_baseline`
  - `gemini-decision-1`: `judge_features` & `judge_layouts` (`client.decisions.create` on `v1beta`; scores summed in Python)
  - Deterministic Python (`shapely` + `numpy`): `check.py` and `arrange.py`

## Files & Commands (always run via `.venv/bin/python`)
- `schemas.py` — Pydantic v2 entities, `to_gemini_response_format`, `validate_with_retry`
- `perceive.py` — Listing photo/plan & current-home video perception (`--listing-photos`, `--plan`, `--hard-plan`, `--video`)
- `check.py` — Deterministic checker (`overlaps`, quarter-circle `door_blocks`, 0.1 m grid + 0.45 m inflation `A*` `clearance_fails`, `outside_room`, `relations_kept`)
- `arrange.py` — Rigid group wall-sliding (0.25 m steps, 90° rotations + mirror), width-5 beam search, 0.1 m refinement, hard-constraint pruning (`~0.11 s` per room pair)
- `judge.py` — Calibrated `noul` + `score` Decisions API rubric evaluator
- `weave.py` — Single-level floor-plan merger (`gemini-4-neon` + deterministic geometric fallback)
- `baseline.py` — Pure-Python top-down PNG renderer + `gemini-4-neon` spatial placement baseline
- `eval.py` — 3-method benchmark with 95% paired bootstrap CIs (`results.csv`, `results.md`) + `--stress` breaking-point ladder (`breaking_points.md`)
- `viewer.html` & `app.py` — Single-file Three.js 3D viewer (top-down ↔ 1.6 m WASD first-person, 400 ms layout transitions, red violation outlines, `/judge` PNG render capture) + FastAPI server

### Quick Start
```bash
export GEMINI_API_KEY="your-key"
.venv/bin/python app.py
# Open http://127.0.0.1:8000
```
