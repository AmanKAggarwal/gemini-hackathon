# Homeweave

**Gemini 4 NYC Hackathon · Element 119 Track**

Homeweave turns apartment listings you like into one composite 3D dream residence, moves the real furniture you already own into it via **Analogical Correspondence**, checks hard architectural constraints (`0` overlaps, unblocked door swings, `>= 0.9 m` `A*` walkways), and lets you orbit or walk through the furnished 3D result.

> Adapted from *Analogical Trajectory Transfer* (Kim et al., [arXiv:2605.14393](https://arxiv.org/abs/2605.14393)), extended from point trajectories to rigid 2D/3D furniture groups with architectural cutaway rendering inspired by [The House at Cornell Tech Residences](https://thehouseatcornelltech.com/residences/).

---

## One-Command Quickstart (100% Reproducible)

**Requirements:** Python `3.11+` (`3.11` or `3.12`).

```bash
git clone https://github.com/AmanKAggarwal/gemini-hackathon.git
cd gemini-hackathon
chmod +x run.sh
./run.sh
```

`./run.sh` creates an isolated `.venv`, installs pinned dependencies from `requirements.txt`, runs `reproduce.py` (generating `./data/home.json`, `./data/layouts.json`, `./data/styles.json`, `./data/results.csv`, `./data/results.md`, and `./data/breaking_points.md` from the bundled NYC apartment schematics & photos in `data/drive_inputs/`), and launches the **Unified Flagship Homeweave Studio** at **http://127.0.0.1:8000**:

- **`http://127.0.0.1:8000/` (Unified Flagship Demo):**
  1. **1 · Pick or Upload:** Choose a real NYC floor plan (`plan1`–`plan3` or `schematic`), pick from `K1–B4` **or upload your own photo** for Kitchen, Living, Bedroom, and Bathroom (`POST /studio/upload-photo` extracts stone/wood/palette/lighting via `gemini-4-hydrogen` or regional vision), and toggle owned furniture cropped from walkthrough videos.
  2. **2 · Weave:** Watch the 8-orientation (`4 rotations × mirror`) geometric search & correspondence engine fit rooms and owned pieces directly over the architectural plan.
  3. **3 · Move in · 3D & Judge:** Watch the 3D mover carry your furniture in with procedural PBR accent walls (`chiseled_limestone`, `calacatta_marble`, `fluted_panel`, `charcoal_gold_patina`), hanging island pendants, camera tour, `.webm` video recording, `★ Judge 3D` (`gemini-decision-1` rubric scoring), and the measured Engine vs. `gemini-4-neon` Baseline benchmark table.
- **`http://127.0.0.1:8000/classic` (Classic 5-Candidate Cutaway):** Direct 5-layout `arrange.py` beam-search residence cutaway inspector.

To re-run live Gemini 4 perception, weaving, judging, and the `gemini-4-neon` spatial placement baseline with your own API key:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
.venv/bin/python reproduce.py --live
```

---

## Model & Engine Split (`google-genai >= 3.2.0`)

| Stage | Model / Engine | Role |
| --- | --- | --- |
| `perceive_listing` & `perceive_home_video` ([perceive.py](perceive.py)) | `gemini-4-hydrogen` (`client.interactions.create`) | High-volume listing photo, floor-plan, and walkthrough video perception with `0.81 m` door-width scale checks |
| `perceive_hard_plan` & `weave` ([perceive.py](perceive.py), [weave.py](weave.py)) | `gemini-4-neon` (`thinking_level="high"`) | Hand-drawn/mixed-unit floor-plan parsing and single-level composite floor-plan synthesis with numeric `conflicts[]` |
| `judge_features` & `judge_layouts` ([judge.py](judge.py)) | `gemini-decision-1` (`client.decisions.create`, `v1beta`) | Calibrated `noul` (yes/no) and 5-level `score` rubric evaluation over shared media (summed strictly in Python) |
| `arrange` & `check_layout` ([arrange.py](arrange.py), [check.py](check.py)) | Deterministic `shapely` + `numpy` engine | Rigid functional-group wall sliding (`0.25 m` steps, `90°` rotations + mirror), width-5 beam search, `0.1 m` refinement, and `A*` `0.9 m` walkway verification (`~0.11 s` per room pair) |
| `run_gemini_baseline` ([baseline.py](baseline.py)) | `gemini-4-neon` | Multi-modal LLM spatial placement baseline scored by the exact same `check_layout` checker |

---

## Included Real NYC Apartment Dataset (`data/drive_inputs/`)

- **`plan1.webp` + `L1..L4.jpg`, `Be1..Be3.jpg`:** Manhattan Park (`30 River Rd, Roosevelt Island, NY`) Plan P (`19'1" x 12'0"` living room + `15'8" x 10'11"` primary bedroom)
- **`plan2.webp` + `K1..K4.jpg`, `B1..B4.jpg`:** The Eastline (`1655 First Ave, Residence 1713`) open island kitchen + spa soaking-tub bathroom
- **`plan3.webp`:** 224 W 124 Apt. 305 (`3BR / 2BA` stress-test schematic)
- **`schematic.png`:** `#4-189` (`700 sq ft` 1BR source home: `LOUNGE 12'x13'`, `RETREAT 11'x10'`)

---

## Individual Stage & Verification Commands

```bash
# 1. Validate Pydantic v2 schemas, null_reason enforcement, and 1x retry-then-flag contract
.venv/bin/python schemas.py

# 2. Run deterministic geometry checker & arrangement engine (< 0.15 s per room pair)
.venv/bin/python check.py
.venv/bin/python arrange.py

# 3. Run offline self-tests for perception, Decisions API rubric judging, weaving, and baseline PNG renderer
.venv/bin/python perceive.py
.venv/bin/python judge.py
.venv/bin/python weave.py
.venv/bin/python baseline.py

# 4. Run full 3-method evaluation + 4-dimension stress breaking-point ladder
.venv/bin/python eval.py --cases samples --out-dir data --stress --offline

# 5. Perceive new listing photos/plans or a phone walkthrough video (requires GEMINI_API_KEY)
.venv/bin/python perceive.py --listing-photos data/drive_inputs/L1.jpg data/drive_inputs/K1.jpg --plan data/drive_inputs/plan1.webp --out data/lst1_home.json
.venv/bin/python perceive.py --video path/to/walkthrough.mp4 --out data/inventory.json
```

---

## Coordinate & Data Conventions
- **Metric internally:** Metres (`m`), right-handed axes, `+y` up, floor at `y = 0`. Angles (`rot_deg`) are clockwise from `+z` viewed from above (`Three.js rotation.y = -radians(rot_deg)`). Imperial (`ft'in"`, `sq ft`) appears only in UI labels.
- **ID prefixes:** `src-` (current home), `new-` (generated home), `lst<N>-` (listing `N`).
- **Provenance & unknowns:** Every object carries a `provenance` trace; unknown measurements are `null` with a required `null_reason`, never guessed defaults.
