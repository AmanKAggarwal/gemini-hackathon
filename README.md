# Homeweave

**by Cornell Movers: Aditi Kumari, Abhinav Barat, Kyuseong Choi and Aman Aggarwal**

Gemini 4 NYC Hackathon · Element 119 track · built on Saturday, November 21, 2026

Homeweave takes the floor plan of a home you are moving into, the room photos you love, and a phone video of the home you live in now. It styles every room on the plan from your photos, moves the furniture you already own into it, has two reviewer agents walk the result and correct it, and lets you walk through the finished home in 3D.

Nothing in that sentence works from a text description. A caption cannot carry a room's dimensions, where the bathtub sits on a plan, or how many throw pillows are on your sofa. Every core input is media.

---

## What you can do in the demo

Start the server and open **http://127.0.0.1:8000** (it opens the Studio).

1. **Pick.** Choose a floor plan, one look for every room on it, and the things you want to bring. A 2-bedroom, 2-bath plan asks for six looks. An agent scores how well each photo suits each room and can pick for you. You can add a new photo or a new floor plan on the spot and Gemini reads it live.
2. **Weave.** The plan's own layout is kept exactly: the bathtub, counters, closets and doors stay where the plan draws them. A photo only gives a room its colours, materials, lighting and loose decor. The agent log shows each step.
3. **Move in.** Your own furniture is matched to the closest piece in the photo and takes its place at its real size. A small mover carries each piece in, in 3D.
4. **Review.** Two reviewers walk every room. Each room's render goes to the Decisions API, and the rule reviewer's corrections happen on screen: no sofas in bedrooms, nothing near a door, a clear walkway, the TV facing the sofa or the bed.
5. **Change it.** Type an instruction such as "remove the coffee table" or "put the TV in the bedroom". Or drag a piece on the 2D plan. Either way the reviewer re-checks the room.
6. **Final design.** Open the photoreal Blender walkthrough and walk around at eye level.

---

## How we use Gemini

Three models, each with one job. Gemini sees and judges; deterministic code places the furniture.

| Model | Job in Homeweave | Input | Where |
| --- | --- | --- | --- |
| `gemini-4-hydrogen` | Gauge a room photo: room size, the items in it, materials and colours, with a confidence and the object it scaled from | image | [studio_live.py](studio_live.py), [perceive.py](perceive.py) |
| `gemini-4-hydrogen` | Watch a home video and list every thing worth moving, one entry per object (furniture, TV, plants, vases, pillows, wall art), with size, colour and the moment it is seen | video | [studio_live.py](studio_live.py), [perceive.py](perceive.py) |
| `gemini-4-neon` | Read a floor-plan image into rooms, doors, windows, drawn fixtures and scale | image | [studio_live.py](studio_live.py), [perceive.py](perceive.py) |
| `gemini-4-neon` | Turn a typed instruction into per-piece directives (move to room, wall, remove) | text | [studio_prompt.py](studio_prompt.py), [layout_prompt.py](layout_prompt.py) |
| `gemini-4-neon` | Baseline: place the furniture by coordinates on its own, so we can compare | image + text | [studio_live.py](studio_live.py), [baseline.py](baseline.py) |
| `gemini-decision-1` (Decisions API) | Judge a render of each finished room against a rubric: clear walking path, blocked windows, balance, comfort | image | [studio_live.py](studio_live.py), [judge.py](judge.py) |

Every Gemini answer is structured JSON validated against a Pydantic schema, retried once on failure, then flagged. Sizes are stored in metres with the reference object the model scaled from.

**Why Gemini does not place the furniture.** The paper we build on found that multi-modal models place points less accurately than a geometric method, and our own check agrees (see below). So placement is a deterministic engine ([studio.py](studio.py), [studio_correspond.py](studio_correspond.py), [arrange.py](arrange.py), [check.py](check.py)), and review is a rule-based agent ([studio_review.py](studio_review.py)) with the Decisions API as a second opinion. If the Decisions API clearly disagrees with the rule check, the layout is changed to agree with it.

---

## The idea underneath

Homeweave adapts *Analogical Trajectory Transfer* (Kim et al., [arXiv:2605.14393](https://arxiv.org/abs/2605.14393)), which finds the analogous place in a new room for a path through an old one. We transfer furniture instead of paths:

- Furniture is rigid. A sofa keeps its size; only built-in runs are re-cut to a new wall.
- The new home starts empty, so the furniture seen in the chosen photo acts as a staging prior. Your own sofa takes the place of the photo's sofa.
- Rooms are assigned first, so bedroom furniture does not land in a living room.
- Pieces that do not fit are left on the truck with a reason, not forced in.

The full design is in [docs/Project_Spec.md](docs/Project_Spec.md).

---

## Proof, and where it breaks

- **Same furniture, two placers, one checker.** On the Eastline plan with 13 of our own pieces, the engine had 0 hard violations; `gemini-4-neon` placing alone had 1 in one run and 2 in another, each a blocked walkway in a busy room. It had no overlaps and nothing in a door zone. This is one plan and a small sample, run live. The endpoint is `POST /studio/baseline`.
- **Plan reading.** Neon's room rectangles landed within a pixel or two of hand measurements on rooms with clear walls, and its scale matched the printed dimensions on two plans (30.2 against 30.3 px/m, 43.5 against 42.2). On the plan with no printed dimensions its scale moved between 59 and 65 px/m across runs.
- **Photo gauging.** Hydrogen's room sizes differ from a person's estimate by up to about 25%, and it reported confidence as low as 0.35 on a photo that shows very little of the room.
- **Video.** Hydrogen finds the pieces and counts them well, but the boxes and timestamps it gives for each piece are often off, so thumbnails show the whole frame with its box drawn on.
- **Decisions verdicts.** On top-down renders the "clear walking path" answer sits near 50% in most rooms. The page reports that as undecided instead of claiming agreement.
- **Evaluation harness.** `eval.py` compares the engine with a Gemini baseline on bundled cases and runs a stress ladder; results are written to [data/results.md](data/results.md) and [data/breaking_points.md](data/breaking_points.md). `reproduce.py` runs it offline by default; add `--live` to call Gemini.

Known limits: the Blender walkthrough is a fixed showcase home, not the plan picked in the Studio; furniture in the Studio's 3D view is built from simple shapes; audio is not used; all sizes are model estimates, not measurements.

---

## Run it

Requires Python 3.12 (the pinned `numpy` needs it).

```bash
git clone https://github.com/AmanKAggarwal/gemini-hackathon.git
cd gemini-hackathon
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
echo "GEMINI_API_KEY=your-key" > .env             # gitignored; never commit it
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Without a key the Studio still runs on the perception already saved in `data/`; the live upload, review verdict, prompt and baseline features need the key.

| Address | What it is |
| --- | --- |
| `/` and `/studio` | The Studio: pick, weave, move in, review, prompt |
| `/walkthrough` | Photoreal Blender walkthrough with its own room-photo studio |
| `/pipeline` | Stage-by-stage JSON inspector for the listing pipeline |

To gauge new media in bulk, drop photos named `K*`, `L*`, `B*`, `Be*` and videos named `Myhome*` in a folder and run:

```bash
.venv/bin/python studio_ingest.py path/to/folder
```

Optional: install Blender and set `BLENDER_EXE` to render the Studio's scene with Cycles (`POST /studio/blender`).

---

## What is in the repo

| Part | Files |
| --- | --- |
| Studio page | [studio.html](studio.html) |
| Fit engine, routes, walkway check | [studio.py](studio.py) |
| Moving your furniture in (correspondence) | [studio_correspond.py](studio_correspond.py) |
| Reviewer agent and its rules | [studio_review.py](studio_review.py) |
| Live Gemini calls for the Studio | [studio_live.py](studio_live.py), [studio_ingest.py](studio_ingest.py) |
| Typed instructions | [studio_prompt.py](studio_prompt.py), [layout_prompt.py](layout_prompt.py) |
| Listing pipeline: perceive, judge, weave, arrange, check | [perceive.py](perceive.py), [judge.py](judge.py), [weave.py](weave.py), [arrange.py](arrange.py), [check.py](check.py), [schemas.py](schemas.py) |
| Room-photo ensemble and three verified layouts | [ensemble.py](ensemble.py) |
| Evaluation and baseline | [eval.py](eval.py), [baseline.py](baseline.py), [reproduce.py](reproduce.py) |
| Blender build and Three.js walkthrough | [walkthrough/](walkthrough/), [blender_render.py](blender_render.py) |
| Plans, photos, perception results | [data/](data/) |

Conventions used everywhere: metres, y up, floor at y = 0; ids prefixed `src-` for the current home and `new-` for the new one; unknown values are `null` with a reason, never a guessed default.

---

## Team

**Cornell Movers**: Aditi Kumari, Abhinav Barat, Kyuseong Choi, Aman Aggarwal.

Everything in this repository was built during the hackathon.
