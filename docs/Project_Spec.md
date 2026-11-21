# Homeweave — Project Spec

Nov 21, 2026 · @Abhinav Barat

## Overview

Homeweave turns the homes you like into one walkable dream home, then moves the furniture you already own into it and shows you the best way to arrange it. The new step, **Correspondence**, maps each furniture group in your current home to the most analogous spot in the new one. It is adapted from *Analogical Trajectory Transfer* (Kim et al., arXiv 2605.14393, May 2026).

**One-liner:** Upload listing photos and floor plans of homes you like, plus a phone video of your current home. Homeweave builds a composite dream home, moves your real furniture into it, and lets you walk through the result.

**The problem has two halves:**

- **Choosing.** No single listing has everything. Buyers like one home's kitchen, another's living room, a third's primary suite, and cannot see whether those parts fit together.
- **Moving in.** Even after choosing, people cannot picture their own sofa, bed and dining set in a new space. They guess, measure by hand, and find out on moving day that the sectional blocks the door.

**What Homeweave does, end to end:**

1. **Perceive** listing photos, schematics and the current-home video.
2. **Judge** features against the user's rubric with the Decisions API.
3. **Weave** the top features into one buildable floor plan.
4. **Generate** a navigable 3D model of that plan.
5. **Correspond** (new): map the user's furniture groups and daily routes from the current home to analogous places in the new home.
6. **Arrange** (new): refine those placements into a collision-free, functional layout, and rank several alternatives.
7. **Walk through** the furnished home in first person.

**Fit with the Element 119 track:** every core input is media: photos, scanned PDFs, video and optional voice notes. A text description cannot carry room dimensions, the scale of a sofa, or how the current living room is actually arranged. The project pushes Gemini 4 on long, messy home videos and dozens of images at once, and it measures where Gemini's spatial reasoning stops and a geometric method has to take over.

**Non-goals:** Homeweave is not an architect's plan, a structural analysis, or a purchasing tool. Its output is a visualization and a planning aid.

## The correspondence concept

The paper shows that a training-free, divide-and-conquer map between two 3D scenes beats LLM, VLM and scene-graph baselines at finding "the analogous place" in a new room. Homeweave reuses that engine to move furniture instead of only paths. Source: [Analogical Trajectory Transfer, arXiv 2605.14393](https://arxiv.org/abs/2605.14393) (Seoul National University, preprint, May 2026).

### What the paper does

Given a trajectory in one 3D scene, it finds the semantically analogous trajectory in another scene that has different object placement, scale and layout. Matching by object labels alone causes collisions and distortions, so the method works in four stages:

1. **Cluster and match.** Each scene is a point cloud with object instances. Per-point features come from Sonata, a pre-trained 3D foundation model. Objects become graph nodes and are grouped into clusters of about 4 objects. Clusters are matched across scenes by graph matching, which compares features and edge lengths. Open-space points inherit the nearest cluster's label.
2. **Fit smooth maps per cluster.** For each matched cluster pair, the method enumerates rotations, reflections, translations and per-axis scaling. It fits a thin-plate spline (TPS) for each and keeps the top 5 by feature discrepancy.
3. **Assemble.** Beam search (width 5) picks one map per cluster. It minimizes feature cost, pairwise distortion between clusters, and the fraction of points that land outside navigable space. The chosen maps merge into one global scene map.
4. **Refine.** Gradient descent cleans the warped result with four terms: shape (keep segment lengths), anchor (stay near the warp), navigability (stay in open space) and feature (stay on semantically matching points). A waypoint mode instead warps sparse points and connects them with A\*.

### Why it matters for us

| Paper result | What it tells Homeweave |
| --- | --- |
| Runs in about 0.6 s on an A6000 GPU, with no training | Fast enough to re-run on every layout edit in an interactive app |
| Best on 3D-FRONT (synthetic) and ARKitScenes (real scans) | Works on both our generated home and the user's phone scan |
| Also best on synthetic-to-real pairs (inlier@1.5 of 0.52 to 0.57) | Our core pair, real current home to generated new home, is a cross-domain case |
| Multi-modal LLM baselines scored worse than the method | Gemini should perceive and judge, not place furniture by coordinates (we will re-test this with Gemini 4) |
| Oracle best of top-5 candidates: inlier@1.5 rises from 0.68 to 0.82 | Generating several layouts and letting a judge pick is worth it |
| Ablations: removing the object graph or the refinement step both hurt | Keep both stages, even under hackathon time pressure |

### How Homeweave adapts it

The paper transfers *paths*, which are points. Homeweave transfers *furniture*, which are rigid objects with footprints, facing directions and relations to each other. Five changes follow:

- **Rigid items, not warped points.** A sofa cannot stretch. Each item is warped as a few anchor points (centroid, footprint corners, front direction), then snapped back to a rigid pose.
- **Functional groups as clusters.** The paper's object clusters map naturally onto how people arrange homes: sofa, coffee table, TV and rug; bed and nightstands; dining table and chairs.
- **The new home starts empty.** The paper assumes both scenes are furnished. The generated home has walls, doors, windows and fixtures, but no furniture. We give it a *staging prior*: furniture positions read from the source listing photos. Your current home is matched to that staged layout, then your real items replace the staging items.
- **Room assignment first.** The paper names cross-type transfer, such as bedroom to living room, as an open limitation. We first assign current rooms to new rooms, then run correspondence within each pair.
- **Routines as trajectories.** The paper's original use still fits: transfer your daily routes, such as bed to bathroom to kitchen, into the new home to check that the arrangement keeps them short and clear.

The paper notes that its method assumes static scenes that share similar semantics, and that its paired evaluation relies on human annotators. Both limits carry over to Homeweave.

## System architecture

Two lanes run in parallel and meet at Correspondence. The top lane builds the dream home from listings; the lower lane turns a video of your current home into a furniture inventory and a 3D scan.

&#91;embedded content: Homeweave pipeline · 2 inputs, 9 stages\]

The Generate stage hands Correspondence the empty new home plus a staging prior taken from the listing photos. The user's rubric drives both Judge steps: first it ranks listing features, then it ranks finished layouts.

| Stage | Input | Output | Fails safe by |
| --- | --- | --- | --- |
| Perceive (listings) | Photos, schematics, voice notes | `Home` per listing, with rooms and fixtures | Flagging rooms whose photo scale disagrees with the plan |
| Judge (features) | Per-listing features, rubric | Scored `Feature` list | Showing low-confidence scores to the user |
| Weave | Top features, constraints | Merged floor plan, staging prior | Listing conflicts instead of forcing a fit |
| Generate | Merged plan, photo textures | Three.js scene, sampled point cloud | Falling back to untextured geometry |
| Perceive (current home) | Walkthrough video, LiDAR scan | `FurnitureItem` list, point cloud, routines | Asking the user to confirm items under 0.6 confidence |
| Correspond | Both scenes, inventory | `CorrespondenceMap` | Sending unmatched items to the leftover list |
| Arrange | Map, constraints | Top 5 `Layout`s | Never shipping a layout with a hard-constraint violation |
| Judge (layouts) | Renders of each layout, rubric | Ranked layouts | Showing all 5 when scores are close |
| Walk through | Chosen layout | First-person scene, optional Genie 4 video | Three.js view if video generation is unavailable |

## Correspondence and Arrange: stage design

The stage takes two furnished-or-staged scenes and returns a rigid pose for every item you own, a list of items that do not fit, and the top 5 alternative layouts ranked by a judge. Each step below names the paper section it comes from, or marks it **new** when Homeweave adds it.

### Inputs

| Input | Where it comes from | Form |
| --- | --- | --- |
| Current home scene `S_src` | Phone LiDAR scan (RoomPlan or Polycam export) or a walkthrough video reconstructed to 3D | Point cloud, 2 cm voxels, with object instances |
| Furniture inventory | Gemini 4 reading the same video | Item list: category, dimensions, front direction, style, "must keep" flag |
| New home scene `S_new` | Generate stage (Three.js mesh) | Point cloud sampled from the mesh, with walls, doors, windows, fixtures |
| Staging prior | Furniture seen in the source listing photos, placed into `S_new` by the Weave stage | Proxy items with poses, marked as replaceable |
| Daily routines (optional) | User taps a sequence of rooms or objects, or narrates them in a voice note | Ordered waypoints in `S_src` |
| Preference rubric | User, as in the Judge stage | Decisions API criteria |

### Steps

1. **Room assignment (new).** Match source rooms to new rooms with the Hungarian algorithm. The cost combines room type, floor area and fixture overlap, so a bedroom does not get mapped to a kitchen. Gemini 4 resolves ambiguous cases, such as an office that could become a guest room, and the user can override any pairing. Open-plan spaces are split into functional zones first.
2. **Features and object graph (paper 3.2).** Run Sonata on both scenes. Each furniture item, staging item and fixed fixture becomes a node with its centroid and mean feature. Walls, doors, windows and built-ins become **fixed nodes**: they can be matched to, but never moved.
3. **Functional clustering (paper 3.2, modified).** The paper clusters by position, about 4 objects per cluster. We seed clusters from Gemini's functional groups ("conversation set", "sleep set", "dining set") and fall back to agglomerative clustering.
4. **Cluster matching (paper 3.2).** Build the affinity matrix from feature similarity (diagonal) and edge-length compatibility (off-diagonal). Solve with spectral graph matching and keep the top K = 3 matches per cluster, the setting the paper used for noisy real scans.
5. **Per-cluster smooth maps (paper 3.3).** Enumerate 8 rotations × 4 reflections × centroid-pair translations, plus per-axis scaling. Fit a TPS for each and keep the top M = 5 by feature discrepancy.
6. **Assembly (paper 3.4, extended).** Beam search with width L = 5 over the paper's three costs, plus three new ones (below). Keep the top 5 assemblies, not only the best one.
7. **Rigidify (new).** For each item, warp its anchor points through the global map and fit the closest rigid pose (rotation about the vertical axis plus translation, by least squares). Snap rotation to the wall direction when within 10°. Furniture never scales.
8. **Refine (paper 3.5, extended).** Gradient descent over item poses, not points, with the cost below. Hard constraints are checked after descent; any item still in violation goes to the leftover list.
9. **Routine check (paper 3.5, waypoint mode).** Transfer each routine's waypoints with the same map, then connect them with A\* in the furnished new home. Report route length and the narrowest clearance on each route.
10. **Judge and pick (new).** Render each of the top 5 layouts as a top-down view and 3 eye-level views. The Decisions API scores each render against the rubric. The user sees the winner first and can flip through the others.

### Arrangement cost

The assembly and refinement costs keep the paper's terms and add furniture-specific ones:

```latex
E(\Theta) = \lambda_{feat}E_{feat} + \lambda_{anchor}E_{anchor} + \lambda_{nav}E_{nav} + \lambda_{rel}E_{rel} + \lambda_{wall}E_{wall} + \lambda_{clear}E_{clear} + \lambda_{route}E_{route}
```

| Term | Source | What it measures | Starting weight |
| --- | --- | --- | --- |
| E\_feat | Paper | Feature distance between each item's old and new surroundings | 1.0 |
| E\_anchor | Paper | Distance from the warped starting pose | 0.1 |
| E\_nav | Paper | Item footprints overlapping walls or other items, via the paper's density term | 1.0 |
| E\_rel | New, replaces paper's E\_shape | Change in distances and facing angles within a group, such as sofa to TV | 1.0 |
| E\_wall | New | Items that stood against a wall drifting away from one | 0.5 |
| E\_clear | New | Clearance violations at doors, windows, walkways and appliances | 1.0 |
| E\_route | New | Extra length of transferred routines versus the empty-home shortest path | 0.3 |

The paper's weights (1.0, 0.1, 1.0, 1.0) are the starting point for its terms; ours are first guesses to tune on the evaluation set.

**Hard constraints**, checked after refinement: no footprint overlaps; no item covers a door swing; walkway clearance of at least 0.9 m (36 in, a common interior design guideline, configurable); every item inside the room it was assigned to.

### Outputs

- **Placement map:** for each item, the room, pose, confidence, and the source neighbors it was matched to (for example, "your armchair keeps its spot beside the sofa, facing the window").
- **Leftover list:** items that do not fit, each with the reason ("sectional is 3.4 m; longest free wall in the new living room is 3.1 m").
- **Gap list:** zones in the new home with no corresponding furniture, such as an empty reading nook, filled with ghosted staging items so the user sees what is missing.
- **Top 5 layouts** with judge scores, route lengths and clearance warnings.
- **Furnished 3D scene** for the walkthrough, with real items shown as boxes textured from the video frames.

## Data model and schemas

Every stage reads and writes JSON with the same coordinate convention: metres, right-handed, y up, floor at y = 0, angles in degrees clockwise from +z when viewed from above. Each object carries a `provenance` field so any number can be traced back to the photo, video frame or schematic it came from.

### Core entities

| Entity | Key fields | Written by | Read by |
| --- | --- | --- | --- |
| `Home` | id, source (listing, current, generated), rooms\[\], fixtures\[\], scale\_check | Perceive, Generate | All later stages |
| `Room` | id, type, polygon (m), ceiling\_height\_m, doors\[\], windows\[\], zone\_ids\[\] | Perceive, Weave | Correspond |
| `Feature` | id, home\_id, room\_id, label, rubric\_scores{}, evidence\[\] | Judge | Weave |
| `FurnitureItem` | id, category, dims\_m {w,d,h}, front\_dir, group\_id, must\_keep, style, texture\_frames\[\] | Perceive (current home) | Correspond, Generate |
| `StagingItem` | Same as FurnitureItem, plus replaceable: true | Weave | Correspond |
| `Routine` | id, name, waypoints\[\] (item or room ids, in order) | User | Correspond |
| `CorrespondenceMap` | room\_pairs\[\], cluster\_pairs\[\], placements\[\], leftovers\[\], gaps\[\] | Correspond | Arrange, Walk through |
| `Layout` | id, rank, placements\[\], costs{}, judge\_scores{}, routes\[\], warnings\[\] | Arrange, Judge | UI |

### Example: one placement

```json
{
  "item_id": "src-sofa-01",
  "room_pair": ["src-living", "new-great-room"],
  "cluster_pair": ["src-c2-conversation", "new-c4-conversation"],
  "pose": {"x": 3.42, "z": 1.10, "rot_deg": 180},
  "confidence": 0.81,
  "matched_neighbors": ["new-fireplace", "new-window-03"],
  "relations_kept": {"faces": "src-tv-01", "distance_change_m": 0.18},
  "provenance": {"dims_from": "video frames 412-460", "scale_check": "door width 0.81 m"}
}
```

### Example: one leftover

```json
{
  "item_id": "src-sectional-01",
  "reason": "no_wall_long_enough",
  "detail": "needs 3.40 m; longest free wall in new-great-room is 3.10 m",
  "suggestions": ["place in new-den (3.6 m wall)", "split into sofa plus chaise if modular"]
}
```

### Conventions for code and prompts

- Ids are prefixed by home: `src-` for the current home, `new-` for the generated home, `lst<N>-` for listing N.
- Dimensions are always metric internally. Imperial appears only in the UI.
- Every Gemini call returns JSON that validates against these schemas; a response that fails validation is retried once, then flagged.
- Unknown values are `null` with a `reason`, never guessed defaults.

## Models and APIs

Gemini 4 does the seeing, hearing and judging; a GPU-hosted geometric engine does the placing. That split follows the paper's finding that multi-modal LLMs placed points less accurately than its geometric method. One stretch experiment tests whether that still holds for Gemini 4.

| Step | Model or tool | Why this one |
| --- | --- | --- |
| Perceive: listing photos and schematics | Gemini 4 Hydrogen (`gemini-4-hydrogen`) | High volume, low cost, available on the free tier |
| Perceive: current-home video to furniture inventory | Gemini 4 Hydrogen; escalate hard frames to Neon | Long, messy footage; only ambiguous items need the larger model |
| Perceive: hand-drawn or low-quality schematics | Gemini 4 Neon (`gemini-4-neon`) | Hardest reading task; Neon supports thinking levels up to xhigh and a 1M-token context |
| Judge: feature rubric, layout renders | Decisions API (`gemini-decision-1`) | Calibrated rubric decisions on images, video and PDFs with zero output tokens |
| Weave: merge features into one plan | Gemini 4 Argon (`gemini-4-argon`) or Neon | Multi-constraint reasoning; pick by A/B on our eval set |
| Generate: Three.js scene code | Gemini 4 Neon | Software engineering focus |
| Room assignment tie-breaks, functional groups | Gemini 4 Hydrogen | Short structured calls |
| Correspond and Arrange | Sonata features + our Python engine (graph matching, TPS, beam search, gradient descent) on one GPU | Training-free; the paper reports about 0.6 s per transfer on an A6000 |
| 3D capture of current home | Phone LiDAR export (RoomPlan or Polycam); video-to-3D reconstruction as fallback | Sonata needs a point cloud, not frames |
| Photoreal walkthrough (stretch) | Genie 4 (`genie-4`) | Streams 24 fps video with 48 kHz stereo audio; not listed as a Gemini 4 model, so confirm it is allowed |

Published per-token prices (per 1M tokens, input / output): Hydrogen $1.50 / $7.50; Carbon $2 / $10 (introductory, through Nov 30, 2026); Neon $3 / $15. The changelog gives no price for Argon, and Decisions API pricing is on a separate page.

### Rough cost per full run (estimate)

These numbers assume about 1,000 input tokens per image or video frame; we must check the real Gemini 4 media token rates before relying on them.

| Workload | Volume assumed | Model | Estimated cost |
| --- | --- | --- | --- |
| Listing photos | 5 homes × 50 photos = 250 images | Hydrogen | about $0.40 |
| Current-home video | 5 min at 1 frame/s = 300 frames | Hydrogen | about $0.45 |
| Schematics | 5 plans, hard ones re-read | Neon | under $0.20 |
| Weave and code generation | about 200k input, 50k output tokens | Neon | about $1.35 |
| Layout judging | 5 layouts × 4 renders = 20 images, about 8 criteria each | Decisions API | price to confirm |
| Correspondence | 1 GPU, seconds per run | Self-hosted | GPU rental only |

The estimate is about $2.50 per full run, excluding the Decisions API and GPU time. That is cheap enough to run the full evaluation set several times.

## Evaluation

The evaluation uses public datasets, real listings and outside raters, so no score depends on examples the team wrote to pass. Every stage has an automatic metric, and the whole system has a human preference study.

### Datasets

| Set | What it is | Used for |
| --- | --- | --- |
| 3D-FRONT (synthetic furnished homes) | Same dataset the paper used | Auto-built furniture transfer pairs (see protocol) |
| ARKitScenes (real phone scans) | Same dataset the paper used; noisy, partial scans | Real-scan source homes; cross-domain pairs |
| Public listings with floor plans | 20+ listings collected from public sources, photos plus plans | Perceive accuracy, Weave buildability |
| Volunteer homes | 5+ people outside the team film their own homes | End-to-end study on messy real input |

### Furniture transfer protocol (automatic, no hand labels)

1. Take a furnished 3D-FRONT room A as the "current home".
2. Take a different furnished room B of the same type. Save B's furniture poses as one valid answer, then remove B's furniture and keep only its shell and fixtures.
3. Swap B's items for A's items where categories match, and ask the system to place A's furniture in empty B.
4. Score against B's original layout. Because many layouts are valid, also collect 2 to 3 layouts per pair from outside annotators and take the best match, as the paper did with multiple ground truths.

This gives hundreds of pairs without the team writing any answers. Repeat with ARKitScenes rooms as A to test real-to-synthetic transfer.

### Metrics and targets

| Stage | Metric | Target |
| --- | --- | --- |
| Perceive | Room dimension error vs. schematic | median under 5% |
| Perceive | Furniture dimension error vs. tape measure (volunteer homes) | median under 10 cm |
| Perceive | Furniture recall from video | 90%+ of items found |
| Weave | Plans rated buildable by outside raters | 80%+ |
| Correspond | Placement inlier rate: item within 0.5 m and 30° of a valid answer | beat best baseline by 10+ points |
| Correspond | Collision ratio (paper's metric, on item footprints) | under 2% |
| Correspond | Relations kept: group pairs whose distance changes under 0.3 m and facing holds | 80%+ |
| Correspond | Placement rate: share of items placed without hard-constraint violations | 90%+ when the new home has room |
| Arrange | Clearance violations per layout | 0 |
| Arrange | Routine length vs. empty-home shortest path | under 1.2× |
| Judge | Agreement between Decisions API pick and majority of human raters | 70%+ |
| System | Correspondence runtime per room pair | under 2 s |
| System | Full pipeline wall time | under 10 min |

Targets are our first proposal; revise after the first baseline run.

### Baselines for the Correspond stage

- **Gemini 4 only:** give Neon a top-down render of both homes plus the item list and ask for poses. This repeats the paper's multi-modal LLM test with a newer model, and it is one of our headline "where it breaks" results.
- **Category nearest neighbor:** put each item where the same category sat in the staging prior.
- **Paper method, unmodified:** points warped with no rigidity, relations or clearance terms.
- **Ours, top-1** and **ours, judge-picked from top 5.**

### Human study

Outside raters see the same room furnished by each method, side by side and unlabeled. They pick the layout they would live in and rate whether it "feels like my old home, adapted". We report win rates with 95% confidence intervals.

## Stress tests and failure modes

The track asks where the system breaks, so each test raises one input dimension until a metric falls below its target. We log that breaking point, the change we made, and the new breaking point. The story for judges is a sequence of "broke at X, changed Y, now holds to Z".

| Dimension pushed | Ladder of difficulty | Expected failure | Planned fix |
| --- | --- | --- | --- |
| Video length and mess | 1 min tidy → 5 min → 15 min handheld, cluttered, night | Items counted twice, missed under clutter, dimensions drift | Track items across frames; check scale against door widths; send only hard frames to Neon |
| Number of listings | 2 → 5 → 10 homes, 50+ photos each | Weave loses features or contradicts itself | Judge features first, pass only the top-ranked to Weave |
| Schematic quality | CAD export → scanned PDF → phone photo of a hand sketch, mixed units | Wrong scale, misread labels | Cross-check plan scale against photo-measured doors; Neon at xhigh thinking |
| Furniture count | 15 → 60 → 150 items | Graph matching slows (affinity matrix grows with the square of node pairs); clusters blur | Match per room pair; cap clusters at about 4 items, as the paper does |
| Scale gap | Same size → 2× → 5× (studio to large house) | Cluster scaling stretches groups apart | Rigid items with a separate group-spacing term; allow one group to map to one zone of a large room |
| Layout mismatch | Similar rooms → open plan to closed rooms → more source rooms than new rooms | Bedroom furniture lands in a living room | Room assignment first; split open plans into zones; send unmatched rooms' items to the leftover list |
| Empty target | Staging prior from listings → sparse staging → no staging | Too little context in the new home; matches become arbitrary | Fixtures as fixed nodes; ghost staging items generated by Weave |
| Scan noise | LiDAR → video reconstruction → partial scan with missing walls | Wrong features, false collisions | Use K = 3 cluster matches, the paper's setting for noisy scans; fill walls from the floor plan |
| Mirror layouts | New home is a mirror of the old one | Correct match found, but with flipped facing | The paper's reflection candidates already cover this; check facing in the relation term |

### Known hard limits we will report, not hide

- **Cross-type rooms.** The paper names bedroom-to-living-room transfer as an open problem. Our room assignment avoids it in most homes, but a home with no matching room type will show it.
- **Gemini spatial placement.** If the Gemini 4 baseline beats our geometric engine, that is a real finding: we will report it and use Gemini where it wins.
- **Dimension accuracy from video alone.** Without LiDAR, a 10 cm target may not hold. We will report accuracy with and without a known-size reference object in frame.
- **Taste.** A layout can pass every constraint and still feel wrong. The human study measures this; the constraint metrics cannot.

## Scope, milestones and roles

The MVP proves the full chain on one real case: two listings in, one volunteer's furniture placed and walkable. Everything above that is measured against the stress ladders.

### Scope tiers

| Tier | Dream home | Move-in (Correspondence) | Evaluation |
| --- | --- | --- | --- |
| MVP | 2 listings, photos plus 1 schematic each, basic 3D walkthrough | 1 room pair, LiDAR scan, rigid placement, leftover list | 3D-FRONT protocol on 50 pairs; Gemini-only baseline |
| Target | 5+ listings, Decisions API scoring, conflict detection, photo textures | Whole home, room assignment, routines, top 5 layouts judged by the Decisions API | 200+ pairs, ARKitScenes cross-domain, human study with 10+ raters |
| Stretch | Voice-driven preference changes | Video-only capture (no LiDAR), live re-layout as the user drags an item | Genie 4 photoreal walkthrough; 5+ volunteer homes end to end |

### Roadmap

Phase lengths depend on the hackathon length, which is still open. Correspondence is the riskiest phase, so its Sonata spike starts on day 1, in parallel with the dream-home work.

&#91;embedded content: Build roadmap · 4 phases, 3 gates\]

Each gate is a demo that works on real input, not on a scripted example. If a gate slips, the next phase starts with the fallback named in the Risks table.

### Workstreams

| Workstream | Owns | Hand-off |
| --- | --- | --- |
| Perception | Gemini prompts and schemas for listings, schematics and home video; furniture inventory | `Home`, `FurnitureItem` JSON |
| Dream home | Judge, Weave, Generate; Three.js scene and staging prior | New-home scene and point cloud |
| Correspondence | Sonata features, matching, TPS maps, assembly, rigidify, refinement, routine check | `CorrespondenceMap`, top 5 `Layout`s |
| Eval and demo | 3D-FRONT protocol, baselines, stress ladders, human study, walkthrough UI, demo video | Results tables, breaking-point log, demo |

Owners are to be assigned once the team size is confirmed.

## Risks, open questions and LLM context

The biggest risk is the correspondence engine: it depends on a GPU, a 3D feature model and a phone scan, none of which the rest of the pipeline needs. The fallback keeps the demo alive if any of them fails.

### Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Sonata or GPU setup takes longer than planned | Correspond stage blocked | Day-1 spike; fallback engine uses 2D floor-plan geometry with the same cluster, assembly and refinement logic |
| Phone scan unavailable for a volunteer | No point cloud for the current home | Video-to-3D reconstruction; worst case, Gemini inventory plus manual room outline |
| Paper code not public | Must re-implement from the description | The method uses standard parts (spectral matching, Hungarian, TPS, beam search, A\*); hyperparameters are in the paper's appendix |
| Decisions API request format unknown | Judge step slips | Read the guide on day 1; fallback to Hydrogen with a JSON rubric |
| Genie 4 not allowed or not available | No photoreal walkthrough | It is a stretch goal; Three.js walkthrough is the baseline |
| Listing photos are copyrighted | Cannot show them in a public demo | Use licensed, public-domain or volunteer-supplied images |
| Home videos show private spaces | Privacy concern for volunteers | Written consent; delete raw video after the inventory is built; never publish faces |

### Open questions

- [ ] Is the paper's code released, and under what license?
- [ ] Which Sonata checkpoint and which GPU do we use?
- [ ] How long is the hackathon, and how many people are on the team?
- [ ] What are Gemini 4's media token rates (needed to confirm the cost estimate)?
- [ ] Does Genie 4 count as an allowed model for this track?
- [ ] Do we require LiDAR for the MVP, or support video-only from the start?

### Glossary for teammates and LLM assistants

| Term | Meaning in this project |
| --- | --- |
| Source home (`S_src`) | The user's current, furnished home |
| New home (`S_new`) | The generated dream home from the Weave and Generate stages |
| Staging prior | Proxy furniture placed in the new home from the listing photos, replaced by the user's items |
| Functional group or cluster | Items used together: conversation set, sleep set, dining set |
| Fixed node | Wall, door, window or built-in; matched to, never moved |
| Smooth map | A warp from one scene to the other: similarity transform plus thin-plate spline |
| Global map | All per-cluster maps merged into one warp for the whole room pair |
| Rigidify | Turning warped anchor points back into a rigid item pose |
| Leftover | An item that could not be placed without breaking a hard constraint |
| Gap | A zone in the new home with no matching furniture |
| Routine | An ordered list of places the user visits daily, used to test circulation |

### Instructions for LLM assistants working on this repo

- Treat this doc as the source of truth for stage names, schemas, coordinate conventions and metrics.
- Do not invent paper hyperparameters; use the values quoted in the Correspondence sections or mark new ones as tunable.
- Keep Gemini responsible for perception and judging, and the geometric engine responsible for placement, unless an evaluation result says otherwise.
- Every number in a UI or report must trace to a `provenance` field.

## Sources

- [Analogical Trajectory Transfer, Kim et al., arXiv 2605.14393 (May 2026)](https://arxiv.org/abs/2605.14393)
- [Google AI Studio changelog: Gemini 4 models, Decisions API, Genie 4](https://aistudio.google.com/docs/changelog?e=0)
