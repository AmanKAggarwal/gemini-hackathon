# Homeweave — Photorealistic First-Person 3D Walkthrough

Eye-level (`1.65 m`) walkable 3D residence built in **Blender 5.2 LTS** and rendered in **Three.js** with WebP-compressed PBR + camera-rectified textures and deterministic 2D/3D sliding collision detection.

---

## Quick Run Instructions

### Option A — Via the Homeweave FastAPI Server (Recommended)
From the repository root:
```bash
./run.sh
```
*(Or directly once `.venv` is created: `.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000`)*

Then open:
- **Interactive Room Upload Studio + Live Tiny-Mover 3D Build + 3 Orientations + POV:** http://127.0.0.1:8000/
- **Direct Eye-Level First-Person Walkthrough:** http://127.0.0.1:8000/walkthrough?mode=walkthrough
- **JSON Pipeline Inspector:** http://127.0.0.1:8000/pipeline

### Option B — Standalone Static Three.js Project
`walkthrough/` has zero backend dependencies and works with any static HTTP server:
```bash
python3 -m http.server 8080 --directory walkthrough
# Open http://127.0.0.1:8080
```

### Rebuilding the Editable Blender Source & GLB Assets
```bash
/opt/homebrew/bin/blender --background --factory-startup --python walkthrough/blender/prepare_textures.py
/opt/homebrew/bin/blender --background --factory-startup --python walkthrough/blender/build_apartment.py
```

---

## Controls

| Input | Action |
| --- | --- |
| **Click viewport** (or drag mouse) | Lock pointer for smooth 360° mouse-look (`Esc` unlocks) |
| **`W A S D` / Arrow keys** | Walk forward, backward, and strafe left/right (`2.35 m/s`) |
| **`Shift`** | Brisk walk (`3.85 m/s`) |
| **`1` – `5`** (or top room pills) | Jump to Living, Kitchen & Foyer, Bedroom, Spa Bath, or Balcony |
| **`L`** | Cycle **Sunlit Day**, **Golden Hour**, and **Evening Cove** lighting |
| **`C`** | Toggle eye level (`1.65 m`) vs joinery/counter inspection height (`1.08 m`) |
| **`M`** | Toggle live floor-plan radar & collider map |
| **Mobile / Touch** | Left virtual thumbstick to walk · Right-screen drag to look |

---

## Deliverables & File Layout

- `walkthrough/index.html` — Three.js first-person walkthrough (PBR `RoomEnvironment` IBL, directional sun shadows, warm interior coves, subtle bloom, PointerLock + mobile touch controls, sliding circle-vs-AABB collisions, live minimap).
- `walkthrough/blender/dream_apartment.blend` — Self-contained editable Blender 5.2 source file (all textures packed, organized into 7 architectural collections).
- `walkthrough/blender/prepare_textures.py` — Homography perspective rectification of aligned reference art/screens + seamless Fourier-domain PBR BaseColor/Normal/Roughness texture synthesis.
- `walkthrough/blender/build_apartment.py` — Parametric metric Blender builder, Cycles eye-level verification renderer, collider exporter, and WebP GLB exporter.
- `walkthrough/assets/dream_apartment.glb` — Optimized ~2.1 MB textured GLB asset.
- `walkthrough/assets/apartment_meta.json` — Synchronized room bounds, spawn pose, presets, assumptions, and 55 collision boxes.

*(Raw reference videos and extracted frames are stored exclusively in git-ignored `private_refs/` to keep source media private.)*

---

## Reference Synthesis & Layout Assumptions

- **Dimensional Anchor (`61.7 m²` interior + `8.4 m²` balcony):** Anchored to Homeweave's `data/home.json`, combining **Manhattan Park Plan P** (`19'1" × 12'0"` Living/Dining + `15'8" × 10'11"` Primary Bedroom + Riverfront Balcony) and **The Eastline Residence 1713** (`11'10" × 12'0"` Open Island Kitchen & Entry Foyer + `11'2" × 10'11"` Spa Bathroom).
- **Camera-Rectified & Projected Reference Surfaces:**
  - *Entry Foyer:* Red-and-black framed abstract painting and smoked-glass console rectified from the mock-up flat walkthrough video.
  - *Living & Dining:* Veined travertine backlit TV slab, flanking walnut display niches, and OLED cinema frame from `L1.jpg`/`L2.jpg`; south-wall gallery canvases and linear LED dining chandelier from the walkthrough video.
  - *Open Kitchen:* Sage-olive shaker cabinetry, mitered Calacatta Gold waterfall island, fluted walnut apron, recessed sink basin, and 3 amber globe pendants from `K1.jpg` & `K3.jpg`.
  - *Primary Bedroom:* Upholstered bouclé platform bed with perimeter LED halo, circular backlit mountain medallion (`Be1.jpg`), glowing lunar nightstand lamps (`Be2.jpg`/`Be3.jpg`), walnut desk, smoked-glass wardrobe, and deep walnut window bench.
  - *Spa Bathroom:* Freestanding soaking tub on LED teak plinth, backlit travertine accent wall, black-framed glass rain shower with lit niche, and floating vanity with circular halo mirror (`B1.jpg`/`B2.jpg`).
- **Labeled Architectural Assumptions:**
  1. Ceiling height standardized to **`2.72 m` (`8'11"`)** across all rooms with `0.14 m` perimeter soffit coves.
  2. Exterior walls modeled at **`0.18 m`** thickness (giving `0.22 m` window sill/reveal depth) and interior walls at **`0.14 m`**.
  3. Shared walls deduplicated with cased openings (`2.56 m` Living–Kitchen portal, `0.92 m` Bedroom doorway, `0.88 m` Bathroom doorway, `0.93 m` open Balcony slider).
  4. Interior doors modeled at `2.22 m` height and swung `86°` open so every room-to-room passage maintains `> 0.82 m` clear walking width.
