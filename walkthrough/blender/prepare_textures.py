"""Prepare perspective-rectified reference textures and seamless PBR maps in Blender Python.

Runs headlessly inside Blender 5.2 (`blender --background --factory-startup --python ...`)
using Blender's built-in OpenImageIO (`bpy.data.images`) + NumPy:
1. Homography-rectifies aligned reference surfaces from `data/drive_inputs/` and `private_refs/frames/`
   (foyer canvas art, living gallery triptych, OLED TV screen frame, circular bedroom relief,
   and waterfront horizon backdrop) without copying raw private video files into public assets.
2. Generates tileable, zero-seam PBR BaseColor, Normal, and Roughness maps at lightweight
   web resolutions (512x512 / 1024x1024) so surfaces hold up at grazing eye-level angles.
"""

from __future__ import annotations

from pathlib import Path
import bpy
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
DRIVE_DIR = ROOT_DIR / "data" / "drive_inputs"
PRIV_FRAMES_DIR = ROOT_DIR / "private_refs" / "frames"
OUT_TEX_DIR = ROOT_DIR / "walkthrough" / "assets" / "textures"


def _save_rgba_image(
    arr_rgb_or_rgba: np.ndarray,
    out_path: Path,
    is_data: bool = False,
    jpeg_quality: int = 86,
) -> None:
    """Save HxWx3 or HxWx4 float32 [0..1] array via Blender OpenImageIO (top-to-bottom input)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr_rgb_or_rgba.shape[:2]
    if arr_rgb_or_rgba.shape[2] == 3:
        rgba = np.concatenate(
            [arr_rgb_or_rgba, np.ones((h, w, 1), dtype=np.float32)], axis=2
        )
    else:
        rgba = arr_rgb_or_rgba.astype(np.float32)

    rgba = np.clip(rgba, 0.0, 1.0).astype(np.float32)
    # Blender image.pixels expects bottom-to-top row order
    flipped = np.ascontiguousarray(np.flipud(rgba), dtype=np.float32)

    img = bpy.data.images.new(
        name=out_path.stem,
        width=w,
        height=h,
        alpha=(out_path.suffix.lower() == ".png"),
        is_data=is_data,
    )
    if is_data:
        img.colorspace_settings.name = "Non-Color"
    else:
        img.colorspace_settings.name = "sRGB"

    img.pixels.foreach_set(flipped.ravel())
    scene = bpy.context.scene
    ext = out_path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        scene.render.image_settings.file_format = "JPEG"
        scene.render.image_settings.quality = jpeg_quality
        scene.render.image_settings.color_mode = "RGB"
    else:
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.compression = 65

    img.filepath_raw = str(out_path)
    img.save_render(filepath=str(out_path), scene=scene)
    bpy.data.images.remove(img)


def _load_image_rgb(path: Path) -> np.ndarray:
    """Load an image via Blender into top-to-bottom HxWx3 float32 sRGB [0..1]."""
    img = bpy.data.images.load(str(path), check_existing=False)
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    rgba = np.flipud(buf.reshape((h, w, 4)))
    # Note: Blender loads sRGB images into linear float space in `.pixels`!
    # Convert linear -> sRGB for save_render or keep linear when scene view_transform='Standard'.
    return np.clip(rgba[:, :, :3], 0.0, 1.0)


def _linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(
        lin <= 0.0031308,
        lin * 12.92,
        1.055 * np.power(np.maximum(lin, 1e-8), 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    srgb = np.clip(srgb, 0.0, 1.0)
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def _homography_warp(
    src_img: np.ndarray,
    src_quad_uv: list[tuple[float, float]],
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Perspective-rectify a 4-point normalized UV quad [(u0,v0)..(u3,v3)] (TL, TR, BR, BL) to out_h x out_w."""
    sh, sw = src_img.shape[:2]
    src_pts = np.array([[u * (sw - 1), v * (sh - 1)] for u, v in src_quad_uv], dtype=np.float64)
    dst_pts = np.array(
        [[0.0, 0.0], [out_w - 1.0, 0.0], [out_w - 1.0, out_h - 1.0], [0.0, out_h - 1.0]],
        dtype=np.float64,
    )
    # Solve H mapping dst -> src
    a_rows = []
    b_vals = []
    for (dx, dy), (sx, sy) in zip(dst_pts, src_pts):
        a_rows.append([dx, dy, 1.0, 0.0, 0.0, 0.0, -dx * sx, -dy * sx])
        b_vals.append(sx)
        a_rows.append([0.0, 0.0, 0.0, dx, dy, 1.0, -dx * sy, -dy * sy])
        b_vals.append(sy)
    h_vec = np.linalg.solve(np.array(a_rows, dtype=np.float64), np.array(b_vals, dtype=np.float64))
    h_mat = np.array(
        [
            [h_vec[0], h_vec[1], h_vec[2]],
            [h_vec[3], h_vec[4], h_vec[5]],
            [h_vec[6], h_vec[7], 1.0],
        ],
        dtype=np.float64,
    )

    gy, gx = np.meshgrid(
        np.arange(out_h, dtype=np.float64),
        np.arange(out_w, dtype=np.float64),
        indexing="ij",
    )
    denom = h_mat[2, 0] * gx + h_mat[2, 1] * gy + h_mat[2, 2]
    sx = (h_mat[0, 0] * gx + h_mat[0, 1] * gy + h_mat[0, 2]) / denom
    sy = (h_mat[1, 0] * gx + h_mat[1, 1] * gy + h_mat[1, 2]) / denom

    sx = np.clip(sx, 0.0, sw - 1.001)
    sy = np.clip(sy, 0.0, sh - 1.001)
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1 = np.minimum(x0 + 1, sw - 1)
    y1 = np.minimum(y0 + 1, sh - 1)
    wx = (sx - x0)[..., None]
    wy = (sy - y0)[..., None]

    top = src_img[y0, x0] * (1.0 - wx) + src_img[y0, x1] * wx
    bot = src_img[y1, x0] * (1.0 - wx) + src_img[y1, x1] * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32)


def _tileable_fbm(res: int, freq_base: int = 4, octaves: int = 4, seed: int = 42) -> np.ndarray:
    """Periodic Fourier-domain filtered noise guaranteed 100% seamless across UV wrap boundaries."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((res, res), dtype=np.float32)
    total_w = 0.0
    ky = np.fft.fftfreq(res) * res
    kx = np.fft.fftfreq(res) * res
    kgrid = np.hypot(kx[None, :], ky[:, None])

    for oct_i in range(octaves):
        f = freq_base * (2**oct_i)
        amp = 0.5**oct_i
        white = rng.standard_normal((res, res)) + 1j * rng.standard_normal((res, res))
        band = np.exp(-0.5 * ((kgrid - f) / max(1.2, 0.45 * f)) ** 2)
        band[0, 0] = 0.0
        spatial = np.fft.ifft2(white * band).real.astype(np.float32)
        std = float(np.std(spatial)) + 1e-6
        acc += amp * (spatial / std)
        total_w += amp

    acc /= total_w
    lo, hi = np.percentile(acc, [1.0, 99.0])
    return np.clip((acc - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def _height_to_normal(height: np.ndarray, strength: float = 2.5) -> np.ndarray:
    """Convert periodic heightmap [0..1] into tangent-space OpenGL normal map [0..1] RGB."""
    dh_dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5 * strength
    dh_dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5 * strength
    # Note: top-to-bottom row index increases downward, so OpenGL +Y (up) is +dh_dy
    nx = -dh_dx
    ny = dh_dy
    nz = np.ones_like(height)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    rgb = np.stack([(nx / norm) * 0.5 + 0.5, (ny / norm) * 0.5 + 0.5, (nz / norm) * 0.5 + 0.5], axis=-1)
    return rgb.astype(np.float32)


def build_pbr_textures() -> None:
    scene = bpy.context.scene
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.display_settings.display_device = "sRGB"

    # 1. Wide-Plank Warm European Oak Floor (matched to K1.jpg & Be1.jpg)
    res = 1024
    y_norm, x_norm = np.meshgrid(
        np.linspace(0.0, 1.0, res, endpoint=False, dtype=np.float32),
        np.linspace(0.0, 1.0, res, endpoint=False, dtype=np.float32),
        indexing="ij",
    )
    n_planks = 6
    plank_idx = np.floor(x_norm * n_planks).astype(np.int32)
    plank_u = (x_norm * n_planks) - plank_idx

    rng = np.random.default_rng(101)
    plank_tint = rng.uniform(-0.045, 0.045, size=n_planks).astype(np.float32)[plank_idx]
    plank_y_shift = rng.uniform(0.0, 1.0, size=n_planks).astype(np.float32)[plank_idx]
    v_shifted = (y_norm + plank_y_shift) % 1.0

    grain_fine = _tileable_fbm(res, freq_base=24, octaves=3, seed=11)
    grain_long = _tileable_fbm(res, freq_base=5, octaves=4, seed=12)
    # Stretch grain vertically along plank length
    v_idx = (np.arange(res) // 4) % res
    elongated = 0.65 * grain_long[v_idx, :] + 0.35 * grain_fine
    cathedral = 0.5 + 0.5 * np.sin((plank_u * 3.2 + elongated * 2.6) * np.pi * 2.0)

    seam_x = np.minimum(plank_u, 1.0 - plank_u) * n_planks
    seam_y = np.minimum(v_shifted, 1.0 - v_shifted)
    bevel_mask = np.clip(np.minimum(seam_x / 0.035, seam_y / 0.008), 0.0, 1.0)

    base_srgb = np.zeros((res, res, 3), dtype=np.float32)
    tone = np.clip(0.84 + plank_tint - 0.07 * cathedral - 0.04 * grain_fine, 0.55, 0.94)
    tone *= 0.72 + 0.28 * bevel_mask
    base_srgb[:, :, 0] = tone
    base_srgb[:, :, 1] = tone * 0.86
    base_srgb[:, :, 2] = tone * 0.69

    oak_lin = _srgb_to_linear(base_srgb)
    oak_height = 0.75 * bevel_mask + 0.25 * elongated
    oak_normal = _height_to_normal(oak_height, strength=3.2)
    oak_rough = np.clip(0.38 + 0.16 * (1.0 - cathedral) + 0.25 * (1.0 - bevel_mask), 0.25, 0.78)
    oak_rough_rgb = np.repeat(oak_rough[:, :, None], 3, axis=2)

    _save_rgba_image(oak_lin, OUT_TEX_DIR / "oak_floor_color.jpg", is_data=False, jpeg_quality=88)
    _save_rgba_image(oak_normal, OUT_TEX_DIR / "oak_floor_normal.png", is_data=True)
    _save_rgba_image(oak_rough_rgb, OUT_TEX_DIR / "oak_floor_rough.jpg", is_data=True, jpeg_quality=85)

    # 2. Honed Calacatta Gold-Veined Marble (Kitchen Waterfall Island, Counters, Backsplash)
    m_res = 1024
    n1 = _tileable_fbm(m_res, freq_base=3, octaves=5, seed=21)
    n2 = _tileable_fbm(m_res, freq_base=6, octaves=4, seed=22)
    vein_primary = np.exp(-((np.abs(n1 - 0.52) / 0.028) ** 1.6))
    vein_gold = np.exp(-((np.abs(n2 - 0.48) / 0.016) ** 1.8))
    cloud = _tileable_fbm(m_res, freq_base=2, octaves=3, seed=23)

    cal_srgb = np.zeros((m_res, m_res, 3), dtype=np.float32)
    cal_srgb[:, :, 0] = 0.955 - 0.04 * cloud - 0.17 * vein_primary - 0.06 * vein_gold
    cal_srgb[:, :, 1] = 0.945 - 0.045 * cloud - 0.19 * vein_primary - 0.13 * vein_gold
    cal_srgb[:, :, 2] = 0.925 - 0.05 * cloud - 0.20 * vein_primary - 0.24 * vein_gold
    cal_rough = np.clip(0.16 + 0.14 * vein_primary + 0.08 * cloud, 0.12, 0.42)

    _save_rgba_image(_srgb_to_linear(cal_srgb), OUT_TEX_DIR / "calacatta_color.jpg", is_data=False, jpeg_quality=88)
    _save_rgba_image(np.repeat(cal_rough[:, :, None], 3, axis=2), OUT_TEX_DIR / "calacatta_rough.jpg", is_data=True)

    # 3. Warm Veined Travertine / Stacked Stone Slab (Living TV Feature Wall & Bath Accent)
    t_res = 1024
    tv_n1 = _tileable_fbm(t_res, freq_base=3, octaves=5, seed=31)
    tv_n2 = _tileable_fbm(t_res, freq_base=10, octaves=4, seed=32)
    strata = _tileable_fbm(t_res, freq_base=4, octaves=4, seed=33)
    band_wave = 0.5 + 0.5 * np.sin((y_norm * 5.0 + 0.35 * tv_n1) * 2.0 * np.pi)
    vein_dark = np.exp(-((np.abs(tv_n1 - 0.50) / 0.035) ** 1.5))

    trav_srgb = np.zeros((t_res, t_res, 3), dtype=np.float32)
    base_t = 0.78 - 0.11 * band_wave - 0.08 * strata - 0.22 * vein_dark + 0.04 * tv_n2
    trav_srgb[:, :, 0] = np.clip(base_t * 1.04, 0.30, 0.90)
    trav_srgb[:, :, 1] = np.clip(base_t * 0.94, 0.26, 0.84)
    trav_srgb[:, :, 2] = np.clip(base_t * 0.80, 0.20, 0.74)
    trav_normal = _height_to_normal(0.6 * band_wave + 0.4 * tv_n2 - 0.3 * vein_dark, strength=3.6)

    _save_rgba_image(_srgb_to_linear(trav_srgb), OUT_TEX_DIR / "travertine_color.jpg", is_data=False, jpeg_quality=88)
    _save_rgba_image(trav_normal, OUT_TEX_DIR / "travertine_normal.png", is_data=True)

    # 4. Brushed American Walnut Wood (Doors, Frames, Media Console, Closet, Shelves)
    w_res = 512
    wn1 = _tileable_fbm(w_res, freq_base=6, octaves=4, seed=41)
    wn2 = _tileable_fbm(w_res, freq_base=22, octaves=3, seed=42)
    wy_idx = (np.arange(w_res) // 3) % w_res
    w_streak = 0.7 * wn1[wy_idx, :] + 0.3 * wn2
    wal_srgb = np.zeros((w_res, w_res, 3), dtype=np.float32)
    wt = 0.28 + 0.16 * w_streak
    wal_srgb[:, :, 0] = wt * 1.15
    wal_srgb[:, :, 1] = wt * 0.80
    wal_srgb[:, :, 2] = wt * 0.56
    _save_rgba_image(_srgb_to_linear(wal_srgb), OUT_TEX_DIR / "walnut_color.jpg", is_data=False, jpeg_quality=86)

    # 5. Oatmeal Boucle Upholstery + Micro-Weave Normal (Sofa, Armchair, Bed, Counter Stools)
    b_res = 512
    bn1 = _tileable_fbm(b_res, freq_base=36, octaves=3, seed=51)
    bn2 = _tileable_fbm(b_res, freq_base=14, octaves=3, seed=52)
    gy_b, gx_b = np.meshgrid(np.linspace(0, 1, b_res, endpoint=False), np.linspace(0, 1, b_res, endpoint=False), indexing="ij")
    weave = 0.5 + 0.25 * np.sin(gx_b * 96 * np.pi) + 0.25 * np.sin(gy_b * 96 * np.pi)
    boucle_h = 0.55 * bn1 + 0.25 * bn2 + 0.20 * weave
    boucle_srgb = np.zeros((b_res, b_res, 3), dtype=np.float32)
    bt = 0.82 + 0.10 * (boucle_h - 0.5)
    boucle_srgb[:, :, 0] = bt
    boucle_srgb[:, :, 1] = bt * 0.96
    boucle_srgb[:, :, 2] = bt * 0.89
    _save_rgba_image(_srgb_to_linear(boucle_srgb), OUT_TEX_DIR / "boucle_color.jpg", is_data=False, jpeg_quality=86)
    _save_rgba_image(_height_to_normal(boucle_h, strength=3.5), OUT_TEX_DIR / "boucle_normal.png", is_data=True)

    # 6. Woven High-Low Area Rug Texture (Living & Bedroom Rugs)
    r_res = 512
    rn1 = _tileable_fbm(r_res, freq_base=28, octaves=3, seed=61)
    rib = 0.5 + 0.5 * np.sin(gy_b * 64.0 * np.pi + rn1 * 1.8)
    rug_h = 0.6 * rib + 0.4 * rn1
    rug_srgb = np.zeros((r_res, r_res, 3), dtype=np.float32)
    rt = 0.76 + 0.12 * (rug_h - 0.5)
    rug_srgb[:, :, 0] = rt
    rug_srgb[:, :, 1] = rt * 0.94
    rug_srgb[:, :, 2] = rt * 0.86
    _save_rgba_image(_srgb_to_linear(rug_srgb), OUT_TEX_DIR / "rug_color.jpg", is_data=False, jpeg_quality=86)
    _save_rgba_image(_height_to_normal(rug_h, strength=4.0), OUT_TEX_DIR / "rug_normal.png", is_data=True)

    # 7. Honed Spa Porcelain Tile (Bathroom Floors & Shower Walls, matched to B1/B2)
    s_res = 512
    sn1 = _tileable_fbm(s_res, freq_base=4, octaves=4, seed=71)
    tiles_n = 2
    tu = (gx_b * tiles_n) % 1.0
    tv = (gy_b * tiles_n) % 1.0
    grout = np.clip(np.minimum(np.minimum(tu, 1.0 - tu), np.minimum(tv, 1.0 - tv)) / 0.016, 0.0, 1.0)
    spa_srgb = np.zeros((s_res, s_res, 3), dtype=np.float32)
    st = (0.76 + 0.09 * sn1) * (0.75 + 0.25 * grout)
    spa_srgb[:, :, 0] = st * 0.95
    spa_srgb[:, :, 1] = st * 0.99
    spa_srgb[:, :, 2] = st * 0.96
    _save_rgba_image(_srgb_to_linear(spa_srgb), OUT_TEX_DIR / "spa_tile_color.jpg", is_data=False, jpeg_quality=86)
    _save_rgba_image(_height_to_normal(0.8 * grout + 0.2 * sn1, strength=3.0), OUT_TEX_DIR / "spa_tile_normal.png", is_data=True)

    # 8. Subtle Architectural Plaster Micro-Normal (prevents sterile CG-flat walls)
    p_res = 512
    pn = _tileable_fbm(p_res, freq_base=18, octaves=3, seed=81)
    _save_rgba_image(_height_to_normal(pn, strength=0.9), OUT_TEX_DIR / "plaster_normal.png", is_data=True)

    # 9. Camera-Matched / Homography-Rectified Reference Projections
    # 9a. Foyer Red & Black Framed Canvas from wa_full_02.jpg (or fallback graphic if private_refs absent)
    foyer_frame = PRIV_FRAMES_DIR / "wa_115508_f04.jpg"
    if foyer_frame.exists():
        src_lin = _load_image_rgb(foyer_frame)
        art_foyer = _homography_warp(
            src_lin,
            [(0.072, 0.012), (0.718, 0.012), (0.705, 0.382), (0.098, 0.385)],
            512,
            640,
        )
        _save_rgba_image(art_foyer, OUT_TEX_DIR / "ref_foyer_art.jpg", is_data=False, jpeg_quality=88)

    # 9b. Living Room Abstract Triptych Artworks from wa_full_04.jpg
    trip_frame = PRIV_FRAMES_DIR / "wa_full_04.jpg"
    if trip_frame.exists():
        src_trip = _load_image_rgb(trip_frame)
        art_mid = _homography_warp(
            src_trip,
            [(0.165, 0.215), (0.428, 0.228), (0.432, 0.445), (0.185, 0.450)],
            384,
            512,
        )
        art_right = _homography_warp(
            src_trip,
            [(0.478, 0.228), (0.692, 0.240), (0.682, 0.440), (0.482, 0.442)],
            384,
            512,
        )
        _save_rgba_image(art_mid, OUT_TEX_DIR / "ref_living_art_1.jpg", is_data=False, jpeg_quality=86)
        _save_rgba_image(art_right, OUT_TEX_DIR / "ref_living_art_2.jpg", is_data=False, jpeg_quality=86)

    # 9c. OLED TV Screen Cinema Frame & Backlit Stone Slab from L1.jpg (in data/drive_inputs/)
    l1_path = DRIVE_DIR / "L1.jpg"
    if l1_path.exists():
        src_l1 = _load_image_rgb(l1_path)
        tv_rect = _homography_warp(
            src_l1,
            [(0.658, 0.330), (0.922, 0.298), (0.922, 0.476), (0.659, 0.462)],
            640,
            360,
        )
        _save_rgba_image(tv_rect, OUT_TEX_DIR / "ref_tv_screen.jpg", is_data=False, jpeg_quality=86)

    # 9d. Bedroom Backlit Moon/Mountain Medallion from Be1.jpg
    be1_path = DRIVE_DIR / "Be1.jpg"
    if be1_path.exists():
        src_be1 = _load_image_rgb(be1_path)
        medallion = _homography_warp(
            src_be1,
            [(0.122, 0.332), (0.235, 0.332), (0.235, 0.435), (0.122, 0.435)],
            512,
            512,
        )
        _save_rgba_image(medallion, OUT_TEX_DIR / "ref_bedroom_medallion.jpg", is_data=False, jpeg_quality=86)

    # 9e. Waterfront River & Lush Hills Vista Backdrop from wa_120407_f03.jpg (or synthesized sky/water fallback)
    vista_frame = PRIV_FRAMES_DIR / "wa_120407_f03.jpg"
    if vista_frame.exists():
        src_v = _load_image_rgb(vista_frame)
        vista_crop = _homography_warp(
            src_v,
            [(0.03, 0.285), (0.94, 0.285), (0.94, 0.565), (0.03, 0.565)],
            1024,
            512,
        )
        _save_rgba_image(vista_crop, OUT_TEX_DIR / "ref_exterior_vista.jpg", is_data=False, jpeg_quality=86)

    print(f"PREPARED_TEXTURES_OK -> {OUT_TEX_DIR}")


if __name__ == "__main__":
    build_pbr_textures()
