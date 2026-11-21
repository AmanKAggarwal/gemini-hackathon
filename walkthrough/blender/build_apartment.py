"""Build the cohesive, photorealistic Dream Apartment in Blender 5.2 and export GLB + colliders.

Usage:
  /opt/homebrew/bin/blender --background --factory-startup --python walkthrough/blender/build_apartment.py

Outputs:
  - walkthrough/blender/dream_apartment.blend   (Organized, editable Blender source scene)
  - walkthrough/assets/dream_apartment.glb      (WebP-compressed textured GLB for Three.js)
  - walkthrough/assets/apartment_meta.json      (Exact wall/furniture 2D/3D colliders, rooms & presets)
  - private_refs/renders_check/*.jpg            (Multi-angle eye-level verification renders)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

ROOT_DIR = Path(__file__).resolve().parents[2]
TEX_DIR = ROOT_DIR / "walkthrough" / "assets" / "textures"
BLEND_OUT = ROOT_DIR / "walkthrough" / "blender" / "dream_apartment.blend"
GLB_OUT = ROOT_DIR / "walkthrough" / "assets" / "dream_apartment.glb"
META_OUT = ROOT_DIR / "walkthrough" / "assets" / "apartment_meta.json"
CHECK_RENDER_DIR = ROOT_DIR / "private_refs" / "renders_check"

CEIL_H = 2.72
EXT_WALL_T = 0.18
INT_WALL_T = 0.14
DOOR_H = 2.22


# ---------------------------------------------------------------------------
# Coordinate Helpers: Homeweave (x East, y Up, z South) -> Blender (X, -Z, Y)
# With glTF +Y Up export, Blender (x, -z, y) exports to Three.js (x, y, z) 1:1.
# ---------------------------------------------------------------------------


def hw_pt(x: float, y: float, z: float) -> tuple[float, float, float]:
    return (float(x), float(-z), float(y))


class SceneBuilder:
    def __init__(self) -> None:
        self.materials: dict[str, bpy.types.Material] = {}
        self.collections: dict[str, bpy.types.Collection] = {}
        self.colliders: list[dict[str, Any]] = []
        self._init_clean_scene()
        self._init_materials()

    def _init_clean_scene(self) -> None:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        scene.unit_settings.system = "METRIC"
        scene.unit_settings.scale_length = 1.0

        for cname in [
            "01_Architecture",
            "02_Living_Dining",
            "03_Kitchen_Foyer",
            "04_Primary_Bedroom",
            "05_Spa_Bathroom",
            "06_Balcony_Exterior",
            "07_Lighting_Cameras",
        ]:
            col = bpy.data.collections.new(cname)
            scene.collection.children.link(col)
            self.collections[cname] = col

    def add_collider(
        self,
        kind: str,
        label: str,
        x0: float,
        z0: float,
        x1: float,
        z1: float,
        y0: float = 0.0,
        y1: float = 2.72,
    ) -> None:
        lo_x, hi_x = min(x0, x1), max(x0, x1)
        lo_z, hi_z = min(z0, z1), max(z0, z1)
        if (hi_x - lo_x) < 0.01 or (hi_z - lo_z) < 0.01:
            return
        self.colliders.append(
            {
                "kind": kind,
                "label": label,
                "x0": round(lo_x, 4),
                "z0": round(lo_z, 4),
                "x1": round(hi_x, 4),
                "z1": round(hi_z, 4),
                "y0": round(y0, 4),
                "y1": round(y1, 4),
            }
        )

    # -----------------------------------------------------------------------
    # PBR Material Factory
    # -----------------------------------------------------------------------

    def _set_input(self, node: bpy.types.Node, candidates: list[str], val: Any) -> None:
        for name in candidates:
            if name in node.inputs:
                node.inputs[name].default_value = val
                return

    def _load_tex_node(
        self,
        nodes: bpy.types.Nodes,
        filename: str,
        non_color: bool = False,
    ) -> Optional[bpy.types.Node]:
        p = TEX_DIR / filename
        if not p.exists():
            return None
        img = bpy.data.images.load(str(p), check_existing=True)
        img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.image = img
        return tex

    def make_pbr_mat(
        self,
        name: str,
        base_rgba: tuple[float, float, float, float] = (0.85, 0.84, 0.81, 1.0),
        roughness: float = 0.55,
        metallic: float = 0.0,
        color_tex: Optional[str] = None,
        normal_tex: Optional[str] = None,
        normal_strength: float = 0.55,
        rough_tex: Optional[str] = None,
        emission_rgb: Optional[tuple[float, float, float]] = None,
        emission_strength: float = 0.0,
        alpha: float = 1.0,
        transmission: float = 0.0,
    ) -> bpy.types.Material:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        out = nodes.new(type="ShaderNodeOutputMaterial")
        out.location = (420, 0)
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (120, 0)
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        self._set_input(bsdf, ["Base Color"], base_rgba)
        self._set_input(bsdf, ["Roughness"], roughness)
        self._set_input(bsdf, ["Metallic"], metallic)

        if alpha < 0.999:
            self._set_input(bsdf, ["Alpha"], alpha)
            if hasattr(mat, "blend_method"):
                mat.blend_method = "BLEND"
            if hasattr(mat, "surface_render_method"):
                mat.surface_render_method = "BLENDED"
        if transmission > 0.0:
            self._set_input(bsdf, ["Transmission Weight", "Transmission"], transmission)

        if emission_rgb is not None and emission_strength > 0.0:
            self._set_input(
                bsdf,
                ["Emission Color", "Emission"],
                (emission_rgb[0], emission_rgb[1], emission_rgb[2], 1.0),
            )
            self._set_input(bsdf, ["Emission Strength"], emission_strength)

        if color_tex:
            ct = self._load_tex_node(nodes, color_tex, non_color=False)
            if ct is not None:
                ct.location = (-360, 180)
                links.new(ct.outputs["Color"], bsdf.inputs["Base Color"])
                if emission_rgb is not None and emission_strength > 0.0 and "Emission Color" in bsdf.inputs:
                    # Self-lit screens/backlit canvases use their own texture color when emission_rgb == (-1,-1,-1)
                    if emission_rgb[0] < 0:
                        links.new(ct.outputs["Color"], bsdf.inputs["Emission Color"])

        if rough_tex:
            rt = self._load_tex_node(nodes, rough_tex, non_color=True)
            if rt is not None:
                rt.location = (-360, -20)
                links.new(rt.outputs["Color"], bsdf.inputs["Roughness"])

        if normal_tex:
            nt = self._load_tex_node(nodes, normal_tex, non_color=True)
            if nt is not None:
                nt.location = (-440, -260)
                nmap = nodes.new(type="ShaderNodeNormalMap")
                nmap.location = (-160, -240)
                nmap.inputs["Strength"].default_value = normal_strength
                links.new(nt.outputs["Color"], nmap.inputs["Color"])
                links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

        self.materials[name] = mat
        return mat

    def _init_materials(self) -> None:
        # Architectural finishes
        self.make_pbr_mat(
            "Wall_WarmPlaster",
            base_rgba=(0.935, 0.918, 0.885, 1.0),
            roughness=0.82,
            normal_tex="plaster_normal.png",
            normal_strength=0.22,
        )
        self.make_pbr_mat(
            "Wall_CeilingMatte",
            base_rgba=(0.965, 0.958, 0.942, 1.0),
            roughness=0.88,
        )
        self.make_pbr_mat(
            "Trim_OffWhite",
            base_rgba=(0.90, 0.88, 0.84, 1.0),
            roughness=0.46,
        )
        self.make_pbr_mat(
            "Floor_HoneyOak",
            color_tex="oak_floor_color.jpg",
            normal_tex="oak_floor_normal.png",
            normal_strength=0.45,
            rough_tex="oak_floor_rough.jpg",
            roughness=0.42,
        )
        self.make_pbr_mat(
            "Wood_Walnut",
            color_tex="walnut_color.jpg",
            normal_tex="oak_floor_normal.png",
            normal_strength=0.18,
            roughness=0.38,
        )
        self.make_pbr_mat(
            "Wood_BalconyTeak",
            color_tex="oak_floor_color.jpg",
            normal_tex="oak_floor_normal.png",
            normal_strength=0.55,
            roughness=0.62,
        )
        self.make_pbr_mat(
            "Stone_Calacatta",
            color_tex="calacatta_color.jpg",
            rough_tex="calacatta_rough.jpg",
            roughness=0.20,
        )
        self.make_pbr_mat(
            "Stone_Travertine",
            color_tex="travertine_color.jpg",
            normal_tex="travertine_normal.png",
            normal_strength=0.65,
            roughness=0.48,
        )
        self.make_pbr_mat(
            "Tile_SpaPorcelain",
            color_tex="spa_tile_color.jpg",
            normal_tex="spa_tile_normal.png",
            normal_strength=0.40,
            roughness=0.26,
        )
        self.make_pbr_mat(
            "Cabinet_SageOlive",
            base_rgba=(0.24, 0.27, 0.21, 1.0),
            roughness=0.42,
            normal_tex="plaster_normal.png",
            normal_strength=0.08,
        )
        self.make_pbr_mat(
            "Metal_DarkBronze",
            base_rgba=(0.11, 0.11, 0.12, 1.0),
            roughness=0.28,
            metallic=0.86,
        )
        self.make_pbr_mat(
            "Metal_WarmBrass",
            base_rgba=(0.76, 0.58, 0.34, 1.0),
            roughness=0.24,
            metallic=0.92,
        )
        self.make_pbr_mat(
            "Metal_DarkStainless",
            base_rgba=(0.20, 0.21, 0.23, 1.0),
            roughness=0.22,
            metallic=0.88,
        )
        self.make_pbr_mat(
            "Glass_Architectural",
            base_rgba=(0.82, 0.91, 0.95, 1.0),
            roughness=0.05,
            metallic=0.08,
            alpha=0.24,
            transmission=0.75,
        )
        self.make_pbr_mat(
            "Glass_SmokedTable",
            base_rgba=(0.12, 0.13, 0.14, 1.0),
            roughness=0.08,
            metallic=0.15,
            alpha=0.68,
        )
        self.make_pbr_mat(
            "Glass_AmberGlobe",
            base_rgba=(0.95, 0.72, 0.42, 1.0),
            roughness=0.12,
            alpha=0.45,
            emission_rgb=(1.0, 0.76, 0.42),
            emission_strength=1.4,
        )
        self.make_pbr_mat(
            "Mirror_Silver",
            base_rgba=(0.92, 0.94, 0.95, 1.0),
            roughness=0.03,
            metallic=0.98,
        )
        self.make_pbr_mat(
            "Fabric_OatmealBoucle",
            color_tex="boucle_color.jpg",
            normal_tex="boucle_normal.png",
            normal_strength=0.55,
            roughness=0.86,
        )
        self.make_pbr_mat(
            "Fabric_OliveVelvet",
            base_rgba=(0.25, 0.30, 0.20, 1.0),
            normal_tex="boucle_normal.png",
            normal_strength=0.35,
            roughness=0.80,
        )
        self.make_pbr_mat(
            "Fabric_TerracottaThrow",
            base_rgba=(0.56, 0.34, 0.22, 1.0),
            normal_tex="rug_normal.png",
            normal_strength=0.45,
            roughness=0.88,
        )
        self.make_pbr_mat(
            "Fabric_CrispLinen",
            base_rgba=(0.95, 0.94, 0.91, 1.0),
            normal_tex="boucle_normal.png",
            normal_strength=0.22,
            roughness=0.84,
        )
        self.make_pbr_mat(
            "Fabric_SheerDrapery",
            base_rgba=(0.90, 0.87, 0.81, 1.0),
            roughness=0.85,
            alpha=0.82,
        )
        self.make_pbr_mat(
            "Rug_WovenCream",
            color_tex="rug_color.jpg",
            normal_tex="rug_normal.png",
            normal_strength=0.65,
            roughness=0.92,
        )
        self.make_pbr_mat(
            "Ceramic_White",
            base_rgba=(0.96, 0.96, 0.95, 1.0),
            roughness=0.14,
        )
        self.make_pbr_mat(
            "Water_Spa",
            base_rgba=(0.56, 0.82, 0.86, 1.0),
            roughness=0.06,
            alpha=0.72,
        )
        self.make_pbr_mat(
            "Foliage_OliveGreen",
            base_rgba=(0.20, 0.32, 0.17, 1.0),
            roughness=0.68,
        )
        self.make_pbr_mat(
            "LED_WarmCove",
            base_rgba=(1.0, 0.87, 0.66, 1.0),
            roughness=0.25,
            emission_rgb=(1.0, 0.84, 0.56),
            emission_strength=4.2,
        )
        # Camera-projected / rectified reference materials
        self.make_pbr_mat(
            "Ref_FoyerArt",
            color_tex="ref_foyer_art.jpg",
            roughness=0.35,
        )
        self.make_pbr_mat(
            "Ref_LivingArt1",
            color_tex="ref_living_art_1.jpg",
            roughness=0.35,
        )
        self.make_pbr_mat(
            "Ref_LivingArt2",
            color_tex="ref_living_art_2.jpg",
            roughness=0.35,
        )
        self.make_pbr_mat(
            "Ref_TVScreen",
            color_tex="ref_tv_screen.jpg",
            roughness=0.12,
            emission_rgb=(-1.0, -1.0, -1.0),
            emission_strength=1.15,
        )
        self.make_pbr_mat(
            "Ref_BedMedallion",
            color_tex="ref_bedroom_medallion.jpg",
            roughness=0.45,
            emission_rgb=(-1.0, -1.0, -1.0),
            emission_strength=0.35,
        )
        self.make_pbr_mat(
            "Ref_ExteriorVista",
            color_tex="ref_exterior_vista.jpg",
            roughness=0.95,
            emission_rgb=(-1.0, -1.0, -1.0),
            emission_strength=1.35,
        )

    # -----------------------------------------------------------------------
    # Geometry Primitives with Metric World-Scale Box UVs (Zero Stretching)
    # -----------------------------------------------------------------------

    def _link_obj(self, obj: bpy.types.Object, col_name: str) -> bpy.types.Object:
        self.collections[col_name].objects.link(obj)
        return obj

    def apply_metric_uv(self, obj: bpy.types.Object, tile_m: float = 1.6) -> None:
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        uv_layer = bm.loops.layers.uv.verify()
        mw = obj.matrix_world
        rot_part = mw.to_3x3()

        for face in bm.faces:
            wn = (rot_part @ face.normal).normalized()
            ax, ay, az = abs(wn.x), abs(wn.y), abs(wn.z)
            for loop in face.loops:
                wco = mw @ loop.vert.co
                if az >= max(ax, ay):
                    u, v = wco.x / tile_m, -wco.y / tile_m
                elif ax >= ay:
                    u, v = -wco.y / tile_m, wco.z / tile_m
                else:
                    u, v = wco.x / tile_m, wco.z / tile_m
                loop[uv_layer].uv = (u, v)

        bm.to_mesh(mesh)
        bm.free()

    def add_box(
        self,
        name: str,
        x0: float,
        y0: float,
        z0: float,
        x1: float,
        y1: float,
        z1: float,
        mat_name: str,
        col_name: str = "01_Architecture",
        tile_m: float = 1.6,
        bevel_m: float = 0.0,
        collider_kind: Optional[str] = None,
    ) -> bpy.types.Object:
        lx0, lx1 = min(x0, x1), max(x0, x1)
        ly0, ly1 = min(y0, y1), max(y0, y1)
        lz0, lz1 = min(z0, z1), max(z0, z1)

        cx = 0.5 * (lx0 + lx1)
        cy = 0.5 * (ly0 + ly1)
        cz = 0.5 * (lz0 + lz1)
        sx = max(0.002, lx1 - lx0)
        sy = max(0.002, ly1 - ly0)
        sz = max(0.002, lz1 - lz0)

        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(sx, sz, sy), verts=bm.verts)
        if bevel_m > 0.0:
            bmesh.ops.bevel(
                bm,
                geom=list(bm.edges),
                offset=min(bevel_m, 0.28 * min(sx, sy, sz)),
                segments=2,
                affect="EDGES",
            )
            for f in bm.faces:
                f.smooth = True
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(name, mesh)
        obj.location = hw_pt(cx, cy, cz)
        obj.data.materials.append(self.materials[mat_name])
        self._link_obj(obj, col_name)
        bpy.context.view_layer.update()
        self.apply_metric_uv(obj, tile_m=tile_m)

        if collider_kind is not None:
            self.add_collider(collider_kind, name, lx0, lz0, lx1, lz1, ly0, ly1)
        return obj

    def add_oriented_box(
        self,
        name: str,
        cx: float,
        cy: float,
        cz: float,
        w: float,
        h: float,
        d: float,
        rot_deg_cw: float,
        mat_name: str,
        col_name: str,
        tile_m: float = 1.0,
        bevel_m: float = 0.015,
    ) -> bpy.types.Object:
        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(w, d, h), verts=bm.verts)
        if bevel_m > 0.0:
            bmesh.ops.bevel(
                bm,
                geom=list(bm.edges),
                offset=min(bevel_m, 0.25 * min(w, h, d)),
                segments=2,
                affect="EDGES",
            )
            for f in bm.faces:
                f.smooth = True
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(name, mesh)
        obj.location = hw_pt(cx, cy, cz)
        # Clockwise rotation from +z in Homeweave corresponds to -rot_deg around Blender +Z
        obj.rotation_euler = Euler((0.0, 0.0, -math.radians(rot_deg_cw)), "XYZ")
        obj.data.materials.append(self.materials[mat_name])
        self._link_obj(obj, col_name)
        bpy.context.view_layer.update()
        self.apply_metric_uv(obj, tile_m=tile_m)
        return obj

    def add_cylinder(
        self,
        name: str,
        cx: float,
        y0: float,
        cz: float,
        radius: float,
        height: float,
        mat_name: str,
        col_name: str,
        segments: int = 24,
        tile_m: float = 0.8,
    ) -> bpy.types.Object:
        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            cap_tris=False,
            segments=segments,
            radius1=radius,
            radius2=radius,
            depth=height,
        )
        for f in bm.faces:
            f.smooth = abs(f.normal.z) < 0.8
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(name, mesh)
        obj.location = hw_pt(cx, y0 + 0.5 * height, cz)
        obj.data.materials.append(self.materials[mat_name])
        self._link_obj(obj, col_name)
        bpy.context.view_layer.update()
        self.apply_metric_uv(obj, tile_m=tile_m)
        return obj

    def add_sphere(
        self,
        name: str,
        cx: float,
        cy: float,
        cz: float,
        radius: float,
        mat_name: str,
        col_name: str,
    ) -> bpy.types.Object:
        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=16, radius=radius)
        for f in bm.faces:
            f.smooth = True
        bm.to_mesh(mesh)
        bm.free()
        obj = bpy.data.objects.new(name, mesh)
        obj.location = hw_pt(cx, cy, cz)
        obj.data.materials.append(self.materials[mat_name])
        self._link_obj(obj, col_name)
        return obj

    def add_uv_quad(
        self,
        name: str,
        corners_hw: list[tuple[float, float, float]],
        mat_name: str,
        col_name: str,
    ) -> bpy.types.Object:
        """Create a 1:1 UV-mapped planar quad (BL, BR, TR, TL in Homeweave world coords)."""
        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()
        verts = [bm.verts.new(hw_pt(*c)) for c in corners_hw]
        face = bm.faces.new(verts)
        uv_layer = bm.loops.layers.uv.verify()
        uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        for loop, uv in zip(face.loops, uvs):
            loop[uv_layer].uv = uv
        bm.to_mesh(mesh)
        bm.free()
        obj = bpy.data.objects.new(name, mesh)
        obj.data.materials.append(self.materials[mat_name])
        self._link_obj(obj, col_name)
        return obj

    def add_uv_disc_west_wall(
        self,
        name: str,
        x_face: float,
        cy: float,
        cz: float,
        radius: float,
        mat_name: str,
        col_name: str,
        segments: int = 40,
    ) -> bpy.types.Object:
        """Create a true circular 3D disc facing +X with circular [0..1] planar UVs."""
        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()
        verts: list[bmesh.types.BMVert] = []
        uv_coords: list[tuple[float, float]] = []
        for i in range(segments):
            ang = 2.0 * math.pi * (i / segments)
            # Viewed facing West (-X): +U goes North (-z), +V goes Up (+y)
            dz = -math.cos(ang) * radius
            dy = math.sin(ang) * radius
            verts.append(bm.verts.new(hw_pt(x_face, cy + dy, cz + dz)))
            uv_coords.append((0.5 + 0.46 * math.cos(ang), 0.5 + 0.46 * math.sin(ang)))
        face = bm.faces.new(verts)
        face.smooth = True
        uv_layer = bm.loops.layers.uv.verify()
        for loop, uv in zip(face.loops, uv_coords):
            loop[uv_layer].uv = uv
        bmesh.ops.solidify(bm, geom=[face], thickness=0.012)
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(name, mesh)
        obj.data.materials.append(self.materials[mat_name])
        self._link_obj(obj, col_name)
        return obj

    def add_olive_bouquet(
        self,
        name: str,
        cx: float,
        y_vase_top: float,
        cz: float,
        scale: float,
        col_name: str,
    ) -> None:
        """Organic multi-stem olive branch bouquet replacing flat geometric plant spheres."""
        stems = [
            (-0.04 * scale, -0.03 * scale, 0.28 * scale, 14.0, -10.0),
            (0.05 * scale, 0.02 * scale, 0.34 * scale, -16.0, 12.0),
            (0.0, 0.04 * scale, 0.25 * scale, 8.0, 18.0),
        ]
        for s_i, (ox, oz, sh, rx_d, ry_d) in enumerate(stems):
            stem = self.add_cylinder(
                f"{name}_stem_{s_i}",
                cx + ox * 0.5,
                y_vase_top - 0.03,
                cz + oz * 0.5,
                0.005 * scale,
                sh,
                "Wood_Walnut",
                col_name,
                segments=6,
            )
            stem.rotation_euler = Euler((math.radians(rx_d), math.radians(ry_d), 0.0), "XYZ")
            for lf_i in range(4):
                t = 0.45 + 0.18 * lf_i
                lx = cx + ox * t + 0.045 * scale * math.cos(s_i * 2.1 + lf_i * 1.7)
                lz = cz + oz * t + 0.045 * scale * math.sin(s_i * 2.1 + lf_i * 1.7)
                ly = y_vase_top + sh * t * 0.85
                leaf = self.add_oriented_box(
                    f"{name}_leaf_{s_i}_{lf_i}",
                    lx,
                    ly,
                    lz,
                    0.075 * scale,
                    0.022 * scale,
                    0.038 * scale,
                    (s_i * 67.0 + lf_i * 43.0) % 360.0,
                    "Foliage_OliveGreen",
                    col_name,
                    bevel_m=0.008,
                )
                leaf.rotation_euler.x = math.radians(18.0 * ((lf_i % 2) * 2 - 1))

    # -----------------------------------------------------------------------
    # Architectural Wall Builder with Real Window Reveals, Door Casings & Baseboards
    # -----------------------------------------------------------------------

    def build_wall_with_openings(
        self,
        name: str,
        axis: str,
        fixed_pos: float,
        span_start: float,
        span_end: float,
        thick: float,
        openings: list[dict[str, Any]],
        baseboard_sides: tuple[bool, bool] = (True, True),
    ) -> None:
        """Build a wall along `axis` ('x' => constant z=fixed_pos; 'z' => constant x=fixed_pos).

        `openings` items: {'s0': float, 's1': float, 'y0': float, 'y1': float, 'kind': 'door'|'portal'|'window'|'slider'}
        """
        ops = sorted(openings, key=lambda o: o["s0"])
        cursor = span_start
        half_t = 0.5 * thick

        def wall_seg(tag: str, a0: float, a1: float, y0: float, y1: float, register_col: bool) -> None:
            if a1 - a0 <= 0.005 or y1 - y0 <= 0.005:
                return
            ck = "wall" if (register_col and y0 < 1.85) else None
            if axis == "x":
                self.add_box(
                    f"{name}_{tag}",
                    a0,
                    y0,
                    fixed_pos - half_t,
                    a1,
                    y1,
                    fixed_pos + half_t,
                    "Wall_WarmPlaster",
                    "01_Architecture",
                    tile_m=2.0,
                    collider_kind=ck,
                )
            else:
                self.add_box(
                    f"{name}_{tag}",
                    fixed_pos - half_t,
                    y0,
                    a0,
                    fixed_pos + half_t,
                    y1,
                    a1,
                    "Wall_WarmPlaster",
                    "01_Architecture",
                    tile_m=2.0,
                    collider_kind=ck,
                )

        def add_baseboard_span(a0: float, a1: float) -> None:
            if a1 - a0 <= 0.03:
                return
            bb_h = 0.105
            bb_p = 0.016
            for idx, enabled in enumerate(baseboard_sides):
                if not enabled:
                    continue
                sign = -1.0 if idx == 0 else 1.0
                if axis == "x":
                    z_face = fixed_pos + sign * half_t
                    self.add_box(
                        f"{name}_bb_{idx}_{a0:.2f}",
                        a0,
                        0.0,
                        z_face,
                        a1,
                        bb_h,
                        z_face + sign * bb_p,
                        "Trim_OffWhite",
                        "01_Architecture",
                        tile_m=1.2,
                        bevel_m=0.003,
                    )
                else:
                    x_face = fixed_pos + sign * half_t
                    self.add_box(
                        f"{name}_bb_{idx}_{a0:.2f}",
                        x_face,
                        0.0,
                        a0,
                        x_face + sign * bb_p,
                        bb_h,
                        a1,
                        "Trim_OffWhite",
                        "01_Architecture",
                        tile_m=1.2,
                        bevel_m=0.003,
                    )

        for idx, op in enumerate(ops):
            s0, s1 = op["s0"], op["s1"]
            y0, y1 = op["y0"], op["y1"]
            kind = op["kind"]
            if s0 > cursor:
                wall_seg(f"solid_{idx}", cursor, s0, 0.0, CEIL_H, True)
                add_baseboard_span(cursor, s0 - (0.075 if kind in ("door", "portal") else 0.0))

            # Below-sill wall for windows
            if y0 > 0.01:
                wall_seg(f"sillwall_{idx}", s0, s1, 0.0, y0, True)
                add_baseboard_span(s0, s1)
            # Lintel header above opening
            if y1 < CEIL_H - 0.01:
                wall_seg(f"lintel_{idx}", s0, s1, y1, CEIL_H, False)

            if kind in ("door", "portal"):
                self._build_door_casing_and_leaf(name, axis, fixed_pos, half_t, op, idx)
            elif kind == "window":
                self._build_recessed_window(name, axis, fixed_pos, half_t, op, idx)
            elif kind == "slider":
                self._build_sliding_balcony_door(name, axis, fixed_pos, half_t, op, idx)

            cursor = s1

        if cursor < span_end:
            wall_seg("solid_end", cursor, span_end, 0.0, CEIL_H, True)
            add_baseboard_span(cursor, span_end)

    def _build_door_casing_and_leaf(
        self,
        wall_name: str,
        axis: str,
        fixed_pos: float,
        half_t: float,
        op: dict[str, Any],
        idx: int,
    ) -> None:
        s0, s1, y1 = op["s0"], op["s1"], op["y1"]
        jamb_t = 0.026
        trim_w = 0.072
        trim_p = 0.018

        if axis == "x":
            # Left, right, top walnut jambs + proud trim on both faces
            self.add_box(f"{wall_name}_jambL_{idx}", s0, 0.0, fixed_pos - half_t - 0.004, s0 + jamb_t, y1, fixed_pos + half_t + 0.004, "Wood_Walnut", "01_Architecture", bevel_m=0.003)
            self.add_box(f"{wall_name}_jambR_{idx}", s1 - jamb_t, 0.0, fixed_pos - half_t - 0.004, s1, y1, fixed_pos + half_t + 0.004, "Wood_Walnut", "01_Architecture", bevel_m=0.003)
            self.add_box(f"{wall_name}_jambT_{idx}", s0, y1 - jamb_t, fixed_pos - half_t - 0.004, s1, y1, fixed_pos + half_t + 0.004, "Wood_Walnut", "01_Architecture", bevel_m=0.003)
            for sgn in (-1.0, 1.0):
                zf = fixed_pos + sgn * half_t
                self.add_box(f"{wall_name}_trimL_{idx}_{sgn}", s0 - trim_w, 0.0, zf, s0 + 0.01, y1 + trim_w, zf + sgn * trim_p, "Wood_Walnut", "01_Architecture", bevel_m=0.003)
                self.add_box(f"{wall_name}_trimR_{idx}_{sgn}", s1 - 0.01, 0.0, zf, s1 + trim_w, y1 + trim_w, zf + sgn * trim_p, "Wood_Walnut", "01_Architecture", bevel_m=0.003)
                self.add_box(f"{wall_name}_trimT_{idx}_{sgn}", s0 - trim_w, y1 - 0.01, zf, s1 + trim_w, y1 + trim_w, zf + sgn * trim_p, "Wood_Walnut", "01_Architecture", bevel_m=0.003)
        else:
            self.add_box(f"{wall_name}_jambL_{idx}", fixed_pos - half_t - 0.004, 0.0, s0, fixed_pos + half_t + 0.004, y1, s0 + jamb_t, "Wood_Walnut", "01_Architecture", bevel_m=0.003)
            self.add_box(f"{wall_name}_jambR_{idx}", fixed_pos - half_t - 0.004, 0.0, s1 - jamb_t, fixed_pos + half_t + 0.004, y1, s1, "Wood_Walnut", "01_Architecture", bevel_m=0.003)
            self.add_box(f"{wall_name}_jambT_{idx}", fixed_pos - half_t - 0.004, y1 - jamb_t, s0, fixed_pos + half_t + 0.004, y1, s1, "Wood_Walnut", "01_Architecture", bevel_m=0.003)
            for sgn in (-1.0, 1.0):
                xf = fixed_pos + sgn * half_t
                self.add_box(f"{wall_name}_trimL_{idx}_{sgn}", xf, 0.0, s0 - trim_w, xf + sgn * trim_p, y1 + trim_w, s0 + 0.01, "Wood_Walnut", "01_Architecture", bevel_m=0.003)
                self.add_box(f"{wall_name}_trimR_{idx}_{sgn}", xf, 0.0, s1 - 0.01, xf + sgn * trim_p, y1 + trim_w, s1 + trim_w, "Wood_Walnut", "01_Architecture", bevel_m=0.003)
                self.add_box(f"{wall_name}_trimT_{idx}_{sgn}", xf, y1 - 0.01, s0 - trim_w, xf + sgn * trim_p, y1 + trim_w, s1 + trim_w, "Wood_Walnut", "01_Architecture", bevel_m=0.003)

        # Optional door leaf ("closed_ext" or "ajar_south")
        leaf_mode = op.get("leaf")
        if leaf_mode == "closed_ext" and axis == "z":
            self.add_box(
                f"{wall_name}_doorleaf_{idx}",
                fixed_pos - 0.022,
                0.01,
                s0 + jamb_t,
                fixed_pos + 0.022,
                y1 - jamb_t,
                s1 - jamb_t,
                "Wood_Walnut",
                "01_Architecture",
                bevel_m=0.004,
                collider_kind="door",
            )
            # Brushed brass lever handle
            self.add_box(
                f"{wall_name}_handle_{idx}",
                fixed_pos - 0.075,
                1.02,
                s0 + 0.11,
                fixed_pos - 0.022,
                1.05,
                s0 + 0.24,
                "Metal_WarmBrass",
                "01_Architecture",
                bevel_m=0.004,
            )
        elif leaf_mode == "ajar_south" and axis == "x":
            # Swung ~86 deg open into +z room from right jamb (s1 - jamb_t) so doorway stays wide open
            leaf_len = (s1 - s0) - 2 * jamb_t
            hx = s1 - jamb_t - 0.02
            hz = fixed_pos + half_t + 0.02
            self.add_box(
                f"{wall_name}_ajarleaf_{idx}",
                hx - 0.042,
                0.01,
                hz,
                hx,
                y1 - jamb_t - 0.01,
                hz + leaf_len,
                "Wood_Walnut",
                "01_Architecture",
                bevel_m=0.004,
                collider_kind="door",
            )
            self.add_box(
                f"{wall_name}_ajarhandle_{idx}",
                hx - 0.085,
                1.02,
                hz + leaf_len - 0.14,
                hx + 0.04,
                1.045,
                hz + leaf_len - 0.08,
                "Metal_WarmBrass",
                "01_Architecture",
                bevel_m=0.003,
            )

    def _build_recessed_window(
        self,
        wall_name: str,
        axis: str,
        fixed_pos: float,
        half_t: float,
        op: dict[str, Any],
        idx: int,
    ) -> None:
        s0, s1, y0, y1 = op["s0"], op["s1"], op["y0"], op["y1"]
        inward_sgn = op.get("inward_sgn", 1.0)
        sill_overhang = 0.045
        frame_d = 0.055
        frame_w = 0.042
        mullion_count = op.get("mullions", 2)

        if axis == "x":
            # Stone sill protruding slightly into the room for real 3D reveal depth
            z_in = fixed_pos + inward_sgn * (half_t + sill_overhang)
            z_out = fixed_pos - inward_sgn * half_t
            self.add_box(
                f"{wall_name}_sill_{idx}",
                s0 - 0.03,
                y0 - 0.032,
                min(z_in, z_out),
                s1 + 0.03,
                y0 + 0.005,
                max(z_in, z_out),
                "Stone_Calacatta",
                "01_Architecture",
                bevel_m=0.004,
            )
            # Outer dark-bronze frame recessed near exterior side of the reveal
            zf = fixed_pos - inward_sgn * (half_t - 0.04)
            self.add_box(f"{wall_name}_wfB_{idx}", s0, y0, zf - 0.5 * frame_d, s1, y0 + frame_w, zf + 0.5 * frame_d, "Metal_DarkBronze", "01_Architecture")
            self.add_box(f"{wall_name}_wfT_{idx}", s0, y1 - frame_w, zf - 0.5 * frame_d, s1, y1, zf + 0.5 * frame_d, "Metal_DarkBronze", "01_Architecture")
            self.add_box(f"{wall_name}_wfL_{idx}", s0, y0, zf - 0.5 * frame_d, s0 + frame_w, y1, zf + 0.5 * frame_d, "Metal_DarkBronze", "01_Architecture")
            self.add_box(f"{wall_name}_wfR_{idx}", s1 - frame_w, y0, zf - 0.5 * frame_d, s1, y1, zf + 0.5 * frame_d, "Metal_DarkBronze", "01_Architecture")
            for m_i in range(1, mullion_count):
                mx = s0 + (s1 - s0) * (m_i / mullion_count)
                self.add_box(f"{wall_name}_wm_{idx}_{m_i}", mx - 0.016, y0, zf - 0.022, mx + 0.016, y1, zf + 0.022, "Metal_DarkBronze", "01_Architecture")
            self.add_box(
                f"{wall_name}_wglass_{idx}",
                s0 + frame_w,
                y0 + frame_w,
                zf - 0.008,
                s1 - frame_w,
                y1 - frame_w,
                zf + 0.008,
                "Glass_Architectural",
                "01_Architecture",
                collider_kind="window",
            )
        else:
            x_in = fixed_pos + inward_sgn * (half_t + sill_overhang)
            x_out = fixed_pos - inward_sgn * half_t
            self.add_box(
                f"{wall_name}_sill_{idx}",
                min(x_in, x_out),
                y0 - 0.032,
                s0 - 0.03,
                max(x_in, x_out),
                y0 + 0.005,
                s1 + 0.03,
                "Stone_Calacatta",
                "01_Architecture",
                bevel_m=0.004,
            )
            xf = fixed_pos - inward_sgn * (half_t - 0.04)
            self.add_box(f"{wall_name}_wfB_{idx}", xf - 0.5 * frame_d, y0, s0, xf + 0.5 * frame_d, y0 + frame_w, s1, "Metal_DarkBronze", "01_Architecture")
            self.add_box(f"{wall_name}_wfT_{idx}", xf - 0.5 * frame_d, y1 - frame_w, s0, xf + 0.5 * frame_d, y1, s1, "Metal_DarkBronze", "01_Architecture")
            self.add_box(f"{wall_name}_wfL_{idx}", xf - 0.5 * frame_d, y0, s0, xf + 0.5 * frame_d, y1, s0 + frame_w, "Metal_DarkBronze", "01_Architecture")
            self.add_box(f"{wall_name}_wfR_{idx}", xf - 0.5 * frame_d, y0, s1 - frame_w, xf + 0.5 * frame_d, y1, s1, "Metal_DarkBronze", "01_Architecture")
            for m_i in range(1, mullion_count):
                mz = s0 + (s1 - s0) * (m_i / mullion_count)
                self.add_box(f"{wall_name}_wm_{idx}_{m_i}", xf - 0.022, y0, mz - 0.016, xf + 0.022, y1, mz + 0.016, "Metal_DarkBronze", "01_Architecture")
            self.add_box(
                f"{wall_name}_wglass_{idx}",
                xf - 0.008,
                y0 + frame_w,
                s0 + frame_w,
                xf + 0.008,
                y1 - frame_w,
                s1 - frame_w,
                "Glass_Architectural",
                "01_Architecture",
                collider_kind="window",
            )

    def _build_sliding_balcony_door(
        self,
        wall_name: str,
        axis: str,
        fixed_pos: float,
        half_t: float,
        op: dict[str, Any],
        idx: int,
    ) -> None:
        """Full-height glass curtain wall with fixed side lights + 1 openable sliding balcony portal."""
        s0, s1, y1 = op["s0"], op["s1"], op["y1"]
        open_x0 = op["open_s0"]
        open_x1 = op["open_s1"]
        zf = fixed_pos - 0.02

        # Outer bronze perimeter track & floor threshold
        self.add_box(f"{wall_name}_trackB_{idx}", s0, 0.0, fixed_pos - half_t, s1, 0.025, fixed_pos + half_t, "Metal_DarkBronze", "01_Architecture")
        self.add_box(f"{wall_name}_trackT_{idx}", s0, y1 - 0.045, fixed_pos - half_t, s1, y1, fixed_pos + half_t, "Metal_DarkBronze", "01_Architecture")
        self.add_box(f"{wall_name}_jambL_{idx}", s0, 0.0, fixed_pos - half_t, s0 + 0.045, y1, fixed_pos + half_t, "Metal_DarkBronze", "01_Architecture")
        self.add_box(f"{wall_name}_jambR_{idx}", s1 - 0.045, 0.0, fixed_pos - half_t, s1, y1, fixed_pos + half_t, "Metal_DarkBronze", "01_Architecture")

        # Left fixed glass bay [s0 .. open_x0]
        self.add_box(
            f"{wall_name}_glassL_{idx}",
            s0 + 0.045,
            0.025,
            zf - 0.01,
            open_x0,
            y1 - 0.045,
            zf + 0.01,
            "Glass_Architectural",
            "01_Architecture",
            collider_kind="window",
        )
        self.add_box(f"{wall_name}_postL_{idx}", open_x0 - 0.022, 0.025, zf - 0.03, open_x0 + 0.022, y1 - 0.045, zf + 0.03, "Metal_DarkBronze", "01_Architecture")

        # Slid-open sliding glass door leaf parked behind left bay [open_x0 - 0.92 .. open_x0]
        slide_w = open_x1 - open_x0
        self.add_box(f"{wall_name}_slideFrameB", open_x0 - slide_w, 0.03, zf + 0.03, open_x0 + 0.03, 0.075, zf + 0.065, "Metal_DarkBronze", "01_Architecture")
        self.add_box(f"{wall_name}_slideFrameT", open_x0 - slide_w, y1 - 0.09, zf + 0.03, open_x0 + 0.03, y1 - 0.045, zf + 0.065, "Metal_DarkBronze", "01_Architecture")
        self.add_box(f"{wall_name}_slideStileR", open_x0 - 0.01, 0.03, zf + 0.03, open_x0 + 0.035, y1 - 0.045, zf + 0.065, "Metal_DarkBronze", "01_Architecture")
        self.add_box(f"{wall_name}_slideGlass", open_x0 - slide_w + 0.03, 0.075, zf + 0.042, open_x0 - 0.01, y1 - 0.09, zf + 0.052, "Glass_Architectural", "01_Architecture")

        # Right fixed glass bay [open_x1 .. s1] (leaves [open_x0 .. open_x1] 100% open to walk onto balcony!)
        self.add_box(f"{wall_name}_postR_{idx}", open_x1 - 0.022, 0.025, zf - 0.03, open_x1 + 0.022, y1 - 0.045, zf + 0.03, "Metal_DarkBronze", "01_Architecture")
        self.add_box(
            f"{wall_name}_glassR_{idx}",
            open_x1,
            0.025,
            zf - 0.01,
            s1 - 0.045,
            y1 - 0.045,
            zf + 0.01,
            "Glass_Architectural",
            "01_Architecture",
            collider_kind="window",
        )

    def add_coffered_ceiling_with_cove(
        self,
        tag: str,
        x0: float,
        z0: float,
        x1: float,
        z1: float,
        col_name: str = "01_Architecture",
    ) -> None:
        """Build architectural soffit perimeter + recessed tray ceiling + warm LED cove strip."""
        soffit_w = 0.42
        soffit_y0 = CEIL_H - 0.14
        # Main upper ceiling plane
        self.add_box(f"{tag}_ceil_top", x0, CEIL_H, z0, x1, CEIL_H + 0.08, z1, "Wall_CeilingMatte", col_name, tile_m=2.5)
        # 4 perimeter soffit beams
        self.add_box(f"{tag}_sofN", x0, soffit_y0, z0, x1, CEIL_H, z0 + soffit_w, "Wall_CeilingMatte", col_name, tile_m=2.0)
        self.add_box(f"{tag}_sofS", x0, soffit_y0, z1 - soffit_w, x1, CEIL_H, z1, "Wall_CeilingMatte", col_name, tile_m=2.0)
        self.add_box(f"{tag}_sofW", x0, soffit_y0, z0 + soffit_w, x0 + soffit_w, CEIL_H, z1 - soffit_w, "Wall_CeilingMatte", col_name, tile_m=2.0)
        self.add_box(f"{tag}_sofE", x1 - soffit_w, soffit_y0, z0 + soffit_w, x1, CEIL_H, z1 - soffit_w, "Wall_CeilingMatte", col_name, tile_m=2.0)

        # Glowing indirect LED cove lip recessed inside tray
        ix0, ix1 = x0 + soffit_w, x1 - soffit_w
        iz0, iz1 = z0 + soffit_w, z1 - soffit_w
        lip_y = soffit_y0 + 0.028
        self.add_box(f"{tag}_ledN", ix0, lip_y, iz0 - 0.015, ix1, lip_y + 0.018, iz0 + 0.01, "LED_WarmCove", col_name)
        self.add_box(f"{tag}_ledS", ix0, lip_y, iz1 - 0.01, ix1, lip_y + 0.018, iz1 + 0.015, "LED_WarmCove", col_name)
        self.add_box(f"{tag}_ledW", ix0 - 0.015, lip_y, iz0, ix0 + 0.01, lip_y + 0.018, iz1, "LED_WarmCove", col_name)
        self.add_box(f"{tag}_ledE", ix1 - 0.01, lip_y, iz0, ix1 + 0.015, lip_y + 0.018, iz1, "LED_WarmCove", col_name)

    def add_pleated_curtain_stack(
        self,
        name: str,
        x0: float,
        x1: float,
        z_center: float,
        y0: float,
        y1: float,
        col_name: str,
        pleats: int = 7,
    ) -> None:
        """Create real 3D sinusoidal pleated drapery folds so curtains have genuine depth."""
        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()
        nx = pleats * 4 + 1
        ny = 8
        verts_grid: list[list[bmesh.types.BMVert]] = []
        for iy in range(ny):
            vy = y0 + (y1 - y0) * (iy / (ny - 1))
            row: list[bmesh.types.BMVert] = []
            for ix in range(nx):
                u = ix / (nx - 1)
                vx = x0 + (x1 - x0) * u
                vz = z_center + 0.048 * math.sin(u * pleats * 2.0 * math.pi)
                row.append(bm.verts.new(hw_pt(vx, vy, vz)))
            verts_grid.append(row)

        for iy in range(ny - 1):
            for ix in range(nx - 1):
                f = bm.faces.new(
                    [
                        verts_grid[iy][ix],
                        verts_grid[iy][ix + 1],
                        verts_grid[iy + 1][ix + 1],
                        verts_grid[iy + 1][ix],
                    ]
                )
                f.smooth = True

        # Solidify slightly (6 mm) so curtains render cleanly from both inside and balcony
        bmesh.ops.solidify(bm, geom=list(bm.faces), thickness=0.006)
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(name, mesh)
        obj.data.materials.append(self.materials["Fabric_SheerDrapery"])
        self._link_obj(obj, col_name)
        bpy.context.view_layer.update()
        self.apply_metric_uv(obj, tile_m=0.8)

    # -----------------------------------------------------------------------
    # Build Full Apartment Shell + All 4 Rooms + Riverfront Balcony
    # -----------------------------------------------------------------------

    def build_shell_and_floors(self) -> None:
        # Continuous wide-plank honey oak floor across Living, Kitchen/Foyer, and Primary Bedroom
        self.add_box("Floor_LivingKitchen", 0.0, -0.06, 0.0, 9.42, 0.0, 3.66, "Floor_HoneyOak", "01_Architecture", tile_m=1.8)
        self.add_box("Floor_Bedroom", 0.0, -0.06, 3.66, 4.78, 0.0, 6.99, "Floor_HoneyOak", "01_Architecture", tile_m=1.8)
        # Spa bathroom honed porcelain tile floor
        self.add_box("Floor_Bathroom", 4.78, -0.06, 3.66, 8.18, 0.0, 6.99, "Tile_SpaPorcelain", "01_Architecture", tile_m=1.2)
        # Brass doorway transition strip at bathroom threshold
        self.add_box("Threshold_Bath", 6.12, 0.0, 3.64, 7.00, 0.006, 3.68, "Metal_WarmBrass", "01_Architecture")

        # Coffered tray ceilings with warm perimeter LED coves in all 4 rooms
        self.add_coffered_ceiling_with_cove("Ceil_Living", 0.0, 0.0, 5.82, 3.66)
        self.add_coffered_ceiling_with_cove("Ceil_Kitchen", 5.82, 0.0, 9.42, 3.66)
        self.add_coffered_ceiling_with_cove("Ceil_Bedroom", 0.0, 3.66, 4.78, 6.99)
        self.add_coffered_ceiling_with_cove("Ceil_Bathroom", 4.78, 3.66, 8.18, 6.99)

        # 1. North Exterior Wall (z = 0.00, x in [0.00, 9.42])
        # Living section [0..5.82] has huge glass curtain wall + open sliding balcony portal [2.95..3.88];
        # Kitchen section [5.82..9.42] is solid behind cooktop & upper shelves.
        self.build_wall_with_openings(
            "Wall_North",
            axis="x",
            fixed_pos=0.0,
            span_start=-0.09,
            span_end=9.51,
            thick=EXT_WALL_T,
            openings=[
                {
                    "s0": 0.52,
                    "s1": 5.30,
                    "y0": 0.0,
                    "y1": 2.46,
                    "kind": "slider",
                    "open_s0": 2.95,
                    "open_s1": 3.88,
                }
            ],
            baseboard_sides=(False, True),
        )

        # 2. West Exterior Wall (x = 0.00, z in [0.00, 6.99])
        self.build_wall_with_openings(
            "Wall_West",
            axis="z",
            fixed_pos=0.0,
            span_start=0.0,
            span_end=7.08,
            thick=EXT_WALL_T,
            openings=[],
            baseboard_sides=(False, True),
        )

        # 3. East Kitchen & Entry Exterior Wall (x = 9.42, z in [0.00, 3.66])
        self.build_wall_with_openings(
            "Wall_East_Kitchen",
            axis="z",
            fixed_pos=9.42,
            span_start=0.0,
            span_end=3.75,
            thick=EXT_WALL_T,
            openings=[
                {
                    "s0": 1.08,
                    "s1": 2.24,
                    "y0": 1.04,
                    "y1": 2.30,
                    "kind": "window",
                    "inward_sgn": -1.0,
                    "mullions": 2,
                },
                {
                    "s0": 2.58,
                    "s1": 3.48,
                    "y0": 0.0,
                    "y1": DOOR_H,
                    "kind": "door",
                    "leaf": "closed_ext",
                },
            ],
            baseboard_sides=(True, False),
        )

        # 4. Central East-West Spine Wall (z = 3.66, x in [0.00, 9.42])
        # Connects Living -> Bedroom at x=[3.38, 4.30], Kitchen/Foyer -> Bathroom at x=[6.12, 7.00],
        # and seals Foyer south wall x=[8.18, 9.51].
        self.build_wall_with_openings(
            "Wall_Central_Spine",
            axis="x",
            fixed_pos=3.66,
            span_start=0.0,
            span_end=8.18,
            thick=INT_WALL_T,
            openings=[
                {
                    "s0": 3.38,
                    "s1": 4.30,
                    "y0": 0.0,
                    "y1": DOOR_H,
                    "kind": "door",
                    "leaf": "ajar_south",
                },
                {
                    "s0": 6.12,
                    "s1": 7.00,
                    "y0": 0.0,
                    "y1": DOOR_H,
                    "kind": "door",
                    "leaf": "ajar_south",
                },
            ],
            baseboard_sides=(True, True),
        )
        # Exterior south wall under Foyer (x in [8.18, 9.51], z = 3.66)
        self.build_wall_with_openings(
            "Wall_Foyer_South",
            axis="x",
            fixed_pos=3.66,
            span_start=8.18,
            span_end=9.51,
            thick=EXT_WALL_T,
            openings=[],
            baseboard_sides=(True, False),
        )

        # 5. Living <-> Kitchen Partition Wall (x = 5.82, z in [0.00, 3.66])
        # Wide cased portal z=[0.74, 3.30] preserves sightlines & full room-to-room flow
        self.build_wall_with_openings(
            "Wall_Living_Kitchen",
            axis="z",
            fixed_pos=5.82,
            span_start=0.0,
            span_end=3.66,
            thick=INT_WALL_T,
            openings=[
                {
                    "s0": 0.74,
                    "s1": 3.30,
                    "y0": 0.0,
                    "y1": 2.40,
                    "kind": "portal",
                }
            ],
            baseboard_sides=(True, True),
        )

        # 6. Bedroom <-> Bathroom Partition Wall (x = 4.78, z in [3.66, 6.99])
        self.build_wall_with_openings(
            "Wall_Bed_Bath",
            axis="z",
            fixed_pos=4.78,
            span_start=3.66,
            span_end=6.99,
            thick=INT_WALL_T,
            openings=[],
            baseboard_sides=(True, True),
        )

        # 7. Bathroom East Exterior Wall (x = 8.18, z in [3.66, 6.99])
        self.build_wall_with_openings(
            "Wall_Bath_East",
            axis="z",
            fixed_pos=8.18,
            span_start=3.66,
            span_end=7.08,
            thick=EXT_WALL_T,
            openings=[],
            baseboard_sides=(True, False),
        )

        # 8. South Bedroom & Bathroom Exterior Wall (z = 6.99, x in [0.00, 8.18])
        self.build_wall_with_openings(
            "Wall_South_BedBath",
            axis="x",
            fixed_pos=6.99,
            span_start=-0.09,
            span_end=8.27,
            thick=EXT_WALL_T,
            openings=[
                {
                    "s0": 0.90,
                    "s1": 3.85,
                    "y0": 0.46,
                    "y1": 2.36,
                    "kind": "window",
                    "inward_sgn": -1.0,
                    "mullions": 3,
                },
                {
                    "s0": 6.68,
                    "s1": 7.78,
                    "y0": 1.02,
                    "y1": 2.30,
                    "kind": "window",
                    "inward_sgn": -1.0,
                    "mullions": 2,
                },
            ],
            baseboard_sides=(True, False),
        )

    def build_living_dining_room(self) -> None:
        col = "02_Living_Dining"

        # 1. Backlit Travertine Feature TV Wall (matched to L1.jpg & L2.jpg)
        # Warm LED halo recessed behind proud stone slab
        self.add_box("TV_LED_Halo", 0.09, 0.56, 0.72, 0.115, 2.44, 2.94, "LED_WarmCove", col)
        self.add_box(
            "TV_Travertine_Slab",
            0.10,
            0.52,
            0.68,
            0.165,
            2.48,
            2.98,
            "Stone_Travertine",
            col,
            tile_m=1.8,
            bevel_m=0.008,
        )

        # Flanking Walnut Illuminated Display Niches (z=[0.20, 0.62] and z=[3.04, 3.46])
        for side_i, (nz0, nz1) in enumerate([(0.20, 0.62), (3.04, 3.46)]):
            self.add_box(f"Niche_Back_{side_i}", 0.08, 0.48, nz0, 0.11, 2.48, nz1, "Wood_Walnut", col, tile_m=1.0)
            self.add_box(f"Niche_SideA_{side_i}", 0.09, 0.48, nz0, 0.28, 2.48, nz0 + 0.028, "Wood_Walnut", col, bevel_m=0.003)
            self.add_box(f"Niche_SideB_{side_i}", 0.09, 0.48, nz1 - 0.028, 0.28, 2.48, nz1, "Wood_Walnut", col, bevel_m=0.003)
            for sh_i, sy in enumerate([0.50, 0.98, 1.46, 1.94, 2.45]):
                self.add_box(f"Niche_Shelf_{side_i}_{sh_i}", 0.10, sy, nz0 + 0.02, 0.27, sy + 0.025, nz1 - 0.02, "Wood_Walnut", col, bevel_m=0.003)
                if sh_i < 4:
                    self.add_box(f"Niche_LED_{side_i}_{sh_i}", 0.12, sy + 0.44, nz0 + 0.05, 0.15, sy + 0.455, nz1 - 0.05, "LED_WarmCove", col)
                    # Sculptural ceramic vessel on shelf
                    mid_z = 0.5 * (nz0 + nz1)
                    self.add_cylinder(f"Niche_Vase_{side_i}_{sh_i}", 0.19, sy + 0.025, mid_z, 0.055, 0.18, "Ceramic_White" if sh_i % 2 == 0 else "Metal_WarmBrass", col, segments=18)

        # 65" Framed OLED TV + Rectified Cinema Screen from L1.jpg + Soundbar
        self.add_box("OLED_TV_Chassis", 0.165, 1.04, 1.10, 0.20, 1.88, 2.56, "Metal_DarkBronze", col, bevel_m=0.004)
        self.add_uv_quad(
            "OLED_TV_Screen",
            [
                (0.203, 1.065, 2.535),
                (0.203, 1.065, 1.125),
                (0.203, 1.855, 1.125),
                (0.203, 1.855, 2.535),
            ],
            "Ref_TVScreen",
            col,
        )
        self.add_box("TV_Soundbar", 0.21, 0.56, 1.38, 0.30, 0.63, 2.28, "Metal_DarkBronze", col, bevel_m=0.006)

        # Floating Walnut Media Console (`src-tv-stand-01` upgraded to finished architectural detail)
        self.add_box(
            "Media_Console_Body",
            0.10,
            0.20,
            0.66,
            0.50,
            0.52,
            3.00,
            "Wood_Walnut",
            col,
            tile_m=1.2,
            bevel_m=0.008,
            collider_kind="furniture",
        )
        self.add_box("Media_Console_Top", 0.10, 0.52, 0.65, 0.51, 0.545, 3.01, "Stone_Calacatta", col, tile_m=1.4, bevel_m=0.004)
        self.add_box("Media_Console_LED", 0.18, 0.182, 0.72, 0.42, 0.198, 2.94, "LED_WarmCove", col)
        for dr_i in range(4):
            dz0 = 0.70 + dr_i * 0.57
            self.add_box(f"Media_Drawer_{dr_i}", 0.50, 0.22, dz0, 0.516, 0.50, dz0 + 0.54, "Wood_Walnut", col, bevel_m=0.004)

        # 2. Woven High-Pile Area Rug
        self.add_box("Living_Area_Rug", 0.78, 0.001, 0.68, 3.42, 0.014, 3.22, "Rug_WovenCream", col, tile_m=1.1, bevel_m=0.004)

        # 3. Oatmeal Boucle 3-Seat Sofa (`src-sofa-01` at x=2.62, z=2.05 facing West toward TV)
        sx0, sx1 = 2.22, 3.16
        sz0, sz1 = 0.92, 3.12
        self.add_collider("furniture", "Living_Sofa", sx0, sz0, sx1, sz1, 0.0, 0.86)
        self.add_box("Sofa_Plinth", sx0 + 0.04, 0.04, sz0 + 0.04, sx1 - 0.04, 0.18, sz1 - 0.04, "Wood_Walnut", col, bevel_m=0.008)
        self.add_box("Sofa_Backrest", sx1 - 0.22, 0.18, sz0, sx1, 0.84, sz1, "Fabric_OatmealBoucle", col, tile_m=0.7, bevel_m=0.045)
        self.add_box("Sofa_ArmL", sx0, 0.18, sz0, sx1 - 0.18, 0.62, sz0 + 0.18, "Fabric_OatmealBoucle", col, tile_m=0.7, bevel_m=0.04)
        self.add_box("Sofa_ArmR", sx0, 0.18, sz1 - 0.18, sx1 - 0.18, 0.62, sz1, "Fabric_OatmealBoucle", col, tile_m=0.7, bevel_m=0.04)
        for c_i in range(3):
            cz_a = sz0 + 0.18 + c_i * 0.613
            self.add_box(f"Sofa_Seat_{c_i}", sx0 + 0.02, 0.18, cz_a + 0.01, sx1 - 0.20, 0.43, cz_a + 0.60, "Fabric_OatmealBoucle", col, tile_m=0.65, bevel_m=0.038)
            self.add_box(f"Sofa_BackCushion_{c_i}", sx1 - 0.34, 0.42, cz_a + 0.03, sx1 - 0.18, 0.80, cz_a + 0.58, "Fabric_OatmealBoucle", col, tile_m=0.65, bevel_m=0.04)
        # Olive & terracotta accent pillows + draped throw
        self.add_oriented_box("Sofa_Pillow_Olive", sx1 - 0.38, 0.56, sz0 + 0.36, 0.14, 0.34, 0.38, 12.0, "Fabric_OliveVelvet", col, bevel_m=0.035)
        self.add_oriented_box("Sofa_Pillow_Terra", sx1 - 0.38, 0.55, sz1 - 0.38, 0.14, 0.32, 0.36, -14.0, "Fabric_TerracottaThrow", col, bevel_m=0.035)
        self.add_box("Sofa_Throw", sx0 - 0.01, 0.26, sz1 - 0.72, sx0 + 0.46, 0.455, sz1 - 0.22, "Fabric_OliveVelvet", col, tile_m=0.5, bevel_m=0.02)

        # 4. Layered Travertine & Walnut Coffee Table (`src-coffee-01` at x=1.42, z=2.02)
        self.add_box("CoffeeTable_Base", 1.16, 0.015, 1.56, 1.66, 0.34, 2.46, "Wood_Walnut", col, bevel_m=0.01, collider_kind="furniture")
        self.add_box("CoffeeTable_Top", 1.08, 0.34, 1.44, 1.74, 0.40, 2.58, "Stone_Calacatta", col, tile_m=1.2, bevel_m=0.012, collider_kind="furniture")
        # Decor tray, art book & warm candles on coffee table
        self.add_box("Coffee_Book", 1.24, 0.40, 1.66, 1.56, 0.435, 1.92, "Fabric_CrispLinen", col, bevel_m=0.004)
        self.add_cylinder("Coffee_Vase", 1.44, 0.40, 2.22, 0.072, 0.20, "Ceramic_White", col)
        self.add_olive_bouquet("Coffee_Branches", 1.44, 0.60, 2.22, 0.85, col)

        # 5. Curved Boucle Accent Armchair (`src-armchair-01` at x=1.46, z=0.88)
        self.add_oriented_box("Accent_Armchair_Seat", 1.46, 0.24, 0.88, 0.78, 0.34, 0.76, -22.0, "Fabric_OatmealBoucle", col, bevel_m=0.06)
        self.add_oriented_box("Accent_Armchair_Back", 1.40, 0.56, 0.62, 0.80, 0.44, 0.22, -22.0, "Fabric_OatmealBoucle", col, bevel_m=0.06)
        self.add_collider("furniture", "Accent_Armchair", 1.04, 0.48, 1.88, 1.28, 0.0, 0.82)

        # 6. Dining Area (`src-dining-table-01` at x=4.58, z=2.12 + 4 woven chairs + linear LED chandelier)
        dx, dz = 4.58, 2.12
        self.add_box("Dining_Table_Top", dx - 0.44, 0.72, dz - 0.70, dx + 0.44, 0.76, dz + 0.70, "Glass_SmokedTable", col, bevel_m=0.008, collider_kind="furniture")
        self.add_box("Dining_Table_Apron", dx - 0.36, 0.67, dz - 0.60, dx + 0.36, 0.72, dz + 0.60, "Wood_Walnut", col, bevel_m=0.006)
        for lx_s in (-0.34, 0.34):
            for lz_s in (-0.58, 0.58):
                self.add_cylinder(f"Dining_Leg_{lx_s}_{lz_s}", dx + lx_s, 0.0, dz + lz_s, 0.024, 0.70, "Wood_Walnut", col, segments=12)

        # 4 tucked dining chairs + ceramic plates (matched to wa_full_06.jpg)
        for ch_i, (cx_off, cz_off, rot_c) in enumerate([(-0.52, -0.34, -90.0), (-0.52, 0.34, -90.0), (0.52, -0.34, 90.0), (0.52, 0.34, 90.0)]):
            self.add_oriented_box(f"DiningChair_Seat_{ch_i}", dx + cx_off, 0.44, dz + cz_off, 0.44, 0.045, 0.44, rot_c, "Fabric_OatmealBoucle", col, bevel_m=0.012)
            self.add_oriented_box(f"DiningChair_Back_{ch_i}", dx + cx_off * 1.32, 0.66, dz + cz_off, 0.46, 0.32, 0.045, rot_c, "Wood_Walnut", col, bevel_m=0.012)
            self.add_cylinder(f"DiningPlate_{ch_i}", dx + cx_off * 0.48, 0.761, dz + cz_off, 0.13, 0.016, "Ceramic_White", col, segments=20)
        self.add_collider("furniture", "Dining_Group", dx - 0.74, dz - 0.74, dx + 0.74, dz + 0.74, 0.0, 0.82)

        # Suspended Linear Warm-LED Chandelier above Dining Table (from wa_full_03.jpg)
        self.add_box("Dining_Chandelier_Canopy", dx - 0.18, CEIL_H - 0.03, dz - 0.05, dx + 0.18, CEIL_H, dz + 0.05, "Metal_WarmBrass", col)
        self.add_box("Dining_Chandelier_Bar1", dx - 0.015, 1.98, dz - 0.56, dx + 0.015, 2.01, dz + 0.56, "LED_WarmCove", col, bevel_m=0.005)
        self.add_box("Dining_Chandelier_Bar2", dx + 0.08, 2.06, dz - 0.46, dx + 0.11, 2.09, dz + 0.46, "LED_WarmCove", col, bevel_m=0.005)

        # 7. Full-Height Pleated Linen Drapery Flanking Window Wall
        self.add_pleated_curtain_stack("Curtain_Living_Left", 0.24, 0.68, 0.16, 0.01, 2.56, col, pleats=6)
        self.add_pleated_curtain_stack("Curtain_Living_Right", 5.14, 5.58, 0.16, 0.01, 2.56, col, pleats=6)

        # 8. Gallery Framed Artworks on South Living Wall (rectified from wa_full_04.jpg)
        for art_i, (ax0, ax1, mat_art) in enumerate([(1.15, 1.73, "Ref_LivingArt1"), (1.92, 2.50, "Ref_LivingArt2")]):
            self.add_box(f"Living_ArtFrame_{art_i}", ax0 - 0.025, 1.18, 3.555, ax1 + 0.025, 1.98, 3.59, "Wood_Walnut", col, bevel_m=0.004)
            self.add_uv_quad(
                f"Living_ArtCanvas_{art_i}",
                [
                    (ax1, 1.21, 3.552),
                    (ax0, 1.21, 3.552),
                    (ax0, 1.95, 3.552),
                    (ax1, 1.95, 3.552),
                ],
                mat_art,
                col,
            )

    def build_kitchen_and_foyer(self) -> None:
        col = "03_Kitchen_Foyer"

        # 1. Tall Walnut Appliance & Fridge Tower (x=[5.92, 6.62], z=[0.09, 0.72], matched to K1.jpg)
        self.add_box("Kitchen_TallTower", 5.92, 0.0, 0.09, 6.62, 2.58, 0.72, "Wood_Walnut", col, tile_m=1.2, bevel_m=0.006, collider_kind="fixture")
        # Built-in double dark-glass ovens
        self.add_box("Kitchen_OvenUpper", 5.98, 1.18, 0.715, 6.56, 1.62, 0.735, "Metal_DarkBronze", col, bevel_m=0.004)
        self.add_box("Kitchen_OvenLower", 5.98, 0.68, 0.715, 6.56, 1.14, 0.735, "Metal_DarkBronze", col, bevel_m=0.004)

        # 2. North Run Base Cabinets + Calacatta Countertop & Backsplash (x=[6.62, 9.33], z=[0.09, 0.69])
        self.add_box("Kitchen_BaseN", 6.62, 0.08, 0.09, 9.33, 0.88, 0.67, "Cabinet_SageOlive", col, bevel_m=0.005, collider_kind="fixture")
        self.add_box("Kitchen_ToeKickN", 6.64, 0.0, 0.09, 9.31, 0.08, 0.60, "Metal_DarkBronze", col)
        self.add_box("Kitchen_ToeLED_N", 6.66, 0.065, 0.605, 8.72, 0.078, 0.62, "LED_WarmCove", col)
        self.add_box("Kitchen_CounterN", 6.62, 0.88, 0.09, 9.33, 0.92, 0.70, "Stone_Calacatta", col, tile_m=1.4, bevel_m=0.005)
        self.add_box("Kitchen_BacksplashN", 6.62, 0.92, 0.09, 9.33, 1.56, 0.12, "Stone_Calacatta", col, tile_m=1.4)

        # Shaker door fronts & matte-black pulls along North run
        for cab_i in range(4):
            cx0 = 6.65 + cab_i * 0.52
            self.add_box(f"Kitchen_DoorN_{cab_i}", cx0, 0.11, 0.67, cx0 + 0.49, 0.85, 0.688, "Cabinet_SageOlive", col, bevel_m=0.004)
            self.add_box(f"Kitchen_PullN_{cab_i}", cx0 + 0.18, 0.80, 0.688, cx0 + 0.31, 0.818, 0.708, "Metal_DarkBronze", col)

        # 5-Burner Gas Cooktop + Sculpted Olive Range Hood + Floating Lit Shelves (K1.jpg)
        self.add_box("Kitchen_Cooktop", 7.36, 0.92, 0.22, 8.16, 0.94, 0.58, "Metal_DarkBronze", col, bevel_m=0.004)
        self.add_box("Kitchen_RangeHood", 7.30, 1.56, 0.09, 8.22, 2.58, 0.56, "Cabinet_SageOlive", col, bevel_m=0.008)
        self.add_box("Kitchen_HoodLight", 7.42, 1.545, 0.24, 8.10, 1.56, 0.44, "LED_WarmCove", col)

        for side_sh, (sx_a, sx_b) in enumerate([(6.66, 7.26), (8.26, 9.26)]):
            for sh_y in (1.56, 1.98):
                self.add_box(f"Kitchen_FloatShelf_{side_sh}_{sh_y}", sx_a, sh_y, 0.10, sx_b, sh_y + 0.032, 0.36, "Wood_Walnut", col, bevel_m=0.004)
                self.add_box(f"Kitchen_ShelfLED_{side_sh}_{sh_y}", sx_a + 0.04, sh_y - 0.012, 0.16, sx_b - 0.04, sh_y, 0.20, "LED_WarmCove", col)
                self.add_cylinder(f"Kitchen_Jar_{side_sh}_{sh_y}", 0.5 * (sx_a + sx_b), sh_y + 0.032, 0.22, 0.06, 0.15, "Ceramic_White", col)

        # 3. East Window Sink Counter Run (x=[8.74, 9.33], z=[0.67, 2.38]) with Real Recessed Sink Basin!
        self.add_box("Kitchen_BaseE", 8.76, 0.08, 0.67, 9.33, 0.88, 2.36, "Cabinet_SageOlive", col, bevel_m=0.005, collider_kind="fixture")
        # Split countertop around sink [z=1.38..1.94] so sink has real 0.20 m recessed 3D depth!
        self.add_box("Kitchen_CounterE_A", 8.73, 0.88, 0.70, 9.33, 0.92, 1.38, "Stone_Calacatta", col, tile_m=1.4, bevel_m=0.004)
        self.add_box("Kitchen_CounterE_B", 8.73, 0.88, 1.94, 9.33, 0.92, 2.38, "Stone_Calacatta", col, tile_m=1.4, bevel_m=0.004)
        self.add_box("Kitchen_CounterE_FrontLip", 8.73, 0.88, 1.38, 8.81, 0.92, 1.94, "Stone_Calacatta", col, bevel_m=0.004)
        self.add_box("Kitchen_CounterE_BackLip", 9.24, 0.88, 1.38, 9.33, 0.92, 1.94, "Stone_Calacatta", col, bevel_m=0.004)
        self.add_box("Kitchen_SinkBasin", 8.80, 0.70, 1.38, 9.24, 0.73, 1.94, "Metal_DarkBronze", col)
        # Matte-black gooseneck faucet
        self.add_cylinder("Kitchen_FaucetStem", 9.26, 0.92, 1.66, 0.015, 0.30, "Metal_DarkBronze", col, segments=12)
        self.add_box("Kitchen_FaucetNeck", 9.06, 1.19, 1.645, 9.27, 1.22, 1.675, "Metal_DarkBronze", col, bevel_m=0.006)

        # 4. Centerpiece Waterfall Calacatta Island (x=[6.58, 8.20], z=[1.58, 2.34]) + 3 Stools + 3 Globe Pendants
        ix0, ix1, iz0, iz1 = 6.58, 8.20, 1.58, 2.34
        self.add_collider("fixture", "Kitchen_Island", ix0, iz0, ix1, iz1 + 0.24, 0.0, 0.93)
        self.add_box("Island_Core", ix0 + 0.05, 0.0, iz0 + 0.03, ix1 - 0.05, 0.88, iz1 - 0.14, "Cabinet_SageOlive", col, bevel_m=0.004)
        # Fluted walnut apron facing south + warm LED cove
        for fl_i in range(22):
            fx = ix0 + 0.08 + fl_i * 0.068
            self.add_box(f"Island_Flute_{fl_i}", fx, 0.06, iz1 - 0.14, fx + 0.042, 0.86, iz1 - 0.115, "Wood_Walnut", col, bevel_m=0.003)
        self.add_box("Island_LED", ix0 + 0.08, 0.855, iz1 - 0.11, ix1 - 0.08, 0.872, iz1 - 0.08, "LED_WarmCove", col)
        # Mitered Calacatta marble top + left/right waterfall legs
        self.add_box("Island_WaterfallTop", ix0, 0.88, iz0, ix1, 0.925, iz1, "Stone_Calacatta", col, tile_m=1.3, bevel_m=0.005)
        self.add_box("Island_WaterfallL", ix0, 0.0, iz0, ix0 + 0.052, 0.88, iz1, "Stone_Calacatta", col, tile_m=1.3, bevel_m=0.004)
        self.add_box("Island_WaterfallR", ix1 - 0.052, 0.0, iz0, ix1, 0.88, iz1, "Stone_Calacatta", col, tile_m=1.3, bevel_m=0.004)

        # Island vase + organic olive branches + wooden fruit bowl (K1.jpg)
        self.add_cylinder("Island_Vase", 7.02, 0.925, 1.88, 0.082, 0.22, "Ceramic_White", col)
        self.add_olive_bouquet("Island_Branches", 7.02, 1.145, 1.88, 1.05, col)
        self.add_cylinder("Island_Bowl", 7.68, 0.925, 1.94, 0.13, 0.055, "Wood_Walnut", col, segments=22)
        for fr_i, (fx_o, fz_o) in enumerate([(-0.04, -0.02), (0.04, -0.01), (0.0, 0.04)]):
            self.add_sphere(f"Island_Pear_{fr_i}", 7.68 + fx_o, 0.995, 1.94 + fz_o, 0.036, "Foliage_OliveGreen", col)

        # 3 Boucle Counter Stools (matched to K1.jpg) & 3 Amber Glass Globe Pendants
        for st_i, sx_c in enumerate([6.88, 7.39, 7.90]):
            self.add_box(f"Stool_Seat_{st_i}", sx_c - 0.19, 0.62, iz1 - 0.02, sx_c + 0.19, 0.68, iz1 + 0.24, "Fabric_OatmealBoucle", col, bevel_m=0.025)
            self.add_box(f"Stool_Back_{st_i}", sx_c - 0.19, 0.68, iz1 + 0.19, sx_c + 0.19, 0.91, iz1 + 0.25, "Fabric_OatmealBoucle", col, bevel_m=0.025)
            for lx_o in (-0.16, 0.16):
                for lz_o in (0.01, 0.22):
                    self.add_cylinder(f"Stool_Leg_{st_i}_{lx_o}_{lz_o}", sx_c + lx_o, 0.0, iz1 + lz_o, 0.011, 0.62, "Metal_DarkBronze", col, segments=8)

            # Hanging globe pendant above island
            self.add_cylinder(f"Pendant_Rod_{st_i}", sx_c, 1.86, 1.94, 0.007, CEIL_H - 1.86, "Metal_DarkBronze", col, segments=8)
            self.add_sphere(f"Pendant_Globe_{st_i}", sx_c, 1.76, 1.94, 0.125, "Glass_AmberGlobe", col)

        # 5. Entry Foyer Console + Camera-Rectified Red/Black Canvas Art from wa_full_02.jpg
        self.add_box("Foyer_Console_Top", 8.36, 0.78, 3.24, 9.22, 0.81, 3.54, "Glass_SmokedTable", col, bevel_m=0.004, collider_kind="furniture")
        for fx_l in (8.38, 9.20):
            for fz_l in (3.26, 3.52):
                self.add_cylinder(f"Foyer_Leg_{fx_l}_{fz_l}", fx_l, 0.0, fz_l, 0.012, 0.78, "Metal_DarkBronze", col, segments=8)
        self.add_cylinder("Foyer_Vase", 8.62, 0.81, 3.39, 0.065, 0.18, "Ceramic_White", col)

        # Framed Foyer Painting on South Foyer Wall (z = 3.57)
        self.add_box("Foyer_Art_Frame", 8.44, 1.04, 3.54, 9.14, 1.92, 3.57, "Metal_DarkBronze", col, bevel_m=0.004)
        self.add_uv_quad(
            "Foyer_Art_Canvas",
            [
                (9.11, 1.07, 3.536),
                (8.47, 1.07, 3.536),
                (8.47, 1.89, 3.536),
                (9.11, 1.89, 3.536),
            ],
            "Ref_FoyerArt",
            col,
        )

    def build_primary_bedroom(self) -> None:
        col = "04_Primary_Bedroom"

        # 1. Plush Bedroom Area Rug
        self.add_box("Bedroom_Rug", 0.35, 0.001, 4.35, 2.85, 0.013, 6.45, "Rug_WovenCream", col, tile_m=1.1, bevel_m=0.004)

        # 2. Upholstered Queen Platform Bed (`src-bed-queen` on West wall, centered at z=5.38, matched to Be1.jpg)
        bz = 5.38
        bw, bd = 1.72, 2.12
        self.add_collider("furniture", "Primary_Queen_Bed", 0.09, bz - 0.5 * bw, 0.09 + bd, bz + 0.5 * bw, 0.0, 1.12)
        self.add_box("Bed_LED_Halo", 0.16, 0.04, bz - 0.82, 2.16, 0.07, bz + 0.82, "LED_WarmCove", col)
        self.add_box("Bed_Platform", 0.10, 0.07, bz - 0.88, 2.22, 0.28, bz + 0.88, "Fabric_OatmealBoucle", col, tile_m=0.8, bevel_m=0.035)
        self.add_box("Bed_Headboard", 0.09, 0.28, bz - 0.94, 0.24, 1.18, bz + 0.94, "Fabric_OatmealBoucle", col, tile_m=0.6, bevel_m=0.03)
        self.add_box("Bed_Mattress_Duvet", 0.24, 0.28, bz - 0.80, 2.14, 0.54, bz + 0.80, "Fabric_CrispLinen", col, tile_m=0.9, bevel_m=0.05)
        self.add_box("Bed_WaffleThrow", 1.52, 0.29, bz - 0.82, 2.16, 0.565, bz + 0.82, "Fabric_TerracottaThrow", col, tile_m=0.6, bevel_m=0.03)
        for p_z in (bz - 0.40, bz + 0.40):
            self.add_oriented_box(f"Bed_PillowBack_{p_z:.2f}", 0.42, 0.62, p_z, 0.22, 0.22, 0.64, 0.0, "Fabric_CrispLinen", col, bevel_m=0.045)
            self.add_oriented_box(f"Bed_PillowFront_{p_z:.2f}", 0.62, 0.59, p_z, 0.18, 0.16, 0.52, 0.0, "Fabric_TerracottaThrow", col, bevel_m=0.04)

        # Backlit Circular Mountain Medallion above Headboard (true circular 3D disc + halo from Be1.jpg)
        bed_halo = self.add_cylinder("Bed_Medallion_Halo", 0.102, 1.86 - 0.007, bz, 0.46, 0.014, "LED_WarmCove", col, segments=40)
        bed_halo.rotation_euler = Euler((0.0, math.radians(90.0), 0.0), "XYZ")
        self.add_uv_disc_west_wall("Bed_Medallion_Art", 0.122, 1.86, bz, 0.435, "Ref_BedMedallion", col, segments=40)

        # 3. Two Walnut Nightstands (`src-nightstand-01`) + Glowing Lunar Lamps (Be2.jpg / Be3.jpg)
        for ns_i, ns_z in enumerate([bz - 1.18, bz + 1.18]):
            self.add_box(f"Nightstand_{ns_i}", 0.10, 0.24, ns_z - 0.24, 0.52, 0.52, ns_z + 0.24, "Wood_Walnut", col, bevel_m=0.008, collider_kind="furniture")
            self.add_sphere(f"MoonLamp_{ns_i}", 0.31, 0.63, ns_z, 0.11, "Glass_AmberGlobe", col)

        # 4. Minimalist Walnut Work Desk (`src-desk-01` along North bedroom wall x=[1.48, 2.68], z=[3.75, 4.32])
        self.add_box("Bedroom_DeskTop", 1.48, 0.72, 3.75, 2.68, 0.76, 4.32, "Wood_Walnut", col, bevel_m=0.008, collider_kind="furniture")
        self.add_box("Bedroom_DeskPanelL", 1.50, 0.0, 3.76, 1.55, 0.72, 4.30, "Wood_Walnut", col, bevel_m=0.004)
        self.add_box("Bedroom_DeskPanelR", 2.61, 0.0, 3.76, 2.66, 0.72, 4.30, "Wood_Walnut", col, bevel_m=0.004)
        self.add_oriented_box("Bedroom_DeskChair", 2.08, 0.44, 4.46, 0.48, 0.06, 0.46, 180.0, "Fabric_OatmealBoucle", col, bevel_m=0.02)
        self.add_box("Bedroom_ArtBook", 1.88, 0.76, 3.92, 2.28, 0.785, 4.18, "Fabric_CrispLinen", col, bevel_m=0.003)

        # 5. Built-in Walnut & Smoked-Glass Wardrobe Closet (x=[4.14, 4.70], z=[4.64, 6.46])
        self.add_box("Bedroom_Wardrobe", 4.14, 0.0, 4.64, 4.70, 2.54, 6.46, "Wood_Walnut", col, tile_m=1.2, bevel_m=0.008, collider_kind="fixture")
        for wd_i in range(3):
            wz0 = 4.68 + wd_i * 0.58
            self.add_box(f"Wardrobe_GlassDoor_{wd_i}", 4.122, 0.08, wz0, 4.14, 2.46, wz0 + 0.55, "Glass_SmokedTable", col, bevel_m=0.003)
            self.add_box(f"Wardrobe_Pull_{wd_i}", 4.105, 0.95, wz0 + 0.04, 4.122, 1.45, wz0 + 0.06, "Metal_WarmBrass", col)

        # 6. Cozy Window Bench Cushion inside South Bedroom Window Reveal (Be3.jpg)
        self.add_box("Bedroom_WindowBench", 0.92, 0.0, 6.56, 3.83, 0.43, 6.90, "Wood_Walnut", col, bevel_m=0.008, collider_kind="furniture")
        self.add_box("Bedroom_BenchCushion", 0.96, 0.43, 6.58, 3.79, 0.50, 6.88, "Fabric_OatmealBoucle", col, bevel_m=0.025)
        self.add_pleated_curtain_stack("Curtain_Bed_L", 0.55, 0.92, 6.82, 0.01, 2.54, col, pleats=5)
        self.add_pleated_curtain_stack("Curtain_Bed_R", 3.83, 4.12, 6.82, 0.01, 2.54, col, pleats=5)

    def build_spa_bathroom(self) -> None:
        col = "05_Spa_Bathroom"

        # 1. Backlit Stacked-Stone Spa Accent Wall + Teak Plinth + Freestanding Soaking Tub (B1.jpg & B2.jpg)
        self.add_box("Bath_AccentStone", 4.88, 0.12, 6.84, 6.62, 2.52, 6.90, "Stone_Travertine", col, tile_m=1.2, bevel_m=0.006)
        self.add_box("Bath_AccentLED", 4.90, 2.50, 6.81, 6.60, 2.53, 6.84, "LED_WarmCove", col)
        self.add_box("Bath_TeakPlinth", 4.90, 0.0, 5.94, 6.68, 0.10, 6.88, "Wood_Walnut", col, bevel_m=0.006)
        self.add_box("Bath_PlinthLED", 4.94, 0.015, 5.925, 6.64, 0.032, 5.945, "LED_WarmCove", col)

        # Sculpted Freestanding Soaking Tub (`new-fix-04-1`) with recessed interior water basin + teak caddy
        tx0, tx1, tz0, tz1 = 5.02, 6.58, 6.04, 6.80
        self.add_collider("fixture", "Spa_Soaking_Tub", tx0, tz0, tx1, tz1, 0.0, 0.68)
        self.add_box("Tub_WallN", tx0, 0.10, tz0, tx1, 0.66, tz0 + 0.09, "Ceramic_White", col, bevel_m=0.03)
        self.add_box("Tub_WallS", tx0, 0.10, tz1 - 0.09, tx1, 0.66, tz1, "Ceramic_White", col, bevel_m=0.03)
        self.add_box("Tub_WallW", tx0, 0.10, tz0, tx0 + 0.11, 0.66, tz1, "Ceramic_White", col, bevel_m=0.03)
        self.add_box("Tub_WallE", tx1 - 0.11, 0.10, tz0, tx1, 0.66, tz1, "Ceramic_White", col, bevel_m=0.03)
        self.add_box("Tub_Bottom", tx0 + 0.05, 0.10, tz0 + 0.05, tx1 - 0.05, 0.18, tz1 - 0.05, "Ceramic_White", col)
        self.add_box("Tub_Water", tx0 + 0.10, 0.50, tz0 + 0.09, tx1 - 0.10, 0.54, tz1 - 0.09, "Water_Spa", col)
        self.add_box("Tub_TeakCaddy", 5.62, 0.66, tz0 - 0.02, 5.88, 0.685, tz1 + 0.02, "Wood_Walnut", col, bevel_m=0.004)
        self.add_cylinder("Tub_Candle", 5.75, 0.685, 0.5 * (tz0 + tz1), 0.035, 0.08, "LED_WarmCove", col, segments=14)
        self.add_cylinder("Tub_FloorFiller", 6.62, 0.10, 6.42, 0.018, 0.76, "Metal_DarkBronze", col, segments=12)

        # 2. Walk-In Frameless Glass Rain Shower (x=[4.86, 5.92], z=[3.74, 5.20], matched to B1.jpg/B2.jpg)
        self.add_box("Shower_Curb", 4.86, 0.0, 5.16, 5.94, 0.055, 5.22, "Stone_Calacatta", col, bevel_m=0.004)
        self.add_box("Shower_WallTileW", 4.85, 0.0, 3.74, 4.875, 2.54, 5.20, "Stone_Calacatta", col, tile_m=1.2)
        self.add_box("Shower_GlassSouth", 4.86, 0.055, 5.18, 5.92, 2.38, 5.20, "Glass_Architectural", col, collider_kind="fixture")
        self.add_box("Shower_FrameTop", 4.86, 2.36, 5.17, 5.93, 2.39, 5.21, "Metal_DarkBronze", col)
        self.add_box("Shower_FramePost", 5.90, 0.0, 5.17, 5.93, 2.39, 5.21, "Metal_DarkBronze", col)
        # Overhead rainfall showerhead + recessed lit niche
        self.add_box("Shower_RainArm", 4.86, 2.26, 4.44, 5.38, 2.285, 4.47, "Metal_DarkBronze", col)
        self.add_box("Shower_RainHead", 5.24, 2.23, 4.34, 5.50, 2.255, 4.60, "Metal_DarkBronze", col, bevel_m=0.004)
        self.add_box("Shower_NicheShelf", 4.86, 1.18, 4.18, 4.96, 1.205, 4.76, "Stone_Calacatta", col)
        self.add_box("Shower_NicheLED", 4.87, 1.52, 4.20, 4.91, 1.535, 4.74, "LED_WarmCove", col)

        # 3. Floating Walnut & Calacatta Vanity (`new-fix-04-2`: x=[7.48, 8.09], z=[4.62, 6.00]) + Halo Mirror (B2.jpg)
        self.add_box("Bath_VanityCabinet", 7.50, 0.24, 4.62, 8.09, 0.82, 6.00, "Wood_Walnut", col, bevel_m=0.008, collider_kind="fixture")
        self.add_box("Bath_VanityTop", 7.47, 0.82, 4.60, 8.09, 0.865, 6.02, "Stone_Calacatta", col, tile_m=1.2, bevel_m=0.005)
        self.add_box("Bath_VanityLED", 7.56, 0.22, 4.68, 8.02, 0.238, 5.94, "LED_WarmCove", col)
        self.add_cylinder("Bath_VesselSink", 7.78, 0.865, 5.31, 0.21, 0.13, "Ceramic_White", col, segments=28)
        self.add_box("Bath_Spout", 7.92, 1.06, 5.29, 8.09, 1.09, 5.33, "Metal_DarkBronze", col, bevel_m=0.004)
        self.add_box("Bath_Mat", 6.68, 0.002, 4.82, 7.36, 0.018, 5.80, "Rug_WovenCream", col, bevel_m=0.004)

        # Circular Backlit Vanity Mirror on East Wall (x = 8.08, centered z = 5.31, y = 1.56)
        mirror_halo = self.add_cylinder("Bath_MirrorHalo", 8.075, 1.56 - 0.008, 5.31, 0.49, 0.016, "LED_WarmCove", col, segments=36)
        mirror_halo.rotation_euler = Euler((0.0, math.radians(90.0), 0.0), "XYZ")
        mirror_disc = self.add_cylinder("Bath_MirrorGlass", 8.062, 1.56 - 0.008, 5.31, 0.46, 0.016, "Mirror_Silver", col, segments=36)
        mirror_disc.rotation_euler = Euler((0.0, math.radians(90.0), 0.0), "XYZ")

        # 4. Modern Wall-Hung Ceramic Toilet (x=[7.54, 8.09], z=[6.28, 6.70])
        self.add_box("Bath_ToiletBowl", 7.54, 0.16, 6.30, 8.09, 0.43, 6.68, "Ceramic_White", col, bevel_m=0.06, collider_kind="fixture")
        self.add_box("Bath_FlushPlate", 8.07, 0.92, 6.40, 8.09, 1.08, 6.58, "Metal_DarkBronze", col)

    def build_balcony_and_exterior_vista(self) -> None:
        col = "06_Balcony_Exterior"

        # 1. Riverfront Balcony Deck (x=[0.36, 5.46], z=[-1.65, -0.09], matched to plan1.webp & wa_full_08.jpg)
        bx0, bx1, bz0, bz1 = 0.36, 5.46, -1.65, -0.09
        self.add_box("Balcony_Slab", bx0 - 0.06, -0.18, bz0 - 0.06, bx1 + 0.06, -0.015, bz1 + 0.05, "Wood_BalconyTeak", col, tile_m=1.4)

        # Steel & Glass Guardrail around North, West, and East Balcony Edges (keeps player safely on balcony!)
        rail_h = 1.08
        self.add_box("Balcony_RailTopN", bx0, rail_h - 0.04, bz0, bx1, rail_h, bz0 + 0.05, "Metal_DarkBronze", col, collider_kind="railing")
        self.add_box("Balcony_RailTopW", bx0, rail_h - 0.04, bz0, bx0 + 0.05, rail_h, bz1, "Metal_DarkBronze", col, collider_kind="railing")
        self.add_box("Balcony_RailTopE", bx1 - 0.05, rail_h - 0.04, bz0, bx1, rail_h, bz1, "Metal_DarkBronze", col, collider_kind="railing")
        self.add_box("Balcony_GlassN", bx0 + 0.04, 0.06, bz0 + 0.015, bx1 - 0.04, rail_h - 0.04, bz0 + 0.03, "Glass_Architectural", col)
        for i_post in range(11):
            p_x = bx0 + (bx1 - bx0) * (i_post / 10.0)
            self.add_box(f"Balcony_Post_{p_x:.2f}", p_x - 0.018, -0.015, bz0, p_x + 0.018, rail_h, bz0 + 0.045, "Metal_DarkBronze", col)

        # Balcony Bistro Set & Planters (leaving center x=[1.95..3.95] open to step outside!)
        self.add_oriented_box("Balcony_ChairW", 1.15, 0.36, -0.95, 0.68, 0.68, 0.66, 28.0, "Fabric_OatmealBoucle", col, bevel_m=0.05)
        self.add_collider("furniture", "Balcony_ChairW", 0.78, -1.32, 1.52, -0.58, 0.0, 0.78)
        self.add_cylinder("Balcony_Table", 4.55, -0.015, -0.98, 0.32, 0.56, "Metal_DarkBronze", col, segments=20)
        self.add_collider("furniture", "Balcony_Table", 4.22, -1.30, 4.88, -0.66, 0.0, 0.58)
        for pl_x in (0.66, 5.12):
            self.add_cylinder(f"Balcony_Pot_{pl_x:.2f}", pl_x, -0.015, -1.38, 0.18, 0.36, "Fabric_TerracottaThrow", col, segments=16)
            self.add_olive_bouquet(f"Balcony_Plant_{pl_x:.2f}", pl_x, 0.34, -1.38, 1.45, col)

        # 2. Wide Wrap-Around Panoramic Waterfront & Lush Hills Horizon Backdrops (zero sky seam at any window angle)
        self.add_uv_quad(
            "Vista_North_River",
            [
                (-8.5, -2.6, -7.2),
                (17.5, -2.6, -7.2),
                (17.5, 7.6, -7.2),
                (-8.5, 7.6, -7.2),
            ],
            "Ref_ExteriorVista",
            col,
        )
        self.add_uv_quad(
            "Vista_West_Wing",
            [
                (-8.5, -2.6, 13.5),
                (-8.5, -2.6, -7.2),
                (-8.5, 7.6, -7.2),
                (-8.5, 7.6, 13.5),
            ],
            "Ref_ExteriorVista",
            col,
        )
        self.add_uv_quad(
            "Vista_South_Hills",
            [
                (15.0, -2.6, 12.8),
                (-8.5, -2.6, 12.8),
                (-8.5, 7.4, 12.8),
                (15.0, 7.4, 12.8),
            ],
            "Ref_ExteriorVista",
            col,
        )
        self.add_uv_quad(
            "Vista_East_Greenery",
            [
                (15.0, -2.4, -7.2),
                (15.0, -2.4, 12.8),
                (15.0, 7.2, 12.8),
                (15.0, 7.2, -7.2),
            ],
            "Ref_ExteriorVista",
            col,
        )

    def setup_lighting_and_cameras(self) -> list[tuple[str, bpy.types.Object]]:
        col = "07_Lighting_Cameras"
        scene = bpy.context.scene

        # Sky / Environment lighting for Blender verification renders
        world = bpy.data.worlds.new("ApartmentWorld")
        scene.world = world
        world.use_nodes = True
        wnodes = world.node_tree.nodes
        bg = wnodes.get("Background")
        if bg is not None:
            bg.inputs[0].default_value = (0.78, 0.86, 0.96, 1.0)
            bg.inputs[1].default_value = 1.15

        # Directional Golden Daylight Sun streaming from North-Northwest across the balcony into Living Room
        sun_data = bpy.data.lights.new(name="Sun_Daylight", type="SUN")
        sun_data.energy = 3.6
        sun_data.color = (1.0, 0.95, 0.87)
        sun_data.angle = math.radians(4.5)
        sun_obj = bpy.data.objects.new("Sun_Daylight", sun_data)
        sun_obj.location = hw_pt(2.8, 7.5, -5.0)
        sun_obj.rotation_euler = Euler((math.radians(52.0), math.radians(-12.0), math.radians(162.0)), "XYZ")
        self._link_obj(sun_obj, col)

        # Warm Interior Area Fill Lights matching the ceiling coves & pendants in Blender renders
        room_lights = [
            ("Light_Living", 2.60, 2.48, 1.95, 140.0, (1.0, 0.91, 0.78), 2.6),
            ("Light_Kitchen", 7.45, 2.46, 1.90, 120.0, (1.0, 0.90, 0.76), 2.2),
            ("Light_Bedroom", 2.25, 2.48, 5.35, 110.0, (1.0, 0.88, 0.74), 2.4),
            ("Light_Bathroom", 6.50, 2.48, 5.30, 95.0, (0.98, 0.95, 0.90), 2.0),
        ]
        for lname, lx, ly, lz, energy, color, sz in room_lights:
            ldata = bpy.data.lights.new(name=lname, type="AREA")
            ldata.energy = energy
            ldata.color = color
            ldata.size = sz
            lobj = bpy.data.objects.new(lname, ldata)
            lobj.location = hw_pt(lx, ly, lz)
            self._link_obj(lobj, col)

        # Eye-level (1.65 m) verification & reference-matching cameras
        cam_specs = [
            ("Cam_01_Living_Overview", (5.15, 1.65, 2.85), (1.35, 1.42, 1.55), 24.0),
            ("Cam_02_Sideways_Window_Dining", (1.20, 1.65, 0.82), (5.40, 1.45, 1.25), 26.0),
            ("Cam_03_Kitchen_Island_Counters", (5.95, 1.65, 2.95), (7.85, 1.35, 1.25), 24.0),
            ("Cam_04_Primary_Bedroom", (3.85, 1.65, 4.15), (0.95, 1.25, 5.65), 24.0),
            ("Cam_05_Spa_Bathroom", (6.38, 1.65, 4.02), (6.85, 1.22, 6.15), 22.0),
        ]
        cams: list[tuple[str, bpy.types.Object]] = []
        for cname, eye_hw, tgt_hw, lens_mm in cam_specs:
            cdata = bpy.data.cameras.new(cname)
            cdata.lens = lens_mm
            cobj = bpy.data.objects.new(cname, cdata)
            eye_v = Vector(hw_pt(*eye_hw))
            tgt_v = Vector(hw_pt(*tgt_hw))
            cobj.location = eye_v
            direction = tgt_v - eye_v
            cobj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
            self._link_obj(cobj, col)
            cams.append((cname, cobj))
        scene.camera = cams[0][1]
        return cams

    def export_metadata(self) -> None:
        META_OUT.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "apartment_name": "Homeweave Composite Dream Residence",
            "eye_height_m": 1.65,
            "player_radius_m": 0.26,
            "ceiling_height_m": CEIL_H,
            "spawn": {
                "x": 5.15,
                "y": 1.65,
                "z": 2.85,
                "yaw_deg": 68.0,
                "pitch_deg": -3.0,
            },
            "rooms": [
                {
                    "id": "new-living",
                    "name": "Living & Dining Room",
                    "key": "1",
                    "bounds": [0.0, 0.0, 5.82, 3.66],
                    "preset": {"x": 4.95, "z": 2.85, "yaw_deg": 66.0, "pitch_deg": -3.0},
                    "source_ref": "Manhattan Park Plan P (19'1\" x 12'0\") + L1/L2 & walkthrough video",
                },
                {
                    "id": "new-kitchen",
                    "name": "Open Island Kitchen & Foyer",
                    "key": "2",
                    "bounds": [5.82, 0.0, 9.42, 3.66],
                    "preset": {"x": 8.65, "z": 2.92, "yaw_deg": 48.0, "pitch_deg": -4.0},
                    "source_ref": "The Eastline Residence 1713 + K1/K3 & Foyer video",
                },
                {
                    "id": "new-primary-bedroom",
                    "name": "Primary Bedroom Suite",
                    "key": "3",
                    "bounds": [0.0, 3.66, 4.78, 6.99],
                    "preset": {"x": 3.75, "z": 4.22, "yaw_deg": 122.0, "pitch_deg": -5.0},
                    "source_ref": "Manhattan Park Plan P (15'8\" x 10'11\") + Be1..Be3",
                },
                {
                    "id": "new-bathroom",
                    "name": "Spa Soaking-Tub Bathroom",
                    "key": "4",
                    "bounds": [4.78, 3.66, 8.18, 6.99],
                    "preset": {"x": 6.56, "z": 4.08, "yaw_deg": 168.0, "pitch_deg": -6.0},
                    "source_ref": "The Eastline Residence 1713 + B1/B2/B4",
                },
                {
                    "id": "new-balcony",
                    "name": "Riverfront Balcony",
                    "key": "5",
                    "bounds": [0.36, -1.65, 5.46, 0.0],
                    "preset": {"x": 3.35, "z": -0.78, "yaw_deg": 195.0, "pitch_deg": -2.0},
                    "source_ref": "Manhattan Park Balcony + Waterfront & Mock-Up Flat Balcony videos",
                },
            ],
            "assumptions": [
                "Ceiling height standardized to 2.72 m (8'11\") across all rooms with 0.14 m perimeter soffit coves.",
                "Exterior walls modeled at 0.18 m thickness (0.22 m window reveal with sill overhang); interior partitions at 0.14 m.",
                "Shared room walls deduplicated with cased openings: 2.56 m Living-Kitchen portal, 0.92 m Bedroom doorway, 0.88 m Bathroom doorway, and 0.93 m open Balcony sliding door.",
                "Interior doors assumed 0.86-0.92 m x 2.22 m and swung 86 deg open so every walkthrough passage exceeds 0.82 m clear width.",
            ],
            "colliders": self.colliders,
        }
        META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def render_verification_views(self, cams: list[tuple[str, bpy.types.Object]]) -> None:
        CHECK_RENDER_DIR.mkdir(parents=True, exist_ok=True)
        scene = bpy.context.scene
        # Use Cycles with Metal/CPU adaptive sampling at 960x540 for crisp shadow/reflection verification
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 48
        scene.cycles.use_denoising = True
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "METAL"
            prefs.get_devices()
            for dev in prefs.devices:
                dev.use = True
            scene.cycles.device = "GPU"
        except Exception:
            scene.cycles.device = "CPU"

        scene.render.resolution_x = 960
        scene.render.resolution_y = 540
        scene.render.image_settings.file_format = "JPEG"
        scene.render.image_settings.quality = 88
        scene.view_settings.view_transform = "AgX" if "AgX" in ("AgX",) else "Standard"
        try:
            scene.view_settings.view_transform = "Filmic"
        except Exception:
            scene.view_settings.view_transform = "Standard"

        for cname, cobj in cams:
            scene.camera = cobj
            scene.render.filepath = str(CHECK_RENDER_DIR / f"{cname}.jpg")
            bpy.ops.render.render(write_still=True)

    def save_blend_and_export_glb(self) -> None:
        BLEND_OUT.parent.mkdir(parents=True, exist_ok=True)
        GLB_OUT.parent.mkdir(parents=True, exist_ok=True)

        # Pack external textures into .blend so the Blender source is 100% self-contained & editable
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_OUT), compress=True)

        # Export mesh objects only (skip Blender helper cameras/area lights so Three.js uses calibrated PBR lights)
        for obj in bpy.data.objects:
            obj.select_set(obj.type == "MESH")

        rna_props = {
            p.identifier for p in bpy.ops.export_scene.gltf.get_rna_type().properties
        }
        export_kwargs: dict[str, Any] = {
            "filepath": str(GLB_OUT),
            "export_format": "GLB",
            "use_selection": True,
            "export_yup": True,
            "export_apply": True,
        }
        if "export_image_format" in rna_props:
            export_kwargs["export_image_format"] = "WEBP"
        if "export_image_quality" in rna_props:
            export_kwargs["export_image_quality"] = 84

        bpy.ops.export_scene.gltf(**export_kwargs)


def main() -> None:
    builder = SceneBuilder()
    builder.build_shell_and_floors()
    builder.build_living_dining_room()
    builder.build_kitchen_and_foyer()
    builder.build_primary_bedroom()
    builder.build_spa_bathroom()
    builder.build_balcony_and_exterior_vista()
    cams = builder.setup_lighting_and_cameras()
    builder.export_metadata()
    builder.render_verification_views(cams)
    builder.save_blend_and_export_glb()
    print(
        f"APARTMENT_BUILD_COMPLETE | blend={BLEND_OUT.stat().st_size // 1024}KB | "
        f"glb={GLB_OUT.stat().st_size // 1024}KB | colliders={len(builder.colliders)}"
    )


if __name__ == "__main__":
    main()
