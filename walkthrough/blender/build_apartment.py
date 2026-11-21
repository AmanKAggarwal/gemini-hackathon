"""Build the 6-Zone Homeweave Dream Residence in Blender 5.2 with Hierarchical Movable Nodes.

Usage:
  /opt/homebrew/bin/blender --background --factory-startup --python walkthrough/blender/build_apartment.py

Outputs:
  - walkthrough/blender/dream_apartment.blend   (Organized, editable 6-zone Blender scene)
  - walkthrough/assets/dream_apartment.glb      (Hierarchical WebP-compressed PBR GLB for Three.js)
  - walkthrough/assets/apartment_meta.json      (Rooms, presets, assumptions, and static/dynamic colliders)
  - walkthrough/assets/samples/O1.jpg, O2.jpg   (Cycles reference renders of the Executive Office)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import bmesh
import bpy
from mathutils import Euler, Vector

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

TEX_DIR = ROOT_DIR / "walkthrough" / "assets" / "textures"
SAMPLES_IMG_DIR = ROOT_DIR / "walkthrough" / "assets" / "samples"
BLEND_OUT = ROOT_DIR / "walkthrough" / "blender" / "dream_apartment.blend"
GLB_OUT = ROOT_DIR / "walkthrough" / "assets" / "dream_apartment.glb"
META_OUT = ROOT_DIR / "walkthrough" / "assets" / "apartment_meta.json"
CHECK_RENDER_DIR = ROOT_DIR / "private_refs" / "renders_check"

CEIL_H = 2.72
EXT_WALL_T = 0.18
INT_WALL_T = 0.14
DOOR_H = 2.22


# Homeweave (x East, y Up, z South) -> Blender (X, -Z, Y)
# Exported with `export_yup=True`, Blender (x, -z, y) maps 1:1 to Three.js (x, y, z).
def hw_pt(x: float, y: float, z: float) -> tuple[float, float, float]:
    return (float(x), float(-z), float(y))


class SceneBuilder:
    def __init__(self) -> None:
        self.materials: dict[str, bpy.types.Material] = {}
        self.collections: dict[str, bpy.types.Collection] = {}
        self.empties: dict[str, bpy.types.Object] = {}
        self.colliders: list[dict[str, Any]] = []
        self._active_parent: Optional[bpy.types.Object] = None
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
            "06_Office_Study",
            "07_Balcony_Exterior",
            "08_Lighting_Cameras",
        ]:
            col = bpy.data.collections.new(cname)
            scene.collection.children.link(col)
            self.collections[cname] = col

    def get_group(self, name: str, col_name: str = "01_Architecture") -> bpy.types.Object:
        if name in self.empties:
            return self.empties[name]
        emp = bpy.data.objects.new(name, None)
        emp.empty_display_type = "PLAIN_AXES"
        emp.empty_display_size = 0.25
        self.collections[col_name].objects.link(emp)
        self.empties[name] = emp
        return emp

    def set_parent_group(self, emp: Optional[bpy.types.Object]) -> None:
        self._active_parent = emp

    def place_group_pose(self, emp: bpy.types.Object, x: float, y: float, z: float, rot_deg_cw: float) -> None:
        emp.location = hw_pt(x, y, z)
        emp.rotation_euler = Euler((0.0, 0.0, -math.radians(rot_deg_cw)), "XYZ")

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
                (max(0.0, emission_rgb[0]), max(0.0, emission_rgb[1]), max(0.0, emission_rgb[2]), 1.0),
            )
            self._set_input(bsdf, ["Emission Strength"], emission_strength)

        if color_tex:
            ct = self._load_tex_node(nodes, color_tex, non_color=False)
            if ct is not None:
                ct.location = (-360, 180)
                links.new(ct.outputs["Color"], bsdf.inputs["Base Color"])
                if emission_rgb is not None and emission_strength > 0.0 and "Emission Color" in bsdf.inputs:
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
        self.make_pbr_mat("Wall_WarmPlaster", base_rgba=(0.935, 0.918, 0.885, 1.0), roughness=0.82, normal_tex="plaster_normal.png", normal_strength=0.22)
        self.make_pbr_mat("Wall_CeilingMatte", base_rgba=(0.965, 0.958, 0.942, 1.0), roughness=0.88)
        self.make_pbr_mat("Trim_OffWhite", base_rgba=(0.90, 0.88, 0.84, 1.0), roughness=0.46)
        self.make_pbr_mat("Floor_HoneyOak", color_tex="oak_floor_color.jpg", normal_tex="oak_floor_normal.png", normal_strength=0.45, rough_tex="oak_floor_rough.jpg", roughness=0.42)
        self.make_pbr_mat("Wood_Walnut", color_tex="walnut_color.jpg", normal_tex="oak_floor_normal.png", normal_strength=0.18, roughness=0.38)
        self.make_pbr_mat("Wood_BalconyTeak", color_tex="oak_floor_color.jpg", normal_tex="oak_floor_normal.png", normal_strength=0.55, roughness=0.62)
        self.make_pbr_mat("Stone_Calacatta", color_tex="calacatta_color.jpg", rough_tex="calacatta_rough.jpg", roughness=0.20)
        self.make_pbr_mat("Stone_Travertine", color_tex="travertine_color.jpg", normal_tex="travertine_normal.png", normal_strength=0.65, roughness=0.48)
        self.make_pbr_mat("Tile_SpaPorcelain", color_tex="spa_tile_color.jpg", normal_tex="spa_tile_normal.png", normal_strength=0.40, roughness=0.26)
        self.make_pbr_mat("Cabinet_SageOlive", base_rgba=(0.24, 0.27, 0.21, 1.0), roughness=0.42, normal_tex="plaster_normal.png", normal_strength=0.08)
        self.make_pbr_mat("Metal_DarkBronze", base_rgba=(0.11, 0.11, 0.12, 1.0), roughness=0.28, metallic=0.86)
        self.make_pbr_mat("Metal_WarmBrass", base_rgba=(0.76, 0.58, 0.34, 1.0), roughness=0.24, metallic=0.92)
        self.make_pbr_mat("Glass_Architectural", base_rgba=(0.82, 0.91, 0.95, 1.0), roughness=0.05, metallic=0.08, alpha=0.24, transmission=0.75)
        self.make_pbr_mat("Glass_SmokedTable", base_rgba=(0.12, 0.13, 0.14, 1.0), roughness=0.08, metallic=0.15, alpha=0.68)
        self.make_pbr_mat("Glass_AmberGlobe", base_rgba=(0.95, 0.72, 0.42, 1.0), roughness=0.12, alpha=0.45, emission_rgb=(1.0, 0.76, 0.42), emission_strength=1.4)
        self.make_pbr_mat("Mirror_Silver", base_rgba=(0.92, 0.94, 0.95, 1.0), roughness=0.03, metallic=0.98)
        self.make_pbr_mat("Fabric_OatmealBoucle", color_tex="boucle_color.jpg", normal_tex="boucle_normal.png", normal_strength=0.55, roughness=0.86)
        self.make_pbr_mat("Fabric_OliveVelvet", base_rgba=(0.25, 0.30, 0.20, 1.0), normal_tex="boucle_normal.png", normal_strength=0.35, roughness=0.80)
        self.make_pbr_mat("Fabric_TerracottaThrow", base_rgba=(0.56, 0.34, 0.22, 1.0), normal_tex="rug_normal.png", normal_strength=0.45, roughness=0.88)
        self.make_pbr_mat("Fabric_CrispLinen", base_rgba=(0.95, 0.94, 0.91, 1.0), normal_tex="boucle_normal.png", normal_strength=0.22, roughness=0.84)
        self.make_pbr_mat("Fabric_SheerDrapery", base_rgba=(0.90, 0.87, 0.81, 1.0), roughness=0.85, alpha=0.82)
        self.make_pbr_mat("Leather_CognacPad", base_rgba=(0.34, 0.20, 0.12, 1.0), roughness=0.42)
        self.make_pbr_mat("Rug_WovenCream", color_tex="rug_color.jpg", normal_tex="rug_normal.png", normal_strength=0.65, roughness=0.92)
        self.make_pbr_mat("Ceramic_White", base_rgba=(0.96, 0.96, 0.95, 1.0), roughness=0.14)
        self.make_pbr_mat("Water_Spa", base_rgba=(0.56, 0.82, 0.86, 1.0), roughness=0.06, alpha=0.72)
        self.make_pbr_mat("Foliage_OliveGreen", base_rgba=(0.20, 0.32, 0.17, 1.0), roughness=0.68)
        self.make_pbr_mat("LED_WarmCove", base_rgba=(1.0, 0.87, 0.66, 1.0), roughness=0.25, emission_rgb=(1.0, 0.84, 0.56), emission_strength=4.2)
        self.make_pbr_mat("Screen_StudioCode", color_tex="plan3_clean.png" if (TEX_DIR / "plan3_clean.png").exists() else "calacatta_color.jpg", base_rgba=(0.14, 0.18, 0.24, 1.0), roughness=0.10, emission_rgb=(0.22, 0.30, 0.42), emission_strength=0.85)

        self.make_pbr_mat("Ref_FoyerArt", color_tex="ref_foyer_art.jpg", roughness=0.35)
        self.make_pbr_mat("Ref_LivingArt1", color_tex="ref_living_art_1.jpg", roughness=0.35)
        self.make_pbr_mat("Ref_LivingArt2", color_tex="ref_living_art_2.jpg", roughness=0.35)
        self.make_pbr_mat("Ref_TVScreen", color_tex="ref_tv_screen.jpg", roughness=0.12, emission_rgb=(-1.0, -1.0, -1.0), emission_strength=1.15)
        self.make_pbr_mat("Ref_BedMedallion", color_tex="ref_bedroom_medallion.jpg", roughness=0.45, emission_rgb=(-1.0, -1.0, -1.0), emission_strength=0.35)
        self.make_pbr_mat("Ref_ExteriorVista", color_tex="ref_exterior_vista.jpg", roughness=0.95, emission_rgb=(-1.0, -1.0, -1.0), emission_strength=1.35)

    # -----------------------------------------------------------------------
    # Metric Geometry Primitives (Parent-Aware)
    # -----------------------------------------------------------------------

    def _link_obj(self, obj: bpy.types.Object, col_name: str) -> bpy.types.Object:
        self.collections[col_name].objects.link(obj)
        if self._active_parent is not None:
            obj.parent = self._active_parent
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
        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new()
        verts: list[bmesh.types.BMVert] = []
        uv_coords: list[tuple[float, float]] = []
        for i in range(segments):
            ang = 2.0 * math.pi * (i / segments)
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
    # Architectural Wall, Reveal, Casing & Coffered Ceiling Builders
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
        ops = sorted(openings, key=lambda o: o["s0"])
        cursor = span_start
        half_t = 0.5 * thick

        def wall_seg(tag: str, a0: float, a1: float, y0: float, y1: float, register_col: bool) -> None:
            if a1 - a0 <= 0.005 or y1 - y0 <= 0.005:
                return
            ck = "wall" if (register_col and y0 < 1.85) else None
            if axis == "x":
                self.add_box(f"{name}_{tag}", a0, y0, fixed_pos - half_t, a1, y1, fixed_pos + half_t, "Wall_WarmPlaster", "01_Architecture", tile_m=2.0, collider_kind=ck)
            else:
                self.add_box(f"{name}_{tag}", fixed_pos - half_t, y0, a0, fixed_pos + half_t, y1, a1, "Wall_WarmPlaster", "01_Architecture", tile_m=2.0, collider_kind=ck)

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
                    self.add_box(f"{name}_bb_{idx}_{a0:.2f}", a0, 0.0, z_face, a1, bb_h, z_face + sign * bb_p, "Trim_OffWhite", "01_Architecture", tile_m=1.2, bevel_m=0.003)
                else:
                    x_face = fixed_pos + sign * half_t
                    self.add_box(f"{name}_bb_{idx}_{a0:.2f}", x_face, 0.0, a0, x_face + sign * bb_p, bb_h, a1, "Trim_OffWhite", "01_Architecture", tile_m=1.2, bevel_m=0.003)

        for idx, op in enumerate(ops):
            s0, s1 = op["s0"], op["s1"]
            y0, y1 = op["y0"], op["y1"]
            kind = op["kind"]
            if s0 > cursor:
                wall_seg(f"solid_{idx}", cursor, s0, 0.0, CEIL_H, True)
                add_baseboard_span(cursor, s0 - (0.075 if kind in ("door", "portal") else 0.0))

            if y0 > 0.01:
                wall_seg(f"sillwall_{idx}", s0, s1, 0.0, y0, True)
                add_baseboard_span(s0, s1)
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

        leaf_mode = op.get("leaf")
        if leaf_mode == "closed_ext" and axis == "z":
            # Pivoted Empty at hinge `(fixed_pos, 0, s0 + jamb_t)` so Entry Door swings open for Tiny Mover!
            prev_parent = self._active_parent
            door_emp = self.get_group("Door__entry_leaf", "01_Architecture")
            leaf_len = (s1 - s0) - 2 * jamb_t
            self.set_parent_group(door_emp)
            self.add_box("Entry_Door_Panel", -0.022, 0.01, 0.0, 0.022, y1 - jamb_t, leaf_len, "Wood_Walnut", "01_Architecture", bevel_m=0.004)
            self.add_box("Entry_Door_Handle", -0.075, 1.02, leaf_len - 0.18, -0.022, 1.05, leaf_len - 0.05, "Metal_WarmBrass", "01_Architecture", bevel_m=0.004)
            self.place_group_pose(door_emp, fixed_pos, 0.0, s0 + jamb_t, 0.0)
            self.set_parent_group(prev_parent)
            self.add_collider("door", "Entry_Door_Closed", fixed_pos - 0.025, s0 + jamb_t, fixed_pos + 0.025, s1 - jamb_t, 0.01, y1 - jamb_t)
        elif leaf_mode in ("ajar_south_right", "ajar_south_left") and axis == "x":
            leaf_len = (s1 - s0) - 2 * jamb_t
            if leaf_mode == "ajar_south_right":
                hx = s1 - jamb_t - 0.02
                x_a, x_b = hx - 0.042, hx
            else:
                hx = s0 + jamb_t + 0.02
                x_a, x_b = hx, hx + 0.042
            hz = fixed_pos + half_t + 0.02
            self.add_box(f"{wall_name}_ajarleaf_{idx}", x_a, 0.01, hz, x_b, y1 - jamb_t - 0.01, hz + leaf_len, "Wood_Walnut", "01_Architecture", bevel_m=0.004, collider_kind="door")
            self.add_box(f"{wall_name}_ajarhandle_{idx}", x_a - 0.03, 1.02, hz + leaf_len - 0.14, x_b + 0.03, 1.045, hz + leaf_len - 0.08, "Metal_WarmBrass", "01_Architecture", bevel_m=0.003)

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
            z_in = fixed_pos + inward_sgn * (half_t + sill_overhang)
            z_out = fixed_pos - inward_sgn * half_t
            self.add_box(f"{wall_name}_sill_{idx}", s0 - 0.03, y0 - 0.032, min(z_in, z_out), s1 + 0.03, y0 + 0.005, max(z_in, z_out), "Stone_Calacatta", "01_Architecture", bevel_m=0.004)
            zf = fixed_pos - inward_sgn * (half_t - 0.04)
            self.add_box(f"{wall_name}_wfB_{idx}", s0, y0, zf - 0.5 * frame_d, s1, y0 + frame_w, zf + 0.5 * frame_d, "Metal_DarkBronze", "01_Architecture")
            self.add_box(f"{wall_name}_wfT_{idx}", s0, y1 - frame_w, zf - 0.5 * frame_d, s1, y1, zf + 0.5 * frame_d, "Metal_DarkBronze", "01_Architecture")
            self.add_box(f"{wall_name}_wfL_{idx}", s0, y0, zf - 0.5 * frame_d, s0 + frame_w, y1, zf + 0.5 * frame_d, "Metal_DarkBronze", "01_Architecture")
            self.add_box(f"{wall_name}_wfR_{idx}", s1 - frame_w, y0, zf - 0.5 * frame_d, s1, y1, zf + 0.5 * frame_d, "Metal_DarkBronze", "01_Architecture")
            for m_i in range(1, mullion_count):
                mx = s0 + (s1 - s0) * (m_i / mullion_count)
                self.add_box(f"{wall_name}_wm_{idx}_{m_i}", mx - 0.016, y0, zf - 0.022, mx + 0.016, y1, zf + 0.022, "Metal_DarkBronze", "01_Architecture")
            self.add_box(f"{wall_name}_wglass_{idx}", s0 + frame_w, y0 + frame_w, zf - 0.008, s1 - frame_w, y1 - frame_w, zf + 0.008, "Glass_Architectural", "01_Architecture", collider_kind="window")
        else:
            x_in = fixed_pos + inward_sgn * (half_t + sill_overhang)
            x_out = fixed_pos - inward_sgn * half_t
            self.add_box(f"{wall_name}_sill_{idx}", min(x_in, x_out), y0 - 0.032, s0 - 0.03, max(x_in, x_out), y0 + 0.005, s1 + 0.03, "Stone_Calacatta", "01_Architecture", bevel_m=0.004)
            xf = fixed_pos - inward_sgn * (half_t - 0.04)
            self.add_box(f"{wall_name}_wfB_{idx}", xf - 0.5 * frame_d, y0, s0, xf + 0.5 * frame_d, y0 + frame_w, s1, "Metal_DarkBronze", "01_Architecture")
            self.add_box(f"{wall_name}_wfT_{idx}", xf - 0.5 * frame_d, y1 - frame_w, s0, xf + 0.5 * frame_d, y1, s1, "Metal_DarkBronze", "01_Architecture")
            self.add_box(f"{wall_name}_wfL_{idx}", xf - 0.5 * frame_d, y0, s0, xf + 0.5 * frame_d, y1, s0 + frame_w, "Metal_DarkBronze", "01_Architecture")
            self.add_box(f"{wall_name}_wfR_{idx}", xf - 0.5 * frame_d, y0, s1 - frame_w, xf + 0.5 * frame_d, y1, s1, "Metal_DarkBronze", "01_Architecture")
            for m_i in range(1, mullion_count):
                mz = s0 + (s1 - s0) * (m_i / mullion_count)
                self.add_box(f"{wall_name}_wm_{idx}_{m_i}", xf - 0.022, y0, mz - 0.016, xf + 0.022, y1, mz + 0.016, "Metal_DarkBronze", "01_Architecture")
            self.add_box(f"{wall_name}_wglass_{idx}", xf - 0.008, y0 + frame_w, s0 + frame_w, xf + 0.008, y1 - frame_w, s1 - frame_w, "Glass_Architectural", "01_Architecture", collider_kind="window")

    def _build_sliding_balcony_door(
        self,
        wall_name: str,
        axis: str,
        fixed_pos: float,
        half_t: float,
        op: dict[str, Any],
        idx: int,
    ) -> None:
        s0, s1, y1 = op["s0"], op["s1"], op["y1"]
        open_x0 = op["open_s0"]
        open_x1 = op["open_s1"]
        zf = fixed_pos - 0.02

        self.add_box(f"{wall_name}_trackB_{idx}", s0, 0.0, fixed_pos - half_t, s1, 0.025, fixed_pos + half_t, "Metal_DarkBronze", "01_Architecture")
        self.add_box(f"{wall_name}_trackT_{idx}", s0, y1 - 0.045, fixed_pos - half_t, s1, y1, fixed_pos + half_t, "Metal_DarkBronze", "01_Architecture")
        self.add_box(f"{wall_name}_jambL_{idx}", s0, 0.0, fixed_pos - half_t, s0 + 0.045, y1, fixed_pos + half_t, "Metal_DarkBronze", "01_Architecture")
        self.add_box(f"{wall_name}_jambR_{idx}", s1 - 0.045, 0.0, fixed_pos - half_t, s1, y1, fixed_pos + half_t, "Metal_DarkBronze", "01_Architecture")

        self.add_box(f"{wall_name}_glassL_{idx}", s0 + 0.045, 0.025, zf - 0.01, open_x0, y1 - 0.045, zf + 0.01, "Glass_Architectural", "01_Architecture", collider_kind="window")
        self.add_box(f"{wall_name}_postL_{idx}", open_x0 - 0.022, 0.025, zf - 0.03, open_x0 + 0.022, y1 - 0.045, zf + 0.03, "Metal_DarkBronze", "01_Architecture")

        slide_w = open_x1 - open_x0
        self.add_box(f"{wall_name}_slideFrameB", open_x0 - slide_w, 0.03, zf + 0.03, open_x0 + 0.03, 0.075, zf + 0.065, "Metal_DarkBronze", "01_Architecture")
        self.add_box(f"{wall_name}_slideFrameT", open_x0 - slide_w, y1 - 0.09, zf + 0.03, open_x0 + 0.03, y1 - 0.045, zf + 0.065, "Metal_DarkBronze", "01_Architecture")
        self.add_box(f"{wall_name}_slideStileR", open_x0 - 0.01, 0.03, zf + 0.03, open_x0 + 0.035, y1 - 0.045, zf + 0.065, "Metal_DarkBronze", "01_Architecture")
        self.add_box(f"{wall_name}_slideGlass", open_x0 - slide_w + 0.03, 0.075, zf + 0.042, open_x0 - 0.01, y1 - 0.09, zf + 0.052, "Glass_Architectural", "01_Architecture")

        self.add_box(f"{wall_name}_postR_{idx}", open_x1 - 0.022, 0.025, zf - 0.03, open_x1 + 0.022, y1 - 0.045, zf + 0.03, "Metal_DarkBronze", "01_Architecture")
        self.add_box(f"{wall_name}_glassR_{idx}", open_x1, 0.025, zf - 0.01, s1 - 0.045, y1 - 0.045, zf + 0.01, "Glass_Architectural", "01_Architecture", collider_kind="window")

    def add_coffered_ceiling_with_cove(
        self,
        group_name: str,
        x0: float,
        z0: float,
        x1: float,
        z1: float,
        col_name: str = "01_Architecture",
    ) -> None:
        prev = self._active_parent
        c_emp = self.get_group(group_name, col_name)
        self.set_parent_group(c_emp)

        soffit_w = 0.40
        soffit_y0 = CEIL_H - 0.14
        self.add_box(f"{group_name}_top", x0, CEIL_H, z0, x1, CEIL_H + 0.08, z1, "Wall_CeilingMatte", col_name, tile_m=2.5)
        self.add_box(f"{group_name}_sofN", x0, soffit_y0, z0, x1, CEIL_H, z0 + soffit_w, "Wall_CeilingMatte", col_name, tile_m=2.0)
        self.add_box(f"{group_name}_sofS", x0, soffit_y0, z1 - soffit_w, x1, CEIL_H, z1, "Wall_CeilingMatte", col_name, tile_m=2.0)
        self.add_box(f"{group_name}_sofW", x0, soffit_y0, z0 + soffit_w, x0 + soffit_w, CEIL_H, z1 - soffit_w, "Wall_CeilingMatte", col_name, tile_m=2.0)
        self.add_box(f"{group_name}_sofE", x1 - soffit_w, soffit_y0, z0 + soffit_w, x1, CEIL_H, z1 - soffit_w, "Wall_CeilingMatte", col_name, tile_m=2.0)

        ix0, ix1 = x0 + soffit_w, x1 - soffit_w
        iz0, iz1 = z0 + soffit_w, z1 - soffit_w
        lip_y = soffit_y0 + 0.028
        self.add_box(f"{group_name}_ledN", ix0, lip_y, iz0 - 0.015, ix1, lip_y + 0.018, iz0 + 0.01, "LED_WarmCove", col_name)
        self.add_box(f"{group_name}_ledS", ix0, lip_y, iz1 - 0.01, ix1, lip_y + 0.018, iz1 + 0.015, "LED_WarmCove", col_name)
        self.add_box(f"{group_name}_ledW", ix0 - 0.015, lip_y, iz0, ix0 + 0.01, lip_y + 0.018, iz1, "LED_WarmCove", col_name)
        self.add_box(f"{group_name}_ledE", ix1 - 0.01, lip_y, iz0, ix1 + 0.015, lip_y + 0.018, iz1, "LED_WarmCove", col_name)

        self.set_parent_group(prev)

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
                f = bm.faces.new([verts_grid[iy][ix], verts_grid[iy][ix + 1], verts_grid[iy + 1][ix + 1], verts_grid[iy + 1][ix]])
                f.smooth = True

        bmesh.ops.solidify(bm, geom=list(bm.faces), thickness=0.006)
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(name, mesh)
        obj.data.materials.append(self.materials["Fabric_SheerDrapery"])
        self._link_obj(obj, col_name)
        bpy.context.view_layer.update()
        self.apply_metric_uv(obj, tile_m=0.8)

    # -----------------------------------------------------------------------
    # 1. Architectural Shell & Room Floors (6 Zones)
    # -----------------------------------------------------------------------

    def build_shell_and_floors(self) -> None:
        # Room-scoped architectural shell groups for 1-by-1 progressive construction
        self.set_parent_group(self.get_group("Arch__living", "01_Architecture"))
        self.add_box("Floor_Living", 0.0, -0.06, 0.0, 5.82, 0.0, 3.66, "Floor_HoneyOak", "01_Architecture", tile_m=1.8)
        self.build_wall_with_openings(
            "Wall_North_Living",
            axis="x",
            fixed_pos=0.0,
            span_start=-0.09,
            span_end=5.82,
            thick=EXT_WALL_T,
            openings=[{"s0": 0.52, "s1": 5.30, "y0": 0.0, "y1": 2.46, "kind": "slider", "open_s0": 2.95, "open_s1": 3.88}],
            baseboard_sides=(False, True),
        )
        self.build_wall_with_openings("Wall_West_Living", axis="z", fixed_pos=0.0, span_start=0.0, span_end=3.66, thick=EXT_WALL_T, openings=[], baseboard_sides=(False, True))
        self.add_pleated_curtain_stack("Curtain_Living_Left", 0.24, 0.68, 0.16, 0.01, 2.56, "01_Architecture", pleats=6)
        self.add_pleated_curtain_stack("Curtain_Living_Right", 5.14, 5.58, 0.16, 0.01, 2.56, "01_Architecture", pleats=6)

        self.set_parent_group(self.get_group("Arch__kitchen", "01_Architecture"))
        self.add_box("Floor_Kitchen", 5.82, -0.06, 0.0, 9.42, 0.0, 3.66, "Floor_HoneyOak", "01_Architecture", tile_m=1.8)
        self.build_wall_with_openings("Wall_North_Kitchen", axis="x", fixed_pos=0.0, span_start=5.82, span_end=9.51, thick=EXT_WALL_T, openings=[], baseboard_sides=(False, True))
        self.build_wall_with_openings(
            "Wall_East_Kitchen",
            axis="z",
            fixed_pos=9.42,
            span_start=0.0,
            span_end=3.57,
            thick=EXT_WALL_T,
            openings=[
                {"s0": 1.08, "s1": 2.24, "y0": 1.04, "y1": 2.30, "kind": "window", "inward_sgn": -1.0, "mullions": 2},
                {"s0": 2.58, "s1": 3.48, "y0": 0.0, "y1": DOOR_H, "kind": "door", "leaf": "closed_ext"},
            ],
            baseboard_sides=(True, False),
        )
        self.build_wall_with_openings(
            "Wall_Living_Kitchen",
            axis="z",
            fixed_pos=5.82,
            span_start=0.0,
            span_end=3.66,
            thick=INT_WALL_T,
            openings=[{"s0": 0.74, "s1": 3.30, "y0": 0.0, "y1": 2.40, "kind": "portal"}],
            baseboard_sides=(True, True),
        )

        self.set_parent_group(self.get_group("Arch__bedroom", "01_Architecture"))
        self.add_box("Floor_Bedroom", 0.0, -0.06, 3.66, 4.78, 0.0, 6.99, "Floor_HoneyOak", "01_Architecture", tile_m=1.8)
        self.build_wall_with_openings("Wall_West_Bed", axis="z", fixed_pos=0.0, span_start=3.66, span_end=7.08, thick=EXT_WALL_T, openings=[], baseboard_sides=(False, True))
        self.build_wall_with_openings(
            "Wall_Spine_Bed",
            axis="x",
            fixed_pos=3.66,
            span_start=0.0,
            span_end=4.78,
            thick=INT_WALL_T,
            openings=[{"s0": 3.38, "s1": 4.30, "y0": 0.0, "y1": DOOR_H, "kind": "door", "leaf": "ajar_south_right"}],
            baseboard_sides=(True, True),
        )
        self.build_wall_with_openings(
            "Wall_South_Bed",
            axis="x",
            fixed_pos=6.99,
            span_start=-0.09,
            span_end=4.78,
            thick=EXT_WALL_T,
            openings=[{"s0": 0.90, "s1": 3.85, "y0": 0.56, "y1": 2.36, "kind": "window", "inward_sgn": -1.0, "mullions": 3}],
            baseboard_sides=(True, False),
        )
        self.add_pleated_curtain_stack("Curtain_Bed_L", 0.55, 0.92, 6.82, 0.01, 2.54, "01_Architecture", pleats=5)
        self.add_pleated_curtain_stack("Curtain_Bed_R", 3.83, 4.12, 6.82, 0.01, 2.54, "01_Architecture", pleats=5)

        self.set_parent_group(self.get_group("Arch__bathroom", "01_Architecture"))
        self.add_box("Floor_Bathroom", 4.78, -0.06, 3.66, 8.18, 0.0, 6.99, "Tile_SpaPorcelain", "01_Architecture", tile_m=1.2)
        self.add_box("Threshold_Bath", 6.12, 0.0, 3.64, 7.00, 0.006, 3.68, "Metal_WarmBrass", "01_Architecture")
        self.build_wall_with_openings("Wall_Bed_Bath", axis="z", fixed_pos=4.78, span_start=3.66, span_end=6.99, thick=INT_WALL_T, openings=[], baseboard_sides=(True, True))
        self.build_wall_with_openings(
            "Wall_Spine_Bath",
            axis="x",
            fixed_pos=3.66,
            span_start=4.78,
            span_end=8.18,
            thick=INT_WALL_T,
            openings=[{"s0": 6.12, "s1": 7.00, "y0": 0.0, "y1": DOOR_H, "kind": "door", "leaf": "ajar_south_left"}],
            baseboard_sides=(True, True),
        )
        self.build_wall_with_openings(
            "Wall_South_Bath",
            axis="x",
            fixed_pos=6.99,
            span_start=4.78,
            span_end=8.18,
            thick=EXT_WALL_T,
            openings=[{"s0": 6.68, "s1": 7.78, "y0": 1.02, "y1": 2.30, "kind": "window", "inward_sgn": -1.0, "mullions": 2}],
            baseboard_sides=(True, False),
        )

        self.set_parent_group(self.get_group("Arch__office", "01_Architecture"))
        self.add_box("Floor_Office", 8.18, -0.06, 3.66, 11.58, 0.0, 6.99, "Floor_HoneyOak", "01_Architecture", tile_m=1.8)
        self.add_box("Threshold_Office", 8.34, 0.0, 3.64, 9.22, 0.006, 3.68, "Metal_WarmBrass", "01_Architecture")
        self.build_wall_with_openings("Wall_Bath_Office", axis="z", fixed_pos=8.18, span_start=3.66, span_end=6.99, thick=INT_WALL_T, openings=[], baseboard_sides=(True, True))
        self.build_wall_with_openings(
            "Wall_Spine_Office",
            axis="x",
            fixed_pos=3.66,
            span_start=8.25,
            span_end=11.67,
            thick=EXT_WALL_T,
            openings=[{"s0": 8.34, "s1": 9.22, "y0": 0.0, "y1": DOOR_H, "kind": "portal"}],
            baseboard_sides=(True, True),
        )
        self.build_wall_with_openings(
            "Wall_East_Office",
            axis="z",
            fixed_pos=11.58,
            span_start=3.66,
            span_end=7.08,
            thick=EXT_WALL_T,
            openings=[{"s0": 4.28, "s1": 6.38, "y0": 0.58, "y1": 2.36, "kind": "window", "inward_sgn": -1.0, "mullions": 3}],
            baseboard_sides=(True, False),
        )
        self.build_wall_with_openings(
            "Wall_South_Office",
            axis="x",
            fixed_pos=6.99,
            span_start=8.18,
            span_end=11.67,
            thick=EXT_WALL_T,
            openings=[],
            baseboard_sides=(True, False),
        )

        # Separate Ceiling Groups (`Ceil__*`) so Dollhouse/Build view can cutaway ceilings cleanly!
        self.set_parent_group(None)
        self.add_coffered_ceiling_with_cove("Ceil__living", 0.0, 0.0, 5.82, 3.66)
        self.add_coffered_ceiling_with_cove("Ceil__kitchen", 5.82, 0.0, 9.42, 3.66)
        self.add_coffered_ceiling_with_cove("Ceil__bedroom", 0.0, 3.66, 4.78, 6.99)
        self.add_coffered_ceiling_with_cove("Ceil__bathroom", 4.78, 3.66, 8.18, 6.99)
        self.add_coffered_ceiling_with_cove("Ceil__office", 8.18, 3.66, 11.58, 6.99)

    # -----------------------------------------------------------------------
    # 2. Built-in Architectural Fixtures & Gallery Art (`Fix__*`, `Art__*`)
    # -----------------------------------------------------------------------

    def build_fixed_joinery_and_art(self) -> None:
        # A. Living Corner Walnut Niches & Gallery Art
        self.set_parent_group(self.get_group("Fix__living_niches", "02_Living_Dining"))
        for side_i, (nz0, nz1) in enumerate([(0.16, 0.54), (3.12, 3.50)]):
            self.add_box(f"Niche_Back_{side_i}", 0.08, 0.48, nz0, 0.105, 2.46, nz1, "Wood_Walnut", "02_Living_Dining", tile_m=1.0)
            self.add_box(f"Niche_SideA_{side_i}", 0.09, 0.48, nz0, 0.22, 2.46, nz0 + 0.025, "Wood_Walnut", "02_Living_Dining", bevel_m=0.003)
            self.add_box(f"Niche_SideB_{side_i}", 0.09, 0.48, nz1 - 0.025, 0.22, 2.46, nz1, "Wood_Walnut", "02_Living_Dining", bevel_m=0.003)
            for sh_i, sy in enumerate([0.50, 1.00, 1.50, 2.00, 2.44]):
                self.add_box(f"Niche_Shelf_{side_i}_{sh_i}", 0.10, sy, nz0 + 0.02, 0.21, sy + 0.024, nz1 - 0.02, "Wood_Walnut", "02_Living_Dining", bevel_m=0.003)
                if sh_i < 4:
                    self.add_box(f"Niche_LED_{side_i}_{sh_i}", 0.11, sy + 0.46, nz0 + 0.04, 0.14, sy + 0.475, nz1 - 0.04, "LED_WarmCove", "02_Living_Dining")
                    self.add_cylinder(f"Niche_Vase_{side_i}_{sh_i}", 0.16, sy + 0.024, 0.5 * (nz0 + nz1), 0.045, 0.16, "Ceramic_White" if sh_i % 2 == 0 else "Metal_WarmBrass", "02_Living_Dining", segments=16)

        for art_i, (ax0, ax1, mat_art) in enumerate([(1.05, 1.63, "Ref_LivingArt1"), (1.84, 2.42, "Ref_LivingArt2")]):
            self.add_box(f"Living_ArtFrame_{art_i}", ax0 - 0.025, 1.24, 3.555, ax1 + 0.025, 2.02, 3.59, "Wood_Walnut", "02_Living_Dining", bevel_m=0.004)
            self.add_uv_quad(f"ArtCanvas_living_{art_i}", [(ax1, 1.27, 3.552), (ax0, 1.27, 3.552), (ax0, 1.99, 3.552), (ax1, 1.99, 3.552)], mat_art, "02_Living_Dining")

        # Synchronized Dining Linear Chandelier (`Light__dining_chandelier` centered at `(4.84, 0, 1.32)`)
        d_light = self.get_group("Light__dining_chandelier", "02_Living_Dining")
        self.set_parent_group(d_light)
        self.add_box("Dining_Chandelier_Canopy", -0.18, CEIL_H - 0.03, -0.05, 0.18, CEIL_H, 0.05, "Metal_WarmBrass", "02_Living_Dining")
        self.add_box("Dining_Chandelier_Bar1", -0.015, 2.02, -0.54, 0.015, 2.05, 0.54, "LED_WarmCove", "02_Living_Dining", bevel_m=0.005)
        self.add_box("Dining_Chandelier_Bar2", 0.075, 2.10, -0.44, 0.105, 2.13, 0.44, "LED_WarmCove", "02_Living_Dining", bevel_m=0.005)
        self.place_group_pose(d_light, 4.84, 0.0, 1.32, 0.0)

        # B. Kitchen Perimeter Cabinetry & Recessed Window Sink
        self.set_parent_group(self.get_group("Fix__kitchen_cabinets", "03_Kitchen_Foyer"))
        self.add_box("Kitchen_TallTower", 5.92, 0.0, 0.09, 6.62, 2.58, 0.70, "Wood_Walnut", "03_Kitchen_Foyer", tile_m=1.2, bevel_m=0.006, collider_kind="fixture")
        self.add_box("Kitchen_OvenUpper", 5.98, 1.18, 0.695, 6.56, 1.62, 0.715, "Metal_DarkBronze", "03_Kitchen_Foyer", bevel_m=0.004)
        self.add_box("Kitchen_OvenLower", 5.98, 0.68, 0.695, 6.56, 1.14, 0.715, "Metal_DarkBronze", "03_Kitchen_Foyer", bevel_m=0.004)

        self.add_box("Kitchen_BaseN", 6.62, 0.08, 0.09, 9.33, 0.88, 0.67, "Cabinet_SageOlive", "03_Kitchen_Foyer", bevel_m=0.005, collider_kind="fixture")
        self.add_box("Kitchen_ToeKickN", 6.64, 0.0, 0.09, 9.31, 0.08, 0.60, "Metal_DarkBronze", "03_Kitchen_Foyer")
        self.add_box("Kitchen_ToeLED_N", 6.66, 0.065, 0.605, 8.72, 0.078, 0.62, "LED_WarmCove", "03_Kitchen_Foyer")
        self.add_box("Kitchen_CounterN", 6.62, 0.88, 0.09, 9.33, 0.92, 0.70, "Stone_Calacatta", "03_Kitchen_Foyer", tile_m=1.4, bevel_m=0.005)
        self.add_box("Kitchen_BacksplashN", 6.62, 0.92, 0.09, 9.33, 1.56, 0.12, "Stone_Calacatta", "03_Kitchen_Foyer", tile_m=1.4)

        for cab_i in range(4):
            cx0 = 6.65 + cab_i * 0.52
            self.add_box(f"Kitchen_DoorN_{cab_i}", cx0, 0.11, 0.67, cx0 + 0.49, 0.85, 0.688, "Cabinet_SageOlive", "03_Kitchen_Foyer", bevel_m=0.004)
            self.add_box(f"Kitchen_PullN_{cab_i}", cx0 + 0.18, 0.80, 0.688, cx0 + 0.31, 0.818, 0.708, "Metal_DarkBronze", "03_Kitchen_Foyer")

        self.add_box("Kitchen_Cooktop", 7.36, 0.92, 0.22, 8.16, 0.94, 0.58, "Metal_DarkBronze", "03_Kitchen_Foyer", bevel_m=0.004)
        self.add_box("Kitchen_RangeHood", 7.30, 1.56, 0.09, 8.22, 2.58, 0.56, "Cabinet_SageOlive", "03_Kitchen_Foyer", bevel_m=0.008)
        self.add_box("Kitchen_HoodLight", 7.42, 1.545, 0.24, 8.10, 1.56, 0.44, "LED_WarmCove", "03_Kitchen_Foyer")

        for side_sh, (sx_a, sx_b) in enumerate([(6.66, 7.26), (8.26, 9.26)]):
            for sh_y in (1.56, 1.98):
                self.add_box(f"Kitchen_FloatShelf_{side_sh}_{sh_y}", sx_a, sh_y, 0.10, sx_b, sh_y + 0.032, 0.36, "Wood_Walnut", "03_Kitchen_Foyer", bevel_m=0.004)
                self.add_box(f"Kitchen_ShelfLED_{side_sh}_{sh_y}", sx_a + 0.04, sh_y - 0.012, 0.16, sx_b - 0.04, sh_y, 0.20, "LED_WarmCove", "03_Kitchen_Foyer")
                self.add_cylinder(f"Kitchen_Jar_{side_sh}_{sh_y}", 0.5 * (sx_a + sx_b), sh_y + 0.032, 0.22, 0.06, 0.15, "Ceramic_White", "03_Kitchen_Foyer")

        self.add_box("Kitchen_BaseE", 8.75, 0.08, 0.67, 9.33, 0.88, 2.36, "Cabinet_SageOlive", "03_Kitchen_Foyer", bevel_m=0.005, collider_kind="fixture")
        self.add_box("Kitchen_CounterE_A", 8.73, 0.88, 0.70, 9.33, 0.92, 1.38, "Stone_Calacatta", "03_Kitchen_Foyer", tile_m=1.4, bevel_m=0.004)
        self.add_box("Kitchen_CounterE_B", 8.73, 0.88, 1.94, 9.33, 0.92, 2.36, "Stone_Calacatta", "03_Kitchen_Foyer", tile_m=1.4, bevel_m=0.004)
        self.add_box("Kitchen_CounterE_FrontLip", 8.73, 0.88, 1.38, 8.81, 0.92, 1.94, "Stone_Calacatta", "03_Kitchen_Foyer", bevel_m=0.004)
        self.add_box("Kitchen_CounterE_BackLip", 9.24, 0.88, 1.38, 9.33, 0.92, 1.94, "Stone_Calacatta", "03_Kitchen_Foyer", bevel_m=0.004)
        self.add_box("Kitchen_SinkBasin", 8.80, 0.70, 1.38, 9.24, 0.73, 1.94, "Metal_DarkBronze", "03_Kitchen_Foyer")
        self.add_cylinder("Kitchen_FaucetStem", 9.26, 0.92, 1.66, 0.015, 0.30, "Metal_DarkBronze", "03_Kitchen_Foyer", segments=12)
        self.add_box("Kitchen_FaucetNeck", 9.06, 1.19, 1.645, 9.27, 1.22, 1.675, "Metal_DarkBronze", "03_Kitchen_Foyer", bevel_m=0.006)

        # Framed Foyer Painting above South Foyer Wall (`x in [7.22, 7.94], z = 3.58`)
        self.add_box("Foyer_Art_Frame", 7.22, 1.08, 3.555, 7.94, 1.94, 3.585, "Metal_DarkBronze", "03_Kitchen_Foyer", bevel_m=0.004)
        self.add_uv_quad("ArtCanvas_kitchen_0", [(7.91, 1.11, 3.551), (7.25, 1.11, 3.551), (7.25, 1.91, 3.551), (7.91, 1.91, 3.551)], "Ref_FoyerArt", "03_Kitchen_Foyer")

        # C. Centerpiece Waterfall Calacatta Island + 3 Amber Globe Pendants (`x=[6.62, 8.14], z=[1.34, 2.04]`)
        self.set_parent_group(self.get_group("Fix__kitchen_island", "03_Kitchen_Foyer"))
        ix0, ix1, iz0, iz1 = 6.62, 8.14, 1.34, 2.04
        self.add_collider("fixture", "Kitchen_Island", ix0, iz0, ix1, iz1, 0.0, 0.93)
        self.add_box("Island_Core", ix0 + 0.05, 0.0, iz0 + 0.03, ix1 - 0.05, 0.88, iz1 - 0.14, "Cabinet_SageOlive", "03_Kitchen_Foyer", bevel_m=0.004)
        for fl_i in range(20):
            fx = ix0 + 0.08 + fl_i * 0.068
            self.add_box(f"Island_Flute_{fl_i}", fx, 0.06, iz1 - 0.14, fx + 0.042, 0.86, iz1 - 0.115, "Wood_Walnut", "03_Kitchen_Foyer", bevel_m=0.003)
        self.add_box("Island_LED", ix0 + 0.08, 0.855, iz1 - 0.11, ix1 - 0.08, 0.872, iz1 - 0.08, "LED_WarmCove", "03_Kitchen_Foyer")
        self.add_box("Island_WaterfallTop", ix0, 0.88, iz0, ix1, 0.925, iz1, "Stone_Calacatta", "03_Kitchen_Foyer", tile_m=1.3, bevel_m=0.005)
        self.add_box("Island_WaterfallL", ix0, 0.0, iz0, ix0 + 0.052, 0.88, iz1, "Stone_Calacatta", "03_Kitchen_Foyer", tile_m=1.3, bevel_m=0.004)
        self.add_box("Island_WaterfallR", ix1 - 0.052, 0.0, iz0, ix1, 0.88, iz1, "Stone_Calacatta", "03_Kitchen_Foyer", tile_m=1.3, bevel_m=0.004)
        self.add_cylinder("Island_Vase", 7.02, 0.925, 1.66, 0.082, 0.22, "Ceramic_White", "03_Kitchen_Foyer")
        self.add_olive_bouquet("Island_Branches", 7.02, 1.145, 1.66, 1.05, "03_Kitchen_Foyer")
        self.add_cylinder("Island_Bowl", 7.68, 0.925, 1.70, 0.13, 0.055, "Wood_Walnut", "03_Kitchen_Foyer", segments=22)
        for fr_i, (fx_o, fz_o) in enumerate([(-0.04, -0.02), (0.04, -0.01), (0.0, 0.04)]):
            self.add_sphere(f"Island_Pear_{fr_i}", 7.68 + fx_o, 0.995, 1.70 + fz_o, 0.036, "Foliage_OliveGreen", "03_Kitchen_Foyer")
        for st_i, sx_c in enumerate([6.92, 7.38, 7.84]):
            self.add_cylinder(f"Pendant_Rod_{st_i}", sx_c, 1.86, 1.69, 0.007, CEIL_H - 1.86, "Metal_DarkBronze", "03_Kitchen_Foyer", segments=8)
            self.add_sphere(f"Pendant_Globe_{st_i}", sx_c, 1.76, 1.69, 0.12, "Glass_AmberGlobe", "03_Kitchen_Foyer")

        # D. Bedroom Built-in Wardrobe & Backlit Medallion
        self.set_parent_group(self.get_group("Fix__bedroom_wardrobe", "04_Primary_Bedroom"))
        self.add_box("Bedroom_Wardrobe", 4.18, 0.0, 4.72, 4.71, 2.54, 6.46, "Wood_Walnut", "04_Primary_Bedroom", tile_m=1.2, bevel_m=0.008, collider_kind="fixture")
        for wd_i in range(3):
            wz0 = 4.76 + wd_i * 0.56
            self.add_box(f"Wardrobe_GlassDoor_{wd_i}", 4.162, 0.08, wz0, 4.18, 2.46, wz0 + 0.53, "Glass_SmokedTable", "04_Primary_Bedroom", bevel_m=0.003)
            self.add_box(f"Wardrobe_Pull_{wd_i}", 4.145, 0.95, wz0 + 0.04, 4.162, 1.45, wz0 + 0.06, "Metal_WarmBrass", "04_Primary_Bedroom")
        bed_halo = self.add_cylinder("Bed_Medallion_Halo", 0.102, 1.88, 5.34, 0.44, 0.014, "LED_WarmCove", "04_Primary_Bedroom", segments=40)
        bed_halo.rotation_euler = Euler((0.0, math.radians(90.0), 0.0), "XYZ")
        self.add_uv_disc_west_wall("ArtCanvas_bedroom_0", 0.122, 1.88, 5.34, 0.415, "Ref_BedMedallion", "04_Primary_Bedroom", segments=40)

        # E. Spa Bathroom Built-in Shower, Soaking Tub & Vanity
        self.set_parent_group(self.get_group("Fix__bath_fixtures", "05_Spa_Bathroom"))
        self.add_box("Bath_AccentStone", 4.88, 0.12, 6.84, 6.62, 2.52, 6.90, "Stone_Travertine", "05_Spa_Bathroom", tile_m=1.2, bevel_m=0.006)
        self.add_box("Bath_AccentLED", 4.90, 2.50, 6.81, 6.60, 2.53, 6.84, "LED_WarmCove", "05_Spa_Bathroom")
        self.add_box("Bath_TeakPlinth", 4.90, 0.0, 5.96, 6.64, 0.10, 6.88, "Wood_Walnut", "05_Spa_Bathroom", bevel_m=0.006)
        self.add_box("Bath_PlinthLED", 4.94, 0.015, 5.945, 6.60, 0.032, 5.965, "LED_WarmCove", "05_Spa_Bathroom")

        tx0, tx1, tz0, tz1 = 4.98, 6.54, 6.02, 6.80
        self.add_collider("fixture", "Spa_Soaking_Tub", tx0, tz0, tx1, tz1, 0.0, 0.68)
        self.add_box("Tub_WallN", tx0, 0.10, tz0, tx1, 0.66, tz0 + 0.09, "Ceramic_White", "05_Spa_Bathroom", bevel_m=0.03)
        self.add_box("Tub_WallS", tx0, 0.10, tz1 - 0.09, tx1, 0.66, tz1, "Ceramic_White", "05_Spa_Bathroom", bevel_m=0.03)
        self.add_box("Tub_WallW", tx0, 0.10, tz0, tx0 + 0.11, 0.66, tz1, "Ceramic_White", "05_Spa_Bathroom", bevel_m=0.03)
        self.add_box("Tub_WallE", tx1 - 0.11, 0.10, tz0, tx1, 0.66, tz1, "Ceramic_White", "05_Spa_Bathroom", bevel_m=0.03)
        self.add_box("Tub_Bottom", tx0 + 0.05, 0.10, tz0 + 0.05, tx1 - 0.05, 0.18, tz1 - 0.05, "Ceramic_White", "05_Spa_Bathroom")
        self.add_box("Tub_Water", tx0 + 0.10, 0.50, tz0 + 0.09, tx1 - 0.10, 0.54, tz1 - 0.09, "Water_Spa", "05_Spa_Bathroom")
        self.add_box("Tub_TeakCaddy", 5.60, 0.66, tz0 - 0.02, 5.86, 0.685, tz1 + 0.02, "Wood_Walnut", "05_Spa_Bathroom", bevel_m=0.004)
        self.add_cylinder("Tub_Candle", 5.73, 0.685, 0.5 * (tz0 + tz1), 0.035, 0.08, "LED_WarmCove", "05_Spa_Bathroom", segments=14)
        self.add_cylinder("Tub_FloorFiller", 6.58, 0.10, 6.42, 0.018, 0.76, "Metal_DarkBronze", "05_Spa_Bathroom", segments=12)

        self.add_box("Shower_Curb", 4.86, 0.0, 5.16, 5.94, 0.055, 5.22, "Stone_Calacatta", "05_Spa_Bathroom", bevel_m=0.004)
        self.add_box("Shower_WallTileW", 4.85, 0.0, 3.74, 4.875, 2.54, 5.20, "Stone_Calacatta", "05_Spa_Bathroom", tile_m=1.2)
        self.add_box("Shower_GlassSouth", 4.86, 0.055, 5.18, 5.92, 2.38, 5.20, "Glass_Architectural", "05_Spa_Bathroom", collider_kind="fixture")
        self.add_box("Shower_FrameTop", 4.86, 2.36, 5.17, 5.93, 2.39, 5.21, "Metal_DarkBronze", "05_Spa_Bathroom")
        self.add_box("Shower_FramePost", 5.90, 0.0, 5.17, 5.93, 2.39, 5.21, "Metal_DarkBronze", "05_Spa_Bathroom")
        self.add_box("Shower_RainArm", 4.86, 2.26, 4.44, 5.38, 2.285, 4.47, "Metal_DarkBronze", "05_Spa_Bathroom")
        self.add_box("Shower_RainHead", 5.24, 2.23, 4.34, 5.50, 2.255, 4.60, "Metal_DarkBronze", "05_Spa_Bathroom", bevel_m=0.004)
        self.add_box("Shower_NicheShelf", 4.86, 1.18, 4.18, 4.96, 1.205, 4.76, "Stone_Calacatta", "05_Spa_Bathroom")
        self.add_box("Shower_NicheLED", 4.87, 1.52, 4.20, 4.91, 1.535, 4.74, "LED_WarmCove", "05_Spa_Bathroom")

        self.add_box("Bath_VanityCabinet", 7.48, 0.24, 4.82, 8.09, 0.82, 6.04, "Wood_Walnut", "05_Spa_Bathroom", bevel_m=0.008, collider_kind="fixture")
        self.add_box("Bath_VanityTop", 7.46, 0.82, 4.80, 8.09, 0.865, 6.06, "Stone_Calacatta", "05_Spa_Bathroom", tile_m=1.2, bevel_m=0.005)
        self.add_box("Bath_VanityLED", 7.54, 0.22, 4.88, 8.02, 0.238, 5.98, "LED_WarmCove", "05_Spa_Bathroom")
        self.add_cylinder("Bath_VesselSink", 7.78, 0.865, 5.43, 0.20, 0.13, "Ceramic_White", "05_Spa_Bathroom", segments=28)
        self.add_box("Bath_Spout", 7.92, 1.06, 5.41, 8.09, 1.09, 5.45, "Metal_DarkBronze", "05_Spa_Bathroom", bevel_m=0.004)
        self.add_box("Bath_Mat", 6.72, 0.002, 4.92, 7.36, 0.016, 5.88, "Rug_WovenCream", "05_Spa_Bathroom", bevel_m=0.004)

        mirror_halo = self.add_cylinder("Bath_MirrorHalo", 8.075, 1.56, 5.43, 0.46, 0.016, "LED_WarmCove", "05_Spa_Bathroom", segments=36)
        mirror_halo.rotation_euler = Euler((0.0, math.radians(90.0), 0.0), "XYZ")
        mirror_disc = self.add_cylinder("Bath_MirrorGlass", 8.062, 1.56, 5.43, 0.43, 0.016, "Mirror_Silver", "05_Spa_Bathroom", segments=36)
        mirror_disc.rotation_euler = Euler((0.0, math.radians(90.0), 0.0), "XYZ")

        self.add_box("Bath_ToiletBowl", 7.52, 0.16, 6.28, 8.09, 0.43, 6.70, "Ceramic_White", "05_Spa_Bathroom", bevel_m=0.06, collider_kind="fixture")
        self.add_box("Bath_FlushPlate", 8.07, 0.92, 6.40, 8.09, 1.08, 6.58, "Metal_DarkBronze", "05_Spa_Bathroom")

        # F. Office Architectural Gallery Frame & Drapery
        self.set_parent_group(self.get_group("Fix__office_trim", "06_Office_Study"))
        self.add_box("Office_ArtFrame", 9.62, 1.24, 3.755, 10.48, 2.02, 3.785, "Wood_Walnut", "06_Office_Study", bevel_m=0.004)
        self.add_uv_quad("ArtCanvas_office_0", [(9.65, 1.27, 3.789), (10.45, 1.27, 3.789), (10.45, 1.99, 3.789), (9.65, 1.99, 3.789)], "Ref_LivingArt1", "06_Office_Study")

        self.set_parent_group(None)

    # -----------------------------------------------------------------------
    # 3. Seventeen Hierarchical Movable Furniture Assemblies (`Item__src_*`)
    #    Every assembly is built in local Homeweave space around `(0, 0, 0)`
    #    with front facing `+Z` (`rot_deg = 0`), then posed to Orientation 1!
    # -----------------------------------------------------------------------

    def build_movable_furniture_items(self) -> None:
        # 1. `Item__src_kitchen_stools` (1.38 x 0.34 x 0.90) -> Pose 1: (7.38, 2.21, 180)
        g = self.get_group("Item__src_kitchen_stools", "03_Kitchen_Foyer")
        self.set_parent_group(g)
        for st_i, lx_c in enumerate([-0.46, 0.0, 0.46]):
            self.add_box(f"Stool_Seat_{st_i}", lx_c - 0.18, 0.62, -0.13, lx_c + 0.18, 0.68, 0.15, "Fabric_OatmealBoucle", "03_Kitchen_Foyer", bevel_m=0.025)
            self.add_box(f"Stool_Back_{st_i}", lx_c - 0.18, 0.68, -0.16, lx_c + 0.18, 0.90, -0.10, "Fabric_OatmealBoucle", "03_Kitchen_Foyer", bevel_m=0.025)
            for ox in (-0.15, 0.15):
                for oz in (-0.12, 0.12):
                    self.add_cylinder(f"Stool_Leg_{st_i}_{ox}_{oz}", lx_c + ox, 0.0, oz, 0.011, 0.62, "Metal_DarkBronze", "03_Kitchen_Foyer", segments=8)
        self.place_group_pose(g, 7.38, 0.0, 2.21, 180.0)

        # 2. `Item__src_foyer_console` (0.80 x 0.26 x 0.80) -> Pose 1: (7.58, 3.44, 180)
        g = self.get_group("Item__src_foyer_console", "03_Kitchen_Foyer")
        self.set_parent_group(g)
        self.add_box("Foyer_Console_Top", -0.40, 0.76, -0.13, 0.40, 0.795, 0.13, "Glass_SmokedTable", "03_Kitchen_Foyer", bevel_m=0.004)
        self.add_box("Foyer_Console_Shelf", -0.36, 0.24, -0.10, 0.36, 0.265, 0.10, "Wood_Walnut", "03_Kitchen_Foyer", bevel_m=0.004)
        for ox in (-0.37, 0.37):
            for oz in (-0.10, 0.10):
                self.add_cylinder(f"Foyer_Leg_{ox}_{oz}", ox, 0.0, oz, 0.012, 0.76, "Metal_DarkBronze", "03_Kitchen_Foyer", segments=8)
        self.add_cylinder("Foyer_Vase", -0.14, 0.795, 0.0, 0.06, 0.17, "Ceramic_White", "03_Kitchen_Foyer")
        self.add_box("Foyer_Tray", 0.08, 0.795, -0.06, 0.28, 0.815, 0.06, "Metal_WarmBrass", "03_Kitchen_Foyer", bevel_m=0.004)
        self.place_group_pose(g, 7.58, 0.0, 3.44, 180.0)

        # 3. `Item__src_living_media` (1.92 x 0.42 x 1.60) -> Pose 1: (0.34, 1.90, 270)
        g = self.get_group("Item__src_living_media", "02_Living_Dining")
        self.set_parent_group(g)
        # Architectural walnut & travertine back-panel + 65" OLED TV facing `+Z`
        self.add_box("Media_BackPanel", -0.84, 0.50, -0.20, 0.84, 1.62, -0.155, "Wood_Walnut", "02_Living_Dining", tile_m=1.1, bevel_m=0.006)
        self.add_box("Media_TravertineInlay", -0.78, 0.54, -0.155, 0.78, 1.58, -0.135, "Stone_Travertine", "02_Living_Dining", tile_m=1.4, bevel_m=0.005)
        self.add_box("Media_LEDHalo", -0.80, 0.56, -0.16, 0.80, 1.56, -0.145, "LED_WarmCove", "02_Living_Dining")
        self.add_box("OLED_TV_Chassis", -0.70, 0.74, -0.135, 0.70, 1.54, -0.10, "Metal_DarkBronze", "02_Living_Dining", bevel_m=0.004)
        self.add_uv_quad("OLED_TV_Screen", [(-0.675, 0.765, -0.097), (0.675, 0.765, -0.097), (0.675, 1.515, -0.097), (-0.675, 1.515, -0.097)], "Ref_TVScreen", "02_Living_Dining")
        self.add_box("TV_Soundbar", -0.42, 0.535, -0.04, 0.42, 0.595, 0.05, "Metal_DarkBronze", "02_Living_Dining", bevel_m=0.006)
        self.add_box("Media_Console_Body", -0.95, 0.16, -0.19, 0.95, 0.50, 0.20, "Wood_Walnut", "02_Living_Dining", tile_m=1.2, bevel_m=0.008)
        self.add_box("Media_Console_Top", -0.96, 0.50, -0.195, 0.96, 0.53, 0.205, "Stone_Calacatta", "02_Living_Dining", tile_m=1.4, bevel_m=0.004)
        self.add_box("Media_Plinth_LED", -0.82, 0.142, -0.12, 0.82, 0.158, 0.14, "LED_WarmCove", "02_Living_Dining")
        for lx_leg in (-0.78, 0.78):
            self.add_box(f"Media_BaseLeg_{lx_leg}", lx_leg - 0.03, 0.0, -0.14, lx_leg + 0.03, 0.16, 0.15, "Metal_DarkBronze", "02_Living_Dining")
        for dr_i in range(4):
            dx0 = -0.91 + dr_i * 0.46
            self.add_box(f"Media_Drawer_{dr_i}", dx0, 0.18, 0.20, dx0 + 0.43, 0.48, 0.214, "Wood_Walnut", "02_Living_Dining", bevel_m=0.004)
        self.place_group_pose(g, 0.34, 0.0, 1.90, 270.0)

        # 4. `Item__src_living_sofa` (2.10 x 0.92 x 0.84) + Conversation Rug -> Pose 1: (2.48, 1.90, 90)
        g = self.get_group("Item__src_living_sofa", "02_Living_Dining")
        self.set_parent_group(g)
        self.add_box("Living_Area_Rug", -1.12, 0.001, -0.36, 1.12, 0.012, 1.42, "Rug_WovenCream", "02_Living_Dining", tile_m=1.1, bevel_m=0.004)
        self.add_box("Sofa_Plinth", -1.00, 0.03, -0.42, 1.00, 0.17, 0.42, "Wood_Walnut", "02_Living_Dining", bevel_m=0.008)
        self.add_box("Sofa_Backrest", -1.04, 0.17, -0.46, 1.04, 0.84, -0.24, "Fabric_OatmealBoucle", "02_Living_Dining", tile_m=0.7, bevel_m=0.045)
        self.add_box("Sofa_ArmL", -1.05, 0.17, -0.26, -0.87, 0.62, 0.45, "Fabric_OatmealBoucle", "02_Living_Dining", tile_m=0.7, bevel_m=0.04)
        self.add_box("Sofa_ArmR", 0.87, 0.17, -0.26, 1.05, 0.62, 0.45, "Fabric_OatmealBoucle", "02_Living_Dining", tile_m=0.7, bevel_m=0.04)
        for c_i in range(3):
            cx_a = -0.86 + c_i * 0.575
            self.add_box(f"Sofa_Seat_{c_i}", cx_a, 0.17, -0.24, cx_a + 0.56, 0.43, 0.44, "Fabric_OatmealBoucle", "02_Living_Dining", tile_m=0.65, bevel_m=0.038)
            self.add_box(f"Sofa_BackCushion_{c_i}", cx_a + 0.02, 0.42, -0.26, cx_a + 0.54, 0.80, -0.10, "Fabric_OatmealBoucle", "02_Living_Dining", tile_m=0.65, bevel_m=0.04)
        self.add_oriented_box("Sofa_Pillow_Olive", -0.68, 0.56, -0.06, 0.36, 0.32, 0.14, -14.0, "Fabric_OliveVelvet", "02_Living_Dining", bevel_m=0.035)
        self.add_oriented_box("Sofa_Pillow_Terra", 0.68, 0.55, -0.06, 0.36, 0.32, 0.14, 14.0, "Fabric_TerracottaThrow", "02_Living_Dining", bevel_m=0.035)
        self.add_box("Sofa_Throw", 0.34, 0.25, -0.02, 0.84, 0.45, 0.45, "Fabric_OliveVelvet", "02_Living_Dining", tile_m=0.5, bevel_m=0.02)
        self.place_group_pose(g, 2.48, 0.0, 1.90, 90.0)

        # 5. `Item__src_living_coffee` (1.10 x 0.60 x 0.42) -> Pose 1: (1.38, 1.90, 90)
        g = self.get_group("Item__src_living_coffee", "02_Living_Dining")
        self.set_parent_group(g)
        self.add_box("CoffeeTable_Base", -0.44, 0.015, -0.22, 0.44, 0.34, 0.22, "Wood_Walnut", "02_Living_Dining", bevel_m=0.01)
        self.add_box("CoffeeTable_Top", -0.54, 0.34, -0.29, 0.54, 0.40, 0.29, "Stone_Calacatta", "02_Living_Dining", tile_m=1.2, bevel_m=0.012)
        self.add_box("Coffee_Book", -0.32, 0.40, -0.12, -0.04, 0.435, 0.14, "Fabric_CrispLinen", "02_Living_Dining", bevel_m=0.004)
        self.add_cylinder("Coffee_Vase", 0.20, 0.40, 0.02, 0.068, 0.19, "Ceramic_White", "02_Living_Dining")
        self.add_olive_bouquet("Coffee_Branches", 0.20, 0.59, 0.02, 0.82, "02_Living_Dining")
        self.place_group_pose(g, 1.38, 0.0, 1.90, 90.0)

        # 6. `Item__src_living_armchair` (0.78 x 0.74 x 0.80) -> Pose 1: (1.38, 0.68, 0)
        g = self.get_group("Item__src_living_armchair", "02_Living_Dining")
        self.set_parent_group(g)
        self.add_box("Armchair_SwivelBase", -0.28, 0.0, -0.28, 0.28, 0.06, 0.28, "Wood_Walnut", "02_Living_Dining", bevel_m=0.02)
        self.add_box("Armchair_Seat", -0.38, 0.06, -0.34, 0.38, 0.40, 0.36, "Fabric_OatmealBoucle", "02_Living_Dining", bevel_m=0.06)
        self.add_box("Armchair_Back", -0.38, 0.38, -0.36, 0.38, 0.79, -0.14, "Fabric_OatmealBoucle", "02_Living_Dining", bevel_m=0.06)
        self.add_box("Armchair_ArmL", -0.39, 0.32, -0.20, -0.25, 0.58, 0.30, "Fabric_OatmealBoucle", "02_Living_Dining", bevel_m=0.04)
        self.add_box("Armchair_ArmR", 0.25, 0.32, -0.20, 0.39, 0.58, 0.30, "Fabric_OatmealBoucle", "02_Living_Dining", bevel_m=0.04)
        self.place_group_pose(g, 1.38, 0.0, 0.68, 0.0)

        # 7. `Item__src_living_dining` (1.36 x 1.42 x 0.82) -> Pose 1: (4.84, 1.32, 0)
        g = self.get_group("Item__src_living_dining", "02_Living_Dining")
        self.set_parent_group(g)
        self.add_box("Dining_Table_Top", -0.44, 0.72, -0.68, 0.44, 0.76, 0.68, "Glass_SmokedTable", "02_Living_Dining", bevel_m=0.008)
        self.add_box("Dining_Table_Apron", -0.36, 0.67, -0.58, 0.36, 0.72, 0.58, "Wood_Walnut", "02_Living_Dining", bevel_m=0.006)
        for lx_s in (-0.34, 0.34):
            for lz_s in (-0.56, 0.56):
                self.add_cylinder(f"Dining_Leg_{lx_s}_{lz_s}", lx_s, 0.0, lz_s, 0.024, 0.70, "Wood_Walnut", "02_Living_Dining", segments=12)
        for ch_i, (cx_o, cz_o, rot_c) in enumerate([(-0.48, -0.34, 270.0), (-0.48, 0.34, 270.0), (0.48, -0.34, 90.0), (0.48, 0.34, 90.0)]):
            self.add_oriented_box(f"DiningChair_Seat_{ch_i}", cx_o, 0.44, cz_o, 0.42, 0.045, 0.42, rot_c, "Fabric_OatmealBoucle", "02_Living_Dining", bevel_m=0.012)
            self.add_oriented_box(f"DiningChair_Back_{ch_i}", cx_o * 1.30, 0.66, cz_o, 0.44, 0.32, 0.045, rot_c, "Wood_Walnut", "02_Living_Dining", bevel_m=0.012)
            for clx in (-0.17, 0.17):
                for clz in (-0.17, 0.17):
                    self.add_cylinder(f"DiningChair_Leg_{ch_i}_{clx}_{clz}", cx_o + clx, 0.0, cz_o + clz, 0.012, 0.42, "Wood_Walnut", "02_Living_Dining", segments=8)
            self.add_cylinder(f"DiningPlate_{ch_i}", cx_o * 0.46, 0.761, cz_o, 0.125, 0.015, "Ceramic_White", "02_Living_Dining", segments=20)
        self.place_group_pose(g, 4.84, 0.0, 1.32, 0.0)

        # 8. `Item__src_balcony_chair` (0.70 x 0.68 x 0.76) -> Pose 1: (1.22, -0.96, 205)
        g = self.get_group("Item__src_balcony_chair", "07_Balcony_Exterior")
        self.set_parent_group(g)
        self.add_box("Balcony_Chair_Seat", -0.34, 0.12, -0.32, 0.34, 0.40, 0.32, "Fabric_OatmealBoucle", "07_Balcony_Exterior", bevel_m=0.05)
        self.add_box("Balcony_Chair_Back", -0.34, 0.38, -0.33, 0.34, 0.75, -0.14, "Fabric_OatmealBoucle", "07_Balcony_Exterior", bevel_m=0.05)
        for ox in (-0.28, 0.28):
            for oz in (-0.26, 0.26):
                self.add_cylinder(f"Balcony_Chair_Leg_{ox}_{oz}", ox, -0.015, oz, 0.014, 0.14, "Metal_DarkBronze", "07_Balcony_Exterior", segments=8)
        self.place_group_pose(g, 1.22, 0.0, -0.96, 205.0)

        # 9. `Item__src_balcony_table` (0.60 x 0.60 x 0.58) -> Pose 1: (4.56, -0.96, 0)
        g = self.get_group("Item__src_balcony_table", "07_Balcony_Exterior")
        self.set_parent_group(g)
        self.add_cylinder("Balcony_Bistro_Base", 0.0, -0.015, 0.0, 0.20, 0.03, "Metal_DarkBronze", "07_Balcony_Exterior", segments=20)
        self.add_cylinder("Balcony_Bistro_Stem", 0.0, 0.015, 0.0, 0.03, 0.52, "Metal_DarkBronze", "07_Balcony_Exterior", segments=14)
        self.add_cylinder("Balcony_Bistro_Top", 0.0, 0.535, 0.0, 0.29, 0.03, "Stone_Calacatta", "07_Balcony_Exterior", segments=24)
        self.place_group_pose(g, 4.56, 0.0, -0.96, 0.0)

        # 10. `Item__src_bed_queen` (1.72 x 2.08 x 1.14) -> Pose 1: (1.16, 5.34, 270)
        g = self.get_group("Item__src_bed_queen", "04_Primary_Bedroom")
        self.set_parent_group(g)
        self.add_box("Bedroom_Rug", -1.10, 0.001, -0.45, 1.10, 0.012, 1.15, "Rug_WovenCream", "04_Primary_Bedroom", tile_m=1.1, bevel_m=0.004)
        self.add_box("Bed_LED_Halo", -0.80, 0.04, -0.92, 0.80, 0.07, 0.96, "LED_WarmCove", "04_Primary_Bedroom")
        self.add_box("Bed_Platform", -0.85, 0.07, -0.94, 0.85, 0.28, 1.02, "Fabric_OatmealBoucle", "04_Primary_Bedroom", tile_m=0.8, bevel_m=0.035)
        self.add_box("Bed_Headboard", -0.86, 0.28, -1.04, 0.86, 1.14, -0.88, "Fabric_OatmealBoucle", "04_Primary_Bedroom", tile_m=0.6, bevel_m=0.03)
        self.add_box("Bed_Mattress_Duvet", -0.78, 0.28, -0.86, 0.78, 0.54, 0.96, "Fabric_CrispLinen", "04_Primary_Bedroom", tile_m=0.9, bevel_m=0.05)
        self.add_box("Bed_WaffleThrow", -0.80, 0.29, 0.38, 0.80, 0.565, 0.98, "Fabric_TerracottaThrow", "04_Primary_Bedroom", tile_m=0.6, bevel_m=0.03)
        for px_o in (-0.38, 0.38):
            self.add_oriented_box(f"Bed_PillowBack_{px_o}", px_o, 0.62, -0.68, 0.62, 0.22, 0.22, 0.0, "Fabric_CrispLinen", "04_Primary_Bedroom", bevel_m=0.045)
            self.add_oriented_box(f"Bed_PillowFront_{px_o}", px_o, 0.59, -0.48, 0.50, 0.16, 0.18, 0.0, "Fabric_TerracottaThrow", "04_Primary_Bedroom", bevel_m=0.04)
        self.place_group_pose(g, 1.16, 0.0, 5.34, 270.0)

        # 11 & 12. `Item__src_bed_nightstand_l` & `Item__src_bed_nightstand_r` (0.46 x 0.42 x 0.64)
        for suffix, pz_1 in [("l", 4.16), ("r", 6.52)]:
            g = self.get_group(f"Item__src_bed_nightstand_{suffix}", "04_Primary_Bedroom")
            self.set_parent_group(g)
            self.add_box(f"Nightstand_Body_{suffix}", -0.22, 0.18, -0.20, 0.22, 0.50, 0.20, "Wood_Walnut", "04_Primary_Bedroom", bevel_m=0.008)
            for ox in (-0.18, 0.18):
                for oz in (-0.16, 0.16):
                    self.add_cylinder(f"Nightstand_Leg_{suffix}_{ox}_{oz}", ox, 0.0, oz, 0.012, 0.18, "Metal_DarkBronze", "04_Primary_Bedroom", segments=8)
            self.add_cylinder(f"MoonLamp_Base_{suffix}", 0.0, 0.50, 0.0, 0.06, 0.025, "Metal_WarmBrass", "04_Primary_Bedroom", segments=16)
            self.add_sphere(f"MoonLamp_Globe_{suffix}", 0.0, 0.62, 0.0, 0.105, "Glass_AmberGlobe", "04_Primary_Bedroom")
            self.place_group_pose(g, 0.33, 0.0, pz_1, 270.0)

        # 13. `Item__src_bed_bench` (1.34 x 0.42 x 0.46) -> Pose 1: (2.86, 6.64, 180)
        g = self.get_group("Item__src_bed_bench", "04_Primary_Bedroom")
        self.set_parent_group(g)
        self.add_box("BedBench_Frame", -0.65, 0.14, -0.19, 0.65, 0.34, 0.19, "Wood_Walnut", "04_Primary_Bedroom", bevel_m=0.008)
        self.add_box("BedBench_Cushion", -0.66, 0.34, -0.20, 0.66, 0.455, 0.20, "Fabric_OatmealBoucle", "04_Primary_Bedroom", bevel_m=0.03)
        for ox in (-0.56, 0.56):
            self.add_box(f"BedBench_Leg_{ox}", ox - 0.025, 0.0, -0.17, ox + 0.025, 0.14, 0.17, "Wood_Walnut", "04_Primary_Bedroom", bevel_m=0.004)
        self.place_group_pose(g, 2.86, 0.0, 6.64, 180.0)

        # 14. `Item__src_bath_stool` (0.58 x 0.36 x 0.46) -> Pose 1: (7.78, 4.10, 90)
        g = self.get_group("Item__src_bath_stool", "05_Spa_Bathroom")
        self.set_parent_group(g)
        self.add_box("SpaBench_Top", -0.28, 0.38, -0.17, 0.28, 0.42, 0.17, "Wood_Walnut", "05_Spa_Bathroom", bevel_m=0.006)
        self.add_box("SpaBench_Shelf", -0.24, 0.12, -0.14, 0.24, 0.15, 0.14, "Wood_Walnut", "05_Spa_Bathroom", bevel_m=0.004)
        for ox in (-0.25, 0.25):
            self.add_box(f"SpaBench_Side_{ox}", ox - 0.02, 0.0, -0.16, ox + 0.02, 0.38, 0.16, "Wood_Walnut", "05_Spa_Bathroom", bevel_m=0.004)
        self.add_box("SpaTowel_1", -0.16, 0.42, -0.11, 0.16, 0.46, 0.11, "Fabric_CrispLinen", "05_Spa_Bathroom", bevel_m=0.015)
        self.place_group_pose(g, 7.78, 0.0, 4.10, 90.0)

        # 15. `Item__src_office_bookshelf` (1.24 x 0.36 x 1.92) -> Pose 1: (8.44, 5.66, 270)
        g = self.get_group("Item__src_office_bookshelf", "06_Office_Study")
        self.set_parent_group(g)
        self.add_box("OfficeBook_Back", -0.60, 0.08, -0.17, 0.60, 1.90, -0.145, "Wood_Walnut", "06_Office_Study", tile_m=1.0)
        for ox in (-0.60, 0.0, 0.60):
            self.add_box(f"OfficeBook_PostF_{ox}", ox - 0.014, 0.0, 0.14, ox + 0.014, 1.92, 0.168, "Metal_WarmBrass", "06_Office_Study", bevel_m=0.003)
            self.add_box(f"OfficeBook_PostB_{ox}", ox - 0.014, 0.0, -0.168, ox + 0.014, 1.92, -0.14, "Metal_WarmBrass", "06_Office_Study", bevel_m=0.003)
        self.add_box("OfficeBook_Credenza", -0.61, 0.08, -0.16, 0.61, 0.52, 0.16, "Wood_Walnut", "06_Office_Study", bevel_m=0.006)
        for sh_i, sy in enumerate([0.52, 0.96, 1.40, 1.88]):
            self.add_box(f"OfficeBook_Shelf_{sh_i}", -0.61, sy, -0.16, 0.61, sy + 0.028, 0.16, "Wood_Walnut", "06_Office_Study", bevel_m=0.004)
            if sh_i < 3:
                self.add_box(f"OfficeBook_LED_{sh_i}", -0.54, sy + 0.41, -0.11, 0.54, sy + 0.425, -0.08, "LED_WarmCove", "06_Office_Study")
                self.add_box(f"OfficeBook_Stack_{sh_i}", -0.44 + sh_i * 0.18, sy + 0.028, -0.08, -0.18 + sh_i * 0.18, sy + 0.24, 0.08, "Fabric_CrispLinen", "06_Office_Study", bevel_m=0.004)
                self.add_cylinder(f"OfficeBook_Obj_{sh_i}", 0.32 - sh_i * 0.14, sy + 0.028, 0.0, 0.055, 0.17, "Ceramic_White" if sh_i % 2 == 0 else "Metal_WarmBrass", "06_Office_Study", segments=16)
        self.place_group_pose(g, 8.44, 0.0, 5.66, 270.0)

        # 16. `Item__src_office_desk` (1.46 x 1.04 x 1.14) -> Pose 1: (10.22, 6.02, 180)
        g = self.get_group("Item__src_office_desk", "06_Office_Study")
        self.set_parent_group(g)
        self.add_box("Office_Rug", -0.92, 0.001, -0.68, 0.92, 0.012, 0.68, "Rug_WovenCream", "06_Office_Study", tile_m=1.0, bevel_m=0.004)
        # Sculpted Executive Walnut Desk Slab (`lz in [-0.08, +0.50]`)
        self.add_box("OfficeDesk_Top", -0.73, 0.72, -0.08, 0.73, 0.76, 0.50, "Wood_Walnut", "06_Office_Study", tile_m=1.2, bevel_m=0.008)
        self.add_box("OfficeDesk_Modesty", -0.64, 0.36, 0.44, 0.64, 0.72, 0.48, "Wood_Walnut", "06_Office_Study", bevel_m=0.004)
        self.add_box("OfficeDesk_LegL", -0.70, 0.0, -0.05, -0.64, 0.72, 0.48, "Wood_Walnut", "06_Office_Study", bevel_m=0.005)
        self.add_box("OfficeDesk_LegR", 0.34, 0.08, -0.05, 0.70, 0.72, 0.48, "Wood_Walnut", "06_Office_Study", bevel_m=0.006)
        self.add_box("OfficeDesk_LegR_Plinth", 0.36, 0.0, -0.03, 0.68, 0.08, 0.46, "Metal_DarkBronze", "06_Office_Study")
        # Cognac leather blotter, keyboard, ultrawide curved monitor (facing `-Z` sitter) & brass lamp
        self.add_box("OfficeDesk_Pad", -0.38, 0.76, -0.02, 0.28, 0.768, 0.32, "Leather_CognacPad", "06_Office_Study", bevel_m=0.003)
        self.add_box("OfficeDesk_Keyboard", -0.18, 0.768, 0.04, 0.14, 0.782, 0.16, "Metal_DarkBronze", "06_Office_Study", bevel_m=0.003)
        self.add_cylinder("OfficeMonitor_Stand", -0.02, 0.76, 0.38, 0.09, 0.015, "Metal_WarmBrass", "06_Office_Study", segments=18)
        self.add_box("OfficeMonitor_Neck", -0.035, 0.775, 0.38, -0.005, 0.94, 0.41, "Metal_WarmBrass", "06_Office_Study", bevel_m=0.003)
        self.add_box("OfficeMonitor_Bezel", -0.44, 0.86, 0.35, 0.40, 1.14, 0.375, "Metal_DarkBronze", "06_Office_Study", bevel_m=0.004)
        self.add_uv_quad("OfficeMonitor_Screen", [(0.385, 0.875, 0.348), (-0.425, 0.875, 0.348), (-0.425, 1.125, 0.348), (0.385, 1.125, 0.348)], "Screen_StudioCode", "06_Office_Study")
        self.add_cylinder("OfficeLamp_Base", -0.56, 0.76, 0.34, 0.075, 0.02, "Metal_WarmBrass", "06_Office_Study", segments=16)
        self.add_cylinder("OfficeLamp_Stem", -0.56, 0.78, 0.34, 0.01, 0.34, "Metal_WarmBrass", "06_Office_Study", segments=10)
        self.add_sphere("OfficeLamp_Globe", -0.48, 1.10, 0.28, 0.065, "Glass_AmberGlobe", "06_Office_Study")

        # Ergonomic Boucle & Walnut Swivel Task Chair (`lz = -0.26` facing `+Z` desk)
        self.add_cylinder("OfficeChair_Hub", -0.04, 0.04, -0.28, 0.025, 0.38, "Metal_DarkBronze", "06_Office_Study", segments=10)
        for sp_i in range(5):
            ang = 2.0 * math.pi * (sp_i / 5.0)
            self.add_oriented_box(f"OfficeChair_Star_{sp_i}", -0.04 + 0.13 * math.sin(ang), 0.04, -0.28 + 0.13 * math.cos(ang), 0.03, 0.025, 0.26, -math.degrees(ang), "Metal_DarkBronze", "06_Office_Study", bevel_m=0.004)
        self.add_box("OfficeChair_Seat", -0.28, 0.42, -0.48, 0.20, 0.49, -0.06, "Fabric_OatmealBoucle", "06_Office_Study", bevel_m=0.03)
        self.add_box("OfficeChair_BackShell", -0.27, 0.48, -0.51, 0.19, 0.90, -0.44, "Wood_Walnut", "06_Office_Study", bevel_m=0.025)
        self.add_box("OfficeChair_BackPad", -0.25, 0.50, -0.45, 0.17, 0.88, -0.40, "Fabric_OatmealBoucle", "06_Office_Study", bevel_m=0.025)
        self.place_group_pose(g, 10.22, 0.0, 6.02, 180.0)

        # 17. `Item__src_office_lounge` (0.86 x 0.76 x 0.80) -> Pose 1: (10.86, 4.30, 42)
        g = self.get_group("Item__src_office_lounge", "06_Office_Study")
        self.set_parent_group(g)
        self.add_box("OfficeLounge_Seat", -0.42, 0.12, -0.34, 0.22, 0.40, 0.34, "Fabric_OliveVelvet", "06_Office_Study", bevel_m=0.05)
        self.add_box("OfficeLounge_Back", -0.42, 0.38, -0.36, 0.22, 0.79, -0.16, "Fabric_OliveVelvet", "06_Office_Study", bevel_m=0.05)
        for ox in (-0.34, 0.14):
            for oz in (-0.26, 0.26):
                self.add_cylinder(f"OfficeLounge_Leg_{ox}_{oz}", ox, 0.0, oz, 0.014, 0.12, "Wood_Walnut", "06_Office_Study", segments=8)
        self.add_cylinder("OfficeSideTable_Base", 0.31, 0.0, 0.06, 0.11, 0.02, "Metal_WarmBrass", "06_Office_Study", segments=16)
        self.add_cylinder("OfficeSideTable_Stem", 0.31, 0.02, 0.06, 0.014, 0.48, "Metal_WarmBrass", "06_Office_Study", segments=10)
        self.add_cylinder("OfficeSideTable_Top", 0.31, 0.50, 0.06, 0.13, 0.02, "Stone_Calacatta", "06_Office_Study", segments=20)
        self.place_group_pose(g, 10.86, 0.0, 4.30, 42.0)

        self.set_parent_group(None)

    # -----------------------------------------------------------------------
    # 4. Balcony Deck, Railings & Panoramic Exterior Vistas
    # -----------------------------------------------------------------------

    def build_balcony_and_exterior_vista(self) -> None:
        self.set_parent_group(self.get_group("Arch__balcony", "07_Balcony_Exterior"))
        bx0, bx1, bz0, bz1 = 0.36, 5.46, -1.65, -0.09
        self.add_box("Balcony_Slab", bx0 - 0.06, -0.18, bz0 - 0.06, bx1 + 0.06, -0.015, bz1 + 0.05, "Wood_BalconyTeak", "07_Balcony_Exterior", tile_m=1.4)

        rail_h = 1.08
        self.add_box("Balcony_RailTopN", bx0, rail_h - 0.04, bz0, bx1, rail_h, bz0 + 0.05, "Metal_DarkBronze", "07_Balcony_Exterior", collider_kind="railing")
        self.add_box("Balcony_RailTopW", bx0, rail_h - 0.04, bz0, bx0 + 0.05, rail_h, bz1, "Metal_DarkBronze", "07_Balcony_Exterior", collider_kind="railing")
        self.add_box("Balcony_RailTopE", bx1 - 0.05, rail_h - 0.04, bz0, bx1, rail_h, bz1, "Metal_DarkBronze", "07_Balcony_Exterior", collider_kind="railing")
        self.add_box("Balcony_GlassN", bx0 + 0.04, 0.06, bz0 + 0.015, bx1 - 0.04, rail_h - 0.04, bz0 + 0.03, "Glass_Architectural", "07_Balcony_Exterior")
        for i_post in range(11):
            p_x = bx0 + (bx1 - bx0) * (i_post / 10.0)
            self.add_box(f"Balcony_Post_{p_x:.2f}", p_x - 0.018, -0.015, bz0, p_x + 0.018, rail_h, bz0 + 0.045, "Metal_DarkBronze", "07_Balcony_Exterior")

        for pl_x in (0.66, 5.16):
            self.add_cylinder(f"Balcony_Pot_{pl_x:.2f}", pl_x, -0.015, -1.38, 0.18, 0.36, "Fabric_TerracottaThrow", "07_Balcony_Exterior", segments=16)
            self.add_olive_bouquet(f"Balcony_Plant_{pl_x:.2f}", pl_x, 0.34, -1.38, 1.45, "07_Balcony_Exterior")

        self.set_parent_group(self.get_group("Arch__vistas", "07_Balcony_Exterior"))
        self.add_uv_quad("Vista_North_River", [(-9.5, -2.8, -7.6), (20.5, -2.8, -7.6), (20.5, 8.2, -7.6), (-9.5, 8.2, -7.6)], "Ref_ExteriorVista", "07_Balcony_Exterior")
        self.add_uv_quad("Vista_West_Wing", [(-9.5, -2.8, 14.5), (-9.5, -2.8, -7.6), (-9.5, 8.2, -7.6), (-9.5, 8.2, 14.5)], "Ref_ExteriorVista", "07_Balcony_Exterior")
        self.add_uv_quad("Vista_South_Hills", [(20.5, -2.8, 13.8), (-9.5, -2.8, 13.8), (-9.5, 8.0, 13.8), (20.5, 8.0, 13.8)], "Ref_ExteriorVista", "07_Balcony_Exterior")
        self.add_uv_quad("Vista_East_Greenery", [(20.5, -2.6, -7.6), (20.5, -2.6, 13.8), (20.5, 8.0, 13.8), (20.5, 8.0, -7.6)], "Ref_ExteriorVista", "07_Balcony_Exterior")
        self.set_parent_group(None)

    # -----------------------------------------------------------------------
    # 5. Blender Cycles Lighting, Office Reference Photos & GLB Export
    # -----------------------------------------------------------------------

    def setup_lighting_and_cameras(self) -> list[tuple[str, bpy.types.Object]]:
        col = "08_Lighting_Cameras"
        scene = bpy.context.scene

        world = bpy.data.worlds.new("ApartmentWorld")
        scene.world = world
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg is not None:
            bg.inputs[0].default_value = (0.78, 0.86, 0.96, 1.0)
            bg.inputs[1].default_value = 1.15

        sun_data = bpy.data.lights.new(name="Sun_Daylight", type="SUN")
        sun_data.energy = 3.6
        sun_data.color = (1.0, 0.95, 0.87)
        sun_data.angle = math.radians(4.5)
        sun_obj = bpy.data.objects.new("Sun_Daylight", sun_data)
        sun_obj.location = hw_pt(3.4, 7.8, -5.2)
        sun_obj.rotation_euler = Euler((math.radians(52.0), math.radians(-12.0), math.radians(162.0)), "XYZ")
        self._link_obj(sun_obj, col)

        room_lights = [
            ("Light_Living", 2.60, 2.48, 1.95, 140.0, (1.0, 0.91, 0.78), 2.6),
            ("Light_Kitchen", 7.45, 2.46, 1.90, 120.0, (1.0, 0.90, 0.76), 2.2),
            ("Light_Bedroom", 2.25, 2.48, 5.35, 110.0, (1.0, 0.88, 0.74), 2.4),
            ("Light_Bathroom", 6.50, 2.48, 5.30, 95.0, (0.98, 0.95, 0.90), 2.0),
            ("Light_Office", 9.88, 2.48, 5.32, 115.0, (1.0, 0.91, 0.78), 2.2),
        ]
        for lname, lx, ly, lz, energy, color, sz in room_lights:
            ldata = bpy.data.lights.new(name=lname, type="AREA")
            ldata.energy = energy
            ldata.color = color
            ldata.size = sz
            lobj = bpy.data.objects.new(lname, ldata)
            lobj.location = hw_pt(lx, ly, lz)
            self._link_obj(lobj, col)

        cam_specs = [
            ("O1", (8.68, 1.58, 4.05), (10.55, 1.12, 5.95), 24.0),
            ("O2", (11.15, 1.52, 6.55), (8.95, 1.24, 4.65), 24.0),
        ]
        cams: list[tuple[str, bpy.types.Object]] = []
        for cname, eye_hw, tgt_hw, lens_mm in cam_specs:
            cdata = bpy.data.cameras.new(cname)
            cdata.lens = lens_mm
            cobj = bpy.data.objects.new(cname, cdata)
            eye_v = Vector(hw_pt(*eye_hw))
            tgt_v = Vector(hw_pt(*tgt_hw))
            cobj.location = eye_v
            cobj.rotation_euler = (tgt_v - eye_v).to_track_quat("-Z", "Y").to_euler()
            self._link_obj(cobj, col)
            cams.append((cname, cobj))
        scene.camera = cams[0][1]
        return cams

    def render_office_samples(self, cams: list[tuple[str, bpy.types.Object]]) -> None:
        SAMPLES_IMG_DIR.mkdir(parents=True, exist_ok=True)
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 40
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

        scene.render.resolution_x = 800
        scene.render.resolution_y = 520
        scene.render.image_settings.file_format = "JPEG"
        scene.render.image_settings.quality = 86
        try:
            scene.view_settings.view_transform = "Filmic"
        except Exception:
            scene.view_settings.view_transform = "Standard"

        for cname, cobj in cams:
            scene.camera = cobj
            scene.render.filepath = str(SAMPLES_IMG_DIR / f"{cname}.jpg")
            bpy.ops.render.render(write_still=True)

    def save_blend_and_export_glb(self) -> None:
        BLEND_OUT.parent.mkdir(parents=True, exist_ok=True)
        GLB_OUT.parent.mkdir(parents=True, exist_ok=True)

        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_OUT), compress=True)

        # Select both MESH and EMPTY nodes so the named `Arch__*`, `Ceil__*`, `Fix__*`, and `Item__*` hierarchy exports intact!
        for obj in bpy.data.objects:
            obj.select_set(obj.type in {"MESH", "EMPTY"})

        rna_props = {p.identifier for p in bpy.ops.export_scene.gltf.get_rna_type().properties}
        export_kwargs: dict[str, Any] = {
            "filepath": str(GLB_OUT),
            "export_format": "GLB",
            "use_selection": True,
            "export_yup": True,
            "export_apply": False,
        }
        if "export_image_format" in rna_props:
            export_kwargs["export_image_format"] = "WEBP"
        if "export_image_quality" in rna_props:
            export_kwargs["export_image_quality"] = 84

        bpy.ops.export_scene.gltf(**export_kwargs)


def main() -> None:
    builder = SceneBuilder()
    builder.build_shell_and_floors()
    builder.build_fixed_joinery_and_art()
    builder.build_movable_furniture_items()
    builder.build_balcony_and_exterior_vista()
    cams = builder.setup_lighting_and_cameras()
    builder.render_office_samples(cams)
    builder.save_blend_and_export_glb()
    print(
        f"APARTMENT_6ZONE_COMPLETE | blend={BLEND_OUT.stat().st_size // 1024}KB | "
        f"glb={GLB_OUT.stat().st_size // 1024}KB | empties={len(builder.empties)}"
    )


if __name__ == "__main__":
    main()
