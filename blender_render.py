"""Render the Homeweave Studio scene in Blender (`blender_render.py`).

Run headless by the server:
    blender -b --factory-startup -P blender_render.py -- scene.glb out.png camera.json

The page exports its exact Three.js scene as glTF, so Blender renders the same rooms, the same
furniture and the same positions; nothing is re-estimated here. Blender only adds what a
rasteriser cannot: path-traced light from the glow strips, soft shadows and bounce light.
"""

import json
import math
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
glb_path, out_path, cam_path = argv[0], argv[1], argv[2]
cfg = json.load(open(cam_path, encoding="utf-8"))

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb_path)
scene = bpy.context.scene


def to_blender(p):
    """three.js (x, y up, z) -> Blender (x, -z, y up)."""
    return Vector((p[0], -p[2], p[1]))


# Lights: keep the page's warm room lights, but give them physical strength and soft shadows.
for ob in list(scene.objects):
    if ob.type == "LIGHT":
        ob.data.energy = cfg.get("room_light_w", 22.0)
        if hasattr(ob.data, "shadow_soft_size"):
            ob.data.shadow_soft_size = 0.35

sun_data = bpy.data.lights.new("sun", "SUN")
sun_data.energy = cfg.get("sun_strength", 1.1)
sun_data.angle = math.radians(12)
sun_data.color = (1.0, 0.9, 0.76)
sun = bpy.data.objects.new("sun", sun_data)
sun.rotation_euler = (math.radians(42), 0.0, math.radians(28))
scene.collection.objects.link(sun)

world = bpy.data.worlds.new("world")
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[0].default_value = (0.045, 0.045, 0.055, 1.0)
    bg.inputs[1].default_value = cfg.get("ambient", 0.9)
scene.world = world

# Walls are single-sided in the page so you can look into a room from outside; do the same here.
for mat in bpy.data.materials:
    if not mat.use_nodes or not mat.name.startswith("wall"):
        continue
    nt = mat.node_tree
    out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if out is None or not out.inputs[0].links:
        continue
    surface = out.inputs[0].links[0].from_socket
    mix = nt.nodes.new("ShaderNodeMixShader")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    nt.links.new(geo.outputs["Backfacing"], mix.inputs[0])
    nt.links.new(surface, mix.inputs[1])
    nt.links.new(clear.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs[0])

# Camera: the same viewpoint the user has in the page.
cam_data = bpy.data.cameras.new("cam")
cam_data.sensor_fit = "VERTICAL"
cam_data.angle_y = math.radians(cfg["fov"])
cam_data.clip_start, cam_data.clip_end = 0.05, 300.0
cam = bpy.data.objects.new("cam", cam_data)
cam.location = to_blender(cfg["position"])
direction = to_blender(cfg["target"]) - cam.location
cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(cam)
scene.camera = cam

scene.render.resolution_x, scene.render.resolution_y = cfg.get("width", 1280), cfg.get("height", 960)
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = out_path

engine = cfg.get("engine", "CYCLES")
if engine == "CYCLES":
    scene.render.engine = "CYCLES"
    scene.cycles.samples = cfg.get("samples", 48)
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    try:
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.05
    except Exception:
        pass
else:
    for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = name
            break
        except TypeError:
            continue
try:
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
except Exception:
    pass
scene.view_settings.exposure = cfg.get("exposure", -0.7)

bpy.ops.render.render(write_still=True)
print("HOMEWEAVE_RENDER_OK", out_path)
