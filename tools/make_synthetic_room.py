"""Render a synthetic before/after room walkthrough with Blender (Cycles, CPU).

Usage:
  blender -b -P tools/make_synthetic_room.py -- --variant before --out data/synth01 [--frames 120]

Writes <out>/<variant>/frames/*.png, <out>/<variant>.mp4 (via ffmpeg, separately),
<out>/<variant>_cameras.json and <out>/ground_truth.json.
"""
import json
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
args = dict(zip(argv[::2], argv[1::2]))
VARIANT = args.get("--variant", "before")
OUT = os.path.abspath(args.get("--out", "data/synth01"))
N_FRAMES = int(args.get("--frames", "120"))
RES = (int(args.get("--w", "640")), int(args.get("--h", "360")))
SAMPLES = int(args.get("--samples", "16"))

ROOM = (5.0, 4.0, 2.6)  # x, y, z extents; room spans [-x/2, x/2] x [-y/2, y/2] x [0, z]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def material(name, base, kind="noise", scale=8.0, base2=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.8
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*base, 1)
    c2 = base2 or tuple(min(1.0, c * 1.8 + 0.08) for c in base)
    ramp.color_ramp.elements[1].color = (*c2, 1)
    coord = nt.nodes.new("ShaderNodeTexCoord")
    if kind == "noise":
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = scale
        tex.inputs["Detail"].default_value = 12.0
    elif kind == "voronoi":
        tex = nt.nodes.new("ShaderNodeTexVoronoi")
        tex.inputs["Scale"].default_value = scale
    elif kind == "brick":
        tex = nt.nodes.new("ShaderNodeTexBrick")
        tex.inputs["Scale"].default_value = scale
    elif kind == "checker":
        tex = nt.nodes.new("ShaderNodeTexChecker")
        tex.inputs["Scale"].default_value = scale
    elif kind == "wave":
        tex = nt.nodes.new("ShaderNodeTexWave")
        tex.inputs["Scale"].default_value = scale
        tex.inputs["Distortion"].default_value = 6.0
        tex.inputs["Detail"].default_value = 8.0
    nt.links.new(coord.outputs["Object"], tex.inputs["Vector"])
    out = tex.outputs["Fac"] if "Fac" in tex.outputs else tex.outputs[0]
    if kind == "voronoi":
        out = tex.outputs["Distance"]
    nt.links.new(out, ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def box(name, center, size, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    o = bpy.context.object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    o.data.materials.append(mat)
    return o


def cylinder(name, center, radius, depth, mat):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=center, vertices=32)
    o = bpy.context.object
    o.name = name
    o.data.materials.append(mat)
    return o


def join(name, objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    o = bpy.context.object
    o.name = name
    return o


# ---------------- static room ----------------
X, Y, Z = ROOM
m_floor = material("floor", (0.25, 0.14, 0.07), "wave", 3.0)
m_wall_a = material("wall_a", (0.35, 0.32, 0.26), "noise", 9.0, (0.85, 0.8, 0.7))
m_wall_b = material("wall_b", (0.35, 0.42, 0.5), "brick", 4.0, (0.7, 0.75, 0.8))
m_ceil = material("ceil", (0.8, 0.8, 0.78), "noise", 3.0, (0.95, 0.95, 0.93))
box("floor", (0, 0, -0.05), (X, Y, 0.1), m_floor)
box("ceiling", (0, 0, Z + 0.05), (X, Y, 0.1), m_ceil)
box("wall_n", (0, Y / 2 + 0.05, Z / 2), (X, 0.1, Z), m_wall_a)
box("wall_s", (0, -Y / 2 - 0.05, Z / 2), (X, 0.1, Z), m_wall_b)
box("wall_e", (X / 2 + 0.05, 0, Z / 2), (0.1, Y, Z), m_wall_b)
box("wall_w", (-X / 2 - 0.05, 0, Z / 2), (0.1, Y, Z), m_wall_a)

# posters / door / window frames for extra texture
box("poster1", (-1.0, Y / 2 - 0.01, 1.6), (0.8, 0.02, 0.6), material("poster1", (0.7, 0.1, 0.1), "voronoi", 10.0, (0.95, 0.85, 0.2)))
box("poster2", (X / 2 - 0.01, 0.8, 1.5), (0.02, 0.7, 0.9), material("poster2", (0.1, 0.3, 0.6), "checker", 6.0, (0.9, 0.9, 0.9)))
box("door", (-X / 2 + 0.01, -1.0, 1.0), (0.04, 0.9, 2.0), material("door", (0.35, 0.2, 0.1), "wave", 2.0))
box("rug", (0.3, -0.2, 0.005), (2.2, 1.5, 0.01), material("rug", (0.5, 0.1, 0.25), "voronoi", 14.0, (0.9, 0.7, 0.4)))

# table (static)
m_wood = material("wood", (0.3, 0.18, 0.08), "wave", 4.0)
legs = [box(f"tleg{i}", (0.6 + dx, 1.2 + dy, 0.37), (0.06, 0.06, 0.74), m_wood) for i, (dx, dy) in enumerate([(-0.65, -0.35), (0.65, -0.35), (-0.65, 0.35), (0.65, 0.35)])]
top = box("ttop", (0.6, 1.2, 0.76), (1.5, 0.9, 0.05), m_wood)
join("table", [top] + legs)

# bookshelf (static)
m_shelf = material("shelf", (0.2, 0.2, 0.22), "noise", 10.0, (0.5, 0.45, 0.4))
parts = [box("sh_frame", (2.25, -1.2, 0.9), (0.4, 1.0, 1.8), m_shelf)]
for i, zz in enumerate([0.4, 0.85, 1.3]):
    parts.append(box(f"books{i}", (2.03, -1.2, zz + 0.15), (0.06, 0.9, 0.3), material(f"books{i}", (0.1 + 0.25 * i, 0.3, 0.6 - 0.15 * i), "checker", 30.0)))
join("bookshelf", parts)

# plant (static)
cylinder("pot", (-2.1, 1.6, 0.2), 0.18, 0.4, material("pot", (0.6, 0.3, 0.15), "noise", 20.0))
bpy.ops.mesh.primitive_ico_sphere_add(radius=0.35, location=(-2.1, 1.6, 0.75), subdivisions=3)
bpy.context.object.data.materials.append(material("leaves", (0.05, 0.3, 0.05), "voronoi", 25.0, (0.3, 0.7, 0.2)))

# ---------------- changing objects ----------------
m_chair = material("chair", (0.1, 0.25, 0.55), "noise", 12.0, (0.3, 0.55, 0.9))


def chair(center, yaw_deg):
    cx, cy = center
    p = [box("seat", (0, 0, 0.45), (0.45, 0.45, 0.05), m_chair), box("back", (0, 0.2, 0.75), (0.45, 0.05, 0.55), m_chair)]
    for dx, dy in [(-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2), (0.2, 0.2)]:
        p.append(box("cleg", (dx, dy, 0.22), (0.04, 0.04, 0.44), m_chair))
    o = join("chair", p)
    o.location = (cx, cy, 0)
    o.rotation_euler = (0, 0, math.radians(yaw_deg))
    return o


GT = {
    "units": "meters",
    "up_axis": "+z",
    "changes": [
        {"type": "moved", "label": "blue chair", "before": {"center": [-0.4, 1.0, 0.45], "yaw_deg": 0}, "after": {"center": [-0.9, -0.9, 0.45], "yaw_deg": 60}},
        {"type": "removed", "label": "green trash bin", "before": {"center": [1.6, 0.2, 0.25], "size": [0.4, 0.4, 0.5]}},
        {"type": "added", "label": "cardboard box", "after": {"center": [0.6, -0.3, 0.2], "size": [0.6, 0.45, 0.4]}},
        {"type": "added", "label": "orange suitcase", "after": {"center": [-1.6, 0.3, 0.35], "size": [0.45, 0.25, 0.7]}},
    ],
}

if VARIANT == "before":
    chair((-0.4, 1.0), 0)
    cylinder("bin", (1.6, 0.2, 0.25), 0.2, 0.5, material("bin", (0.05, 0.4, 0.15), "noise", 15.0))
else:
    chair((-0.9, -0.9), 60)
    box("cardboard_box", (0.6, -0.3, 0.2), (0.6, 0.45, 0.4), material("cardboard", (0.55, 0.38, 0.2), "noise", 18.0, (0.75, 0.55, 0.3)))
    box("suitcase", (-1.6, 0.3, 0.35), (0.45, 0.25, 0.7), material("suitcase", (0.9, 0.35, 0.05), "voronoi", 12.0, (1.0, 0.6, 0.2)))

# ---------------- lights ----------------
bpy.ops.object.light_add(type="AREA", location=(0, 0, Z - 0.05))
bpy.context.object.data.energy = 150
bpy.context.object.data.size = 2.5
bpy.ops.object.light_add(type="POINT", location=(-1.5, -1.2, 2.0))
bpy.context.object.data.energy = 60
world = bpy.data.worlds.new("w")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.3

# ---------------- camera path ----------------
cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 26.0  # ~phone main camera (35mm-equivalent ~26mm)
cam_data.sensor_width = 36.0
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

# Handheld walk: before goes along the south side looking north-ish; after starts east and arcs around.
if VARIANT == "before":
    a0, a1, rad, h, look = math.radians(-150), math.radians(-30), 1.7, 1.45, Vector((0.1, 0.4, 0.6))
else:
    a0, a1, rad, h, look = math.radians(-10), math.radians(-160), 1.5, 1.35, Vector((-0.1, -0.1, 0.5))

cams = []
for f in range(N_FRAMES):
    t = f / max(1, N_FRAMES - 1)
    a = a0 + (a1 - a0) * t
    wob = 0.03 * math.sin(7.0 * t * math.pi)
    pos = Vector((rad * 1.25 * math.cos(a), rad * math.sin(a), h + wob))
    target = look + Vector((0.6 * math.sin(2 * math.pi * t), 0.3 * math.cos(3 * math.pi * t), 0))
    d = (target - pos).normalized()
    cam.location = pos
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cam.keyframe_insert("location", frame=f + 1)
    cam.keyframe_insert("rotation_euler", frame=f + 1)

scene.frame_start, scene.frame_end = 1, N_FRAMES
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.view_settings.view_transform = "Standard"
scene.render.resolution_x, scene.render.resolution_y = RES
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
os.makedirs(f"{OUT}/{VARIANT}/frames", exist_ok=True)
scene.render.filepath = f"{OUT}/{VARIANT}/frames/"

# ground-truth cameras (OpenCV convention: x right, y down, z forward; world->camera)
fx = cam_data.lens / cam_data.sensor_width * RES[0]
K = [[fx, 0, RES[0] / 2], [0, fx, RES[1] / 2], [0, 0, 1]]
flip = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
frames = []
for f in range(N_FRAMES):
    scene.frame_set(f + 1)
    c2w = cam.matrix_world.copy()
    w2c_bl = c2w.inverted()
    R = [[sum(flip[i][k] * w2c_bl[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    tvec = [sum(flip[i][k] * w2c_bl[k][3] for k in range(3)) for i in range(3)]
    frames.append({"frame": f + 1, "R_w2c": R, "t_w2c": tvec})
with open(f"{OUT}/{VARIANT}_cameras.json", "w") as fh:
    json.dump({"K": K, "width": RES[0], "height": RES[1], "frames": frames}, fh)
with open(f"{OUT}/ground_truth.json", "w") as fh:
    json.dump(GT, fh, indent=2)

if args.get("--render", "1") == "1":
    if "--single" in args:
        scene.frame_start = scene.frame_end = int(args["--single"])
    bpy.ops.render.render(animation=True)
