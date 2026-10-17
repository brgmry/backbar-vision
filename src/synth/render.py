"""Render synthetic bar-shelf photos of spirit bottles from flat TTB label artwork (Blender, headless).

Each render: one hero bottle (label wrapped on a generic shape chosen from the spirit type) on a shelf, with
distractor bottles around it, lit by a random indoor HDRI, random camera, fill level, glass tint and depth of field.
The output is a crop-like portrait image of the hero bottle, the same shape as a detector crop from a bar photo.

Usage: blender -b -P src/synth/render.py -- <labels.csv> <out_dir> <per_label> <shard> <n_shards>
"""
import csv
import math
import random
import sys
from pathlib import Path

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
LABELS, OUT, PER, SHARD, NSHARDS = args[0], Path(args[1]), int(args[2]), int(args[3]), int(args[4])
OUT.mkdir(parents=True, exist_ok=True)
HDRIS = sorted(Path("data/synth/hdri").glob("*.hdr"))

# profiles: (radius, height) points from base to lip, metres
SHAPES = {
    "round":   [(0.0, 0), (0.040, 0), (0.042, 0.005), (0.042, 0.19), (0.036, 0.215), (0.016, 0.245), (0.014, 0.30), (0.0, 0.30)],
    "tall":    [(0.0, 0), (0.034, 0), (0.036, 0.005), (0.036, 0.24), (0.030, 0.26), (0.013, 0.285), (0.012, 0.33), (0.0, 0.33)],
    "squat":   [(0.0, 0), (0.048, 0), (0.050, 0.006), (0.050, 0.15), (0.044, 0.175), (0.018, 0.195), (0.016, 0.24), (0.0, 0.24)],
    "flask":   [(0.0, 0), (0.044, 0), (0.046, 0.006), (0.046, 0.17), (0.026, 0.215), (0.015, 0.235), (0.014, 0.285), (0.0, 0.285)],
    "square":  None,  # bevelled box, handled separately
}
BODY = {"round": (0.012, 0.185), "tall": (0.015, 0.235), "squat": (0.012, 0.145), "flask": (0.012, 0.165), "square": (0.012, 0.19)}


def shapes_for(category: str) -> list[str]:
    c = category.lower()
    if "vodka" in c:
        return ["tall", "round"]
    if "bourbon" in c or "tennessee" in c:
        return ["square", "round"]
    if "tequila" in c or "mezcal" in c:
        return ["squat", "round"]
    if "cognac" in c or "brandy" in c:
        return ["flask", "squat"]
    if "liqueur" in c or "cream" in c:
        return ["round", "squat"]
    return ["round", "square", "squat"]


def liquid_colour(category: str) -> tuple:
    c = category.lower()
    if any(k in c for k in ("vodka", "gin", "tequila fb", "blanco", "silver")):
        return (0.95, 0.95, 0.95, 1)
    if "coffee" in c or "cream" in c:
        return (0.12, 0.05, 0.02, 1)
    return random.choice([(0.75, 0.38, 0.08, 1), (0.6, 0.25, 0.05, 1), (0.85, 0.55, 0.15, 1)])


def principled(name, **kw):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    for k, v in kw.items():
        if k in p.inputs:
            p.inputs[k].default_value = v
    return m, p


def lathe(name, profile, segments=48, scale=1.0, zmax=None):
    verts, faces = [], []
    prof = [(r * scale, z) for r, z in profile if zmax is None or z <= zmax]
    if zmax is not None:
        prof.append((prof[-1][0], zmax))
        prof.append((0.0, zmax))
    n = len(prof)
    for s in range(segments):
        a = 2 * math.pi * s / segments
        for r, z in prof:
            verts.append((r * math.cos(a), r * math.sin(a), z))
    for s in range(segments):
        for i in range(n - 1):
            a, b = s * n + i, ((s + 1) % segments) * n + i
            faces.append((a, b, b + 1, a + 1))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    for p in me.polygons:
        p.use_smooth = True
    return ob


def label_band(name, radius, z0, z1, arc, img_mat):
    """Partial cylinder just outside the glass, UV-mapped 0..1 across the arc and height."""
    seg = 32
    verts, faces = [], []
    for i in range(seg + 1):
        a = -arc / 2 + arc * i / seg - math.pi / 2  # centred on -Y, facing the camera at azimuth 0
        for z in (z0, z1):
            verts.append((radius * math.cos(a), radius * math.sin(a), z))
    for i in range(seg):
        faces.append((2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    uv = me.uv_layers.new()
    for f in me.polygons:
        for li, vi in zip(f.loop_indices, f.vertices, strict=True):
            uv.data[li].uv = ((vi // 2) / seg, vi % 2)
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(img_mat)
    for p in me.polygons:
        p.use_smooth = True
    return ob


def image_material(path):
    m = bpy.data.materials.new("label")
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(Path(path).resolve()), check_existing=True)
    tex.extension = "CLIP"
    nt.links.new(tex.outputs["Color"], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = random.uniform(0.25, 0.6)
    return m, tex.image


def bottle(label_path, category, x, y, rot, scale=1.0):
    shape = random.choice(shapes_for(category))
    glass, _ = principled("glass", **{"Base Color": random.choice([(1, 1, 1, 1), (0.85, 0.95, 0.85, 1), (0.95, 0.9, 0.8, 1)]),
                                       "Transmission Weight": 1.0, "Roughness": random.uniform(0.0, 0.08), "IOR": 1.5})
    liquid, _ = principled("liquid", **{"Base Color": liquid_colour(category), "Transmission Weight": 1.0,
                                         "Roughness": 0.02, "IOR": 1.36})
    cap, _ = principled("cap", **{"Base Color": random.choice([(0.02, 0.02, 0.02, 1), (0.6, 0.45, 0.1, 1), (0.5, 0.05, 0.05, 1),
                                                               (0.8, 0.8, 0.8, 1)]), "Metallic": random.choice([0, 1]),
                                   "Roughness": 0.3})
    lab, img = image_material(label_path)
    aspect = img.size[0] / max(img.size[1], 1)
    zb0, zb1 = BODY[shape]
    parts = []
    if shape == "square":
        bpy.ops.mesh.primitive_cube_add(size=1)
        b = bpy.context.object
        w = 0.085 * scale
        b.scale = (w, w * 0.75, 0.21 * scale)
        b.location = (0, 0, 0.105 * scale)
        bev = b.modifiers.new("bev", "BEVEL")
        bev.width, bev.segments = 0.008, 4
        b.data.materials.append(glass)
        neck = lathe("neck", [(0, 0.21), (0.016, 0.21), (0.014, 0.29), (0, 0.29)], scale=scale)
        neck.data.materials.append(glass)
        fill = random.uniform(0.1, 1.0)
        bpy.ops.mesh.primitive_cube_add(size=1)
        liq = bpy.context.object
        liq.scale = (w * 0.94, w * 0.7, 0.2 * scale * fill)
        liq.location = (0, 0, 0.005 + 0.1 * scale * fill)
        liq.data.materials.append(liquid)
        h = min((zb1 - zb0) * scale, w * 0.9 / aspect)
        bpy.ops.mesh.primitive_plane_add(size=1)
        lp = bpy.context.object
        lp.scale = (w * 0.9, h, 1)
        lp.rotation_euler = (math.pi / 2, 0, 0)
        lp.location = (0, -w * 0.375 - 0.0015, (zb0 * scale + zb1 * scale) / 2)
        lp.data.materials.append(lab)
        parts = [b, neck, liq, lp]
        top = 0.29 * scale
    else:
        prof = SHAPES[shape]
        b = lathe("glass", prof, scale=scale)
        b.data.materials.append(glass)
        fill_z = zb0 + random.uniform(0.05, 1.0) * (prof[-2][1] * 0.8 - zb0)
        liq = lathe("liquid", prof[:-1], scale=0.93 * scale, zmax=fill_z)
        liq.data.materials.append(liquid)
        r = max(p[0] for p in prof) * scale
        arc = random.uniform(1.6, 2.4)
        h = min((zb1 - zb0), r * arc / aspect)
        zc = random.uniform(zb0 + h / 2, zb1 - h / 2) if zb1 - zb0 > h else (zb0 + zb1) / 2
        lb = label_band("label", r + 0.0015, zc - h / 2, zc + h / 2, arc, lab)
        parts = [b, liq, lb]
        top = prof[-1][1]
    c = lathe("cap", [(0, top), (0.017 * scale, top), (0.017 * scale, top + 0.03), (0, top + 0.03)])
    c.data.materials.append(cap)
    parts.append(c)
    root = bpy.data.objects.new("bottle", None)
    bpy.context.collection.objects.link(root)
    for p in parts:
        p.parent = root
    root.location = (x, y, 0)
    root.rotation_euler = (0, 0, rot)
    return root, top + 0.03


def setup_render():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    sc.cycles.device = "GPU"
    sc.cycles.samples = 20
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6
    sc.cycles.transmission_bounces = 6
    sc.render.resolution_x, sc.render.resolution_y = 320, 448
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 90
    sc.view_settings.view_transform = "AgX" if "AgX" in [v.identifier for v in
                                                        type(sc.view_settings).bl_rna.properties["view_transform"].enum_items] else "Filmic"


def scene(row, all_rows, i):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_render()
    sc = bpy.context.scene
    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(random.choice(HDRIS).resolve()))
    mapn, coord = nt.nodes.new("ShaderNodeMapping"), nt.nodes.new("ShaderNodeTexCoord")
    mapn.inputs["Rotation"].default_value = (0, 0, random.uniform(0, 2 * math.pi))
    nt.links.new(coord.outputs["Generated"], mapn.inputs["Vector"])
    nt.links.new(mapn.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], nt.nodes["Background"].inputs["Color"])
    nt.nodes["Background"].inputs["Strength"].default_value = random.uniform(0.4, 1.6)
    # shelf
    bpy.ops.mesh.primitive_plane_add(size=3, location=(0, 0.3, 0))
    shelf, _ = principled("shelf", **{"Base Color": random.choice([(0.25, 0.12, 0.05, 1), (0.05, 0.05, 0.05, 1), (0.6, 0.6, 0.6, 1),
                                                                    (0.4, 0.25, 0.1, 1)]), "Roughness": random.uniform(0.1, 0.7)})
    bpy.context.object.data.materials.append(shelf)
    if random.random() < 0.5:  # back-bar mirror or wall
        bpy.ops.mesh.primitive_plane_add(size=3, location=(0, random.uniform(0.25, 0.6), 1), rotation=(math.pi / 2, 0, 0))
        wall, _ = principled("wall", **{"Base Color": (random.random() * 0.3,) * 3 + (1,), "Metallic": random.choice([0, 1]),
                                         "Roughness": random.uniform(0.02, 0.5)})
        bpy.context.object.data.materials.append(wall)
    # hero + distractors
    hero, top = bottle(row["label"], row["category"], 0, 0, random.uniform(-0.6, 0.6))
    others = [r for r in all_rows if r["brand_key"] != row["brand_key"]]
    for _ in range(random.randint(2, 6)):
        o = random.choice(others)
        side = random.choice([-1, 1])
        bottle(o["label"], o["category"], side * random.uniform(0.09, 0.3), random.uniform(-0.05, 0.25),
               random.uniform(-math.pi, math.pi), scale=random.uniform(0.85, 1.15))
    # camera framed on the hero
    outs = []
    for v in range(VIEWS):
        outs.append(shoot(sc, row, top, i, v, nt, mapn))
    return outs


VIEWS = 4


def shoot(sc, row, top, i, v, nt, mapn):
    for o in [o for o in sc.objects if o.name.startswith(("cam", "tgt"))]:
        bpy.data.objects.remove(o)
    mapn.inputs["Rotation"].default_value = (0, 0, random.uniform(0, 2 * math.pi))
    nt.nodes["Background"].inputs["Strength"].default_value = random.uniform(0.4, 1.6)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = random.uniform(35, 85)
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    az = random.gauss(0, 0.35)
    el = random.uniform(-0.05, 0.3)
    fov_h = 2 * math.atan(cam_data.sensor_width / 2 / cam_data.lens) * 448 / 320
    dist = (top * random.uniform(1.05, 1.35)) / 2 / math.tan(fov_h / 2)
    tz = top / 2
    cam.location = (dist * math.sin(az) * math.cos(el), -dist * math.cos(az) * math.cos(el), tz + dist * math.sin(el))
    tgt = bpy.data.objects.new("tgt", None)
    sc.collection.objects.link(tgt)
    tgt.location = (random.gauss(0, 0.005), 0, tz)
    con = cam.constraints.new("TRACK_TO")
    con.target, con.track_axis, con.up_axis = tgt, "TRACK_NEGATIVE_Z", "UP_Y"
    if random.random() < 0.6:
        cam_data.dof.use_dof = True
        cam_data.dof.focus_object = tgt
        cam_data.dof.aperture_fstop = random.uniform(1.8, 8)
    out = OUT / f"{row['brand_key'].replace(' ', '_')}__{i:05d}_{v}.jpg"
    sc.render.filepath = str(out.resolve())
    bpy.ops.render.render(write_still=True)
    return out


with open(LABELS, newline="") as f:
    rows = list(csv.DictReader(f))
jobs = [r for r in rows for _ in range(PER)]
random.seed(SHARD)
for j, r in enumerate(jobs):
    if j % NSHARDS != SHARD:
        continue
    out = OUT / f"{r['brand_key'].replace(' ', '_')}__{j:05d}_{VIEWS - 1}.jpg"
    if out.exists():
        continue
    random.seed(j)
    scene(r, rows, j)
    print(f"RENDERED {j}/{len(jobs)} {out.name}", flush=True)
