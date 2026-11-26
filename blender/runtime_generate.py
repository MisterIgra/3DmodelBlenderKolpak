"""Runtime Blender model generator for AutoKolpak and METIZLAB.
Run with Blender: blender.exe -b --python runtime_generate.py -- --input request.json --output model.glb
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from geometry import clear_scene, make_hubcap


def args_parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


def material(name, color):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1.0)
    m.metallic = 0.05 if name == "plastic" else 0.82
    m.roughness = 0.29 if name in {"zinc", "stainless"} else 0.23
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Metallic"].default_value = m.metallic
        bsdf.inputs["Roughness"].default_value = m.roughness
    return m



def add_thread(name, radius, pitch, length, z_start, mat, internal=False):
    """Create a visible helical thread ridge around a shaft or inside a bore."""
    pitch = max(float(pitch), 0.0002)
    length = max(float(length), pitch)
    turns = max(1, int(length / pitch * 24))
    curve = bpy.data.curves.new(name + "Curve", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = max(radius * 0.055, 0.00012)
    curve.bevel_resolution = 2
    spline = curve.splines.new("POLY")
    spline.points.add(turns)
    thread_radius = radius * (0.96 if internal else 1.0)
    for i, point in enumerate(spline.points):
        t = i / turns
        angle = 2.0 * math.pi * (length / pitch) * t
        point.co = (thread_radius * math.cos(angle), thread_radius * math.sin(angle), z_start + length * t, 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def add_bevel(obj, width, segments=3):
    modifier = obj.modifiers.new("Machined edge", "BEVEL")
    modifier.width = max(0.00005, width)
    modifier.segments = segments
    modifier.limit_method = "ANGLE"
    return modifier


def cylinder(name, radius, depth, z, mat, vertices=64):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=(0, 0, z))
    o = bpy.context.object
    o.name = name
    o.data.materials.append(mat)
    bevel = o.modifiers.new("Edge bevel", "BEVEL")
    bevel.width = min(radius * 0.08, depth * 0.12)
    bevel.segments = 3
    return o


def ring(name, outer, inner, depth, z, mat):
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=outer, depth=depth, location=(0, 0, z))
    outer_obj = bpy.context.object
    outer_obj.name = name
    outer_obj.data.materials.append(mat)
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=inner, depth=depth * 1.3, location=(0, 0, z))
    cutter = bpy.context.object
    mod = outer_obj.modifiers.new("Bore", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.solver = "EXACT"
    mod.object = cutter
    bpy.context.view_layer.objects.active = outer_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)
    return outer_obj


def make_metiz(part_type, p, material_name):
    colors = {"steel": (0.48, 0.53, 0.60), "stainless": (0.70, 0.74, 0.80), "zinc": (0.62, 0.67, 0.70), "brass": (0.62, 0.34, 0.09), "aluminum": (0.72, 0.75, 0.79), "plastic": (0.12, 0.16, 0.19)}
    mat = material(material_name, colors.get(material_name, colors["steel"]))
    length = float(p.get("length", 40)) / 1000
    if part_type == "angle_bracket":
        width = float(p.get("width", 40)) / 1000
        height = float(p.get("height", 40)) / 1000
        wall = float(p.get("wall_thickness", 4)) / 1000
        hole_r = float(p.get("hole_diameter", 6)) / 1000 / 2
        # Two perpendicular plates joined along the inner corner.
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, height / 2))
        vertical = bpy.context.object
        vertical.name = "Angle vertical plate"
        vertical.dimensions = (wall, width, height)
        vertical.data.materials.append(mat)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bpy.ops.mesh.primitive_cube_add(location=(length / 2, 0, wall / 2))
        horizontal = bpy.context.object
        horizontal.name = "Angle horizontal plate"
        horizontal.dimensions = (length, width, wall)
        horizontal.data.materials.append(mat)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        # Visible through-holes on both legs.
        for x in (length * 0.25, length * 0.75):
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=hole_r, depth=wall * 2, location=(x, 0, wall / 2), rotation=(0, 0, 0))
            cutter = bpy.context.object
            cutter.name = "Angle hole cutter"
            mod = horizontal.modifiers.new("Mounting hole", "BOOLEAN")
            mod.operation = "DIFFERENCE"
            mod.object = cutter
            bpy.context.view_layer.objects.active = horizontal
            bpy.ops.object.modifier_apply(modifier=mod.name)
            bpy.data.objects.remove(cutter, do_unlink=True)
        for y in (-width * 0.25, width * 0.25):
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=hole_r, depth=wall * 2, location=(wall / 2, y, height * 0.65), rotation=(0, math.pi / 2, 0))
            cutter = bpy.context.object
            mod = vertical.modifiers.new("Mounting hole", "BOOLEAN")
            mod.operation = "DIFFERENCE"
            mod.object = cutter
            bpy.context.view_layer.objects.active = vertical
            bpy.ops.object.modifier_apply(modifier=mod.name)
            bpy.data.objects.remove(cutter, do_unlink=True)
        return
    diameter = float(p.get("diameter", 10)) / 1000
    r = diameter / 2
    if part_type in {"washer"} or "bushing" in part_type or part_type in {"spacer", "flanged_bushing", "flanged_spacer", "embedded_bushing"}:
        inner = float(p.get("inner_diameter", p.get("hole_diameter", diameter * 0.55))) / 1000 / 2
        body_length = float(p.get("height", 2) if part_type == "washer" else p.get("length", 40)) / 1000
        ring("Body", r, max(0.0001, inner), body_length, 0, mat)
        add_bevel(bpy.context.object, min(body_length * 0.08, (r - inner) * 0.12))
        if "flange" in part_type or part_type in {"flanged_bushing", "flanged_spacer", "embedded_bushing"}:
            flange = float(p.get("flange_diameter", diameter * 1.45)) / 1000 / 2
            fh = float(p.get("flange_height", 4)) / 1000
            flange_obj = ring("Flange", flange, max(0.0001, inner), fh, body_length / 2 - fh / 2, mat)
            add_bevel(flange_obj, min(fh * 0.08, (flange - inner) * 0.12))
        if part_type == "threaded_bushing":
            pitch = float(p.get("thread_pitch", 1.5)) / 1000
            tlen = min(float(p.get("thread_length", body_length * 1000)) / 1000, body_length)
            add_thread("Internal thread", inner, pitch, tlen, -tlen / 2, mat, internal=True)
        if part_type == "expansion_bushing":
            for angle in (0, math.pi):
                bpy.ops.mesh.primitive_cube_add(size=1, location=(r * 0.94 * math.cos(angle), r * 0.94 * math.sin(angle), 0))
                slot = bpy.context.object
                slot.name = "Expansion slot"
                slot.dimensions = (r * 0.18, r * 0.18, body_length * 0.72)
                slot.rotation_euler[2] = angle
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                boolean = bpy.context.object
                body = next(o for o in bpy.context.scene.objects if o.name == "Body")
                mod = body.modifiers.new("Expansion slit", "BOOLEAN")
                mod.operation = "DIFFERENCE"
                mod.object = slot
                bpy.context.view_layer.objects.active = body
                bpy.ops.object.modifier_apply(modifier=mod.name)
                bpy.data.objects.remove(slot, do_unlink=True)
        return
    threaded_types = {"hex_bolt", "anchor", "socket_bolt", "countersunk_screw", "rivet", "self_tapping_screw", "stud"}
    thread_len = min(float(p.get("thread_length", diameter * 4)) / 1000, length) if part_type in threaded_types else 0
    thread_start = -length / 2 if part_type == "stud" else length / 2 - thread_len
    plain_len = max(0.0, length - thread_len)
    shaft = cylinder("Shaft", r * (0.92 if thread_len else 1.0), length, 0, mat)
    if part_type == "stud":
        return shaft
    if thread_len:
        add_thread("External thread", r, float(p.get("thread_pitch", 1.5)) / 1000, thread_len, thread_start, mat)
    if part_type in {"hex_bolt", "anchor", "socket_bolt", "countersunk_screw", "rivet", "self_tapping_screw"}:
        head_r = float(p.get("head_diameter", diameter * 1.6)) / 1000 / 2
        head_h = float(p.get("head_height", 5)) / 1000
        if part_type == "countersunk_screw":
            bpy.ops.mesh.primitive_cone_add(vertices=64, radius1=head_r, radius2=r, depth=head_h, location=(0, 0, length / 2 + head_h / 2))
            bpy.context.object.data.materials.append(mat)
        else:
            head = cylinder("Head", head_r, head_h, length / 2 + head_h / 2, mat, 6 if part_type in {"hex_bolt", "anchor"} else 64)
            add_bevel(head, min(head_h * 0.10, head_r * 0.08))
            if part_type == "socket_bolt":
                bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=head_r * 0.42, depth=head_h * 0.3, location=(0, 0, length / 2 + head_h * 0.9))
                socket = bpy.context.object
                socket.name = "Hex drive recess"
                mod = head.modifiers.new("Hex socket", "BOOLEAN")
                mod.operation = "DIFFERENCE"
                mod.object = socket
                bpy.context.view_layer.objects.active = head
                bpy.ops.object.modifier_apply(modifier=mod.name)
                bpy.data.objects.remove(socket, do_unlink=True)
    elif part_type == "hex_nut":
        width = float(p.get("width", diameter * 1.7)) / 1000
        h = float(p.get("height", 8)) / 1000
        inner = float(p.get("hole_diameter", diameter * 0.85)) / 1000 / 2
        bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=width / 1.732, depth=h, location=(0, 0, 0))
        nut = bpy.context.object
        nut.name = "Hex nut"
        nut.data.materials.append(mat)
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=inner, depth=h * 1.5, location=(0, 0, 0))
        cutter = bpy.context.object
        mod = nut.modifiers.new("Thread bore", "BOOLEAN")
        mod.operation = "DIFFERENCE"
        mod.solver = "EXACT"
        mod.object = cutter
        bpy.context.view_layer.objects.active = nut
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
        add_bevel(nut, min(h * 0.12, (width / 1.732 - inner) * 0.16))
        add_thread("Nut internal thread", inner, float(p.get("thread_pitch", 1.5)) / 1000, h * 0.9, -h * 0.45, mat, internal=True)
    return shaft


def join_mesh_objects(name="Generated_Model"):
    """Export one connected assembly instead of a pile of visible loose parts."""
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        return
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    joined = bpy.context.object
    joined.name = name


def export(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", export_materials="EXPORT")


def main():
    args = args_parse()
    # PowerShell and some Windows editors may write UTF-8 JSON with a BOM.
    # Accept both BOM and plain UTF-8 request files.
    request = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    clear_scene()
    if request.get("category") == "hubcap":
        p = dict(request.get("params") or {})
        # A single image may not contain an absolute scale or may hide the bore.
        # Keep generation alive with conservative visual defaults in those cases.
        defaults = {
            "outer_diameter_mm": 380,
            "rim_depth_mm": 120,
            "face_thickness_mm": 7,
            "bulge_depth_mm": 15,
            "center_hole_diameter_mm": 60,
            "spoke_count": 7,
            "family": "classic",
            "finish": "solid_silver",
            "center_type": "open",
            "bolt_count": 5,
            "design_variant": "straight",
            "spoke_width_scale": 1.0,
            "spoke_sweep_deg": 0.0,
            "spoke_mid_radius_ratio": 0.53,
            "spoke_outer_radius_ratio": 0.82,
            "rim_lip_scale": 1.0,
            "hole_radius_scale": 1.0,
        }
        for key, value in defaults.items():
            if p.get(key) is None:
                p[key] = value
        if p.get("photo_reconstruction") and not isinstance(p.get("spoke_profiles"), list):
            raise ValueError("Фото-реконструкция требует индивидуальные spoke_profiles; шаблонная геометрия отключена")
        if p.get("photo_reconstruction") and not p["spoke_profiles"]:
            raise ValueError("Фото-реконструкция получила пустые spoke_profiles; шаблонная геометрия отключена")
        # In the interactive editor, spoke_count means visible tips; branch styles use two tips/root.
        if p.get("spoke_count") is not None:
            p["spoke_roots_count"] = max(1, int(p["spoke_count"]) // (2 if p.get("family") in {"split", "v_shaped", "y_shaped"} else 1))
        else:
            p["spoke_roots_count"] = p.get("spoke_roots_count") or 7
        make_hubcap(p, __import__("random").Random(int(p.get("seed", 42))))
    else:
        make_metiz(request["part_type"], request.get("params") or {}, request.get("material", "steel"))
    join_mesh_objects("Generated_Model")
    export(Path(args.output))


if __name__ == "__main__":
    main()
