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


def _apply_scene_geometry():
    """Turn all procedural helpers into clean, exportable mesh geometry."""
    # Thread ridges are created as curves for speed and then converted here;
    # glTF/STL exporters otherwise ignore them when objects are joined.
    for obj in list(bpy.context.scene.objects):
        if obj.type != "CURVE":
            continue
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.convert(target="MESH")

    for obj in list(bpy.context.scene.objects):
        if obj.type != "MESH":
            continue
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        # Blender evaluates modifiers as a stack. Applying the last one first
        # avoids "modifier was not first" warnings and keeps Boolean results
        # deterministic for castle nuts and slotted washers.
        for modifier in reversed(list(obj.modifiers)):
            try:
                bpy.ops.object.modifier_apply(modifier=modifier.name)
            except RuntimeError:
                # A non-critical display modifier must not prevent export.
                pass
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        obj.select_set(False)


def _validate_meshes():
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("Генератор не создал ни одного Mesh-объекта")
    for obj in meshes:
        if not obj.data.vertices or not obj.data.polygons:
            raise RuntimeError(f"Пустая геометрия объекта: {obj.name}")
        if any(not all(math.isfinite(value) for value in vertex.co) for vertex in obj.data.vertices):
            raise RuntimeError(f"Некорректные координаты в объекте: {obj.name}")
        for polygon in obj.data.polygons:
            polygon.use_smooth = False
        obj.data.update()


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


def rod_between(name, start, end, radius, mat, vertices=32):
    from mathutils import Vector
    start = Vector(start)
    end = Vector(end)
    direction = end - start
    length = direction.length
    if length <= 0:
        return None
    midpoint = (start + end) * 0.5
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=length, location=midpoint)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    bevel = obj.modifiers.new("Rod edge bevel", "BEVEL")
    bevel.width = min(radius * 0.14, length * 0.08)
    bevel.segments = 2
    return obj


def _cut_object(body, cutter, name):
    modifier = body.modifiers.new(name, "BOOLEAN")
    modifier.operation = "DIFFERENCE"
    modifier.solver = "EXACT"
    modifier.object = cutter
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def _add_nut_body(name, outer_radius, height, bore_radius, mat, width=None):
    vertices = 6
    radius = (width / 1.732) if width else outer_radius
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=height, location=(0, 0, 0))
    body = bpy.context.object
    body.name = name
    body.data.materials.append(mat)
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=bore_radius, depth=height * 1.5, location=(0, 0, 0))
    _cut_object(body, bpy.context.object, "Thread bore")
    add_bevel(body, min(height * .10, max(.0001, radius - bore_radius) * .14))
    return body


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
        # Distribute the requested total number of holes between both legs.
        # The old implementation used two fixed positions per leg, so every
        # angle bracket incorrectly contained four holes regardless of the
        # value of holes_count.
        requested_holes = max(1, int(round(float(p.get("holes_count", 4)))))
        # Do not place overlapping holes when a small bracket receives an
        # excessive count. Keep the user-visible count as close as possible
        # while preserving at least one hole diameter of material between cuts.
        pitch_x = max(hole_r * 2.4, hole_r * 2 + hole_r * 1.2)
        pitch_y = max(hole_r * 2.4, hole_r * 2 + hole_r * 1.2)
        max_horizontal = max(1, int((length - hole_r * 4) / pitch_x) + 1)
        max_vertical = max(0, int((width - hole_r * 4) / pitch_y) + 1)
        holes_count = min(requested_holes, max_horizontal + max_vertical)
        horizontal_count = (holes_count + 1) // 2
        vertical_count = holes_count - horizontal_count
        edge_margin = max(hole_r * 2.2, length * 0.12)
        usable_length = max(0.0, length - edge_margin * 2)
        horizontal_positions = [
            length / 2 if horizontal_count == 1 else edge_margin + usable_length * i / (horizontal_count - 1)
            for i in range(horizontal_count)
        ]
        for x in horizontal_positions:
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=hole_r, depth=wall * 2, location=(x, 0, wall / 2), rotation=(0, 0, 0))
            cutter = bpy.context.object
            cutter.name = "Angle hole cutter"
            mod = horizontal.modifiers.new("Mounting hole", "BOOLEAN")
            mod.operation = "DIFFERENCE"
            mod.object = cutter
            bpy.context.view_layer.objects.active = horizontal
            bpy.ops.object.modifier_apply(modifier=mod.name)
            bpy.data.objects.remove(cutter, do_unlink=True)
        edge_margin_y = max(hole_r * 2.2, width * 0.12)
        usable_width = max(0.0, width - edge_margin_y * 2)
        vertical_positions = [
            0.0 if vertical_count == 1 else -usable_width / 2 + usable_width * i / (vertical_count - 1)
            for i in range(vertical_count)
        ]
        for y in vertical_positions:
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
    if part_type == "split_pin":
        # A split pin is a bent wire with two separate legs, not a solid rod.
        wire = max(r * 0.42, 0.00035)
        head = max(diameter * 0.75, wire * 2.2)
        rod_between("Split pin bridge", (-wire, 0, length * 0.45), (wire, 0, length * 0.45), wire, mat)
        rod_between("Split pin leg A", (-wire, 0, length * 0.45), (-wire, 0, -length * 0.50), wire, mat)
        rod_between("Split pin leg B", (wire, 0, length * 0.45), (wire, 0, -length * 0.42), wire, mat)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=12, radius=head, location=(0, 0, length * 0.45))
        bpy.context.object.name = "Split pin eye"
        bpy.context.object.data.materials.append(mat)
        return
    if part_type == "pin":
        # Machined dowel with small chamfers at both ends.
        shaft = cylinder("Dowel pin", r, length * 0.96, 0, mat, 48)
        for z in (-length * 0.49, length * 0.49):
            bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=r * 0.82, radius2=r, depth=length * 0.04, location=(0, 0, z))
            tip = bpy.context.object
            tip.name = "Dowel pin chamfer"
            tip.data.materials.append(mat)
        return shaft
    if part_type == "rivet":
        shaft = cylinder("Rivet shank", r, length, 0, mat, 48)
        head_r = float(p.get("head_diameter", diameter * 1.8)) / 1000 / 2
        head_h = float(p.get("head_height", diameter * 0.45)) / 1000
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=48, ring_count=20, radius=head_r,
            location=(0, 0, length / 2 + head_h * 0.28),
        )
        dome = bpy.context.object
        dome.name = "Rivet domed head"
        dome.scale.z = min(0.62, max(0.28, head_h / head_r))
        dome.data.materials.append(mat)
        return shaft
    if part_type == "anchor":
        # A visible expansion sleeve makes the anchor different from a plain
        # hex bolt while keeping the model robust for preview and export.
        sleeve_len = min(length * 0.42, max(length * 0.18, 0.012))
        sleeve_r = r * 1.18
        ring("Anchor sleeve", sleeve_r, r * 0.96, sleeve_len, -length * 0.12, mat)
        for angle in (0, math.pi):
            bpy.ops.mesh.primitive_cube_add(size=1, location=(sleeve_r * 0.82 * math.cos(angle), sleeve_r * 0.82 * math.sin(angle), -length * 0.12))
            slot = bpy.context.object
            slot.name = "Anchor expansion slot"
            slot.dimensions = (r * 0.18, sleeve_r * 0.7, sleeve_len * 0.72)
            slot.rotation_euler[2] = angle
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            body = next((obj for obj in bpy.context.scene.objects if obj.name == "Anchor sleeve"), None)
            if body:
                mod = body.modifiers.new("Anchor slot", "BOOLEAN")
                mod.operation = "DIFFERENCE"
                mod.object = slot
                bpy.context.view_layer.objects.active = body
                bpy.ops.object.modifier_apply(modifier=mod.name)
            bpy.data.objects.remove(slot, do_unlink=True)
        # Continue through the common branch to add the threaded head.
    if part_type in {"washer", "spring_washer", "lock_washer"} or "bushing" in part_type or part_type in {"spacer", "flanged_bushing", "flanged_spacer", "embedded_bushing"}:
        if p.get("wall_thickness") is not None and p.get("inner_diameter") is not None:
            inner_diameter = max(0.2, float(p["diameter"]) - 2 * float(p["wall_thickness"]))
        else:
            inner_diameter = float(p.get("inner_diameter", p.get("hole_diameter", diameter * 0.55)))
        inner = inner_diameter / 1000 / 2
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
        if part_type == "spring_washer":
            gap = float(p.get("spring_gap", 2)) / 1000
            # Cut a narrow diagonal opening and tilt the washer to show its
            # spring action in the preview and exported model.
            bpy.ops.mesh.primitive_cube_add(size=1, location=(r, 0, 0))
            cutter = bpy.context.object
            cutter.dimensions = (r * .45, gap, body_length * 2)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            body = next(o for o in bpy.context.scene.objects if o.name == "Body")
            _cut_object(body, cutter, "Spring washer gap")
            body.rotation_euler[1] = math.radians(7)
        if part_type == "lock_washer":
            teeth = max(4, int(p.get("teeth_count", 12)))
            tooth_height = float(p.get("tooth_height", 2)) / 1000
            for index in range(teeth):
                angle = 2 * math.pi * index / teeth
                bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=tooth_height * .45, radius2=tooth_height * .08,
                                                depth=body_length, location=((r + tooth_height * .5) * math.cos(angle),
                                                                             (r + tooth_height * .5) * math.sin(angle), 0))
                tooth = bpy.context.object
                tooth.name = "Lock washer tooth"
                tooth.rotation_euler[2] = angle
                tooth.data.materials.append(mat)
        return
    threaded_types = {"hex_bolt", "anchor", "socket_bolt", "countersunk_screw", "self_tapping_screw", "stud"}
    thread_len = min(float(p.get("thread_length", diameter * 4)) / 1000, length) if part_type in threaded_types else 0
    thread_start = -length / 2 if part_type == "stud" else length / 2 - thread_len
    plain_len = max(0.0, length - thread_len)
    shaft = cylinder("Shaft", r * (0.92 if thread_len else 1.0), length, 0, mat)
    if part_type == "self_tapping_screw":
        # Replace the blunt end with a short conical self-tapping point.
        point_len = min(length * 0.16, max(r * 2.2, 0.0015))
        bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=r * 0.02, radius2=r * 0.92, depth=point_len, location=(0, 0, -length / 2 - point_len / 2 + 0.00005))
        point = bpy.context.object
        point.name = "Self tapping point"
        point.data.materials.append(mat)
    if thread_len:
        add_thread("External thread", r, float(p.get("thread_pitch", 1.5)) / 1000, thread_len, thread_start, mat)
    if part_type == "stud":
        return shaft
    if part_type in {"hex_bolt", "anchor", "socket_bolt", "countersunk_screw", "rivet", "self_tapping_screw"}:
        head_r = float(p.get("head_diameter", diameter * 1.6)) / 1000 / 2
        head_h = float(p.get("head_height", 5)) / 1000
        if part_type == "countersunk_screw":
            bpy.ops.mesh.primitive_cone_add(vertices=64, radius1=head_r, radius2=r, depth=head_h, location=(0, 0, length / 2 + head_h / 2))
            bpy.context.object.name = "Countersunk head"
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
    elif part_type in {"hex_nut", "nyloc_nut", "flange_nut", "castle_nut", "cap_nut"}:
        width = float(p.get("width", diameter * 1.7)) / 1000
        h = float(p.get("height", 8)) / 1000
        inner = float(p.get("hole_diameter", diameter * 0.85)) / 1000 / 2
        nut = _add_nut_body("Hex nut", width / 1.732, h, inner, mat, width=width)
        if part_type == "flange_nut":
            flange_r = float(p.get("flange_diameter", diameter * 2.2)) / 1000 / 2
            flange_h = float(p.get("flange_height", 2)) / 1000
            ring("Nut flange", flange_r, inner, flange_h, -h / 2 + flange_h / 2, mat)
        if part_type == "nyloc_nut":
            nylon_h = float(p.get("nylon_height", 3)) / 1000
            bpy.ops.mesh.primitive_torus_add(major_radius=max(inner * 1.03, width / 3.2), minor_radius=max(.0003, nylon_h * .18), major_segments=64, minor_segments=12, location=(0, 0, h / 2 - nylon_h / 2))
            bpy.context.object.name = "Nylon locking ring"
            bpy.context.object.data.materials.append(material("nylon", (0.08, 0.10, 0.12)))
        if part_type == "castle_nut":
            slots = max(2, int(p.get("slots_count", 6)))
            slot_w = float(p.get("slot_width", 3)) / 1000
            for index in range(slots):
                angle = 2 * math.pi * index / slots
                bpy.ops.mesh.primitive_cube_add(size=1, location=((width / 2) * math.cos(angle), (width / 2) * math.sin(angle), h * .62))
                cutter = bpy.context.object
                cutter.name = "Castle nut slot"
                cutter.dimensions = (slot_w, width * .30, h * .85)
                cutter.rotation_euler[2] = angle
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                _cut_object(nut, cutter, "Castle slot")
        if part_type == "cap_nut":
            cap_h = float(p.get("cap_height", 5)) / 1000
            bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=16, radius=width / 2, location=(0, 0, h / 2 + cap_h * .22))
            cap = bpy.context.object
            cap.name = "Cap nut dome"
            cap.scale.z = min(.65, cap_h / max(width / 2, .0001))
            cap.data.materials.append(mat)
        add_thread("Nut internal thread", inner, float(p.get("thread_pitch", 1.5)) / 1000, h * 0.9, -h * 0.45, mat, internal=True)
    return shaft


def join_mesh_objects(name="Generated_Model"):
    """Export one connected assembly instead of a pile of visible loose parts."""
    _apply_scene_geometry()
    _validate_meshes()
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("После подготовки не осталось Mesh-объектов для экспорта")
    if len(meshes) == 1:
        meshes[0].name = name
        meshes[0].select_set(True)
        return meshes[0]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    joined = bpy.context.object
    joined.name = name
    for polygon in joined.data.polygons:
        polygon.use_smooth = True
    joined.data.update()
    return joined


def export(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(path), export_format="GLB", export_materials="EXPORT",
        use_selection=True, export_apply=True,
    )


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
