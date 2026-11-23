"""Parametric alloy-wheel geometry for AutoKolpak Blender Generator v4."""

import math
import bpy
import bmesh
from mathutils import Vector

from materials import preset_material, make_material


def clear_scene():
    """Delete scene objects and remove unused materials between generated models."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablock in list(bpy.data.materials):
        if datablock.users == 0:
            bpy.data.materials.remove(datablock)


def _recalculate_normals(mesh):
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        mesh.update()
    finally:
        bm.free()


def _annulus(name, outer_r, inner_r, z_back, z_front, material, segments=96):
    """Create a closed ring/barrel section with a real central opening."""
    verts = []
    # Ordered rings: outer-front, inner-front, outer-back, inner-back.
    for radius, z in ((outer_r, z_front), (inner_r, z_front),
                      (outer_r, z_back), (inner_r, z_back)):
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            verts.append((radius * math.cos(a), radius * math.sin(a), z))

    faces = []
    of, inf, ob, ib = 0, segments, 2 * segments, 3 * segments
    for i in range(segments):
        j = (i + 1) % segments
        # Front annular face, outer wall, inner wall and rear face.
        faces.append((of + i, of + j, inf + j, inf + i))
        faces.append((ob + i, ob + j, of + j, of + i))
        faces.append((ib + j, ib + i, inf + i, inf + j))
        faces.append((ib + i, ib + j, ob + j, ob + i))

    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    _recalculate_normals(mesh)
    # Smooth only the cylindrical walls; keep the annular front/back faces planar.
    for index, polygon in enumerate(mesh.polygons):
        if index % 4 in (1, 2):
            polygon.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def _torus(name, major_radius, minor_radius, z, material, major_segments=96, minor_segments=12):
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_segments=major_segments,
        minor_segments=minor_segments,
        location=(0.0, 0.0, z),
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def _bevel(obj, width, segments=3):
    if width <= 0:
        return
    modifier = obj.modifiers.new("Machined_edge_bevel", "BEVEL")
    modifier.width = width
    modifier.segments = segments
    modifier.limit_method = "ANGLE"
    modifier.angle_limit = math.radians(25.0)
    normal = obj.modifiers.new("Weighted_corner_normals", "WEIGHTED_NORMAL")
    normal.keep_sharp = True


def _cylinder(name, radius, depth, location, material, vertices=48, bevel_width=0.0):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices, radius=radius, depth=depth, location=location
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    for poly in obj.data.polygons:
        poly.use_smooth = len(poly.vertices) <= 4
    _bevel(obj, bevel_width)
    return obj


def _polar(radius, angle):
    return (radius * math.cos(angle), radius * math.sin(angle))


def _spoke_prism(name, path, half_widths, top_heights, thickness, material, bevel_width):
    """Make a tapered, swept polygonal spoke, with real thickness and bevels.

    path uses (radius, angle) samples; half_widths and top_heights have one value
    per sample. This is not a flat face line or a scaled cube.
    """
    xy = [_polar(radius, angle) for radius, angle in path]
    left_side = []
    right_side = []
    for i, (x, y) in enumerate(xy):
        if i == 0:
            dx, dy = xy[1][0] - x, xy[1][1] - y
        elif i == len(xy) - 1:
            dx, dy = x - xy[i - 1][0], y - xy[i - 1][1]
        else:
            dx, dy = xy[i + 1][0] - xy[i - 1][0], xy[i + 1][1] - xy[i - 1][1]
        length = max(math.hypot(dx, dy), 1e-8)
        nx, ny = -dy / length, dx / length
        w = half_widths[i]
        left_side.append((x + nx * w, y + ny * w, top_heights[i]))
        right_side.append((x - nx * w, y - ny * w, top_heights[i]))

    outline = left_side + list(reversed(right_side))
    n = len(outline)
    verts = list(outline) + [(x, y, z - thickness) for x, y, z in outline]
    faces = [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))

    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    _recalculate_normals(mesh)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    _bevel(obj, bevel_width, segments=3)
    return obj


def _path(root_r, mid_r, outer_r, root_angle, mid_offset, outer_offset):
    return [
        (root_r, root_angle),
        (mid_r, root_angle + mid_offset),
        (outer_r, root_angle + outer_offset),
    ]


def _add_spokes(p, rng, radius, bore_r, face_mat, accent_mat):
    """Create real, volumetric spokes whose visible tip count matches metadata."""
    family = p["family"]
    root_count = int(p.get("spoke_roots_count", p["spoke_count"]))
    variant = p.get("design_variant", "straight")
    width_scale = float(p.get("spoke_width_scale", 1.0))
    sweep = math.radians(float(p.get("spoke_sweep_deg", 0.0)))
    mid_ratio = float(p.get("spoke_mid_radius_ratio", 0.53))
    outer_ratio = float(p.get("spoke_outer_radius_ratio", 0.82))

    # Variants affect actual geometry, not just the label.
    variant_factors = {
        "straight": (0.65, 1.08, -0.015),
        "swept": (1.35, 0.92, 0.010),
        "wide": (0.72, 1.35, -0.025),
        "slim": (1.12, 0.66, 0.025),
        "angular": (1.65, 0.94, -0.005),
        "turbine": (2.00, 0.84, 0.040),
        "petal": (0.82, 1.24, 0.015),
        "concave": (1.10, 0.91, 0.030),
    }
    sweep_factor, variant_width, mid_adjust = variant_factors.get(variant, (1.0, 1.0, 0.0))
    width_scale *= variant_width
    root_r = max(bore_r * 1.55, radius * 0.19)
    outer_r = radius * max(0.76, min(0.89, outer_ratio))
    mid_r = radius * max(0.40, min(0.70, mid_ratio + mid_adjust))
    mid_r = max(root_r + radius * 0.09, min(mid_r, outer_r - radius * 0.10))
    # Labelled face thickness is the real vertical thickness of each spoke.
    thickness = max(0.0045, float(p["face_thickness_mm"]) / 1000.0)
    bevel = min(radius * 0.009, thickness * 0.28)
    bulge = float(p["bulge_depth_mm"]) / 1000.0
    z_root = radius * 0.11 + bulge
    z_mid = radius * 0.095 + bulge * 0.70
    z_tip = radius * 0.075 + bulge * 0.30

    if family == "mesh":
        half_root, half_mid, half_tip = (radius * 0.012, radius * 0.018, radius * 0.014)
    elif family == "split":
        half_root, half_mid, half_tip = (radius * 0.021, radius * 0.025, radius * 0.019)
    elif family == "y_shaped":
        half_root, half_mid, half_tip = (radius * 0.038, radius * 0.036, radius * 0.022)
    elif family == "v_shaped":
        half_root, half_mid, half_tip = (radius * 0.035, radius * 0.033, radius * 0.020)
    else:
        half_root, half_mid, half_tip = (radius * 0.032, radius * 0.042, radius * 0.029)
    half_root *= width_scale
    half_mid *= width_scale
    half_tip *= width_scale

    for i in range(root_count):
        # A tiny per-root jitter prevents an overly CAD-perfect look but is reproducible from seed.
        a = 2.0 * math.pi * i / root_count + rng.uniform(-0.006, 0.006)
        local_sweep = sweep * sweep_factor + rng.uniform(-0.025, 0.025)
        accent = accent_mat if p["finish"] == "diamond_cut" and i % 2 == 0 else face_mat

        if family == "classic":
            path = _path(root_r, mid_r, outer_r, a, local_sweep * 0.45, local_sweep)
            _spoke_prism(f"Spoke_{i:02d}", path,
                [half_root, half_mid, half_tip], [z_root, z_mid, z_tip],
                thickness, accent, bevel)

        elif family == "v_shaped":
            fork_r = radius * max(0.36, min(0.49, mid_ratio - 0.035))
            _spoke_prism(f"V_Root_{i:02d}", [(root_r, a), (fork_r, a)],
                [radius * 0.034 * width_scale, radius * 0.045 * width_scale],
                [z_root, z_mid], thickness, face_mat, bevel)
            fork_angle = math.radians(rng.uniform(3.7, 6.2))
            for side in (-1, 1):
                path = [(fork_r * 0.98, a + side * 0.008),
                        (radius * 0.62, a + side * fork_angle * 0.56 + local_sweep * 0.25),
                        (outer_r, a + side * fork_angle + local_sweep)]
                _spoke_prism(f"V_Arm_{i:02d}_{side}", path,
                    [radius*0.022*width_scale, radius*0.021*width_scale, half_tip],
                    [z_mid, (z_mid+z_tip)*0.5, z_tip], thickness*0.92,
                    accent if side == 1 else face_mat, bevel*0.72)

        elif family == "split":
            split_angle = math.radians(rng.uniform(1.5, 3.0))
            for side in (-1, 1):
                path = _path(root_r, mid_r, outer_r, a + side*split_angle,
                             local_sweep*0.32, local_sweep + side*split_angle*0.35)
                _spoke_prism(f"Split_Spoke_{i:02d}_{side}", path,
                    [radius*0.020*width_scale, radius*0.025*width_scale, half_tip],
                    [z_root, z_mid, z_tip], thickness,
                    accent if side == 1 else face_mat, bevel*0.8)

        elif family == "y_shaped":
            fork_r = radius * rng.uniform(0.39, 0.47)
            _spoke_prism(f"Y_Trunk_{i:02d}",
                [(root_r, a), (radius*0.31, a), (fork_r, a + local_sweep*0.25)],
                [radius*0.038*width_scale, radius*0.047*width_scale, radius*0.042*width_scale],
                [z_root, z_root*0.92, z_mid], thickness, face_mat, bevel)
            fork_angle = math.radians(rng.uniform(4.3, 7.0))
            for side in (-1, 1):
                path = [(fork_r*0.96, a + local_sweep*0.25),
                        (radius*0.62, a + side*fork_angle*0.55 + local_sweep*0.5),
                        (outer_r, a + side*fork_angle + local_sweep)]
                _spoke_prism(f"Y_Branch_{i:02d}_{side}", path,
                    [radius*0.031*width_scale, radius*0.025*width_scale, half_tip],
                    [z_mid, (z_mid+z_tip)*0.5, z_tip], thickness*0.92,
                    accent if side == 1 else face_mat, bevel*0.7)

        else:  # dense, narrow multi-spoke face
            path = _path(root_r, mid_r, outer_r, a, local_sweep*0.5, local_sweep)
            _spoke_prism(f"Mesh_Spoke_{i:02d}", path,
                [half_root, half_mid, half_tip], [z_root*0.92, z_mid*0.92, z_tip*0.95],
                thickness*0.82, accent, bevel*0.55)

def make_hubcap(p, rng):
    """Generate a 3D alloy-wheel face and barrel in Blender metres.

    STL/OBJ exporters in generate_dataset.py scale this geometry to millimetres.
    The wheel front is +Z, which the camera setup treats as the front face.
    """
    radius = float(p["outer_diameter_mm"]) / 2000.0
    barrel_depth = float(p["rim_depth_mm"]) / 1000.0
    face_thickness = float(p["face_thickness_mm"]) / 1000.0
    bulge = float(p["bulge_depth_mm"]) / 1000.0
    center_hole = p.get("center_hole_diameter_mm")
    # A capped centre has no measurable hole in the source image; use a hidden
    # construction bore so the procedural model remains valid.
    if center_hole is None:
        center_hole = float(p.get("center_hole_fallback_mm", 60.0))
    bore_r = max(float(center_hole) / 2000.0, radius * 0.075)
    bore_r = min(bore_r, radius * 0.19)

    finish = p["finish"]
    if finish == "solid_black":
        base = preset_material("Face_Black", "black")
        accent = preset_material("Machined_Silver", "chrome")
    elif finish == "diamond_cut":
        base = preset_material("Face_Dark", "black", color=(0.075, 0.085, 0.10), metallic=0.82, roughness=0.23)
        accent = preset_material("Diamond_Cut", "chrome", color=(0.72, 0.75, 0.80), roughness=0.19)
    else:
        base = preset_material("Face_Silver", "silver",
            color=(rng.uniform(0.43, 0.68), rng.uniform(0.46, 0.70), rng.uniform(0.51, 0.75)),
            metallic=rng.uniform(0.76, 0.91), roughness=rng.uniform(0.20, 0.32))
        accent = preset_material("Edge_Chrome", "chrome")

    barrel_mat = preset_material("Inner_Barrel", "barrel")
    dark_mat = preset_material("Recess_Dark", "hole")
    bolt_mat = preset_material("Lug_Metal", "chrome")

    # Deep annular barrel: unlike the previous version, this is not a solid disc.
    lip_scale = float(p.get("rim_lip_scale", 1.0))
    _annulus("Wheel_Barrel", radius*0.965, radius*0.735,
             -barrel_depth, radius*0.025, barrel_mat, segments=128)
    lip_minor = radius * 0.023 * lip_scale
    _torus("Outer_Polished_Lip", radius - lip_minor, lip_minor,
           radius*0.021, accent, major_segments=128, minor_segments=16)
    _torus("Inner_Bead_Ring", radius*0.755, radius*0.010*lip_scale,
           radius*0.020, accent, major_segments=112, minor_segments=12)

    # Raised hub flange with an actual centre bore (or a separate centre cap).
    hub_outer = radius*0.305
    hub_front = radius*0.14 + bulge
    hub_back = hub_front - face_thickness*0.9
    hub_outer = max(hub_outer, bore_r*1.55)
    if hub_outer > bore_r * 1.12:
        _annulus("Hub_Flange", hub_outer, bore_r, hub_back, hub_front,
                 base, segments=96)
        _torus("Hub_Flange_Bead", hub_outer*0.94, radius*0.012,
               hub_front, accent, major_segments=96, minor_segments=12)

    # Real volumetric spokes or branching arms. Their top surfaces are domed upward.
    _add_spokes(p, rng, radius, bore_r, base, accent)

    # Lug wells are dark inset-looking bores with metallic rims; separate geometry
    # avoids destructive Boolean operations across thousands of dataset samples.
    bolt_count = int(p.get("bolt_count", 5))
    bolt_circle = max(hub_outer*0.73, bore_r*1.22)
    bolt_circle = min(bolt_circle, radius*0.255)
    hole_r = radius * rng.uniform(0.020, 0.026) * float(p.get("hole_radius_scale", 1.0))
    for i in range(bolt_count):
        angle = 2.0*math.pi*i/bolt_count + math.radians(float(p.get("bolt_phase_deg", 18.0)))
        x, y = _polar(bolt_circle, angle)
        z = hub_front + radius*0.002
        _cylinder(f"Lug_Recess_{i:02d}", hole_r, radius*0.012,
                  (x, y, z), dark_mat, vertices=48, bevel_width=radius*0.002)
        _torus(f"Lug_Bezel_{i:02d}", hole_r*1.12, radius*0.003,
               z + radius*0.006, accent, major_segments=40, minor_segments=8)

    # Decorative centre cap is optional; without it, the central bore remains open.
    if p.get("center_type") == "cap":
        cap_r = bore_r * 0.91
        cap_z = hub_front + radius*0.012
        cap_sides = int(p.get("center_sides") or 8)
        cap = _cylinder("Centre_Cap", cap_r, radius*0.035,
                        (0.0, 0.0, cap_z), accent if finish != "solid_black" else base,
                        vertices=max(5, min(12, cap_sides)), bevel_width=radius*0.008)
        _torus("Centre_Cap_Ring", cap_r*0.92, radius*0.008,
               cap_z + radius*0.018, base, major_segments=96, minor_segments=12)

    # A handful of shallow machined rings add readable detail in close-up renders.
    _torus("Hub_Accent_Ring", radius*0.355, radius*0.0045,
           radius*0.045, accent, major_segments=112, minor_segments=8)

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    return mesh_objects
