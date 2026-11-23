"""Stable conceptual hubcap mesh generation in millimetres."""
from __future__ import annotations
import math
import numpy as np
import trimesh
from .constants import FAMILIES, SPOKE_COUNTS


def sanitize_parameters(params):
    p = dict(params)
    if p.get("outer_diameter_mm") is None:
        raise ValueError("Для 3D-экспорта требуется измеренный наружный диаметр")
    d = float(p["outer_diameter_mm"])
    if not 330 <= d <= 460:
        raise ValueError("outer_diameter_mm должен быть в диапазоне 330–460 мм")
    p["outer_diameter_mm"] = d
    for name, default, low, high in [("center_hole_diameter_mm", d*.15, 50., min(74., d*.19)), ("bulge_depth_mm", d*.05, 8., 28.), ("face_thickness_mm", d*.02, 5., 11.), ("rim_depth_mm", d*.35, 90., 170.)]:
        value = p.get(name)
        p[name] = float(np.clip(float(default if value is None else value), low, high))
    p["spoke_count"] = int(np.clip(int(p.get("spoke_count", 10)), min(SPOKE_COUNTS), max(SPOKE_COUNTS)))
    p["family"] = p.get("family") if p.get("family") in FAMILIES else "classic"
    p["center_type"] = p.get("center_type") if p.get("center_type") in ("open", "cap") else "cap"
    return p


def _ring(inner, outer, height, z, sections):
    angles = np.linspace(0., 2.*math.pi, sections, endpoint=False)
    verts = []
    for zz, rr in ((z-height/2, outer), (z-height/2, inner), (z+height/2, outer), (z+height/2, inner)):
        verts.extend([(rr*math.cos(a), rr*math.sin(a), zz) for a in angles])
    faces = []
    of, ib, ot, it = 0, sections, 2*sections, 3*sections
    for i in range(sections):
        j = (i+1) % sections
        faces += [(of+i, of+j, ot+j, ot+i), (ib+j, ib+i, it+i, it+j), (of+j, of+i, ib+i, ib+j), (ot+i, ot+j, it+j, it+i)]
    return trimesh.Trimesh(np.asarray(verts), np.asarray(faces), process=True)


def _tapered_spoke(start, end, width_start, width_end, z, thickness):
    sx, sy = start; ex, ey = end
    dx, dy = ex-sx, ey-sy
    length = math.hypot(dx, dy)
    nx, ny = -dy/length, dx/length
    corners = [(sx+nx*width_start/2, sy+ny*width_start/2), (ex+nx*width_end/2, ey+ny*width_end/2), (ex-nx*width_end/2, ey-ny*width_end/2), (sx-nx*width_start/2, sy-ny*width_start/2)]
    verts = [(x, y, z-thickness/2) for x,y in corners] + [(x, y, z+thickness/2) for x,y in corners]
    faces = [(0,1,2,3),(4,7,6,5),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)]
    return trimesh.Trimesh(np.asarray(verts), np.asarray(faces), process=True)


def create_parametric_mesh(params, sections=96):
    """Build a symmetric, millimetre-scale conceptual hubcap assembly."""
    p = sanitize_parameters(params)
    radius = p["outer_diameter_mm"]/2.; bore = p["center_hole_diameter_mm"]/2.
    face, depth, dish = p["face_thickness_mm"], p["rim_depth_mm"], p["bulge_depth_mm"]
    inner = max(bore*1.35, radius*.20); outer = radius*(.91 if p["family"] == "mesh" else .86)
    width = max(5., radius*(.045 if p["family"] == "mesh" else .065))
    solids = [_ring(radius*.94, radius, depth, -depth/2, sections), _ring(inner, radius*.97, face, dish, sections)]
    hub = trimesh.creation.cylinder(radius=inner, height=face, sections=sections); hub.apply_translation([0,0,dish]); solids.append(hub)
    if p["center_type"] == "cap":
        cap = trimesh.creation.cylinder(radius=bore*1.04, height=max(2., face*.35), sections=sections); cap.apply_translation([0,0,dish+face*.65]); solids.append(cap)
    for i in range(p["spoke_count"]):
        angle = 2*math.pi*i/p["spoke_count"]
        offset = (0.12 if i%2 else -0.12) if p["family"] == "v_shaped" else 0.
        if p["family"] in ("split", "y_shaped"): start_r, offset = (inner if i%2 == 0 else radius*.38), (-.13 if i%2 else .13)
        elif p["family"] == "mesh": start_r, offset = radius*.24, (-.08 if i%2 else .08)
        else: start_r = inner
        a0, a1 = angle+offset*.20, angle+offset
        start = (start_r*math.cos(a0), start_r*math.sin(a0)); end = (outer*math.cos(a1), outer*math.sin(a1))
        solids.append(_tapered_spoke(start, end, width*.82, width, dish+face*.55, max(2., face*1.45)))
        if p["family"] in ("split", "y_shaped"):
            branch = (outer*.82*math.cos(a1 + (.035 if i % 2 else -.035)), outer*.82*math.sin(a1 + (.035 if i % 2 else -.035)))
            solids.append(_tapered_spoke(end, branch, width*.42, width*.30, dish+face*.62, max(2., face*1.1)))
    mesh = trimesh.util.concatenate(solids)
    mesh.remove_infinite_values(); mesh.remove_unreferenced_vertices(); mesh.fix_normals()
    return mesh, p
