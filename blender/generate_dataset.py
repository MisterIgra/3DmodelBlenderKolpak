"""AutoKolpak Blender Generator v4.0 - diverse, metadata-consistent synthetic wheels.

Run:
  blender.exe -b --python generate_dataset.py -- --output ./generated_dataset --count 10000 --renders-per-model 1 --resolution 512

Blender builds geometry in metres; STL/OBJ exports are scaled to millimetres.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import traceback
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from geometry import clear_scene, make_hubcap

VERSION = "4.0.0"
CURRENT_BACKDROP = (0.12, 0.13, 0.15)
BACKDROP_PALETTES = (
    (0.018, 0.022, 0.028),   # graphite studio
    (0.075, 0.083, 0.095),   # dark grey
    (0.20, 0.215, 0.23),     # mid grey
    (0.42, 0.44, 0.46),      # light grey
    (0.68, 0.69, 0.70),      # off-white
    (0.20, 0.23, 0.27),      # blue-grey
    (0.30, 0.25, 0.20),      # warm neutral
)
FAMILIES = ["classic", "v_shaped", "split", "y_shaped", "mesh"]
FINISHES = ["solid_silver", "solid_black", "diamond_cut"]
DESIGN_VARIANTS = ["straight", "swept", "wide", "slim", "angular", "turbine", "petal", "concave"]


def args_parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Generate diverse, labelled wheel/hubcap assets")
    parser.add_argument("--output", default="./generated_dataset")
    parser.add_argument("--count", type=int, default=10000, help="Number of models, not rendered images")
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic seed; omit for random models")
    parser.add_argument("--renders-per-model", type=int, default=1, help="Use 1 for speed, 2-3 for view diversity")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--export-formats", nargs="+", choices=["stl", "obj"], default=["stl", "obj"])
    parser.add_argument("--no-export-models", action="store_true", help="Skip STL/OBJ export to speed up image dataset generation")
    parser.add_argument("--resume", action="store_true", help="Continue in the output folder")
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--masks", action="store_true", help="Reserved; masks are not implemented")
    parser.add_argument("--depth", action="store_true", help="Reserved; depth maps are not implemented")
    return parser.parse_args(argv)


def params(index, seed):
    """Generate observable labels first; geometry consumes these saved values."""
    rng = random.Random(seed)
    family = rng.choice(FAMILIES)
    finish = rng.choice(FINISHES)
    variant = rng.choice(DESIGN_VARIANTS)
    diameter = round(rng.uniform(330.0, 460.0), 2)
    # Nominal size is derived from the physical diameter, rather than sampled independently.
    nominal = max(13, min(18, int(round(diameter / 25.4))))
    if family == "mesh":
        spoke_count = rng.randint(11, 24)
        roots_count = spoke_count
    elif family == "classic":
        spoke_count = rng.randint(5, 10)
        roots_count = spoke_count
    elif family == "split":
        roots_count = rng.randint(5, 9)
        spoke_count = roots_count * 2  # visible outer tips, matching what a person counts
    else:  # V/Y spokes also bifurcate into two visible ends
        roots_count = rng.randint(5, 8)
        spoke_count = roots_count * 2

    # Keep geometry inside meaningful ranges and record every global shape parameter.
    max_center_hole = min(74.0, diameter * 0.19)
    center_hole = min(round(rng.uniform(50.0, max_center_hole), 2), math.floor(max_center_hole * 100.0) / 100.0)
    bulge = round(rng.uniform(8.0, 28.0), 2)
    face_thickness = round(rng.uniform(5.0, 11.0), 2)
    rim_depth = round(rng.uniform(90.0, 170.0), 2)
    bolt_count = rng.choice([4, 5, 6])
    center_type = rng.choice(["open", "cap"])
    center_sides = rng.choice([5, 6, 8, 10])
    return {
        "model_id": f"hubcap_{index:06d}",
        "family": family,
        "design_variant": variant,
        "outer_diameter_mm": diameter,
        "rim_nominal_diameter_in": nominal,
        "spoke_count": spoke_count,
        "spoke_roots_count": roots_count,
        "center_hole_diameter_mm": center_hole,
        "bulge_depth_mm": bulge,
        "face_thickness_mm": face_thickness,
        "rim_depth_mm": rim_depth,
        "latch_count": bolt_count,
        "bolt_count": bolt_count,
        "spoke_style": family,
        "center_sides": center_sides,
        "finish": finish,
        "center_type": center_type,
        "spoke_width_scale": round(rng.uniform(0.65, 1.40), 4),
        "spoke_sweep_deg": round(rng.uniform(-13.0, 19.0), 3),
        "spoke_mid_radius_ratio": round(rng.uniform(0.44, 0.64), 4),
        "spoke_outer_radius_ratio": round(rng.uniform(0.77, 0.87), 4),
        "bolt_phase_deg": round(rng.uniform(0.0, 360.0), 3),
        "hole_radius_scale": round(rng.uniform(0.80, 1.20), 4),
        "rim_lip_scale": round(rng.uniform(0.75, 1.35), 4),
        "seed": int(seed),
        "units": "mm",
        "generator_version": VERSION,
        "modeling_note": "Synthetic visual asset; lug wells are visual recesses, not Boolean-cut mounting holes.",
    }


def model_fingerprint(p):
    """Quantized signature; avoids repeated parameter sets in a generated collection."""
    keys = ("family", "design_variant", "outer_diameter_mm", "spoke_count", "spoke_roots_count",
            "center_hole_diameter_mm", "bulge_depth_mm", "face_thickness_mm", "rim_depth_mm",
            "bolt_count", "center_type", "center_sides", "finish", "spoke_width_scale",
            "spoke_sweep_deg", "spoke_mid_radius_ratio", "spoke_outer_radius_ratio", "bolt_phase_deg",
            "hole_radius_scale", "rim_lip_scale")
    values = []
    for key in keys:
        value = p.get(key)
        if isinstance(value, float):
            value = round(value, 2)
        values.append(value)
    return hashlib.sha256(json.dumps(values, ensure_ascii=True).encode()).hexdigest()


def look_at(obj, target=(0.0, 0.0, 0.0)):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _new_area_light(name, location, energy, size, target=(0.0, 0.0, 0.02)):
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = energy
    light.data.shape = "DISK"
    light.data.size = size
    look_at(light, target)
    return light


def setup_render(scene, output_resolution, model_rng):
    global CURRENT_BACKDROP
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = output_resolution
    scene.render.resolution_y = output_resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.image_settings.compression = 10
    scene.render.use_file_extension = True
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    scene.view_settings.exposure = -0.65
    scene.view_settings.gamma = 1.0

    if not scene.world:
        scene.world = bpy.data.worlds.new("AutoKolpak_World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        # Broad domain randomization: the model must not learn that every object has a black backdrop.
        palette = model_rng.choice(BACKDROP_PALETTES)
        tint = model_rng.uniform(0.88, 1.12)
        CURRENT_BACKDROP = tuple(max(0.005, min(0.85, float(channel) * tint)) for channel in palette)
        bg.inputs["Color"].default_value = (*CURRENT_BACKDROP, 1.0)
        bg.inputs["Strength"].default_value = model_rng.uniform(0.45, 0.95)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Dataset_Camera"
    scene.camera = camera
    camera.data.lens = 50
    camera.data.sensor_width = 36
    camera.data.dof.use_dof = False

    # Per-model light rig variation, with bounded energy to keep the face readable.
    _new_area_light("Key_Softbox", (model_rng.uniform(-0.55, -0.25), model_rng.uniform(-0.55, -0.20), model_rng.uniform(0.55, 0.85)), model_rng.uniform(75, 115), model_rng.uniform(0.38, 0.68))
    _new_area_light("Fill_Softbox", (model_rng.uniform(0.28, 0.58), model_rng.uniform(-0.15, 0.15), model_rng.uniform(0.35, 0.58)), model_rng.uniform(35, 70), model_rng.uniform(0.35, 0.60))
    _new_area_light("Rim_Softbox", (model_rng.uniform(-0.30, 0.05), model_rng.uniform(0.35, 0.60), model_rng.uniform(0.28, 0.52)), model_rng.uniform(45, 85), model_rng.uniform(0.28, 0.48))
    return camera


def setup_ground(radius, depth, model_rng):
    mat = bpy.data.materials.new("Studio_Ground_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    # The ground/background sheet gets its own lightly shifted version of the sampled palette.
    # This creates light, neutral, warm, cool and dark examples without changing wheel labels.
    shade_scale = model_rng.uniform(0.78, 1.18)
    ground_color = tuple(max(0.005, min(0.90, c * shade_scale)) for c in CURRENT_BACKDROP)
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*ground_color, 1.0)
        bsdf.inputs["Roughness"].default_value = model_rng.uniform(0.62, 0.94)
        bsdf.inputs["Metallic"].default_value = 0.0
    bpy.ops.mesh.primitive_plane_add(size=radius * 14.0, location=(0.0, 0.0, -depth - radius * 0.045))
    plane = bpy.context.object
    plane.name = "Studio_Ground"
    plane.data.materials.append(mat)
    return plane


def camera_view(view_index, model_rng):
    # First view is face-focused; second and later views expose depth and side profile.
    if view_index == 0:
        return model_rng.uniform(-18, 18), model_rng.uniform(68, 88)
    if view_index == 1:
        return model_rng.uniform(-180, 180), model_rng.uniform(28, 48)
    if view_index == 2:
        return model_rng.uniform(-180, 180), model_rng.uniform(48, 65)
    return model_rng.uniform(-180, 180), model_rng.uniform(25, 82)


def export_model(objects, destination, file_format):
    bpy.ops.object.select_all(action="DESELECT")
    valid = [obj for obj in objects if obj and obj.name in bpy.context.scene.objects]
    for obj in valid:
        obj.select_set(True)
    if not valid:
        raise RuntimeError("No mesh objects available for export")
    bpy.context.view_layer.objects.active = valid[0]
    if file_format == "stl":
        bpy.ops.wm.stl_export(filepath=str(destination), export_selected_objects=True,
            global_scale=1000.0, use_scene_unit=False, apply_modifiers=True,
            forward_axis="Y", up_axis="Z", ascii_format=False)
    else:
        bpy.ops.wm.obj_export(filepath=str(destination), export_selected_objects=True,
            global_scale=1000.0, apply_modifiers=True, export_materials=True,
            export_normals=True, export_uv=True, forward_axis="Y", up_axis="Z", path_mode="AUTO")


def load_existing(output, resume, renders_per_model):
    """Load a resumable output and discard partial model records safely.

    A model is complete only when the expected image rows, their files, and its
    metadata JSON all exist. This prevents interrupted renders becoming valid
    but incomplete training samples.
    """
    annotations = output / "metadata" / "annotations.jsonl"
    done = set()
    fingerprints = set()
    if annotations.exists() and annotations.stat().st_size:
        if not resume:
            raise SystemExit(f"{annotations} already exists; use --resume or choose a new output folder")

        grouped = {}
        for line_no, line in enumerate(annotations.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                mid = row.get("model_id")
                if mid:
                    grouped.setdefault(str(mid), []).append(row)
            except (ValueError, TypeError):
                print(f"WARNING: ignoring malformed existing annotation line {line_no}", file=sys.stderr)

        kept_rows = []
        for mid, rows in grouped.items():
            metadata_path = output / "metadata" / f"{mid}.json"
            expected_rows = rows[:renders_per_model]
            signature = None
            complete = len(rows) == renders_per_model and metadata_path.is_file()
            if complete:
                for row in expected_rows:
                    image_path = (output / str(row.get("image_path", ""))).resolve()
                    try:
                        image_path.relative_to(output.resolve())
                    except ValueError:
                        complete = False
                        break
                    if not image_path.is_file():
                        complete = False
                        break
            if complete:
                try:
                    signature = model_fingerprint(json.loads(metadata_path.read_text(encoding="utf-8")))
                except (ValueError, OSError, TypeError):
                    complete = False
            if complete:
                done.add(mid)
                kept_rows.extend(expected_rows)
                fingerprints.add(signature)
            else:
                print(f"WARNING: removing incomplete annotation records for {mid}; it will be regenerated", file=sys.stderr)
                metadata_path.unlink(missing_ok=True)
                for row in rows:
                    image_path = output / str(row.get("image_path", ""))
                    try:
                        image_path.resolve().relative_to(output.resolve())
                        image_path.unlink(missing_ok=True)
                    except (ValueError, OSError):
                        pass

        # Rewrite atomically so a later interruption won't preserve partial rows.
        temporary = annotations.with_suffix(".jsonl.tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept_rows),
            encoding="utf-8",
        )
        temporary.replace(annotations)
    return annotations, done, fingerprints


def main():
    args = args_parse()
    if args.count < 1 or args.renders_per_model < 1 or args.resolution < 64:
        raise SystemExit("count and renders-per-model must be >= 1; resolution must be >= 64")
    output = Path(args.output).resolve()
    for relative in ("images", "metadata", "models/stl", "models/obj"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    annotations_path, done, fingerprints = load_existing(output, args.resume, args.renders_per_model)

    successful = images_created = 0
    errors = []
    mode = "a" if args.resume else "w"
    with annotations_path.open(mode, encoding="utf-8") as annotation_file:
        for index in range(args.count):
            model_id = f"hubcap_{index:06d}"
            if model_id in done:
                continue
            base_seed = args.seed + index if args.seed is not None else random.SystemRandom().randrange(1, 2**31 - 1)
            try:
                # Retry rare parameter-signature collisions so a 10k dataset has no exact/near-exact rows.
                for attempt in range(200):
                    model_seed = (base_seed + attempt * 104729) % (2**31 - 1) or 1
                    p = params(index, model_seed)
                    fingerprint = model_fingerprint(p)
                    if fingerprint not in fingerprints:
                        break
                else:
                    raise RuntimeError("could not create a unique parameter signature after 200 attempts")
                p["seed"] = int(model_seed)
                fingerprints.add(fingerprint)
                clear_scene()
                model_rng = random.Random(model_seed ^ 0x5A17C9)
                mesh_objects = make_hubcap(p, random.Random(model_seed))
                scene = bpy.context.scene
                camera = setup_render(scene, args.resolution, model_rng)
                radius = p["outer_diameter_mm"] / 2000.0
                depth = p["rim_depth_mm"] / 1000.0
                setup_ground(radius, depth, model_rng)
                model_path = None

                for view_index in range(args.renders_per_model):
                    view_rng = random.Random(model_seed * 1009 + view_index * 9176)
                    azimuth, elevation = camera_view(view_index, view_rng)
                    azr, elr = math.radians(azimuth), math.radians(elevation)
                    distance = radius * (4.1 + view_rng.uniform(-0.15, 0.25))
                    camera.location = (distance * math.cos(elr) * math.sin(azr),
                                       -distance * math.cos(elr) * math.cos(azr),
                                       distance * math.sin(elr))
                    look_at(camera)
                    camera.data.lens = view_rng.uniform(48, 62)
                    image_rel = f"images/{model_id}_view_{view_index:03d}.png"
                    scene.render.filepath = str(output / image_rel)
                    bpy.ops.render.render(write_still=True)
                    if view_index == 0 and not args.no_export_models:
                        for file_format in args.export_formats:
                            destination = output / "models" / file_format / f"{model_id}.{file_format}"
                            export_model(mesh_objects, destination, file_format)
                            model_path = f"models/{file_format}/{model_id}.{file_format}"
                    row = {**p,
                           "image_id": f"{model_id}_view_{view_index:03d}",
                           "image_path": image_rel,
                           "model_path": model_path if view_index == 0 else None,
                           "camera_azimuth_deg": round(azimuth, 3),
                           "camera_elevation_deg": round(elevation, 3),
                           "lighting": "randomized_three_area_softboxes",
                           "resolution": [args.resolution, args.resolution],
                           "export_scale": 1000.0}
                    annotation_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    annotation_file.flush()
                    images_created += 1

                (output / "metadata" / f"{model_id}.json").write_text(
                    json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")
                successful += 1
                if successful % 25 == 0 or successful == 1:
                    print(f"[{successful}/{args.count}] generated {model_id}: {p['family']}/{p['design_variant']}, spokes={p['spoke_count']}", flush=True)
            except Exception as exc:
                errors.append({"model_id": model_id, "error": str(exc)})
                print(f"ERROR {model_id}: {exc}", file=sys.stderr, flush=True)
                traceback.print_exc()

    report = {
        "requested_models_this_run": args.count,
        "successful_models_this_run": successful,
        "images_created_this_run": images_created,
        "errors": errors,
        "seed": args.seed,
        "generator_version": VERSION,
        "blender_version": bpy.app.version_string,
        "unique_parameter_signatures": len(fingerprints),
        "units": "mm in exported STL/OBJ; metres internally in Blender",
        "notes": [
            "spoke_count means visible spoke tips; spoke_roots_count is the number of spoke trunks",
            "rim_nominal_diameter_in is derived from outer_diameter_mm, not sampled independently",
            "bulge_depth_mm and face_thickness_mm directly affect the generated geometry",
            "Randomized studio backdrops reduce background shortcuts but do not replace real-photo training",
            "Masks and depth-map outputs are not implemented",
            "Lug wells are visual recesses, not Boolean-cut through-holes; assets are not engineering-certified",
            "For ML, split train/validation/test by model_id, never by individual image rows",
        ],
    }
    (output / "metadata" / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
