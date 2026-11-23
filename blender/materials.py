"""Material presets used by the AutoKolpak synthetic wheel generator."""

import bpy


PRESETS = {
    "silver": {"metallic": 0.82, "roughness": 0.24, "color": (0.58, 0.62, 0.68)},
    "black": {"metallic": 0.72, "roughness": 0.29, "color": (0.035, 0.043, 0.055)},
    "matte": {"metallic": 0.16, "roughness": 0.68, "color": (0.11, 0.12, 0.135)},
    "chrome": {"metallic": 0.96, "roughness": 0.14, "color": (0.72, 0.76, 0.81)},
    "barrel": {"metallic": 0.48, "roughness": 0.42, "color": (0.055, 0.061, 0.072)},
    "hole": {"metallic": 0.18, "roughness": 0.55, "color": (0.008, 0.01, 0.014)},
}


def material_presets():
    """Return a copy of the available PBR material settings."""
    return {key: value.copy() for key, value in PRESETS.items()}


def make_material(name, color, metallic=0.5, roughness=0.3):
    """Create a Blender Principled BSDF material."""
    mat = bpy.data.materials.new(name=name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Metallic"].default_value = float(metallic)
        bsdf.inputs["Roughness"].default_value = float(roughness)
    return mat


def preset_material(name, preset, color=None, metallic=None, roughness=None):
    """Create a material from a named preset with optional randomization."""
    settings = PRESETS[preset]
    return make_material(
        name,
        color if color is not None else settings["color"],
        settings["metallic"] if metallic is None else metallic,
        settings["roughness"] if roughness is None else roughness,
    )
