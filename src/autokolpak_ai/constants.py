"""Canonical label schema shared by Blender generation, training and inference."""
from __future__ import annotations

FAMILIES = ["classic", "v_shaped", "split", "y_shaped", "mesh"]
SPOKE_COUNTS = list(range(5, 25))  # all counts 5–24 are sampled by at least one v4 family
CENTER_TYPES = ["open", "cap"]
FINISHES = ["solid_silver", "solid_black", "diamond_cut"]
DESIGN_VARIANTS = ["straight", "swept", "wide", "slim", "angular", "turbine", "petal", "concave"]

# Absolute millimetre measurements are not identifiable from an uncalibrated
# single photograph. Learn dimensionless ratios and scale them only when a
# known outer diameter is supplied by the user.
REGRESSION_FIELDS = [
    "center_hole_ratio",
    "bulge_depth_ratio",
    "face_thickness_ratio",
    "rim_depth_ratio",
]
REGRESSION_SOURCE_FIELDS = [
    "center_hole_diameter_mm",
    "bulge_depth_mm",
    "face_thickness_mm",
    "rim_depth_mm",
]
REGRESSION_RANGES = {
    "center_hole_ratio": (50.0 / 460.0, 0.19),
    "bulge_depth_ratio": (8.0 / 460.0, 28.0 / 330.0),
    "face_thickness_ratio": (5.0 / 460.0, 11.0 / 330.0),
    "rim_depth_ratio": (90.0 / 460.0, 170.0 / 330.0),
}
RAW_FIELD_RANGES = {
    "outer_diameter_mm": (330.0, 460.0),
    "center_hole_diameter_mm": (50.0, 74.0),
    "bulge_depth_mm": (8.0, 28.0),
    "face_thickness_mm": (5.0, 11.0),
    "rim_depth_mm": (90.0, 170.0),
}
IMAGE_SIZE = 320
SCHEMA_VERSION = 2
GENERATOR_VERSION = "4.0.0"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ratios_from_row(row: dict) -> list[float]:
    diameter = float(row["outer_diameter_mm"])
    if diameter <= 0:
        raise ValueError("outer_diameter_mm must be > 0")
    return [float(row[key]) / diameter for key in REGRESSION_SOURCE_FIELDS]


def normalize_ratios(ratios):
    """Map ratio labels to approximately [-1, 1] for balanced regression loss."""
    import numpy as np

    values = np.asarray(ratios, dtype=np.float32)
    low = np.asarray([REGRESSION_RANGES[k][0] for k in REGRESSION_FIELDS], dtype=np.float32)
    high = np.asarray([REGRESSION_RANGES[k][1] for k in REGRESSION_FIELDS], dtype=np.float32)
    return 2.0 * (values - low) / (high - low) - 1.0


def denormalize_ratios(values):
    """Map network output in [-1, 1] back into dimensionless ratios."""
    import numpy as np

    values = np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)
    low = np.asarray([REGRESSION_RANGES[k][0] for k in REGRESSION_FIELDS], dtype=np.float32)
    high = np.asarray([REGRESSION_RANGES[k][1] for k in REGRESSION_FIELDS], dtype=np.float32)
    return low + (values + 1.0) * 0.5 * (high - low)
