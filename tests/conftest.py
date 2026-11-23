from __future__ import annotations
import json
from pathlib import Path
import pytest
from PIL import Image, ImageDraw

FAMILIES = ["classic", "v_shaped", "split", "y_shaped", "mesh"]
FINISHES = ["solid_silver", "solid_black", "diamond_cut"]
DESIGN_VARIANTS = ["straight", "swept", "wide", "slim", "angular", "turbine", "petal", "concave"]


@pytest.fixture
def tiny_dataset(tmp_path: Path):
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    (root / "metadata").mkdir()
    rows = []
    for family_index, family in enumerate(FAMILIES):
        for j in range(4):
            i = family_index * 4 + j
            mid = f"test_{i:03d}"
            diameter = 350.0 + i * 2.0
            if family == "classic": spokes = 5 + j
            elif family in ("v_shaped", "split", "y_shaped"): spokes = 10 + 2 * j
            else: spokes = 14 + j
            row = {
                "model_id": mid,
                "family": family,
                "design_variant": DESIGN_VARIANTS[i % len(DESIGN_VARIANTS)],
                "outer_diameter_mm": diameter,
                "rim_nominal_diameter_in": max(13, min(18, int(round(diameter / 25.4)))),
                "spoke_count": spokes,
                "spoke_roots_count": max(5, spokes // 2) if family in ("v_shaped", "split", "y_shaped") else spokes,
                "center_hole_diameter_mm": 50.0 + (i % 5) * 1.5,
                "bulge_depth_mm": 8.0 + (i % 6) * 2.0,
                "face_thickness_mm": 5.0 + (i % 4) * 1.5,
                "rim_depth_mm": 90.0 + (i % 5) * 15.0,
                "latch_count": 6,
                "bolt_count": 4 + (i % 3),
                "spoke_style": family,
                "center_sides": 6,
                "finish": FINISHES[i % len(FINISHES)],
                "center_type": "open" if i % 2 else "cap",
                "spoke_width_scale": 0.7 + (i % 6) * 0.1,
                "spoke_sweep_deg": -10.0 + i,
                "spoke_mid_radius_ratio": 0.45 + (i % 5) * 0.02,
                "spoke_outer_radius_ratio": 0.78 + (i % 5) * 0.015,
                "bolt_phase_deg": float(i * 13),
                "hole_radius_scale": 0.8 + (i % 5) * 0.05,
                "rim_lip_scale": 0.8 + (i % 5) * 0.1,
                "seed": 1000 + i,
                "units": "mm",
                "generator_version": "4.0.0",
            }
            for view in range(2):
                rel = f"images/{mid}_view_{view:03d}.png"
                img = Image.new("RGB", (96 + view * 12, 96), (32 + 4 * view, 35, 38))
                draw = ImageDraw.Draw(img)
                draw.ellipse((13, 13, 82, 82), fill=(170, 174, 180), outline=(220, 220, 225), width=3)
                draw.ellipse((37, 37, 58, 58), fill=(30, 30, 30))
                img.save(root / rel)
                rows.append({**row, "image_id": f"{mid}_view_{view:03d}", "image_path": rel,
                             "model_path": None, "camera_azimuth_deg": 0 if view == 0 else 45,
                             "camera_elevation_deg": 80 if view == 0 else 40, "resolution": [96, 96]})
    with (root / "metadata" / "annotations.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return root
