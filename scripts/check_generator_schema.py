"""Preflight the Blender parameter generator without requiring Blender to render.

This checks the generator's parameter sampling and label schema only. It does NOT
validate actual Blender geometry or rendered pixels; run Blender to do that.
"""
from __future__ import annotations

import argparse
import sys
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLENDER_DIR = ROOT / "blender"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BLENDER_DIR))

# The sampled params()/fingerprint functions don't call Blender, but module imports
# otherwise require bpy, bmesh and mathutils. Stub only those import-time symbols.
if "bpy" not in sys.modules:
    bpy_stub = types.ModuleType("bpy")
    bpy_stub.app = types.SimpleNamespace(version_string="schema-preflight")
    sys.modules["bpy"] = bpy_stub
if "bmesh" not in sys.modules:
    sys.modules["bmesh"] = types.ModuleType("bmesh")
if "mathutils" not in sys.modules:
    mathutils_stub = types.ModuleType("mathutils")
    mathutils_stub.Vector = object
    sys.modules["mathutils"] = mathutils_stub

from generate_dataset import model_fingerprint, params  # noqa: E402
from autokolpak_ai.constants import (  # noqa: E402
    CENTER_TYPES, DESIGN_VARIANTS, FAMILIES, FINISHES, RAW_FIELD_RANGES, SPOKE_COUNTS,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight Blender Generator v4 labels and uniqueness")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    fingerprints: set[str] = set()
    family_counts: Counter[str] = Counter()
    spoke_counts: Counter[int] = Counter()
    errors: list[str] = []
    for index in range(args.count):
        row = params(index, args.seed + index)
        fingerprint = model_fingerprint(row)
        if fingerprint in fingerprints:
            errors.append(f"duplicate geometry signature at model index {index}")
        fingerprints.add(fingerprint)
        if row.get("family") not in FAMILIES:
            errors.append(f"model {index}: invalid family={row.get('family')!r}")
        if row.get("design_variant") not in DESIGN_VARIANTS:
            errors.append(f"model {index}: invalid design_variant={row.get('design_variant')!r}")
        if row.get("center_type") not in CENTER_TYPES or row.get("finish") not in FINISHES:
            errors.append(f"model {index}: invalid center/finish label")
        if row.get("spoke_count") not in SPOKE_COUNTS:
            errors.append(f"model {index}: invalid spoke_count={row.get('spoke_count')!r}")
        roots = int(row["spoke_roots_count"])
        expected = roots * 2 if row["family"] in ("v_shaped", "split", "y_shaped") else roots
        if row["spoke_count"] != expected:
            errors.append(f"model {index}: spoke_count does not match spoke_roots_count/family")
        for field, (low, high) in RAW_FIELD_RANGES.items():
            value = float(row[field])
            if not low <= value <= high:
                errors.append(f"model {index}: {field}={value} outside [{low}, {high}]")
        family_counts[row["family"]] += 1
        spoke_counts[int(row["spoke_count"])] += 1

    print(f"Models sampled: {args.count}")
    print(f"Unique signatures: {len(fingerprints)}")
    print(f"Family counts: {dict(sorted(family_counts.items()))}")
    print(f"Spoke counts represented: {len(spoke_counts)}/{len(SPOKE_COUNTS)}")
    if errors:
        print(f"FAIL: {len(errors)} errors")
        for error in errors[:30]:
            print(f" - {error}")
        return 2
    print("PASS: sampled labels/ranges/signatures are consistent. Blender renders were not tested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
