from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BLENDER_DIR = ROOT / "blender"


def _import_generator_without_blender():
    # The generator's parameter sampling does not invoke Blender; only imports need stubs.
    bpy_stub = types.ModuleType("bpy")
    bpy_stub.app = types.SimpleNamespace(version_string="pytest-stub")
    sys.modules.setdefault("bpy", bpy_stub)
    sys.modules.setdefault("bmesh", types.ModuleType("bmesh"))
    if "mathutils" not in sys.modules:
        mathutils_stub = types.ModuleType("mathutils")
        mathutils_stub.Vector = object
        sys.modules["mathutils"] = mathutils_stub
    if str(BLENDER_DIR) not in sys.path:
        sys.path.insert(0, str(BLENDER_DIR))
    return importlib.import_module("generate_dataset")


def test_generator_parameter_sampler_has_full_label_schema():
    generator = _import_generator_without_blender()
    from autokolpak_ai.constants import CENTER_TYPES, DESIGN_VARIANTS, FAMILIES, FINISHES, SPOKE_COUNTS

    signatures = set()
    observed_spokes = set()
    for index in range(1000):
        row = generator.params(index, 20261021 + index)
        signature = generator.model_fingerprint(row)
        assert signature not in signatures
        signatures.add(signature)
        assert row["family"] in FAMILIES
        assert row["design_variant"] in DESIGN_VARIANTS
        assert row["center_type"] in CENTER_TYPES
        assert row["finish"] in FINISHES
        assert row["spoke_count"] in SPOKE_COUNTS
        roots = row["spoke_roots_count"]
        expected_count = roots * 2 if row["family"] in ("v_shaped", "split", "y_shaped") else roots
        assert row["spoke_count"] == expected_count
        observed_spokes.add(row["spoke_count"])
    assert observed_spokes == set(SPOKE_COUNTS)


def test_generator_resume_discards_partial_models(tmp_path: Path):
    generator = _import_generator_without_blender()
    output = tmp_path / "dataset"
    images = output / "images"
    metadata = output / "metadata"
    images.mkdir(parents=True)
    metadata.mkdir()

    complete = generator.params(0, 10001)
    complete_rows = []
    for view in range(2):
        relative = f"images/{complete['model_id']}_view_{view:03d}.png"
        Image.new("RGB", (16, 16), (50, 50, 50)).save(output / relative)
        complete_rows.append({**complete, "image_id": f"{complete['model_id']}_view_{view:03d}", "image_path": relative})
    (metadata / f"{complete['model_id']}.json").write_text(json.dumps(complete), encoding="utf-8")

    partial = generator.params(1, 10002)
    partial_image = f"images/{partial['model_id']}_view_000.png"
    Image.new("RGB", (16, 16), (20, 20, 20)).save(output / partial_image)
    partial_row = {**partial, "image_id": f"{partial['model_id']}_view_000", "image_path": partial_image}

    annotation_path = metadata / "annotations.jsonl"
    annotation_path.write_text(
        "".join(json.dumps(row) + "\n" for row in complete_rows + [partial_row]), encoding="utf-8"
    )
    _, done, fingerprints = generator.load_existing(output, resume=True, renders_per_model=2)
    assert done == {complete["model_id"]}
    assert len(fingerprints) == 1
    persisted = [json.loads(line) for line in annotation_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(persisted) == 2
    assert all(row["model_id"] == complete["model_id"] for row in persisted)
    assert not (output / partial_image).exists()
    assert not (metadata / f"{partial['model_id']}.json").exists()
