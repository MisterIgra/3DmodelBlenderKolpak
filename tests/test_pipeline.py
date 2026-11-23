from __future__ import annotations
import json
from pathlib import Path
import pytest
import numpy as np
import torch
from PIL import Image

from autokolpak_ai.constants import (
    CENTER_TYPES, FAMILIES, FINISHES, REGRESSION_FIELDS, REGRESSION_RANGES, SCHEMA_VERSION,
    SPOKE_COUNTS, denormalize_ratios, normalize_ratios,
)
from autokolpak_ai.dataset import HubcapDataset, load_annotations, split_by_model
from autokolpak_ai.losses import multitask_loss
from autokolpak_ai.model import HubcapNet
from autokolpak_ai.inference import load_checkpoint
from autokolpak_ai.transforms import build_transforms
from autokolpak_ai.validate import validate_dataset


def test_schema_matches_blender_v4():
    assert FAMILIES == ["classic", "v_shaped", "split", "y_shaped", "mesh"]
    assert SPOKE_COUNTS == list(range(5, 25))
    assert CENTER_TYPES == ["open", "cap"]
    assert FINISHES == ["solid_silver", "solid_black", "diamond_cut"]


def test_ratio_normalization_roundtrip():
    values = np.asarray([(a+b)/2 for a,b in REGRESSION_RANGES.values()], dtype=np.float32)
    restored = denormalize_ratios(normalize_ratios(values))
    np.testing.assert_allclose(values, restored, rtol=1e-6, atol=1e-7)


def test_split_never_leaks_views(tiny_dataset):
    rows = load_annotations(tiny_dataset)
    tr, va, te, ids = split_by_model(rows, .15, .15, 42)
    sets = [set(ids[x]) for x in ("train", "validation", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(tr) + len(va) + len(te) == len(rows)
    for model_id in {r["model_id"] for r in rows}:
        locations = sum(any(r["model_id"] == model_id for r in split) for split in (tr, va, te))
        assert locations == 1


def test_non_square_preprocessing_preserves_square_shape():
    image = Image.new("RGB", (240, 100), (100, 110, 120))
    tensor = build_transforms(128, train=False)(image)
    assert tuple(tensor.shape) == (3, 128, 128)
    assert torch.isfinite(tensor).all()


def test_dataset_validation_accepts_v4_and_groups_views(tiny_dataset):
    report = validate_dataset(tiny_dataset)
    assert report["ok"], report["errors_preview"]
    assert report["unique_models"] == 20
    assert report["annotation_rows"] == 40
    assert report["duplicate_geometry_signatures"] == 0


def test_model_outputs_and_loss_are_finite():
    torch.set_num_threads(1)
    model = HubcapNet(pretrained=False)
    x = torch.rand(2, 3, 64, 64)
    output = model(x)
    assert output["family"].shape == (2, len(FAMILIES))
    assert output["spokes"].shape == (2, len(SPOKE_COUNTS))
    assert output["center"].shape == (2, len(CENTER_TYPES))
    assert output["finish"].shape == (2, len(FINISHES))
    assert output["regression"].shape == (2, len(REGRESSION_FIELDS))
    batch = {
        "family": torch.tensor([0, 1]), "spokes": torch.tensor([0, 2]),
        "center": torch.tensor([0, 1]), "finish": torch.tensor([0, 2]),
        "regression": torch.zeros(2, len(REGRESSION_FIELDS)),
    }
    loss, components = multitask_loss(output, batch)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    assert model.family_head[0].weight.grad is not None


def test_checkpoint_rejects_mismatched_class_order(tmp_path):
    checkpoint = tmp_path / "wrong-label-order.pt"
    torch.save({
        "schema_version": SCHEMA_VERSION,
        "architecture": HubcapNet.architecture,
        "families": list(reversed(FAMILIES)),
    }, checkpoint)
    with pytest.raises(ValueError, match="Несовместимый чекпойнт"):
        load_checkpoint(checkpoint, device="cpu")


def test_inference_bounds_mm_to_training_physical_ranges():
    from autokolpak_ai.inference import _build_result

    outputs = {
        "family": torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]]),
        "spokes": torch.tensor([[0.0] * 19 + [1.0]]),
        "center": torch.tensor([[1.0, 0.0]]),
        "finish": torch.tensor([[0.0, 1.0, 0.0]]),
        # Extreme normalized outputs should not produce out-of-range mm estimates.
        "regression": torch.ones((1, len(REGRESSION_FIELDS))),
    }
    ckpt = {
        "architecture": HubcapNet.architecture,
        "families": FAMILIES,
        "spoke_counts": SPOKE_COUNTS,
        "center_types": CENTER_TYPES,
        "finishes": FINISHES,
        "regression_fields": REGRESSION_FIELDS,
    }
    result = _build_result(outputs, ckpt, reference_outer_diameter_mm=460)
    values = result["parameters"]
    assert values["center_hole_diameter_mm"] <= 74.0
    assert values["center_hole_diameter_mm"] <= 460 * 0.19
    assert 8 <= values["bulge_depth_mm"] <= 28
    assert 5 <= values["face_thickness_mm"] <= 11
    assert 90 <= values["rim_depth_mm"] <= 170
    assert result["geometry_ratios"]["center_hole_ratio"] == values["center_hole_diameter_mm"] / 460


def test_hidden_center_hole_is_not_reported_as_measured():
    from autokolpak_ai.inference import _build_result

    outputs = {
        "family": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
        "spokes": torch.zeros((1, len(SPOKE_COUNTS))),
        "center": torch.tensor([[0.0, 1.0]]),
        "finish": torch.zeros((1, len(FINISHES))),
        "regression": torch.zeros((1, len(REGRESSION_FIELDS))),
    }
    ckpt = {
        "architecture": HubcapNet.architecture,
        "families": FAMILIES, "spoke_counts": SPOKE_COUNTS,
        "center_types": CENTER_TYPES, "finishes": FINISHES,
        "regression_fields": REGRESSION_FIELDS,
    }
    result = _build_result(outputs, ckpt, reference_outer_diameter_mm=430)
    assert result["parameters"]["center_type"] == "cap"
    assert result["parameters"]["center_hole_diameter_mm"] is None
    assert result["geometry_ratios"]["center_hole_ratio"] is None
    assert "скрыт декоративной крышкой" in result["warning"]


def test_export_parameter_sanitizer_matches_training_ranges():
    from autokolpak_ai.geometry import sanitize_parameters

    result = sanitize_parameters({
        "outer_diameter_mm": 460,
        "center_hole_diameter_mm": 120,
        "bulge_depth_mm": 50,
        "face_thickness_mm": 30,
        "rim_depth_mm": 300,
        "spoke_count": 24,
        "family": "mesh",
        "center_type": "open",
    })
    assert 50 <= result["center_hole_diameter_mm"] <= 74
    assert result["center_hole_diameter_mm"] <= 460 * 0.19
    assert 8 <= result["bulge_depth_mm"] <= 28
    assert 5 <= result["face_thickness_mm"] <= 11
    assert 90 <= result["rim_depth_mm"] <= 170


def test_parametric_mesh_uses_millimetres_and_family_geometry():
    from autokolpak_ai.geometry import create_parametric_mesh

    params = {
        "outer_diameter_mm": 430,
        "center_hole_diameter_mm": 58,
        "bulge_depth_mm": 18,
        "face_thickness_mm": 7,
        "rim_depth_mm": 120,
        "spoke_count": 10,
        "center_type": "open",
    }
    classic, _ = create_parametric_mesh({**params, "family": "classic"})
    split, _ = create_parametric_mesh({**params, "family": "split"})
    assert np.isclose(classic.extents[0], 430, atol=1)
    assert np.isclose(classic.extents[1], 430, atol=1)
    assert len(split.faces) > len(classic.faces)
    assert np.isfinite(split.vertices).all()
