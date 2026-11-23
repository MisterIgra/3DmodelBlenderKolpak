from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

from .constants import (
    CENTER_TYPES, DESIGN_VARIANTS, FAMILIES, FINISHES, GENERATOR_VERSION, RAW_FIELD_RANGES, SPOKE_COUNTS,
)
from .dataset import load_annotations

LABEL_KEYS = (
    "family", "design_variant", "outer_diameter_mm", "rim_nominal_diameter_in",
    "spoke_count", "spoke_roots_count", "center_hole_diameter_mm", "bulge_depth_mm",
    "face_thickness_mm", "rim_depth_mm", "latch_count", "bolt_count", "spoke_style",
    "center_sides", "finish", "center_type", "spoke_width_scale", "spoke_sweep_deg",
    "spoke_mid_radius_ratio", "spoke_outer_radius_ratio", "bolt_phase_deg",
    "hole_radius_scale", "rim_lip_scale", "units", "generator_version",
)


def validate_dataset(data_dir: str | Path, report_path: str | Path | None = None) -> dict:
    root = Path(data_dir).resolve()
    rows = load_annotations(root)
    errors: list[str] = []
    warnings: list[str] = []
    seen_paths: set[str] = set()
    model_labels: dict[str, tuple] = {}
    models = set()
    family_by_model = {}
    spoke_by_model = {}
    finish_by_model = {}
    fingerprints: dict[tuple, str] = {}
    model_fingerprints: dict[str, tuple] = {}
    duplicate_fingerprints = 0

    if not rows:
        errors.append("metadata/annotations.jsonl пуст")
    for row in rows:
        line = row.get("_annotation_line", "?")
        mid = str(row.get("model_id", ""))
        rel = str(row.get("image_path", ""))
        if not mid:
            errors.append(f"строка {line}: отсутствует model_id")
            continue
        if not rel:
            errors.append(f"строка {line}: отсутствует image_path")
            continue
        models.add(mid)
        try:
            if rel in seen_paths:
                errors.append(f"строка {line}: image_path повторяется: {rel}")
                continue
            seen_paths.add(rel)
            fp = (root / rel).resolve()
            fp.relative_to(root)
            if not fp.is_file():
                errors.append(f"строка {line}: файл не найден: {rel}")
                continue
            with Image.open(fp) as im:
                im.verify()
        except (ValueError, OSError) as exc:
            errors.append(f"строка {line}: недопустимое/повреждённое изображение {rel}: {exc}")
            continue

        generator_version = str(row.get("generator_version", ""))
        if generator_version != GENERATOR_VERSION:
            errors.append(f"строка {line}: generator_version={generator_version!r}; ожидается {GENERATOR_VERSION!r}; не смешивайте версии генератора")

        family = row.get("family")
        if family not in FAMILIES:
            errors.append(f"строка {line}: family={family!r}; ожидается одно из {FAMILIES}")
        if row.get("design_variant") not in DESIGN_VARIANTS:
            errors.append(f"строка {line}: неизвестный design_variant={row.get('design_variant')!r}")
        if row.get("center_type") not in CENTER_TYPES:
            errors.append(f"строка {line}: неизвестный center_type={row.get('center_type')!r}")
        if row.get("finish") not in FINISHES:
            errors.append(f"строка {line}: неизвестный finish={row.get('finish')!r}")
        try:
            spokes = int(row.get("spoke_count"))
            if spokes not in SPOKE_COUNTS:
                errors.append(f"строка {line}: spoke_count={spokes}; ожидается диапазон 5–24")
        except (ValueError, TypeError):
            errors.append(f"строка {line}: неверный spoke_count={row.get('spoke_count')!r}")
            spokes = -1

        try:
            roots = int(row["spoke_roots_count"])
            expected_spokes = roots * 2 if family in ("v_shaped", "split", "y_shaped") else roots
            if roots < 1 or spokes != expected_spokes:
                errors.append(
                    f"строка {line}: spoke_count={spokes} противоречит spoke_roots_count={roots} для family={family!r}"
                )
        except (KeyError, ValueError, TypeError):
            errors.append(f"строка {line}: отсутствует/неверное spoke_roots_count")

        numeric = {}
        for field, (low, high) in RAW_FIELD_RANGES.items():
            try:
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError("not finite")
                numeric[field] = value
                if value < low - 1e-4 or value > high + 1e-4:
                    errors.append(f"строка {line}: {field}={value} вне диапазона [{low}, {high}]")
            except (KeyError, ValueError, TypeError):
                errors.append(f"строка {line}: отсутствует/неверное поле {field}")

        if numeric.get("outer_diameter_mm", 0) > 0 and row.get("rim_nominal_diameter_in") is not None:
            expected = max(13, min(18, int(round(numeric["outer_diameter_mm"] / 25.4))))
            try:
                actual = int(row["rim_nominal_diameter_in"])
                if actual != expected:
                    warnings.append(f"строка {line}: rim_nominal_diameter_in={actual}; по генератору ожидается {expected}")
            except (ValueError, TypeError):
                warnings.append(f"строка {line}: неверный rim_nominal_diameter_in")

        current_labels = tuple(row.get(k) for k in LABEL_KEYS)
        previous = model_labels.setdefault(mid, current_labels)
        if previous != current_labels:
            errors.append(f"model_id={mid}: у разных ракурсов различаются метки/параметры")
        family_by_model[mid] = family
        spoke_by_model[mid] = spokes
        finish_by_model[mid] = row.get("finish")
        # Signature excludes camera/image fields and random seed. Rounding mirrors generator fingerprint logic.
        sig_fields = ("family", "design_variant", "outer_diameter_mm", "spoke_count", "spoke_roots_count",
                      "center_hole_diameter_mm", "bulge_depth_mm", "face_thickness_mm", "rim_depth_mm",
                      "bolt_count", "center_type", "center_sides", "finish", "spoke_width_scale",
                      "spoke_sweep_deg", "spoke_mid_radius_ratio", "spoke_outer_radius_ratio", "bolt_phase_deg",
                      "hole_radius_scale", "rim_lip_scale")
        sig = tuple(round(v, 2) if isinstance(v, float) else v for v in (row.get(k) for k in sig_fields))
        if mid not in model_fingerprints:
            model_fingerprints[mid] = sig
            previous_mid = fingerprints.get(sig)
            if previous_mid is not None and previous_mid != mid:
                duplicate_fingerprints += 1
            else:
                fingerprints[sig] = mid

    if len(models) < 3:
        errors.append(f"Нужно минимум 3 уникальные модели; найдено {len(models)}")
    fam_counts = Counter(family_by_model.values())
    spoke_counts = Counter(spoke_by_model.values())
    finish_counts = Counter(finish_by_model.values())
    if set(fam_counts) != set(FAMILIES):
        warnings.append(f"В датасете не представлены все семейства: {dict(fam_counts)}")
    if duplicate_fingerprints:
        errors.append(f"Найдено повторяющихся геометрических сигнатур: {duplicate_fingerprints}; для чистого обучения пересоздайте/очистите датасет")
    for family in FAMILIES:
        if fam_counts.get(family, 0) < 30:
            warnings.append(f"Мало моделей семейства {family}: {fam_counts.get(family, 0)} (желательно >=30)")
    for spoke_count in SPOKE_COUNTS:
        n_spoke = spoke_counts.get(spoke_count, 0)
        if n_spoke < 30:
            warnings.append(f"Мало моделей с spoke_count={spoke_count}: {n_spoke} (желательно >=30)")

    report = {
        "schema_version": 2,
        "generator_version_expected": GENERATOR_VERSION,
        "data_dir": str(root),
        "annotation_rows": len(rows),
        "valid_unique_images": len(seen_paths),
        "unique_models": len(models),
        "family_models": dict(fam_counts),
        "spoke_count_models": dict(sorted(spoke_counts.items())),
        "finish_models": dict(finish_counts),
        "duplicate_geometry_signatures": duplicate_fingerprints,
        "errors_count": len(errors),
        "errors_preview": errors[:300],
        "warnings_count": len(warnings),
        "warnings_preview": warnings[:100],
        "ok": not errors,
        "notes": [
            "Ракурсы одной модели обязаны оставаться в одной выборке.",
            "Абсолютный масштаб по одному фото не наблюдаем без известного размера-эталона.",
        ],
    }
    if report_path:
        dest = Path(report_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
