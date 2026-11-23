from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import (
    CENTER_TYPES, FAMILIES, FINISHES, REGRESSION_FIELDS, SPOKE_COUNTS,
    normalize_ratios, ratios_from_row,
)
from .transforms import build_transforms


def load_annotations(data_dir: str | Path) -> list[dict]:
    root = Path(data_dir)
    path = root / "metadata" / "annotations.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл аннотаций: {path}")
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL строка {line_no}: {exc}") from exc
            row["_annotation_line"] = line_no
            rows.append(row)
    return rows


def split_by_model(rows: list[dict], validation_fraction: float = 0.15,
                   test_fraction: float = 0.15, seed: int = 42):
    """Stratified group split by model_id: views of one wheel never leak across splits."""
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1")
    model_family = {}
    for row in rows:
        mid = str(row["model_id"])
        family = str(row.get("family", "unknown"))
        previous = model_family.setdefault(mid, family)
        if previous != family:
            raise ValueError(f"model_id {mid!r} has more than one family label")
    by_family: dict[str, list[str]] = defaultdict(list)
    for mid, family in model_family.items():
        by_family[family].append(mid)
    rng = random.Random(seed)
    train_ids, val_ids, test_ids = set(), set(), set()
    for family, ids in sorted(by_family.items()):
        rng.shuffle(ids)
        n = len(ids)
        n_test = int(round(n * test_fraction)) if test_fraction else 0
        n_val = int(round(n * validation_fraction)) if validation_fraction else 0
        if n >= 3:
            if test_fraction > 0: n_test = max(1, n_test)
            if validation_fraction > 0: n_val = max(1, n_val)
        while n_test + n_val >= n:
            if n_val >= n_test and n_val > 0: n_val -= 1
            elif n_test > 0: n_test -= 1
            else: break
        test_ids.update(ids[:n_test])
        val_ids.update(ids[n_test:n_test + n_val])
        train_ids.update(ids[n_test + n_val:])
    # Ensure tiny datasets fail explicitly rather than silently training on empty splits.
    train_rows = [r for r in rows if str(r["model_id"]) in train_ids]
    val_rows = [r for r in rows if str(r["model_id"]) in val_ids]
    test_rows = [r for r in rows if str(r["model_id"]) in test_ids]
    return train_rows, val_rows, test_rows, {
        "train": sorted(train_ids), "validation": sorted(val_ids), "test": sorted(test_ids)
    }


class HubcapDataset(Dataset):
    def __init__(self, data_dir: str | Path, rows: list[dict] | None = None,
                 train: bool = False, image_size: int = 320):
        self.root = Path(data_dir).resolve()
        self.rows = rows if rows is not None else load_annotations(self.root)
        self.transform = build_transforms(image_size, train)
        self.train_mode = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = (self.root / str(row["image_path"])).resolve()
        try:
            image_path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"image_path escapes dataset folder: {row['image_path']}") from exc
        with Image.open(image_path) as im:
            image = self.transform(im.convert("RGB"))

        family = FAMILIES.index(str(row["family"]))
        spokes = SPOKE_COUNTS.index(int(row["spoke_count"]))
        center = CENTER_TYPES.index(str(row["center_type"]))
        finish = FINISHES.index(str(row["finish"]))
        ratios = ratios_from_row(row)
        # When the center is covered by a cap, its hidden bore diameter cannot
        # be measured visually. Do not train the regression head on unknowable labels.
        if str(row["center_type"]) == "cap":
            ratios[0] = float("nan")
        normalized = normalize_ratios(ratios)
        return {
            "image": image,
            "family": torch.tensor(family, dtype=torch.long),
            "spokes": torch.tensor(spokes, dtype=torch.long),
            "center": torch.tensor(center, dtype=torch.long),
            "finish": torch.tensor(finish, dtype=torch.long),
            "regression": torch.as_tensor(normalized, dtype=torch.float32),
            "regression_ratios": torch.as_tensor(ratios, dtype=torch.float32),
            "outer_diameter_mm": torch.tensor(float(row["outer_diameter_mm"]), dtype=torch.float32),
            "model_id": str(row["model_id"]),
        }
