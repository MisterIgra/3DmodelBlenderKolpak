from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, mean_absolute_error

from .constants import CENTER_TYPES, FAMILIES, FINISHES, IMAGE_SIZE, REGRESSION_FIELDS, REGRESSION_RANGES, SPOKE_COUNTS, denormalize_ratios
from .dataset import HubcapDataset, load_annotations, split_by_model
from .inference import load_checkpoint


def evaluate(data_dir: str | Path, checkpoint: str | Path, output: str | Path = "evaluation.json") -> dict:
    model, device, ckpt = load_checkpoint(checkpoint)
    rows = load_annotations(data_dir)
    ids = ckpt.get("split_ids")
    if ids and "test" in ids:
        test_ids = set(str(x) for x in ids["test"])
        test_rows = [r for r in rows if str(r["model_id"]) in test_ids]
        if not test_rows:
            raise ValueError("Чекпойнт содержит test split, но в указанном датасете нет его model_id. Используйте исходный датасет.")
    else:
        _, _, test_rows, _ = split_by_model(rows, ckpt.get("config", {}).get("validation_fraction", .15),
                                             ckpt.get("config", {}).get("test_fraction", .15),
                                             ckpt.get("config", {}).get("seed", 42))
    if not test_rows:
        raise ValueError("Test split пуст; добавьте уникальные модели")
    loader = DataLoader(HubcapDataset(data_dir, test_rows, False, int(ckpt.get("image_size", IMAGE_SIZE))),
                        batch_size=32, shuffle=False, num_workers=0)
    labels = {"family": FAMILIES, "spokes": SPOKE_COUNTS, "center": CENTER_TYPES, "finish": FINISHES}
    y_true = defaultdict(list)
    y_pred = defaultdict(list)
    ratio_true, ratio_pred = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            outputs = model(batch["image"].to(device))
            for key in labels:
                y_true[key].extend(batch[key].numpy().tolist())
                y_pred[key].extend(outputs[key].argmax(1).cpu().numpy().tolist())
            normalized_pred = outputs["regression"].float().clamp(-1, 1).cpu().numpy()
            ratio_pred.extend(denormalize_ratios(normalized_pred).tolist())
            ratio_true.extend(batch["regression_ratios"].numpy().tolist())
    result = {"checkpoint": str(Path(checkpoint).resolve()), "test_images": len(test_rows),
              "test_models": len({str(r["model_id"]) for r in test_rows}), "classification": {}}
    for key, names in labels.items():
        true = np.asarray(y_true[key], dtype=np.int64)
        pred = np.asarray(y_pred[key], dtype=np.int64)
        result["classification"][key] = {
            "accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
            "labels": names,
            "classification_report": classification_report(true, pred, labels=list(range(len(names))),
                                                              target_names=[str(x) for x in names],
                                                              output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(true, pred, labels=list(range(len(names)))).tolist(),
        }
    yt = np.asarray(ratio_true, dtype=np.float64)
    yp = np.asarray(ratio_pred, dtype=np.float64)
    regression_metrics = {}
    for i, field in enumerate(REGRESSION_FIELDS):
        valid = np.isfinite(yt[:, i])
        regression_metrics[field] = {
            "mae_ratio": float(mean_absolute_error(yt[valid, i], yp[valid, i])) if bool(valid.any()) else None,
            "valid_images": int(valid.sum()),
            "unit": "unitless ratio",
        }
    result["regression_ratios"] = regression_metrics
    result["regression_note"] = (
        "Ratio MAE is not millimetres. Convert to mm only using a verified outer diameter. "
        "Center-hole regression is evaluated only when the center hole is visually open; "
        "a decorative center cap hides its diameter."
    )
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description="Evaluate AutoKolpak AI v2 on a fixed model-level test split")
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", default="./runs/best.pt")
    p.add_argument("--output", default="./runs/evaluation.json")
    args = p.parse_args()
    print(json.dumps(evaluate(args.data, args.checkpoint, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
