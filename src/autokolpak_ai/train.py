from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .constants import (
    CENTER_TYPES, FAMILIES, FINISHES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    REGRESSION_FIELDS, REGRESSION_RANGES, SCHEMA_VERSION, SPOKE_COUNTS,
)
from .dataset import HubcapDataset, load_annotations, split_by_model
from .losses import CLASSIFICATION_TASKS, multitask_loss
from .model import HubcapNet
from .validate import validate_dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Avoid forcing deterministic CUDA kernels that can be much slower; seed and split remain reproducible.
    torch.backends.cudnn.benchmark = torch.cuda.is_available()


def _move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def _epoch(model, loader, device, loss_weights, label_smoothing, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    seen = 0
    correct = {key: 0 for key in CLASSIFICATION_TASKS}
    totals = {key: 0 for key in CLASSIFICATION_TASKS}
    ratio_abs_sum = np.zeros(len(REGRESSION_FIELDS), dtype=np.float64)
    ratio_count = np.zeros(len(REGRESSION_FIELDS), dtype=np.int64)
    for batch in loader:
        batch = _move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                outputs = model(batch["image"])
                loss, _ = multitask_loss(outputs, batch, loss_weights, label_smoothing)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
        batch_size = int(batch["image"].shape[0])
        loss_sum += float(loss.detach().item()) * batch_size
        seen += batch_size
        with torch.no_grad():
            for key in CLASSIFICATION_TASKS:
                targets = batch[key]
                valid = targets != -100
                if bool(valid.any()):
                    preds = outputs[key].argmax(dim=1)
                    correct[key] += int((preds[valid] == targets[valid]).sum().item())
                    totals[key] += int(valid.sum().item())
            # outputs are normalized ratios in [-1, 1]. Decode for interpretable ratio MAE.
            pred_n = outputs["regression"].float().clamp(-1, 1).cpu().numpy()
            true_ratio = batch["regression_ratios"].cpu().numpy()
            low = np.asarray([REGRESSION_RANGES[k][0] for k in REGRESSION_FIELDS])
            high = np.asarray([REGRESSION_RANGES[k][1] for k in REGRESSION_FIELDS])
            pred_ratio = low + (pred_n + 1.0) * 0.5 * (high - low)
            valid_ratio = np.isfinite(true_ratio)
            ratio_abs_sum += np.where(valid_ratio, np.abs(pred_ratio - true_ratio), 0.0).sum(axis=0)
            ratio_count += valid_ratio.sum(axis=0)
    result = {"loss": loss_sum / max(1, seen), "images": seen}
    for key in CLASSIFICATION_TASKS:
        result[f"{key}_accuracy"] = correct[key] / max(1, totals[key])
    result["ratio_mae"] = {
        key: (float(ratio_abs_sum[i] / ratio_count[i]) if ratio_count[i] else None)
        for i, key in enumerate(REGRESSION_FIELDS)
    }
    valid_maes = [ratio_abs_sum[i] / ratio_count[i] for i in range(len(REGRESSION_FIELDS)) if ratio_count[i]]
    result["mean_ratio_mae"] = float(np.mean(valid_maes)) if valid_maes else None
    result["ratio_valid_counts"] = {
        key: int(ratio_count[i]) for i, key in enumerate(REGRESSION_FIELDS)
    }
    return result


def _loader(dataset, batch_size, shuffle, workers, device):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0, drop_last=False,
    )


def run_training(config_path: str | Path, data_dir: str | Path | None = None) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    data_dir = Path(data_dir or cfg.get("data_dir", "./generated_dataset")).resolve()
    out_dir = Path(cfg.get("output_dir", "./runs")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = validate_dataset(data_dir, out_dir / "dataset_report.json")
    if not report["ok"]:
        preview = "\n".join(report["errors_preview"][:15])
        raise ValueError(f"Датасет не прошёл проверку ({report['errors_count']} ошибок). Первые ошибки:\n{preview}")
    rows = load_annotations(data_dir)
    tr, va, te, split_ids = split_by_model(
        rows,
        float(cfg.get("validation_fraction", 0.15)),
        float(cfg.get("test_fraction", 0.15)), seed,
    )
    if not tr or not va or not te:
        raise ValueError(f"Пустая выборка: train={len(tr)}, validation={len(va)}, test={len(te)}; добавьте больше уникальных моделей")
    split_path = out_dir / "splits.json"
    split_path.write_text(json.dumps(split_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    split_report = {
        name: {"images": len(part), "models": len({str(r['model_id']) for r in part})}
        for name, part in (("train", tr), ("validation", va), ("test", te))
    }
    (out_dir / "split_report.json").write_text(json.dumps(split_report, ensure_ascii=False, indent=2), encoding="utf-8")

    size = int(cfg.get("image_size", IMAGE_SIZE))
    batch_size = int(cfg.get("batch_size", 16))
    workers = int(cfg.get("num_workers", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.get("mixed_precision", True)) and device.type == "cuda"
    train_loader = _loader(HubcapDataset(data_dir, tr, True, size), batch_size, True, workers, device)
    val_loader = _loader(HubcapDataset(data_dir, va, False, size), batch_size, False, workers, device)

    model = HubcapNet(pretrained=bool(cfg.get("pretrained", True))).to(device)
    head_params = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    backbone_params = [p for name, p in model.named_parameters() if name.startswith("backbone.")]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": float(cfg.get("backbone_learning_rate", 2e-5))},
        {"params": head_params, "lr": float(cfg.get("head_learning_rate", 3e-4))},
    ], weight_decay=float(cfg.get("weight_decay", 1e-4)))
    max_epochs = int(cfg.get("epochs", 45))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, max_epochs), eta_min=float(cfg.get("min_learning_rate", 1e-6))
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):  # compatibility with PyTorch 2.2
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_val = float("inf")
    best_epoch = 0
    start_epoch = 0
    patience = int(cfg.get("patience", 8))
    min_delta = float(cfg.get("min_delta", 0.0005))
    history_path = out_dir / "history.json"
    history = []

    resume_path = cfg.get("resume")
    if resume_path:
        resume_ck = torch.load(resume_path, map_location=device, weights_only=False)
        if resume_ck.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Нельзя продолжить обучение из старого несовместимого чекпойнта. Начните новую обучающую сессию.")
        model.load_state_dict(resume_ck["model_state"])
        if "optimizer_state" in resume_ck:
            optimizer.load_state_dict(resume_ck["optimizer_state"])
        if "scheduler_state" in resume_ck:
            scheduler.load_state_dict(resume_ck["scheduler_state"])
        start_epoch = int(resume_ck.get("epoch", -1)) + 1
        best_val = float(resume_ck.get("best_val_loss", float("inf")))
        best_epoch = int(resume_ck.get("best_epoch", 0))
        if history_path.is_file():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                history = []

    weights = cfg.get("loss_weights", {"family": 1.0, "spokes": 1.2, "center": 0.6, "finish": 0.5, "regression": 1.0})
    no_improve = 0
    if resume_path:
        no_improve = int(resume_ck.get("no_improve", 0))
    for epoch in range(start_epoch, max_epochs):
        train_metrics = _epoch(model, train_loader, device, weights, float(cfg.get("label_smoothing", 0.02)), optimizer, scaler, amp)
        val_metrics = _epoch(model, val_loader, device, weights, float(cfg.get("label_smoothing", 0.02)), None, None, amp)
        scheduler.step()

        improved = val_metrics["loss"] < best_val - min_delta
        if improved:
            best_val = val_metrics["loss"]
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1

        row = {
            "epoch": epoch + 1,
            "train_loss": round(train_metrics["loss"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "best_val_loss": round(best_val, 6),
            "best_epoch": best_epoch,
            **{f"val_{key}_accuracy": round(val_metrics[f"{key}_accuracy"], 6) for key in CLASSIFICATION_TASKS},
            "val_mean_ratio_mae": round(val_metrics["mean_ratio_mae"], 6),
            "val_ratio_mae": val_metrics["ratio_mae"],
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_heads": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        common = {
            "schema_version": SCHEMA_VERSION,
            "architecture": model.architecture,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "no_improve": no_improve,
            "val_metrics": val_metrics,
            "image_size": size,
            "imagenet_mean": IMAGENET_MEAN,
            "imagenet_std": IMAGENET_STD,
            "families": FAMILIES,
            "spoke_counts": SPOKE_COUNTS,
            "center_types": CENTER_TYPES,
            "finishes": FINISHES,
            "regression_fields": REGRESSION_FIELDS,
            "regression_ranges": REGRESSION_RANGES,
            "split_ids": split_ids,
            "split_report": split_report,
            "generator_version_expected": "4.0.0",
            "config": {**cfg, "data_dir": str(data_dir), "output_dir": str(out_dir)},
        }
        if improved:
            torch.save({**common, "model_state": model.state_dict()}, out_dir / "best.pt")
        torch.save({**common, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict()}, out_dir / "last.pt")
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

        if no_improve >= patience:
            print(json.dumps({"early_stopping": True, "epoch": epoch + 1, "best_epoch": best_epoch}, ensure_ascii=False), flush=True)
            break

    if not (out_dir / "best.pt").is_file():
        raise RuntimeError("Не создан best.pt: обучение не выполнило ни одной эпохи")
    return {
        "best_checkpoint": str(out_dir / "best.pt"), "last_checkpoint": str(out_dir / "last.pt"),
        "epochs_completed": len(history), "best_epoch": best_epoch, "best_val_loss": best_val,
        "device": str(device), "amp_enabled": amp, "split_report": split_report,
        "test_models": len(split_ids["test"]), "test_images": len(te),
    }


def main():
    parser = argparse.ArgumentParser(description="Train AutoKolpak AI v2")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    print(json.dumps(run_training(args.config, args.data), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
