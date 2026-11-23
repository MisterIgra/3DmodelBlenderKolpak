from __future__ import annotations
import json
from pathlib import Path
import yaml
import torch
from autokolpak_ai.train import run_training
from autokolpak_ai.evaluate import evaluate


def test_one_epoch_train_checkpoint_and_evaluate(tiny_dataset, tmp_path):
    torch.set_num_threads(1)
    out = tmp_path / "runs"
    cfg = {
        "seed": 42, "data_dir": str(tiny_dataset), "output_dir": str(out),
        "epochs": 1, "patience": 2, "batch_size": 8, "num_workers": 0,
        "image_size": 64, "validation_fraction": .15, "test_fraction": .15,
        "pretrained": False, "mixed_precision": False,
        "backbone_learning_rate": 0.00001, "head_learning_rate": 0.0001,
        "min_learning_rate": 0.000001, "weight_decay": 0.0001,
        "label_smoothing": 0.02,
        "loss_weights": {"family": 1, "spokes": 1, "center": .5, "finish": .5, "regression": 1},
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    result = run_training(config_path, tiny_dataset)
    checkpoint = out / "best.pt"
    assert checkpoint.is_file()
    assert result["epochs_completed"] == 1
    metrics = evaluate(tiny_dataset, checkpoint, out / "evaluation.json")
    assert metrics["test_models"] > 0
    assert 0 <= metrics["classification"]["family"]["accuracy"] <= 1
    assert (out / "evaluation.json").is_file()
