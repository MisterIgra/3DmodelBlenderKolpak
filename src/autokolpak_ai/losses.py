from __future__ import annotations
import torch
import torch.nn.functional as F

CLASSIFICATION_TASKS = ("family", "spokes", "center", "finish")


def multitask_loss(outputs, batch, weights=None, label_smoothing: float = 0.02):
    weights = weights or {}
    losses = {}
    for key in CLASSIFICATION_TASKS:
        target = batch[key].long()
        mask = target != -100
        losses[key] = (F.cross_entropy(outputs[key][mask], target[mask], label_smoothing=label_smoothing)
                       if bool(mask.any()) else outputs[key].sum() * 0.0)
    target = batch["regression"].float()
    pred = outputs["regression"].float()
    mask = torch.isfinite(target)
    losses["regression"] = F.smooth_l1_loss(pred[mask], target[mask], beta=0.10) if bool(mask.any()) else pred.sum() * 0.0
    total = sum(float(weights.get(key, 1.0)) * value for key, value in losses.items())
    return total, {key: value.detach() for key, value in losses.items()}
