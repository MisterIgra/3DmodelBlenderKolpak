from __future__ import annotations

from pathlib import Path
import torch
from PIL import Image

from .constants import (
    CENTER_TYPES, FAMILIES, FINISHES, IMAGE_SIZE, RAW_FIELD_RANGES, REGRESSION_FIELDS,
    REGRESSION_RANGES, REGRESSION_SOURCE_FIELDS, SCHEMA_VERSION, SPOKE_COUNTS,
    denormalize_ratios,
)
from .model import HubcapNet
from .transforms import build_transforms


def load_checkpoint(checkpoint: str | Path, device: str | torch.device | None = None):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint не найден: {path}. Сначала обучите модель v2.")
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(path, map_location=selected_device, weights_only=False)
    if int(ckpt.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(
            "Этот чекпойнт создан старой несовместимой версией AutoKolpak AI. "
            "Нужно заново обучить модель пакетом v2; старые веса нельзя использовать с новыми метками."
        )
    if ckpt.get("architecture") != HubcapNet.architecture:
        raise ValueError(f"Неизвестная архитектура в чекпойнте: {ckpt.get('architecture')!r}")
    expected_labels = {
        "families": FAMILIES,
        "spoke_counts": SPOKE_COUNTS,
        "center_types": CENTER_TYPES,
        "finishes": FINISHES,
        "regression_fields": REGRESSION_FIELDS,
    }
    for field, expected in expected_labels.items():
        if ckpt.get(field) != expected:
            raise ValueError(
                f"Несовместимый чекпойнт: поле {field!r} равно {ckpt.get(field)!r}, "
                f"ожидалось {expected!r}. Переобучите модель текущим пакетом."
            )
    model = HubcapNet(pretrained=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(selected_device).eval()
    return model, selected_device, ckpt


def _build_result(outputs: dict, ckpt: dict, reference_outer_diameter_mm: float | None = None) -> dict:
    family = FAMILIES[int(outputs["family"].argmax(1).item())]
    spokes = SPOKE_COUNTS[int(outputs["spokes"].argmax(1).item())]
    center = CENTER_TYPES[int(outputs["center"].argmax(1).item())]
    finish = FINISHES[int(outputs["finish"].argmax(1).item())]
    normalized = outputs["regression"][0].detach().float().cpu().numpy()
    ratios = denormalize_ratios(normalized).tolist()
    ratio_values = {key: float(value) for key, value in zip(REGRESSION_FIELDS, ratios)}

    known_diameter = None
    if reference_outer_diameter_mm is not None:
        known_diameter = float(reference_outer_diameter_mm)
        if not 330.0 <= known_diameter <= 460.0:
            raise ValueError("reference_outer_diameter_mm должен быть в диапазоне 330–460 мм: за этими пределами модель не обучалась")
    # Deliberately refuse to invent absolute dimensions when no scale reference was supplied.
    # With a reference scale, bound dimensional estimates to the physical ranges represented
    # in training and update the returned ratios so they agree with the displayed mm values.
    dimensions = {}
    if center == "cap":
        # The actual bore is occluded by the center cap and is not recoverable from this image.
        ratio_values["center_hole_ratio"] = None
    if known_diameter is not None:
        for ratio_field, source_field in zip(REGRESSION_FIELDS, REGRESSION_SOURCE_FIELDS):
            if ratio_field == "center_hole_ratio" and center == "cap":
                dimensions[source_field] = None
                continue
            low, high = RAW_FIELD_RANGES[source_field]
            if source_field == "center_hole_diameter_mm":
                high = min(high, known_diameter * 0.19)
            estimate = ratio_values[ratio_field] * known_diameter
            bounded = float(max(low, min(high, estimate)))
            dimensions[source_field] = bounded
            ratio_values[ratio_field] = bounded / known_diameter
    else:
        dimensions = {field: None for field in REGRESSION_SOURCE_FIELDS}
    params = {
        "outer_diameter_mm": known_diameter,
        **dimensions,
        "family": family,
        "spoke_count": int(spokes),
        "center_type": center,
        "finish": finish,
    }
    confidence = {}
    for key in ("family", "spokes", "center", "finish"):
        probs = outputs[key].softmax(1)[0]
        confidence_key = {"spokes": "spoke_count"}.get(key, key)
        confidence[confidence_key] = {
            "score": float(probs.max().item()),
            "top3": [
                {"label": (FAMILIES if key == "family" else SPOKE_COUNTS if key == "spokes" else CENTER_TYPES if key == "center" else FINISHES)[int(i)],
                 "score": float(probs[int(i)].item())}
                for i in torch.topk(probs, k=min(3, probs.numel())).indices.tolist()
            ],
        }
    note = (
        "Оценки основаны на синтетических изображениях. Размеры в мм ограничены диапазонами обучающего датасета "
        "и пересчитаны по указанному вами наружному диаметру; проверьте результат инструментально перед изготовлением. "
        + ("Диаметр центрального отверстия скрыт декоративной крышкой и не выдаётся. " if center == "cap" else "")
        if known_diameter is not None else
        "Абсолютный масштаб нельзя надёжно восстановить по одной некалиброванной фотографии. "
        "Укажите измеренный наружный диаметр в мм, чтобы пересчитать относительные пропорции в размеры. "
        "Относительные оценки глубины особенно ненадёжны на строго фронтальном фото. "
        + ("Диаметр центрального отверстия скрыт декоративной крышкой и не выдаётся." if center == "cap" else "")
    )
    return {
        "model_version": ckpt.get("architecture", "unknown"),
        "parameters": params,
        "geometry_ratios": ratio_values,
        "confidence": confidence,
        "reference_outer_diameter_mm": known_diameter,
        "units": "mm for parameters; unitless ratios for geometry_ratios",
        "confidence_interpretation": "Softmax scores are ranking scores and are not calibrated probabilities of correctness.",
        "warning": note,
    }


@torch.inference_mode()
def predict_pil(image: Image.Image, model, device, checkpoint_meta: dict,
                reference_outer_diameter_mm: float | None = None) -> dict:
    transform = build_transforms(int(checkpoint_meta.get("image_size", IMAGE_SIZE)), train=False)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    outputs = model(tensor)
    return _build_result(outputs, checkpoint_meta, reference_outer_diameter_mm)


@torch.inference_mode()
def predict_image(image_path: str | Path, checkpoint: str | Path,
                  reference_outer_diameter_mm: float | None = None,
                  export_dir: str | Path | None = None) -> dict:
    model, device, ckpt = load_checkpoint(checkpoint)
    with Image.open(image_path) as im:
        result = predict_pil(im, model, device, ckpt, reference_outer_diameter_mm)
    if export_dir:
        if reference_outer_diameter_mm is None:
            raise ValueError("Для STL-экспорта укажите --reference-diameter-mm: без масштаба физический размер неизвестен")
        from .export import export_model
        result["export"] = export_model(result["parameters"], export_dir)
    return result
