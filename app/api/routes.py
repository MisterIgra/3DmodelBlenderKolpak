import io
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from app.metizlab.blender_service import BlenderUnavailable, generate_model
from app.metizlab.cad.catalog import BUSHINGS, FASTENERS, normalize_params
from app.metizlab.config import get_settings
from app.services.ai_service import AIService, ModelNotConfigured

router = APIRouter(prefix="/api")
ai_service = AIService()
ai_service.load()
blender_settings = get_settings()


class UnifiedModelPayload(BaseModel):
    category: str = Field(pattern="^(hubcap|fastener|bushing|bracket)$")
    part_type: str = Field(min_length=1, max_length=64)
    material: str = Field(default="steel", max_length=32)
    params: dict[str, Any] = Field(default_factory=dict)


HUBCAP_RANGES = {
    "outer_diameter_mm": (330, 460), "center_hole_diameter_mm": (50, 74),
    "bulge_depth_mm": (8, 50), "face_thickness_mm": (5, 11), "rim_depth_mm": (90, 170),
    "spoke_count": (5, 24), "bolt_count": (3, 8), "spoke_width_scale": (0.55, 1.65),
    "spoke_sweep_deg": (-22, 22), "spoke_mid_radius_ratio": (0.40, 0.70),
    "spoke_outer_radius_ratio": (0.76, 0.89), "rim_lip_scale": (0.65, 1.45),
}
HUBCAP_ENUMS = {
    "family": {"classic", "v_shaped", "split", "y_shaped", "mesh"},
    "center_type": {"open", "cap"}, "finish": {"solid_silver", "solid_black", "diamond_cut"},
    "design_variant": {"straight", "swept", "wide", "slim", "angular", "turbine", "petal", "concave"},
}


def normalize_hubcap_params(params: dict[str, Any]) -> dict[str, Any]:
    result = dict(params)
    for key, allowed in HUBCAP_ENUMS.items():
        if key in result and result[key] not in allowed:
            raise ValueError(f"Недопустимое значение параметра {key}")
    for key, (low, high) in HUBCAP_RANGES.items():
        if key not in result or result[key] is None:
            continue
        try:
            value = float(result[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Параметр {key} должен быть числом") from exc
        if not low <= value <= high:
            raise ValueError(f"Параметр {key} должен быть в диапазоне {low}–{high}")
        if key in {"spoke_count", "bolt_count"} and not value.is_integer():
            raise ValueError(f"Параметр {key} должен быть целым числом")
        result[key] = int(value) if key in {"spoke_count", "bolt_count"} else value
    result.setdefault("outer_diameter_mm", 380)
    result.setdefault("center_hole_diameter_mm", 60)
    result.setdefault("family", "classic")
    result.setdefault("spoke_count", 7)
    # The geometry generator consumes root count; branch families have two tips per root.
    result["spoke_roots_count"] = int(result["spoke_count"] / 2) if result["family"] in {"split", "v_shaped", "y_shaped"} else int(result["spoke_count"])
    result.setdefault("center_type", "open")
    result.setdefault("finish", "solid_silver")
    result.setdefault("design_variant", "straight")
    result.setdefault("spoke_width_scale", 1.0)
    result.setdefault("spoke_sweep_deg", 0.0)
    result.setdefault("spoke_mid_radius_ratio", 0.53)
    result.setdefault("spoke_outer_radius_ratio", 0.82)
    return result



MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


@router.get("/health")
def health():
    info = ai_service.info()
    return {"status": "ok", "model_loaded": ai_service.ready, "backend": "claude-api", "model": info.get("model"),
            "model_version": info.get("model_version"), "error": None if ai_service.ready else info.get("status")}


@router.get("/model-info")
def model_info():
    return ai_service.info()


@router.post("/predict")
async def predict(file: UploadFile = File(...), reference_outer_diameter_mm: float | None = Form(default=None, ge=330, le=460)):
    if not file.filename:
        raise HTTPException(400, "Не указано имя файла")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "Файл пуст")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Размер изображения не должен превышать 15 МБ")

    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Разрешение изображения не должно превышать 25 Мп")
            image.verify()
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, UnidentifiedImageError, OSError):
        raise HTTPException(400, "Файл не является корректным изображением")
    try:
        analysis = ai_service.predict_image(data, file.filename, file.content_type, reference_outer_diameter_mm)
        parameters = analysis["parameters"]
        try:
            model = await generate_model(blender_settings, {
                "category": "hubcap",
                "part_type": "hubcap",
                "material": "steel",
                "params": parameters,
            })
            return {**analysis, "success": True, "analysis": analysis, "model": model}
        except (BlenderUnavailable, RuntimeError) as exc:
            return {**analysis, "success": True, "analysis": analysis, "model_error": str(exc)}
    except ModelNotConfigured as exc:
        raise HTTPException(503, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))



@router.post("/hubcaps/generate")
async def generate_hubcap(payload: dict[str, Any]):
    """Перестроить колпак по параметрам, исправленным пользователем."""
    try:
        parameters = normalize_hubcap_params(payload.get("params") or {})
        model = await generate_model(blender_settings, {
            "category": "hubcap", "part_type": "hubcap", "material": "steel", "params": parameters,
        })
    except BlenderUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"success": True, "model": model, "parameters": parameters}


@router.post("/models/generate")
async def generate_unified_model(payload: UnifiedModelPayload):
    """Единая точка построения: параметры API передаются в Blender для любого изделия."""
    if payload.category != "hubcap" and payload.part_type not in {item["id"] for item in FASTENERS + BUSHINGS}:
        raise HTTPException(422, "Неизвестный тип изделия")
    data = payload.model_dump()
    if payload.category != "hubcap":
        data["params"] = normalize_params(payload.part_type, data["params"])
    try:
        model = await generate_model(blender_settings, data)
    except BlenderUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"success": True, "model": model, "category": payload.category, "part_type": payload.part_type, "params": data["params"]}
