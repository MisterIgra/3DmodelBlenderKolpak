"""Optional standalone API for the trained model."""
from __future__ import annotations
import io, os
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from .inference import load_checkpoint, predict_pil

app = FastAPI(title="АвтоКолпак AI", version="2.0.0")
CHECKPOINT = Path(os.getenv("AUTOKOLPAK_CHECKPOINT", "./runs/best.pt"))
MAX_BYTES = int(float(os.getenv("AUTOKOLPAK_MAX_UPLOAD_MB", "15")) * 1024 * 1024)
_model = _device = _meta = None
_load_error = None


def _load():
    global _model, _device, _meta, _load_error
    try:
        _model, _device, _meta = load_checkpoint(CHECKPOINT)
        _load_error = None
    except Exception as exc:
        _model = _device = _meta = None
        _load_error = f"{type(exc).__name__}: {exc}"


_load()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "checkpoint": str(CHECKPOINT.resolve()), "error": _load_error}


@app.post("/predict")
async def predict(file: UploadFile = File(...), reference_outer_diameter_mm: float | None = Form(default=None, ge=330, le=460)):
    if _model is None:
        raise HTTPException(503, _load_error or "Модель v2 не загружена")
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise HTTPException(415, "Поддерживаются JPEG, PNG, WEBP и BMP")
    raw = await file.read(MAX_BYTES + 1)
    if not raw:
        raise HTTPException(400, "Файл пуст")
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"Размер изображения не должен превышать {MAX_BYTES // (1024*1024)} МБ")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            return predict_pil(image, _model, _device, _meta, reference_outer_diameter_mm)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "Файл не является корректным изображением") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
