from __future__ import annotations

import mimetypes
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.metizlab.cad.catalog import BUSHINGS, FASTENERS, MATERIALS, defaults_for, normalize_params
from app.metizlab.blender_service import BlenderUnavailable, generate_model
from app.metizlab.config import get_settings
from app.metizlab.database.db import ProjectDatabase
from app.metizlab.services.ai_service import ALLOWED_TYPES, analyze_images, check_api_connection

router = APIRouter(prefix="/api/metiz", tags=["METIZLAB"])
settings = get_settings()
database = ProjectDatabase(settings.database_path)


class ProjectPayload(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    category: str = Field(pattern="^(fastener|bushing|bracket)$")
    part_type: str = Field(min_length=1, max_length=64)
    material: str = Field(default="steel", max_length=32)
    params: dict[str, Any] = Field(default_factory=dict)
    notes: str = Field(default="", max_length=2000)


class ModelPayload(BaseModel):
    category: str = Field(pattern="^(fastener|bushing|bracket|hubcap)$")
    part_type: str = Field(min_length=1, max_length=64)
    material: str = Field(default="steel", max_length=32)
    params: dict[str, Any] = Field(default_factory=dict)


@router.get("/health")
async def health():
    return {"status": "ok", "app": "METIZLAB", "ai_configured": bool(settings.ai_api_key), "ai_model": settings.ai_model}


@router.get("/catalog")
async def catalog():
    return {"fasteners": FASTENERS, "bushings": BUSHINGS, "materials": MATERIALS}


@router.post("/models/generate")
async def generate_part_model(payload: ModelPayload):
    if payload.category != "hubcap" and payload.part_type not in {item["id"] for item in FASTENERS + BUSHINGS}:
        raise HTTPException(status_code=422, detail="Неизвестный тип изделия")
    data = payload.model_dump()
    data["params"] = data["params"] if payload.category == "hubcap" else normalize_params(payload.part_type, data["params"])
    try:
        model = await generate_model(settings, data)
    except BlenderUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"success": True, "model": model, "part_type": payload.part_type, "params": data["params"]}


@router.get("/defaults/{part_type}")
async def part_defaults(part_type: str):
    if part_type not in {item["id"] for item in FASTENERS + BUSHINGS}:
        raise HTTPException(status_code=404, detail="Неизвестный тип изделия")
    return {"params": defaults_for(part_type)}


@router.get("/projects")
async def list_projects():
    return {"projects": database.list_projects()}


@router.post("/projects")
async def save_project(payload: ProjectPayload):
    if payload.part_type not in {item["id"] for item in FASTENERS + BUSHINGS}:
        raise HTTPException(status_code=422, detail="Неизвестный тип изделия")
    if payload.category == "bracket" and payload.part_type != "angle_bracket":
        raise HTTPException(status_code=422, detail="Для категории bracket нужен angle_bracket")
    data = payload.model_dump()
    data["params"] = normalize_params(payload.part_type, data["params"])
    return {"project": database.save_project(data)}


@router.put("/projects/{project_id}")
async def update_project(project_id: str, payload: ProjectPayload):
    if not database.get_project(project_id):
        raise HTTPException(status_code=404, detail="Проект не найден")
    if payload.part_type not in {item["id"] for item in FASTENERS + BUSHINGS}:
        raise HTTPException(status_code=422, detail="Неизвестный тип изделия")
    data = payload.model_dump()
    data["params"] = normalize_params(payload.part_type, data["params"])
    return {"project": database.save_project(data, project_id=project_id)}


@router.get("/projects/{project_id}")
async def get_project(project_id: str):
    project = database.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Проект не найден")
    return {"project": project}


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    if not database.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Проект не найден")
    return {"success": True}


@router.post("/ai/analyze")
async def ai_analyze(photos: list[UploadFile] = File(...), reference_mm: float | None = Form(default=None), extra_notes: str = Form(default="")):
    if not photos or len(photos) > 5:
        raise HTTPException(status_code=400, detail="Можно загрузить от 1 до 5 фотографий")
    images: list[tuple[bytes, str]] = []
    max_bytes = settings.max_upload_mb * 1024 * 1024
    for photo in photos:
        mime = photo.content_type or mimetypes.guess_type(photo.filename or "")[0] or ""
        if mime not in ALLOWED_TYPES:
            raise HTTPException(status_code=415, detail="Поддерживаются JPG, PNG и WEBP")
        data = await photo.read()
        if not data or len(data) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Файл пустой или превышает лимит {settings.max_upload_mb} МБ")
        images.append((data, mime))
    if reference_mm is not None and not 0 < reference_mm <= 10000:
        raise HTTPException(status_code=422, detail="Контрольный размер должен быть от 0 до 10000 мм")
    try:
        analysis = await analyze_images(settings, images, reference_mm, extra_notes)
        try:
            analysis["model"] = await generate_model(settings, {
                "category": analysis["category"], "part_type": analysis["type"],
                "material": analysis.get("material", "steel"), "params": analysis["params"],
            })
        except BlenderUnavailable as exc:
            analysis["model_error"] = str(exc)
        return {"success": True, "analysis": analysis}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ai/check")
async def ai_check():
    try:
        return await check_api_connection(settings)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
