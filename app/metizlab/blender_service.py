from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path

from app.metizlab.config import Settings

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "blender" / "runtime_generate.py"


class BlenderUnavailable(RuntimeError):
    pass


def blender_binary(settings: Settings) -> str | None:
    configured = os.getenv("BLENDER_EXECUTABLE", "").strip() or getattr(settings, "blender_executable", "")
    if configured and Path(configured).is_file():
        return configured
    for name in ("blender", "blender.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


async def generate_model(settings: Settings, payload: dict) -> dict:
    binary = blender_binary(settings)
    if not binary:
        raise BlenderUnavailable("Blender не найден. Укажите BLENDER_EXECUTABLE в .env.")
    output_dir = Path(settings.generated_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    job = uuid.uuid4().hex
    request_path = output_dir / f"{job}.json"
    model_path = output_dir / f"{job}.glb"
    request_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    command = [binary, "-b", "--python", str(SCRIPT), "--", "--input", str(request_path), "--output", str(model_path)]
    try:
        process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=max(30, settings.request_timeout))
        if process.returncode != 0 or not model_path.is_file():
            detail = (stderr or stdout).decode("utf-8", errors="replace")[-1000:]
            raise RuntimeError(f"Blender не смог создать модель: {detail}")
        if model_path.stat().st_size < 20:
            model_path.unlink(missing_ok=True)
            raise RuntimeError("Blender создал пустой GLB")
        with model_path.open("rb") as model_file:
            header = model_file.read(12)
        if len(header) != 12 or header[:4] != b"glTF" or int.from_bytes(header[4:8], "little") != 2:
            model_path.unlink(missing_ok=True)
            raise RuntimeError("Blender создал некорректный GLB-файл")
    finally:
        request_path.unlink(missing_ok=True)
    return {"url": f"/generated/{model_path.name}", "path": str(model_path), "format": "glb", "size_bytes": model_path.stat().st_size, "validated": True}
