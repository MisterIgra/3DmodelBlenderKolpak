import base64
import json
import re
from typing import Any

import httpx

from app.metizlab.cad.catalog import FASTENERS, BUSHINGS, PARAM_LABELS, normalize_params
from app.metizlab.config import Settings

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

ANALYSIS_SCHEMA = {
    "category": "fastener, bushing или bracket",
    "type": "ID подходящей модели из каталога METIZLAB; для L-образной пластины используй angle_bracket",
    "material": "steel, stainless, zinc, brass, aluminum, plastic или unknown",
    "params": {
        "length": "number или null", "diameter": "number или null",
        "inner_diameter": "number или null", "hole_diameter": "number или null",
        "wall_thickness": "number или null", "width": "number или null",
        "height": "number или null", "head_diameter": "number или null",
        "head_height": "number или null", "wrench_size": "number или null",
        "flange_diameter": "number или null", "flange_height": "number или null",
        "thread_pitch": "number или null", "thread_length": "number или null",
        "holes_count": "number или null",
    },
    "confidence": "число от 0 до 1, уверенность классификации",
    "unknown_dimensions": ["массив названий неизвестных размеров"],
    "observations": "краткое описание видимых особенностей",
}


def _extract_json(text: str) -> dict:
    text = text.strip()
    # Remove optional Markdown code fences without requiring them.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("AI вернул ответ без JSON-объекта")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("AI вернул JSON не в виде объекта")
    return value


def _anthropic_messages_url(base_url: str) -> str:
    """Build the Messages endpoint from ANTHROPIC_BASE_URL.

    Accepts https://host, https://host/v1, or a full /v1/messages URL.
    """
    base = base_url.strip().rstrip("/")
    if not base:
        raise ValueError("Не задан ANTHROPIC_BASE_URL")
    if base.endswith("/v1/messages"):
        return base
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def _anthropic_text(response_json: dict) -> str:
    blocks = response_json.get("content") or []
    return "\n".join(
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
    ).strip()


def _safe_error(exc: Exception, secret: str) -> str:
    message = str(exc)
    if secret:
        message = message.replace(secret, "[СКРЫТЫЙ API-КЛЮЧ]")
    return message[:450] or exc.__class__.__name__


async def _anthropic_request(
    settings: Settings,
    *,
    system_prompt: str,
    user_content: list[dict],
    max_tokens: int,
) -> str:
    if not settings.ai_api_key:
        raise RuntimeError("Не задан API-ключ. Заполните ANTHROPIC_AUTH_TOKEN в файле .env и перезапустите приложение.")

    url = _anthropic_messages_url(settings.ai_base_url)
    headers = {
        "Authorization": f"Bearer {settings.ai_api_key}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model": settings.ai_model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_content}],
    }

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            response = await client.post(url, headers=headers, json=body)
        if response.is_error:
            try:
                error_payload = response.json()
                detail = error_payload.get("error", {}).get("message") or error_payload.get("message") or response.text
            except Exception:
                detail = response.text
            safe_detail = str(detail).replace(settings.ai_api_key, "[СКРЫТЫЙ API-КЛЮЧ]")[:350]
            raise RuntimeError(f"API вернул HTTP {response.status_code}: {safe_detail}")
        result = response.json()
        text = _anthropic_text(result)
        if not text:
            raise RuntimeError("AI API ответил, но не вернул текстовый блок")
        return text
    except httpx.TimeoutException as exc:
        raise RuntimeError("Истекло время ожидания ответа AI API. Попробуйте ещё раз.") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Не удалось подключиться к AI API: {_safe_error(exc, settings.ai_api_key)}") from exc


async def _openai_analyze(settings: Settings, images: list[tuple[bytes, str]], instructions: str) -> dict:
    # Kept as a compatibility option for existing users of the original project.
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("Библиотека OpenAI не установлена. Выполните: pip install -r requirements.txt") from exc
    client = AsyncOpenAI(api_key=settings.ai_api_key, base_url=settings.ai_base_url, timeout=settings.request_timeout)
    content = [{"type": "input_text", "text": instructions}]
    for raw, mime in images:
        encoded = base64.b64encode(raw).decode("ascii")
        content.append({"type": "input_image", "image_url": f"data:{mime};base64,{encoded}", "detail": "high"})
    try:
        response = await client.responses.create(
            model=settings.ai_model,
            input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_object"}},
            max_output_tokens=1800,
        )
        value = getattr(response, "output_text", None)
        if not value:
            chunks = []
            for item in getattr(response, "output", []) or []:
                for block in getattr(item, "content", []) or []:
                    if getattr(block, "text", None):
                        chunks.append(block.text)
            value = "\n".join(chunks)
        return _extract_json(value or "")
    except Exception as exc:
        raise RuntimeError(f"Не удалось получить результат от AI API: {_safe_error(exc, settings.ai_api_key)}") from exc


def _build_prompt(reference_mm: float | None, extra_notes: str) -> str:
    catalog_types = [f"{item['id']} — {item['name']}" for item in FASTENERS + BUSHINGS]
    return (
        "Ты инженер по визуальному анализу крепежа, уголков, кронштейнов и втулок. Проанализируй загруженные фотографии. "
        "Если деталь имеет L-образную форму из двух перпендикулярных пластин, это angle_bracket и category=bracket; никогда не называй её втулкой. "
        "Если у детали нет цилиндрического корпуса с осевым отверстием, запрещено выбирать любой bushing. "
        "Верни только корректный JSON-объект, без Markdown и комментариев, со следующими полями и типами: "
        f"{json.dumps(ANALYSIS_SCHEMA, ensure_ascii=False)}. "
        f"Выбери ближайший type из каталога: {', '.join(catalog_types)}. "
        "Все размеры указывай в миллиметрах. Не выдумывай точность: одна фотография без масштаба не позволяет достоверно "
        "определить абсолютные размеры. Если размер невозможно определить, верни null и добавь его в unknown_dimensions. "
        "confidence от 0 до 1 означает уверенность визуальной классификации, а не точность размеров. "
        "Параметры, не относящиеся к данной детали, заполни null. Обрати внимание на отверстия, буртики, резьбу, "
        "форму головки, внутреннюю полость, пропорции и количество видимых элементов. "
        f"Контрольный размер, указанный пользователем: {reference_mm if reference_mm is not None else 'не задан'} мм. "
        f"Дополнительные сведения пользователя: {extra_notes or 'нет'}."
    )


async def analyze_images(
    settings: Settings,
    images: list[tuple[bytes, str]],
    reference_mm: float | None,
    extra_notes: str = "",
) -> dict:
    if not settings.ai_api_key:
        raise RuntimeError("Не задан API-ключ. Заполните ANTHROPIC_AUTH_TOKEN в файле .env и перезапустите приложение.")
    if not images:
        raise ValueError("Загрузите хотя бы одну фотографию")

    for _raw, mime in images:
        if mime not in ALLOWED_TYPES:
            raise ValueError(f"Формат {mime} не поддерживается. Используйте JPG, PNG или WEBP.")

    instructions = _build_prompt(reference_mm, extra_notes)
    if settings.uses_anthropic_api:
        user_content = []
        for raw, mime in images:
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": base64.b64encode(raw).decode("ascii"),
                },
            })
        user_content.append({"type": "text", "text": instructions})
        raw_text = await _anthropic_request(
            settings,
            system_prompt="Ты точный инженерный помощник по анализу изображений деталей. Соблюдай формат JSON в инструкции пользователя.",
            user_content=user_content,
            max_tokens=1800,
        )
        payload = _extract_json(raw_text)
    else:
        payload = await _openai_analyze(settings, images, instructions)

    part_type = payload.get("type", "hex_bolt")
    category = payload.get("category", "fastener")
    known = {item["id"] for item in FASTENERS + BUSHINGS}
    if part_type not in known:
        raise ValueError(f"AI вернул неизвестный тип детали: {part_type}")
    if part_type == "angle_bracket":
        canonical_category = "bracket"
    elif part_type in {item["id"] for item in BUSHINGS}:
        canonical_category = "bushing"
    else:
        canonical_category = "fastener"
    params = normalize_params(part_type, payload.get("params") or {})
    unknown = list(payload.get("unknown_dimensions") or [])
    raw_params = payload.get("params") or {}
    unknown_parameter_keys = [key for key in raw_params if key in PARAM_LABELS and raw_params.get(key) is None]
    for key in unknown_parameter_keys:
        label = PARAM_LABELS[key][0]
        if label not in unknown and key not in unknown:
            unknown.append(label)
    return {
        "category": canonical_category,
        "type": part_type,
        "material": payload.get("material", "unknown"),
        "params": params,
        "confidence": max(0.0, min(1.0, float(payload.get("confidence", 0.0)))),
        "unknown_dimensions": unknown,
        "unknown_parameter_keys": unknown_parameter_keys,
        "observations": payload.get("observations", ""),
    }


async def check_api_connection(settings: Settings) -> dict:
    if not settings.ai_api_key:
        raise RuntimeError("Не задан API-ключ. Заполните ANTHROPIC_AUTH_TOKEN в файле .env и перезапустите приложение.")

    if settings.uses_anthropic_api:
        text = await _anthropic_request(
            settings,
            system_prompt="Ответь кратко и не добавляй пояснений.",
            user_content=[{"type": "text", "text": "Ответь одним словом: OK."}],
            max_tokens=12,
        )
        return {
            "ok": True,
            "model": settings.ai_model,
            "message": f"Anthropic Messages API ответил: {text[:80]}",
        }

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("Библиотека OpenAI не установлена. Выполните: pip install -r requirements.txt") from exc
    client = AsyncOpenAI(api_key=settings.ai_api_key, base_url=settings.ai_base_url, timeout=settings.request_timeout)
    try:
        await client.responses.create(model=settings.ai_model, input="Ответь одним словом: OK.", max_output_tokens=8)
        return {"ok": True, "model": settings.ai_model, "message": "Responses API ответил успешно."}
    except Exception as exc:
        raise RuntimeError(f"Проверка AI API не прошла: {_safe_error(exc, settings.ai_api_key)}") from exc
