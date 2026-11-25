
"""AI-сервис AutoKolpak: распознавание колпака по фото через Claude API (без локальной нейросети).

Настройки берутся из переменных окружения или из файла .env в корне проекта:
  ANTHROPIC_BASE_URL    адрес API-шлюза (по умолчанию https://api.anthropic.com)
  ANTHROPIC_AUTH_TOKEN  ключ (отправляется как Authorization: Bearer ...)
  ANTHROPIC_API_KEY     альтернативный ключ (отправляется как x-api-key)
  AUTOKOLPAK_MODEL      модель (по умолчанию claude-sonnet-5)
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import base64, io, json, os, re, urllib.error, urllib.request
from PIL import Image

FAMILIES = ["classic", "v_shaped", "split", "y_shaped", "mesh"]
SPOKE_COUNTS = list(range(5, 25))
CENTER_TYPES = ["open", "cap"]
FINISHES = ["solid_silver", "solid_black", "diamond_cut"]
RATIO_FIELDS = ["center_hole_ratio", "bulge_depth_ratio", "face_thickness_ratio", "rim_depth_ratio"]
SOURCE_FIELDS = ["center_hole_diameter_mm", "bulge_depth_mm", "face_thickness_mm", "rim_depth_mm"]
SHAPE_FIELDS = {
    "spoke_width_scale": (0.55, 1.65, 1.0),
    "spoke_sweep_deg": (-22.0, 22.0, 0.0),
    "spoke_mid_radius_ratio": (0.40, 0.70, 0.53),
    "spoke_outer_radius_ratio": (0.76, 0.89, 0.82),
    "rim_lip_scale": (0.65, 1.45, 1.0),
    "hole_radius_scale": (0.70, 1.35, 1.0),
}
DESIGN_VARIANTS = ["straight", "swept", "wide", "slim", "angular", "turbine", "petal", "concave"]
RATIO_RANGES = {
    "center_hole_ratio": (50.0 / 460.0, 0.19),
    "bulge_depth_ratio": (8.0 / 460.0, 28.0 / 330.0),
    "face_thickness_ratio": (5.0 / 460.0, 11.0 / 330.0),
    "rim_depth_ratio": (90.0 / 460.0, 170.0 / 330.0),
}
RAW_RANGES = {
    "center_hole_diameter_mm": (50.0, 74.0),
    "bulge_depth_mm": (8.0, 28.0),
    "face_thickness_mm": (5.0, 11.0),
    "rim_depth_mm": (90.0, 170.0),
}
MAX_SIDE = 1568


class ModelNotConfigured(RuntimeError):
    pass


class AIUpstreamError(RuntimeError):
    """An upstream API responded, but did not accept the request."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class AIServiceUnavailable(ModelNotConfigured):
    """The configured upstream cannot be reached or credentials are absent."""

    status_code = 503

    pass


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


PROMPT = f"""Ты — модуль компьютерного зрения для подбора автомобильных колпаков (декоративных колпаков/дисков) в системе AutoKolpak.
Проанализируй фотографию колпака и верни ТОЛЬКО один JSON-объект (без markdown и пояснений) такой структуры:

{{
  "family":  [{{"label": "<одно из {FAMILIES}>", "score": 0.0}}, ... до 3 вариантов по убыванию],
  "spoke_count": [{{"label": <целое {SPOKE_COUNTS[0]}-{SPOKE_COUNTS[-1]}, полное число внешних концов спиц на симметричном колпаке>, "score": 0.0}}, ... до 3 вариантов],
  "center": [{{"label": "<одно из {CENTER_TYPES}>", "score": 0.0}}, ... ],
  "shape": {{
    "design_variant": "<одно из {DESIGN_VARIANTS}>",
    "spoke_width_scale": <число 0.55-1.65>,
    "spoke_sweep_deg": <число -22..22>,
    "spoke_mid_radius_ratio": <число 0.40-0.70>,
    "spoke_outer_radius_ratio": <число 0.76-0.89>,
    "rim_lip_scale": <число 0.65-1.45>,
    "hole_radius_scale": <число 0.70-1.35>,
    "spoke_profiles": [
      {{"root_angle_deg": <0-360>, "mid_offset_deg": <-30..30>, "outer_offset_deg": <-30..30>, "width_scale": <0.55-1.65>}}
    ]
  }}
}},


  "finish":  [{{"label": "<одно из {FINISHES}>", "score": 0.0}}, ... до 3 вариантов],
  "ratios": {{
    "center_hole_ratio": <диаметр центрального отверстия / наружный диаметр, {RATIO_RANGES['center_hole_ratio'][0]:.3f}-{RATIO_RANGES['center_hole_ratio'][1]:.3f}>,
    "bulge_depth_ratio": <выпуклость / наружный диаметр, {RATIO_RANGES['bulge_depth_ratio'][0]:.3f}-{RATIO_RANGES['bulge_depth_ratio'][1]:.3f}>,
    "face_thickness_ratio": <толщина лицевой части / наружный диаметр, {RATIO_RANGES['face_thickness_ratio'][0]:.3f}-{RATIO_RANGES['face_thickness_ratio'][1]:.3f}>,
    "rim_depth_ratio": <глубина обода / наружный диаметр, {RATIO_RANGES['rim_depth_ratio'][0]:.3f}-{RATIO_RANGES['rim_depth_ratio'][1]:.3f}>
  }}
}}

Описание классов:
- family: classic — прямые классические спицы; v_shaped — спицы V-образной формы; split — раздвоенные спицы; y_shaped — Y-образные спицы; mesh — сетчатый/ячеистый рисунок.
- center: open — центральное отверстие открыто; cap — закрыто декоративной крышкой.
- finish: solid_silver — серебристая однотонная; solid_black — чёрная; diamond_cut — алмазная проточка (контрастные блестящие грани на тёмном фоне).
Правила подсчёта spoke_count:
1. Сначала определи ось центра и мысленно раздели полный круг на одинаковые угловые сектора.
2. Считай полный симметричный рисунок колпака, а не только концы, видимые в передней половине изображения.
3. Один счёт = один отдельный элемент, который доходит до внешнего обода. Считай именно внешние концы по кругу.
4. Не считай центральные корни, внутренние ответвления, промежутки, отверстия, болты, грани и отражения.
5. Для split/Y/V каждый внешний конец считается отдельно. Если 5 корней раздваиваются и дают 10 внешних концов, верни 10.
6. Выполни два независимых прохода: по часовой стрелке и через противоположные сектора. Если результаты различаются, выбери наиболее вероятное симметричное значение и снизь score.
7. Не путай количество внешних концов с количеством корней, отверстий или лепестков центральной крышки.
Оценки score — относительные вероятности от 0 до 1 (в сумме по каждому списку не больше 1). Не определяй физические размеры в миллиметрах: размеры вводит пользователь отдельно. Возвращай только относительные пропорции формы и признаки рисунка.
Если на фото нет автомобильного колпака, верни {{"error": "краткая причина по-русски"}}."""


class AIService:
    def __init__(self):
        self.root = Path(os.getenv("AUTOKOLPAK_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
        _load_dotenv(self.root / ".env")
        self.base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").strip().rstrip("/")
        self.auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip()
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.model_name = os.getenv("AUTOKOLPAK_MODEL", "claude-sonnet-5").strip()
        try:
            configured_timeout = float(os.getenv("AUTOKOLPAK_API_TIMEOUT", "90"))
        except ValueError:
            configured_timeout = 90.0
        self.timeout = max(1.0, min(configured_timeout, 300.0))
        self.load_error: str | None = None

    # --- совместимость с routes.py ---
    @property
    def weights_exist(self):
        return True  # локальные веса больше не нужны

    @property
    def ready(self):
        return bool(self.auth_token or self.api_key)

    def load(self):
        self.load_error = None if self.ready else (
            "Не задан ключ API: укажите ANTHROPIC_AUTH_TOKEN в файле .env (см. .env.example)")

    def info(self):
        return {
            "ready": self.ready,
            "backend": "claude-api",
            "model": self.model_name,
            "base_url": self.base_url,
            "model_version": f"claude-api:{self.model_name}",
            "capabilities": ["family", "visible_spoke_count", "center_type", "finish", "geometry_ratios"] if self.ready else [],
            "status": f"Подключено к API ({self.model_name})" if self.ready else self.load_error,
        }

    # --- запрос к API ---
    def _headers(self):
        h = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
        if self.auth_token:
            h["authorization"] = f"Bearer {self.auth_token}"
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    @staticmethod
    def _prepare_image(image_bytes: bytes) -> tuple[str, str]:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im.load()
            im = im.convert("RGB")
            if max(im.size) > MAX_SIDE:
                im.thumbnail((MAX_SIDE, MAX_SIDE))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=92)
        return "image/jpeg", base64.b64encode(buf.getvalue()).decode("ascii")

    def _call_api(self, media_type: str, b64: str) -> str:
        body = {
            "model": self.model_name,
            "max_tokens": 1000,
            "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": PROMPT},
            ]}],
        }
        req = urllib.request.Request(f"{self.base_url}/v1/messages", data=json.dumps(body).encode("utf-8"),
                                     headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            message = f"Upstream API вернул HTTP {exc.code}: {detail}"
            if exc.code in (401, 403):
                raise AIUpstreamError(f"{message} Проверьте ключ и доступ к модели.", 502) from exc
            if exc.code == 429:
                raise AIUpstreamError(f"{message} Проверьте лимит запросов или баланс аккаунта.", 429) from exc
            if exc.code >= 500:
                raise AIServiceUnavailable(message) from exc
            raise AIUpstreamError(message, 502) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AIServiceUnavailable(f"Не удалось связаться с API {self.base_url}: {exc}") from exc
        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        text = "".join(parts).strip()
        if not text:
            raise ValueError("API вернул пустой ответ")
        return text

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.S)
            if not m:
                raise ValueError("Модель вернула ответ не в формате JSON")
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Не удалось разобрать ответ модели: {exc}")

    @staticmethod
    def _pick(items, allowed, key):
        """Нормализует список {label, score}: оставляет допустимые метки, сортирует, возвращает топ-3."""
        out, seen = [], set()
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                continue
            label = it.get("label")
            if allowed and isinstance(allowed[0], int):
                try:
                    numeric_label = float(label)
                    if numeric_label.is_integer():
                        label = int(numeric_label)
                except (TypeError, ValueError):
                    pass
            try:
                score = float(it.get("score", 0))
            except (TypeError, ValueError):
                continue
            if label in allowed and label not in seen:
                seen.add(label)
                out.append({"label": label, "score": max(0.0, min(1.0, score))})
        if not out:
            raise ValueError(f"Модель не определила поле «{key}»")
        out.sort(key=lambda x: -x["score"])
        return out[:3]

    def predict_image(self, image_bytes: bytes, filename: str, content_type: str | None,
                      reference_outer_diameter_mm: float | None = None):
        if not self.ready:
            raise ModelNotConfigured(self.load_error or "Не задан ключ API")
        try:
            media_type, b64 = self._prepare_image(image_bytes)
        except OSError as exc:
            raise ValueError(f"Не удалось обработать изображение: {exc}") from exc
        raw = self._parse_json(self._call_api(media_type, b64))
        if raw.get("error"):
            raise ValueError(str(raw["error"]))

        top = {
            "family": self._pick(raw.get("family"), FAMILIES, "family"),
            "spoke_count": self._pick(raw.get("spoke_count"), SPOKE_COUNTS, "spoke_count"),
            "center": self._pick(raw.get("center"), CENTER_TYPES, "center"),
            "finish": self._pick(raw.get("finish"), FINISHES, "finish"),
        }
        family, spokes = top["family"][0]["label"], top["spoke_count"][0]["label"]
        center, finish = top["center"][0]["label"], top["finish"][0]["label"]

        rr = raw.get("ratios") if isinstance(raw.get("ratios"), dict) else {}
        ratio_values: dict[str, float | None] = {}
        for k in RATIO_FIELDS:
            lo, hi = RATIO_RANGES[k]
            try:
                v = float(rr.get(k))
            except (TypeError, ValueError):
                v = (lo + hi) / 2
            ratio_values[k] = max(lo, min(hi, v))

        known = None
        if reference_outer_diameter_mm is not None:
            known = float(reference_outer_diameter_mm)
            if not 330.0 <= known <= 460.0:
                raise ValueError("reference_outer_diameter_mm должен быть в диапазоне 330–460 мм")
        if center == "cap":
            ratio_values["center_hole_ratio"] = None
        dimensions = {f: None for f in SOURCE_FIELDS}
        if known is not None:
            for rf, sf in zip(RATIO_FIELDS, SOURCE_FIELDS):
                if rf == "center_hole_ratio" and center == "cap":
                    continue
                lo, hi = RAW_RANGES[sf]
                if sf == "center_hole_diameter_mm":
                    hi = min(hi, known * 0.19)
                est = max(lo, min(hi, ratio_values[rf] * known))
                dimensions[sf] = est
                ratio_values[rf] = est / known

        shape_raw = raw.get("shape") if isinstance(raw.get("shape"), dict) else {}
        shape: dict[str, Any] = {}
        for field, (lo, hi, fallback) in SHAPE_FIELDS.items():
            try:
                value = float(shape_raw.get(field, fallback))
            except (TypeError, ValueError):
                value = fallback
            shape[field] = max(lo, min(hi, value))
        design_variant = shape_raw.get("design_variant", "straight")
        if design_variant not in DESIGN_VARIANTS:
            design_variant = "straight"
        profiles_raw = shape_raw.get("spoke_profiles") if isinstance(shape_raw.get("spoke_profiles"), list) else []
        spoke_profiles = []
        for index, item in enumerate(profiles_raw[:24]):
            if not isinstance(item, dict):
                continue
            try:
                angle = float(item.get("root_angle_deg", index * 360.0 / max(1, len(profiles_raw))))
                mid_offset = float(item.get("mid_offset_deg", 0.0))
                outer_offset = float(item.get("outer_offset_deg", 0.0))
                width = float(item.get("width_scale", 1.0))
            except (TypeError, ValueError):
                continue
            spoke_profiles.append({"root_angle_deg": angle % 360.0, "mid_offset_deg": max(-30.0, min(30.0, mid_offset)), "outer_offset_deg": max(-30.0, min(30.0, outer_offset)), "width_scale": max(0.55, min(1.65, width))})
        if not spoke_profiles:
            raise ValueError("AI не извлёк индивидуальные профили спиц по фотографии; готовая шаблонная модель не используется")


        hidden = "Диаметр центрального отверстия скрыт декоративной крышкой и не выдаётся. " if center == "cap" else ""
        warning = (
            ("Оценка выполнена нейросетью Claude по фотографии; размеры в мм — по визуальным пропорциям и указанному наружному диаметру. "
             "Проверьте результат инструментально перед изготовлением. ") + hidden
            if known is not None else
            ("Оценка выполнена нейросетью Claude по фотографии. Абсолютный масштаб по одной фотографии определить нельзя: "
             "укажите измеренный наружный диаметр в мм, чтобы пересчитать пропорции в размеры. ") + hidden
        ).strip()
        return {
            "model_version": f"claude-api:{self.model_name}",
            "parameters": {"outer_diameter_mm": known, **dimensions, "family": family, "spoke_count": int(spokes),
                           "visible_outer_end_count": int(spokes), "center_type": center, "finish": finish,
                           "design_variant": design_variant, **shape},
            "geometry_ratios": ratio_values,
            "geometry": {
                "outer_diameter_mm": known,
                "center_hole_diameter_mm": dimensions["center_hole_diameter_mm"],
                "bulge_depth_mm": dimensions["bulge_depth_mm"],
                "face_thickness_mm": dimensions["face_thickness_mm"],
                "rim_depth_mm": dimensions["rim_depth_mm"],
                "geometry_ratios": ratio_values,
                "spoke_profiles": spoke_profiles,

                "units": "mm",
                "family": family,
                "spoke_count": int(spokes),
                "center_type": center,
                "design_variant": design_variant,
                **shape,
            },
            "confidence": {
                "family": {"score": top["family"][0]["score"], "top3": top["family"]},
                "spoke_count": {"score": top["spoke_count"][0]["score"], "top3": top["spoke_count"]},
                "center": {"score": top["center"][0]["score"], "top3": top["center"]},
                "finish": {"score": top["finish"][0]["score"], "top3": top["finish"]},
            },
            "reference_outer_diameter_mm": known,
            "units": "mm for parameters; unitless ratios for geometry_ratios",
            "confidence_interpretation": "Scores are the model's self-reported relative confidence, not calibrated probabilities.",
            "warning": "Форма и относительные пропорции определены по фото; физические размеры не определяются AI и должны быть введены пользователем.",
        }
