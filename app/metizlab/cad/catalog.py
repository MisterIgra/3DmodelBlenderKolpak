"""Parametric part catalog and safe parameter normalization."""

FASTENERS = [
    {"id": "hex_bolt", "name": "Болт с шестигранной головкой", "group": "Болты", "default": {"length": 50, "diameter": 10, "head_diameter": 16, "head_height": 6, "thread_pitch": 1.5, "thread_length": 32}},
    {"id": "socket_bolt", "name": "Винт с цилиндрической головкой", "group": "Винты", "default": {"length": 40, "diameter": 8, "head_diameter": 13, "head_height": 8, "thread_pitch": 1.25, "thread_length": 28}},
    {"id": "countersunk_screw", "name": "Потайной винт", "group": "Винты", "default": {"length": 35, "diameter": 6, "head_diameter": 12, "head_height": 3.5, "thread_pitch": 1, "thread_length": 30}},
    {"id": "stud", "name": "Шпилька резьбовая", "group": "Шпильки", "default": {"length": 70, "diameter": 10, "thread_pitch": 1.5, "thread_length": 70}},
    {"id": "hex_nut", "name": "Гайка шестигранная", "group": "Гайки", "default": {"height": 8, "diameter": 10, "width": 17, "hole_diameter": 8.5, "thread_pitch": 1.5}},
    {"id": "washer", "name": "Шайба плоская", "group": "Шайбы", "default": {"height": 2, "diameter": 20, "hole_diameter": 10.5}},
    {"id": "rivet", "name": "Заклепка", "group": "Заклепки", "default": {"length": 18, "diameter": 5, "head_diameter": 9, "head_height": 2}},
    {"id": "anchor", "name": "Анкерный болт", "group": "Анкеры", "default": {"length": 80, "diameter": 12, "head_diameter": 18, "head_height": 8, "thread_pitch": 1.75, "thread_length": 35}},
    {"id": "pin", "name": "Цилиндрический штифт", "group": "Штифты", "default": {"length": 30, "diameter": 8}},
    {"id": "split_pin", "name": "Шплинт", "group": "Шплинты", "default": {"length": 35, "diameter": 3}},
    {"id": "self_tapping_screw", "name": "Саморез", "group": "Саморезы", "default": {"length": 40, "diameter": 4.2, "head_diameter": 8, "head_height": 3, "thread_pitch": 2, "thread_length": 35}},
    {"id": "angle_bracket", "name": "Крепежный уголок", "group": "Уголки и кронштейны", "default": {"length": 50, "width": 40, "height": 40, "wall_thickness": 4, "holes_count": 4, "hole_diameter": 6}},
]

BUSHINGS = [
    {"id": "cyl_bushing", "name": "Цилиндрическая втулка", "group": "Цилиндрические", "default": {"length": 40, "diameter": 20, "inner_diameter": 12, "wall_thickness": 4}},
    {"id": "flanged_bushing", "name": "Втулка с буртиком", "group": "С буртиком", "default": {"length": 45, "diameter": 20, "inner_diameter": 12, "flange_diameter": 30, "flange_height": 5}},
    {"id": "spacer", "name": "Дистанционная втулка", "group": "Дистанционные", "default": {"length": 25, "diameter": 16, "inner_diameter": 8}},
    {"id": "threaded_bushing", "name": "Резьбовая втулка", "group": "Резьбовые", "default": {"length": 30, "diameter": 18, "inner_diameter": 10, "thread_pitch": 1.5, "thread_length": 20}},
    {"id": "embedded_bushing", "name": "Закладная втулка", "group": "Закладные", "default": {"length": 35, "diameter": 18, "inner_diameter": 10, "flange_diameter": 22, "flange_height": 3}},
    {"id": "expansion_bushing", "name": "Распорная втулка", "group": "Распорные", "default": {"length": 50, "diameter": 14, "inner_diameter": 8, "wall_thickness": 3}},
    {"id": "mounting_bushing", "name": "Монтажная втулка", "group": "Монтажные", "default": {"length": 35, "diameter": 22, "inner_diameter": 12}},
    {"id": "long_bushing", "name": "Удлиненная втулка", "group": "Удлиненные", "default": {"length": 80, "diameter": 22, "inner_diameter": 12}},
    {"id": "flanged_spacer", "name": "Фланцевая втулка", "group": "Фланцевые", "default": {"length": 40, "diameter": 18, "inner_diameter": 10, "flange_diameter": 32, "flange_height": 4}},
    {"id": "custom_bushing", "name": "Пользовательская втулка", "group": "Специальные", "default": {"length": 40, "diameter": 20, "inner_diameter": 10}},
]

PARAM_LABELS = {
    "length": ("Длина", 1, 500), "diameter": ("Наружный диаметр", 0.5, 200),
    "inner_diameter": ("Внутренний диаметр", 0.2, 199), "hole_diameter": ("Диаметр отверстия", 0.2, 199),
    "wall_thickness": ("Толщина стенки", 0.2, 50), "width": ("Ширина / размер под ключ", 1, 300),
    "height": ("Высота", 0.5, 200), "head_diameter": ("Диаметр головки", 1, 200),
    "head_height": ("Высота головки", 0.5, 100), "wrench_size": ("Размер под ключ", 1, 300),
    "flange_diameter": ("Диаметр буртика", 1, 250), "flange_height": ("Высота буртика", 0.5, 100),
    "thread_pitch": ("Шаг резьбы", 0.2, 10), "thread_length": ("Длина резьбы", 1, 500),
    "holes_count": ("Количество отверстий", 1, 32),
}

MATERIALS = {
    "steel": "Сталь", "stainless": "Нержавеющая сталь", "zinc": "Оцинкованная сталь",
    "brass": "Латунь", "aluminum": "Алюминий", "plastic": "Пластик"
}

CATALOG = {item["id"]: item for item in FASTENERS + BUSHINGS}

def defaults_for(part_type: str) -> dict:
    item = CATALOG.get(part_type, CATALOG["hex_bolt"])
    values = dict(item["default"])
    values.setdefault("diameter", 10)
    return values

def normalize_params(part_type: str, params: dict) -> dict:
    base = defaults_for(part_type)
    result = base.copy()
    for key, value in (params or {}).items():
        if key not in PARAM_LABELS or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        lo, hi = PARAM_LABELS[key][1], PARAM_LABELS[key][2]
        number = max(lo, min(hi, number))
        result[key] = int(number) if key == "holes_count" else round(number, 3)
    # Keep hollow components physically meaningful.
    if "inner_diameter" in result:
        result["inner_diameter"] = min(result["inner_diameter"], result.get("diameter", 20) - 0.2)
    if "hole_diameter" in result:
        result["hole_diameter"] = min(result["hole_diameter"], result.get("diameter", 20) - 0.2)
    if "wall_thickness" in result and "diameter" in result:
        result["wall_thickness"] = min(result["wall_thickness"], max(0.2, (result["diameter"] - result.get("inner_diameter", 0)) / 2))
    if "thread_length" in result and "length" in result:
        result["thread_length"] = min(result["thread_length"], result["length"])
    return result
