"""Parametric part catalog and safe parameter normalization."""

FASTENERS = [
    {"id": "hex_bolt", "name": "Болт с шестигранной головкой", "group": "Болты", "default": {"length": 50, "diameter": 10, "head_diameter": 16, "head_height": 6, "thread_pitch": 1.5, "thread_length": 32}},
    {"id": "socket_bolt", "name": "Винт с цилиндрической головкой", "group": "Винты", "default": {"length": 40, "diameter": 8, "head_diameter": 13, "head_height": 8, "thread_pitch": 1.25, "thread_length": 28}},
    {"id": "countersunk_screw", "name": "Потайной винт", "group": "Винты", "default": {"length": 35, "diameter": 6, "head_diameter": 12, "head_height": 3.5, "thread_pitch": 1, "thread_length": 30}},
    {"id": "stud", "name": "Шпилька резьбовая", "group": "Шпильки", "default": {"length": 70, "diameter": 10, "thread_pitch": 1.5, "thread_length": 70}},
    {"id": "hex_nut", "name": "Гайка шестигранная", "group": "Гайки", "default": {"height": 8, "diameter": 10, "width": 17, "hole_diameter": 8.5, "thread_pitch": 1.5}},
    {"id": "nyloc_nut", "name": "Гайка самоконтрящаяся", "group": "Гайки", "default": {"height": 10, "diameter": 10, "width": 17, "hole_diameter": 8.5, "thread_pitch": 1.5, "nylon_height": 3}},
    {"id": "flange_nut", "name": "Гайка с фланцем", "group": "Гайки", "default": {"height": 10, "diameter": 10, "width": 17, "hole_diameter": 8.5, "thread_pitch": 1.5, "flange_diameter": 22, "flange_height": 2}},
    {"id": "castle_nut", "name": "Корончатая гайка", "group": "Гайки", "default": {"height": 12, "diameter": 10, "width": 17, "hole_diameter": 8.5, "thread_pitch": 1.5, "slots_count": 6, "slot_width": 3}},
    {"id": "cap_nut", "name": "Колпачковая гайка", "group": "Гайки", "default": {"height": 14, "diameter": 10, "width": 17, "hole_diameter": 8.5, "thread_pitch": 1.5, "cap_height": 5}},
    {"id": "washer", "name": "Шайба плоская", "group": "Шайбы", "default": {"height": 2, "diameter": 20, "hole_diameter": 10.5}},
    {"id": "spring_washer", "name": "Шайба пружинная", "group": "Шайбы", "default": {"height": 2.5, "diameter": 18, "hole_diameter": 10.5, "spring_gap": 2}},
    {"id": "lock_washer", "name": "Шайба стопорная", "group": "Шайбы", "default": {"height": 2, "diameter": 20, "hole_diameter": 10.5, "teeth_count": 12, "tooth_height": 2}},
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
    "holes_count": ("Количество отверстий", 1, 32), "nylon_height": ("Высота нейлонового кольца", .5, 30),
    "slots_count": ("Количество прорезей", 2, 16), "slot_width": ("Ширина прорези", .5, 20),
    "cap_height": ("Высота колпачка", 1, 50), "spring_gap": ("Зазор пружинной шайбы", .2, 10),
    "teeth_count": ("Количество зубьев", 4, 32), "tooth_height": ("Высота зуба", .2, 20),
}

# ISO metric coarse thread defaults. The value is the nominal diameter in mm
# and the recommended coarse pitch in mm. Custom pitches remain supported.
METRIC_THREADS = {
    "M3": {"diameter": 3, "pitch": 0.5}, "M4": {"diameter": 4, "pitch": 0.7},
    "M5": {"diameter": 5, "pitch": 0.8}, "M6": {"diameter": 6, "pitch": 1.0},
    "M8": {"diameter": 8, "pitch": 1.25}, "M10": {"diameter": 10, "pitch": 1.5},
    "M12": {"diameter": 12, "pitch": 1.75}, "M14": {"diameter": 14, "pitch": 2.0},
    "M16": {"diameter": 16, "pitch": 2.0}, "M20": {"diameter": 20, "pitch": 2.5},
}
PARAM_LABELS["thread_standard"] = ("Метрический стандарт", 0, 0)

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
        if key == "thread_standard" and isinstance(value, str) and value in METRIC_THREADS:
            result[key] = value
            continue
        if key not in PARAM_LABELS or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        lo, hi = PARAM_LABELS[key][1], PARAM_LABELS[key][2]
        number = max(lo, min(hi, number))
        if key in {"holes_count", "slots_count", "teeth_count"}:
            result[key] = max(1, int(round(number)))
        else:
            result[key] = round(number, 3)
    standard = result.get("thread_standard")
    if standard in METRIC_THREADS:
        thread = METRIC_THREADS[standard]
        result["diameter"] = thread["diameter"]
        result["thread_pitch"] = thread["pitch"]
    # Keep hollow components physically meaningful.
    if "inner_diameter" in result:
        result["inner_diameter"] = min(result["inner_diameter"], result.get("diameter", 20) - 0.2)
    if "hole_diameter" in result:
        result["hole_diameter"] = min(result["hole_diameter"], result.get("diameter", 20) - 0.2)
    if "wall_thickness" in result and "diameter" in result:
        max_wall = max(0.2, (result["diameter"] - result.get("inner_diameter", 0)) / 2)
        result["wall_thickness"] = min(result["wall_thickness"], max_wall)
        # Wall thickness is authoritative for hollow parts. Keep the inner
        # diameter consistent with it instead of silently ignoring the field.
        result["inner_diameter"] = round(max(0.2, result["diameter"] - result["wall_thickness"] * 2), 3)
    if "thread_length" in result and "length" in result:
        result["thread_length"] = min(result["thread_length"], result["length"])
    if "thread_pitch" in result and "diameter" in result:
        # A pitch larger than the nominal diameter cannot produce a usable
        # ISO-like thread and creates self-intersecting procedural ridges.
        result["thread_pitch"] = min(result["thread_pitch"], max(0.2, result["diameter"] * 0.28))
    if "head_diameter" in result and "diameter" in result:
        result["head_diameter"] = max(result["head_diameter"], result["diameter"] * 1.15)
    if "flange_diameter" in result and "diameter" in result:
        result["flange_diameter"] = max(result["flange_diameter"], result["diameter"] * 1.05)
    if "flange_height" in result:
        result["flange_height"] = min(result["flange_height"], result.get("length", result.get("height", 200)) * 0.5)
    if "cap_height" in result and "height" in result:
        result["cap_height"] = min(result["cap_height"], result["height"] * 0.7)
    if "nylon_height" in result and "height" in result:
        result["nylon_height"] = min(result["nylon_height"], result["height"] * 0.6)
    return result
