import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

class ProjectDatabase:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self):
        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    part_type TEXT NOT NULL,
                    material TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()

    @staticmethod
    def _serialize(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["params"] = json.loads(item.pop("params_json"))
        return item

    def list_projects(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        return [self._serialize(row) for row in rows]

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return self._serialize(row) if row else None

    def save_project(self, data: dict[str, Any], project_id: str | None = None) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        project_id = project_id or str(uuid.uuid4())
        with self.connect() as conn:
            existing = conn.execute("SELECT created_at FROM projects WHERE id = ?", (project_id,)).fetchone()
            created_at = existing["created_at"] if existing else now
            conn.execute("""
                INSERT INTO projects (id, name, category, part_type, material, params_json, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name, category=excluded.category, part_type=excluded.part_type,
                    material=excluded.material, params_json=excluded.params_json, notes=excluded.notes,
                    updated_at=excluded.updated_at
            """, (project_id, data["name"], data["category"], data["part_type"], data["material"],
                  json.dumps(data["params"], ensure_ascii=False), data.get("notes", ""), created_at, now))
            conn.commit()
        return self.get_project(project_id)

    def delete_project(self, project_id: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            conn.commit()
            return cur.rowcount > 0
