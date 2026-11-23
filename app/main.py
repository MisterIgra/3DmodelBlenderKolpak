from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from app.api.routes import router
from app.metizlab.routes import router as metiz_router

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}

app = FastAPI(title="AutoKolpak AI", version="2.0.0")
app.include_router(router)
app.include_router(metiz_router)
# Локальные библиотеки (Tailwind, Three.js, FontAwesome): интерфейс работает без интернета
app.mount("/vendor", StaticFiles(directory=FRONTEND / "vendor"), name="vendor")
GENERATED = ROOT / "generated_models"
GENERATED.mkdir(parents=True, exist_ok=True)
app.mount("/generated", StaticFiles(directory=GENERATED), name="generated")


@app.get("/", include_in_schema=False)
def index():
    # no-store: браузер не должен показывать закэшированную старую версию интерфейса
    return FileResponse(FRONTEND / "index.html", headers=NO_CACHE)


@app.get("/api/whoami", include_in_schema=False)
def whoami():
    """Диагностика: какой именно проект и какой index.html сейчас отдаёт сервер."""
    import os, time
    idx = FRONTEND / "index.html"
    return {
        "root": str(ROOT),
        "index_html": str(idx),
        "index_size": idx.stat().st_size,
        "index_modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(idx.stat().st_mtime)),
        "ui_version_in_file": "2.1-local" in idx.read_text(encoding="utf-8", errors="ignore"),
        "cwd": os.getcwd(),
    }


@app.on_event("startup")
def _banner():
    print(f"[AutoKolpak] project root: {ROOT}")
    print(f"[AutoKolpak] serving: {FRONTEND / 'index.html'}")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)
