from functools import lru_cache
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    path = ROOT_DIR / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class Settings:
    def __init__(self):
        _load_dotenv()
        self.app_name = os.getenv("METIZ_APP_NAME", "METIZLAB AI")
        self.host = os.getenv("HOST", "127.0.0.1")
        self.port = int(os.getenv("PORT", "8000"))
        self.anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL", "https://token-shop.online")
        self.anthropic_auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.request_timeout = int(os.getenv("METIZ_REQUEST_TIMEOUT", "120"))
        self.max_upload_mb = int(os.getenv("METIZ_MAX_UPLOAD_MB", "12"))
        self.database_path = os.getenv("METIZ_DATABASE_PATH", str(ROOT_DIR / "metizlab.db"))
        self.uploads_dir = os.getenv("METIZ_UPLOADS_DIR", str(ROOT_DIR / "uploads"))
        self.generated_dir = os.getenv("METIZ_GENERATED_DIR", str(ROOT_DIR / "generated_models"))
        self.blender_executable = os.getenv("BLENDER_EXECUTABLE", "")

    @property
    def uses_anthropic_api(self) -> bool:
        return bool(self.anthropic_auth_token.strip() or self.anthropic_api_key.strip())

    @property
    def ai_api_key(self) -> str:
        return self.anthropic_auth_token.strip() or self.anthropic_api_key.strip() or self.openai_api_key.strip()

    @property
    def ai_base_url(self) -> str:
        return (self.anthropic_base_url if self.uses_anthropic_api else self.openai_base_url).rstrip("/")

    @property
    def ai_model(self) -> str:
        return self.anthropic_model if self.uses_anthropic_api else self.openai_model

    @property
    def ai_provider(self) -> str:
        return "Anthropic Messages API" if self.uses_anthropic_api else "OpenAI Responses API"


@lru_cache
def get_settings() -> Settings:
    return Settings()
