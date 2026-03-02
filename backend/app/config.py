from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# .env lives at the project root (one level above backend/)
_env_file = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(_env_file),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Google OAuth2
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""

    # JWT / session secret
    SECRET_KEY: str = "change-me-in-production"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/app.db"

    # File storage
    UPLOAD_DIR: str = "./data/uploads"
    MODEL_DIR: str = "./data/models"

    # Gemini API (for high-quality OCR via multimodal LLM)
    GEMINI_API_KEY: str = ""

    # Frontend
    FRONTEND_URL: str = "http://localhost:5173"

    # JWT config
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days


settings = Settings()
