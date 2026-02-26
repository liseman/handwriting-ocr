from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
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

    # Frontend
    FRONTEND_URL: str = "http://localhost:5173"

    # JWT config
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days


settings = Settings()
