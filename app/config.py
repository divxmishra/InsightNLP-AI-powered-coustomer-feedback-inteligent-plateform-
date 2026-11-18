from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    max_upload_rows: int = 5000
    models_dir: str = "models"
    # Render Postgres internal URL in production; local SQLite file otherwise.
    database_url: str = "sqlite:///./insightnlp.db"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
