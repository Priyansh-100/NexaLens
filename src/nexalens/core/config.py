from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_SECRET = "dev-secret-change-in-production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "nexalens"
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    postgres_user: str = "postgres"
    postgres_password: str = "change-me-in-production"
    postgres_db: str = "nexalens"
    postgres_readonly_user: str = "nexalens_readonly"
    postgres_readonly_password: str = "change-me-in-production"
    database_url: str = Field(default="", validation_alias="DATABASE_URL")
    database_pool_size: int = 10
    database_max_overflow: int = 20

    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")

    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "nexalens_docs"

    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    llm_embedding_model: str = "nomic-embed-text"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    llm_context_window: int = 8192

    sql_dialect: str = "postgresql"
    sql_max_rows: int = 1000
    sql_timeout: int = 30
    sql_max_query_cost: float = 10000.0

    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_doc_size_mb: int = 50
    supported_extensions: list[str] = [".pdf", ".txt", ".md", ".csv", ".xlsx", ".docx"]

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8501"]

    secret_key: str = DEFAULT_SECRET
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7

    enable_metrics: bool = True
    metrics_port: int = 9090

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("supported_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        if self.app_env == "production":
            if self.secret_key == DEFAULT_SECRET:
                raise ValueError("Production SECRET_KEY must be configured (not default)")
            if self.postgres_password == "change-me-in-production":
                raise ValueError("Production POSTGRES_PASSWORD must be configured")
            if self.postgres_readonly_password == "change-me-in-production":
                raise ValueError("Production POSTGRES_READONLY_PASSWORD must be configured")
        return self

    @model_validator(mode="after")
    def build_database_url(self) -> "Settings":
        if not self.database_url:
            self.database_url = (
                f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
                f"@localhost:5432/{self.postgres_db}"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()