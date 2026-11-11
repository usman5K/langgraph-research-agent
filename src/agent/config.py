"""Typed configuration loaded from environment variables / a .env file."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr | None = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    tavily_api_key: SecretStr | None = None
    max_steps: int = Field(default=6, ge=1, le=25)
    docs_dir: str = "docs"
    embeddings: Literal["auto", "openai", "hashing"] = "auto"


@lru_cache
def get_settings() -> Settings:
    return Settings()
