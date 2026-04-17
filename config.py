import logging
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


def _find_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Project root not found")


PROJECT_ROOT = _find_root()


class Settings(BaseSettings):
    PROJECT_ROOT: Path = PROJECT_ROOT
    DOCS_DIR: Path = PROJECT_ROOT / "docs"
    VECTOR_DB_DIR: Path = PROJECT_ROOT / "vector_db"
    EMBEDDER_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_DB_COLLECTION_NAME: str = "pdf_collection"
    LLM_BACKEND: str = "ollama"
    LLM_MODEL: Optional[str] = "llama3"
    GEMINI_API_KEY: Optional[str] = None

    @field_validator("DOCS_DIR", "VECTOR_DB_DIR", mode="before")
    @classmethod
    def make_absolute(cls, v: str | Path) -> Path:
        path = Path(v)
        if not path.is_absolute():
            return PROJECT_ROOT / path
        return path


def load_settings(env: Optional[str] = None) -> Settings:
    if env is None:
        env = os.getenv("APP_ENV", "prod")

    config_file = PROJECT_ROOT / "config" / f"{env}.yaml"

    if not config_file.exists():
        logger.warning("Config file %s not found. Using default settings.", config_file)
        return Settings()

    try:
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f) or {}
            return Settings(**config_data)
    except Exception as e:
        logger.error("Failed to load config from %s: %s. Using default settings.", config_file, e)
        return Settings()
