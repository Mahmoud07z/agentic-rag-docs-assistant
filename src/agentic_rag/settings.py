"""Environment-backed settings for the demo application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    docs_dir: Path
    index_dir: Path
    embedding_model: str
    llm_api_key: str | None
    llm_base_url: str | None
    llm_model: str | None
    tavily_api_key: str | None


def get_settings() -> Settings:
    """Read settings from the environment and a local, git-ignored .env file."""
    load_dotenv()
    return Settings(
        docs_dir=Path(os.getenv("RAG_DOCS_DIR", "data/docs")),
        index_dir=Path(os.getenv("RAG_INDEX_DIR", "data/index/chroma")),
        embedding_model=os.getenv(
            "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        llm_api_key=os.getenv("OPENAI_API_KEY"),
        llm_base_url=os.getenv("OPENAI_BASE_URL"),
        llm_model=os.getenv("OPENAI_MODEL"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
    )
