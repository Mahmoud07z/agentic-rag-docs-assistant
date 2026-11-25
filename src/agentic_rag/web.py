"""FastAPI endpoints and a minimal browser interface for the demo."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from agentic_rag.assistant import RAGAssistant
from agentic_rag.documents import load_documents
from agentic_rag.retrieval import LocalVectorStore
from agentic_rag.settings import get_settings


app = FastAPI(title="Agentic RAG Documentation Assistant")
INDEX_HTML = Path(__file__).parent / "static" / "index.html"


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@lru_cache(maxsize=1)
def get_assistant() -> RAGAssistant:
    return RAGAssistant()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ok",
        "llm_configured": bool(settings.llm_api_key and settings.llm_model),
        "web_search_configured": bool(settings.tavily_api_key),
    }


@app.post("/api/index")
def index_documents() -> dict[str, object]:
    settings = get_settings()
    try:
        chunks = load_documents(settings.docs_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not chunks:
        raise HTTPException(status_code=400, detail=f"No supported document text found in {settings.docs_dir}.")
    store = LocalVectorStore(settings.index_dir, settings.embedding_model)
    count = store.rebuild(chunks)
    get_assistant.cache_clear()
    return {"indexed_chunks": count, "documents_directory": str(settings.docs_dir)}


@app.post("/api/ask")
def ask(request: QuestionRequest) -> dict[str, object]:
    try:
        result = get_assistant().answer(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Assistant could not complete the request: {exc}") from exc
    return {
        "answer": result.answer,
        "sources": [asdict(source) for source in result.sources],
        "route": result.route,
        "review_attempts": result.review_attempts,
    }


def run() -> None:
    uvicorn.run("agentic_rag.web:app", host="127.0.0.1", port=8000)
