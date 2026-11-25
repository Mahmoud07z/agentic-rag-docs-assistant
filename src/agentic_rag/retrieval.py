"""Local embedding generation and persistent Chroma vector search."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from agentic_rag.documents import DocumentChunk
from agentic_rag.settings import get_settings


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    content: str
    source: str
    score: float
    chunk_index: int
    page: int | None = None
    section: str | None = None


class LocalVectorStore:
    """A small persistent vector index backed by local embeddings and Chroma."""

    def __init__(self, index_dir: str | Path | None = None, model_name: str | None = None):
        settings = get_settings()
        self.index_dir = Path(index_dir or settings.index_dir)
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None
        client = chromadb.PersistentClient(path=str(self.index_dir))
        self._client = client
        self._collection = client.get_or_create_collection(
            name="technical_docs",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def rebuild(self, chunks: list[DocumentChunk]) -> int:
        """Replace the index contents with the supplied document chunks."""
        self._collection = self._client.get_or_create_collection(
            name="technical_docs",
            metadata={"hnsw:space": "cosine"},
        )
        existing_ids = self._collection.get(include=["metadatas"]).get("ids", [])
        if existing_ids:
            self._collection.delete(ids=existing_ids)
        if not chunks:
            return 0

        texts = [chunk.content for chunk in chunks]
        vectors = self.model.encode_document(texts, normalize_embeddings=True).tolist()
        ids = [f"{i}:{chunk.source}:{chunk.chunk_index}" for i, chunk in enumerate(chunks)]
        metadatas = []
        for chunk in chunks:
            metadatas.append({
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "page": chunk.page if chunk.page is not None else -1,
                "section": chunk.section or "",
            })
        self._collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(self, query: str, *, limit: int = 5) -> list[RetrievedPassage]:
        if not query.strip() or self._collection.count() == 0:
            return []
        query_vector = self.model.encode_query(query, normalize_embeddings=True).tolist()
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(limit, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        passages: list[RetrievedPassage] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            page = metadata.get("page", -1)
            passages.append(RetrievedPassage(
                content=document,
                source=metadata["source"],
                score=max(0.0, 1.0 - float(distance)),
                chunk_index=int(metadata.get("chunk_index", 0)),
                page=page if page >= 1 else None,
                section=metadata.get("section") or None,
            ))
        return passages
