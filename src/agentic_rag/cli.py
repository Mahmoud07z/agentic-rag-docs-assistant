"""Command-line entry point for the documentation assistant."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentic_rag.documents import load_documents
from agentic_rag.retrieval import LocalVectorStore
from agentic_rag.settings import get_settings


def main() -> None:
    """Load local documents and show the citation metadata for each chunk."""
    parser = argparse.ArgumentParser(description="Load and inspect documentation chunks.")
    parser.add_argument(
        "--docs",
        type=Path,
        default=Path("data/docs"),
        help="Directory containing Markdown and PDF documents (default: data/docs)",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument(
        "--index",
        action="store_true",
        help="Rebuild the local Chroma vector index after loading the chunks",
    )
    args = parser.parse_args()

    chunks = load_documents(
        args.docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    if not chunks:
        print(f"No supported document text found in {args.docs}.")
        return

    print(f"Loaded {len(chunks)} chunks from {args.docs}:")
    for chunk in chunks:
        location = chunk.source
        if chunk.page is not None:
            location += f", page {chunk.page}"
        if chunk.section:
            location += f", section: {chunk.section}"
        preview = " ".join(chunk.content.split())
        if len(preview) > 120:
            preview = preview[:117] + "..."
        print(f"- [{chunk.chunk_index}] {location}: {preview}")
    if args.index:
        settings = get_settings()
        store = LocalVectorStore(settings.index_dir, settings.embedding_model)
        count = store.rebuild(chunks)
        print(f"Indexed {count} chunks in {settings.index_dir}.")
