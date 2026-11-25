"""Load Markdown and PDF files into citation-friendly text chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".md", ".markdown", ".pdf"}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A passage and the location needed to cite it later."""

    content: str
    source: str
    chunk_index: int
    page: int | None = None
    section: str | None = None


def load_documents(
    directory: str | Path,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[DocumentChunk]:
    """Load supported files recursively from ``directory`` and split them.

    PDF page numbers in returned chunks are one-based. Markdown section names
    are taken from the nearest preceding heading.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {root}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    for path in files:
        source = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".pdf":
            chunks.extend(_load_pdf(path, source, chunk_size, chunk_overlap))
        else:
            chunks.extend(_load_markdown(path, source, chunk_size, chunk_overlap))
    return chunks


def _load_markdown(
    path: Path, source: str, chunk_size: int, chunk_overlap: int
) -> list[DocumentChunk]:
    text = path.read_text(encoding="utf-8-sig")
    chunks: list[DocumentChunk] = []
    section: str | None = None
    section_lines: list[str] = []

    def flush() -> None:
        nonlocal section_lines
        body = "\n".join(section_lines).strip()
        if body:
            for part in _split_text(body, chunk_size, chunk_overlap):
                chunks.append(DocumentChunk(
                    content=part,
                    source=source,
                    chunk_index=len(chunks),
                    section=section,
                ))
        section_lines = []

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            section = match.group(2)
        else:
            section_lines.append(line)
    flush()
    return chunks


def _load_pdf(
    path: Path, source: str, chunk_size: int, chunk_overlap: int
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    reader = PdfReader(path)
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for part in _split_text(text, chunk_size, chunk_overlap):
            chunks.append(DocumentChunk(
                content=part,
                source=source,
                chunk_index=len(chunks),
                page=page_number,
            ))
    return chunks


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterator[str]:
    """Split text near whitespace, with a bounded overlap between passages."""
    text = text.strip()
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary
        part = text[start:end].strip()
        if part:
            yield part
        if end >= len(text):
            break
        next_start = max(start + 1, end - chunk_overlap)
        while next_start < end and text[next_start].isspace():
            next_start += 1
        start = next_start
