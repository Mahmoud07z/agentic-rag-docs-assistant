# Agentic RAG Documentation Assistant

A learning project that answers questions from technical documentation. It loads local Markdown and PDF files, retrieves relevant passages, and returns answers with source locations. If local evidence looks weak, it can broaden retrieval and optionally search the web. A bounded review step checks generated answers against their evidence.

## Run locally

Requires Python 3.11 or newer. On first indexing, Sentence Transformers downloads the configured embedding model; this can take a little while.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Add an LLM API key and the model name for your provider to `.env` to enable generated answers and model-based evidence review. Set `OPENAI_BASE_URL` too when using an OpenAI-compatible endpoint. Add `TAVILY_API_KEY` to enable web search when local documents are insufficient. Without an LLM key/model, the app returns retrieved passages directly.

Place Markdown or text-based PDF files in `data/docs/`, then build the local index:

```powershell
agentic-rag --docs data/docs --index
```

Start the demo and open the address shown by Uvicorn (normally `http://127.0.0.1:8000`):

```powershell
agentic-rag-web
```

The page can also rebuild the index using its **Index documents** button. The index is stored under `data/index/` and ignored by Git. Scanned image-only PDFs need OCR, which is not included.

## How it works

1. `documents.py` reads Markdown/PDF files and splits them into chunks with source and page/section metadata.
2. `retrieval.py` embeds chunks locally and stores vectors in Chroma for semantic search.
3. `assistant.py` decides whether the retrieved evidence is sufficient, widens retrieval when needed, and can use Tavily search if configured.
4. With an LLM configured, it drafts an answer with citations and checks its claims against retrieved passages. It makes at most one additional retrieval/review attempt.
5. `web.py` exposes indexing and question-answering endpoints; the browser page renders answers and cited sources.

See [the project walkthrough](docs/PROJECT_WALKTHROUGH.md) for a longer explanation of the design and the concepts behind each step. That document is temporary and can be removed when the project is finished.

## Configuration

Copy `.env.example` to `.env`, then fill in the values you use. Never commit API keys. `RAG_DOCS_DIR`, `RAG_INDEX_DIR`, and `RAG_EMBEDDING_MODEL` can be set to customize the document folder, vector index location, and local embedding model.
