# Project walkthrough (temporary learning notes)

This file explains the parts we are building and why they are there. It is meant to help you understand the project as we finish it; we can delete it then.

## The problem this project solves

A language model answers from patterns it learned during training. It may not know the exact details in a project's technical docs, and it can confidently fill gaps with guesses. Retrieval-augmented generation (RAG) gives it relevant passages from your own documents at question time. The model then writes an answer grounded in that retrieved material.

The "agentic" part is the small decision loop around retrieval. The app checks whether its evidence is useful, searches again when needed, and reviews the generated answer before returning it. The loop is deliberately bounded so one question cannot cause endless retries.

## The full path through the application

```text
Markdown/PDF files
        |
        v
read and split into chunks ---- source/page/section metadata
        |
        v
local embedding model ---------- each chunk becomes a vector
        |
        v
Chroma persistent index
        |
question -> question vector -> nearest evidence passages
        |                         |
        +----- evidence check ----+---- weak? broaden retrieval / optional web search
                                  |
                                  v
                         grounded answer with labels
                                  |
                                  v
                       answer evidence review
                                  |
                     accepted or one retry, then return
```

## Important files

### `src/agentic_rag/documents.py`

This is the ingestion layer. It reads `.md`, `.markdown`, and `.pdf` files. It breaks text into overlapping pieces so a passage is not cut at an arbitrary word boundary and some nearby context remains available. Each chunk records its source filename and a chunk number. PDF chunks also record a one-based page number; Markdown chunks keep the nearest heading as their section.

Why keep this metadata? Once the model has answered, we need to show where its evidence came from. A PDF page or Markdown section is much more useful than an anonymous paragraph.

The PDF reader extracts text from a PDF's text layer. A scanned page is just an image, so it needs OCR before this loader can read it.

### `src/agentic_rag/retrieval.py`

An embedding model converts text into a list of numbers (a vector) that represents its meaning. Similar passages tend to have nearby vectors. During indexing, the project embeds each chunk and stores the vector, text, and metadata in Chroma. During a question, it embeds the question and asks Chroma for the nearest chunks.

The default embedding model runs locally. This keeps the document embedding step separate from the answer-generating LLM. The vector index persists at `data/index/chroma`, so it does not have to be rebuilt for every question. Re-index after changing the source files so the stored vectors match the current documents.

### `src/agentic_rag/assistant.py`

This file coordinates the workflow:

1. Search the local vector index for passages related to the question.
2. Judge whether the evidence can answer the question. Without an LLM configured, it uses a simple similarity-score cutoff instead.
3. If evidence appears weak, broaden the local search. If `TAVILY_API_KEY` is set, search the web too.
4. If an LLM is configured, ask it to answer from the selected evidence and cite passage labels such as `[L1]` or `[W1]`.
5. Ask the model to check that the answer is supported. If it fails review, retrieve once more and regenerate. If review still fails, return an uncertainty message.

The model gets a reminder that document text is data, not instructions. This reduces the chance that malicious text inside a document can redirect the assistant, but it does not make prompt injection impossible.

### `src/agentic_rag/web_search.py`

This is an optional adapter for Tavily. The assistant only calls it when local evidence is judged insufficient and a Tavily key is configured. Web passages are marked as web sources and returned with URLs so the interface can link to them.

### `src/agentic_rag/web.py` and `static/index.html`

FastAPI provides four routes: `/` serves the page, `/api/health` reports which integrations are configured, `/api/index` rebuilds the local index, and `/api/ask` runs the question workflow. The HTML page calls those endpoints and displays the answer, source locations, route taken, and review count.

## Configuration and data flow

Copy `.env.example` to `.env` and enter credentials locally. The app uses:

- `OPENAI_API_KEY` and `OPENAI_MODEL` for answer generation and model-based grading/review;
- optional `OPENAI_BASE_URL` for an OpenAI-compatible endpoint;
- optional `TAVILY_API_KEY` for web search;
- optional `RAG_DOCS_DIR`, `RAG_INDEX_DIR`, and `RAG_EMBEDDING_MODEL` for paths and model selection.

The OpenAI-compatible setting means the code can point at more than one provider, but compatible providers may differ in supported models and API details. The configured model name must be one the endpoint accepts.

## What the current approach does not guarantee

- Semantic similarity is not the same as truth. A close vector match can still be irrelevant.
- The evidence grader and answer reviewer are themselves language model judgments, not formal proofs.
- The app shows source passages so a person can check the citations; it does not independently establish that a source is authoritative or current.
- Web results are external content and can be incomplete or wrong.
- There is no OCR, authentication, multi-user session storage, production rate limiting, or automated quality benchmark yet.

These limits are useful learning points: a real system needs a representative evaluation set, retrieval-quality measurements, source-quality rules, and operational controls before people rely on it for important decisions.

## How to experiment

Use the included `data/docs/sample-api.md` first. Index it and ask questions whose answers are present, such as the request limit, and absent, such as the API's data-retention policy. Then add your own technical docs, re-index, and compare which sources the assistant retrieves. This makes it easier to see whether a wrong answer came from missing evidence, poor retrieval, or answer generation.
