# AGENTS.md — Development Guide for `research-rag`

This file is the working contract for any AI agent (Claude Code, Cursor, Copilot Workspace, etc.) building or modifying this project. Read it before writing code. When in doubt, follow this file over your own defaults.

---

## 1. Project summary

A modular RAG application: ingest PDF/DOCX/TXT/CSV/XLSX/web documents → chunk → embed locally → store in Qdrant → hybrid retrieve (vector + BM25) → rerank → generate an answer with citations via a free-tier LLM API, optionally augmented with live web search. Developed in Google Colab (imports from `app/`), deployed on Streamlit Community Cloud.

**Stack decisions already made — do not re-litigate these without being asked:**
- Embeddings: local Sentence Transformers (`all-MiniLM-L6-v2` unless told otherwise)
- Reranker: local cross-encoder (`ms-marco-MiniLM-L-6-v2`), added last, behind a config flag
- Vector store: Qdrant (local for dev, hosted for deployment)
- LLM: Groq (primary) → Google AI Studio Gemini Flash-Lite (fallback)
- Web search: Tavily
- Keyword search: BM25 (`rank-bm25`)

---

## 2. Non-negotiable constraints

An agent must never violate these, even if a task description seems to imply otherwise:

1. **No hardcoded secrets.** No API keys, ever, in any file that gets committed. Read from `st.secrets` (Streamlit) or `os.environ` (Colab/local via `.env` + `python-dotenv`). If a key is needed to test something, stop and ask the human to provide it via `.env`, don't invent a placeholder that looks real.
2. **Memory budget: assume ~1GB RAM in production.** Before adding any model (embedding, reranker, or otherwise) to the deployed path, check its footprint. If it's not clearly small (under a few hundred MB loaded), flag it instead of silently including it.
3. **Every LLM/search call must handle failure.** No bare `requests.post(...)` or SDK call without a try/except that handles rate limits (429) and falls back or raises a typed exception. See §9.
4. **No module reaches into another layer's internals.** `retrieval/` calls `embeddings/` and `vectorstore/` through their public functions only, never touches Qdrant's raw client directly, etc. If you find yourself importing a private helper from another package, stop — that's a sign the interface is missing something, not a reason to bypass it.
5. **Colab notebooks import from `app/`, they don't contain logic.** If a notebook has more than ~10 lines of real logic (not calls + prints), that logic belongs in `app/` and the notebook should import it.
6. **Don't add dependencies not listed in §6 without checking they're needed.** Every new package in `requirements.txt` is a deploy-time risk on Streamlit Community Cloud — prefer stdlib or an already-included package first.

---

## 3. Repository structure

```
research-rag/
├── app/
│   ├── streamlit_app.py
│   ├── config.py
│   ├── ingestion/{pipeline,cleaner,chunker,metadata}.py + loaders/{pdf,docx,txt,csv,xlsx,web}_loader.py
│   ├── embeddings/model.py
│   ├── vectorstore/qdrant_store.py
│   ├── retrieval/{vector_search,keyword_search,hybrid_search,reranker}.py
│   ├── generation/{llm,prompts,answer_generator}.py
│   ├── query/{query_rewriter,query_router}.py
│   ├── research/{web_search,source_manager,research_pipeline}.py
│   └── ui/{sidebar,upload,chat,citations}.py
├── notebooks/01_document_ingestion.ipynb … 07_llm_testing.ipynb
├── tests/test_{ingestion,chunking,retrieval,generation}.py
├── data/.gitkeep
├── .streamlit/config.toml
├── .gitignore, requirements.txt, README.md, LICENSE
```

Don't restructure this without being asked. If a new file is genuinely needed, put it in the package it logically belongs to.

---

## 4. Build order & definition of done

Work through phases in order. **Do not start a phase until the previous one has a passing test in `tests/` or a working Colab notebook cell.** This is the single most important rule for agent-driven development on this repo — it prevents debugging four broken layers at once.

| Phase | Deliverable | Definition of done |
|---|---|---|
| 1. Scaffolding | Repo structure, `.gitignore`, `requirements.txt`, `.env.example` | A Colab notebook can `import app.X` successfully |
| 2. Ingestion | All 6 loaders + cleaner + chunker + metadata | Each loader tested against one real sample file in `01_document_ingestion.ipynb`; chunk size/overlap are explicit named constants, not magic numbers |
| 3. Embeddings | `embeddings/model.py` | Batch-embeds real chunks in `03_embeddings.ipynb`; memory/time logged |
| 4. Vector store | `vectorstore/qdrant_store.py` | Same code path works against local Qdrant *and* a hosted Qdrant Cloud instance |
| 5. Retrieval | vector, BM25, hybrid, reranker | Each layer tested independently before combining; reranker latency measured before it's turned on by default |
| 6. Generation | `llm.py`, `prompts.py`, `answer_generator.py` | Provider fallback (Groq → Gemini) actually triggers under a simulated 429, verified by a test, not just written |
| 7. Query & research | rewriter, router, Tavily integration, research pipeline | Router correctly chooses local-docs vs web-search on at least 3 test queries of each type |
| 8. UI & deployment | Streamlit UI, secrets wiring, Community Cloud deploy | End-to-end query works on the deployed app against hosted Qdrant, not just localhost |

A phase is **not done** if it "looks right" — it's done when there's a runnable test or notebook cell proving it against real (not synthetic/mocked) input at least once.

---

## 5. Coding conventions

- Python 3.11+, type hints on all public functions, docstrings with Args/Returns.
- Every function that calls an external service (LLM, search, Qdrant) is `async` or clearly synchronous-and-blocking — don't mix silently.
- Config values (model names, chunk size, top-k, rate limits) live in `app/config.py` as named constants or env-driven settings — never inline magic numbers in logic files.
- Errors from external providers get wrapped in project-specific exceptions (e.g. `LLMProviderError`, `SearchProviderError`) so callers can catch one thing instead of guessing SDK-specific exception types.
- Logging over printing, everywhere except quick Colab exploration cells.

---

## 6. Configuration & secrets

`app/config.py` should expose one settings object (e.g. via `pydantic-settings` or a plain dataclass reading `os.environ`/`st.secrets`) with at least:

```
LLM_PROVIDER_PRIMARY=groq
LLM_MODEL_PRIMARY=llama-4-maverick        # or gpt-oss-120b
LLM_PROVIDER_FALLBACK=gemini
LLM_MODEL_FALLBACK=gemini-2.5-flash-lite
GROQ_API_KEY=
GEMINI_API_KEY=
SEARCH_PROVIDER=tavily
TAVILY_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
EMBEDDING_MODEL=all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

Local/Colab: `.env` + `python-dotenv`. Deployed: Streamlit Secrets (`st.secrets`). `config.py` should check both without the rest of the app knowing which environment it's in.

**requirements.txt baseline** (add to this list deliberately, don't remove without checking what breaks):
```
streamlit
langchain
langchain-community
langchain-text-splitters
pymupdf
python-docx
pandas
openpyxl
beautifulsoup4
qdrant-client
sentence-transformers
rank-bm25
requests
python-dotenv
groq
google-generativeai
tavily-python
```

---

## 7. Testing requirements

- Every module in `app/` gets a corresponding test in `tests/` before being considered done, even if minimal (one real-input smoke test beats zero tests).
- LLM/search-calling code must have a test that mocks a 429/rate-limit response and asserts the fallback path is taken — this is the part most likely to silently rot.
- Retrieval quality isn't unit-testable in the traditional sense; instead, keep a small fixed set of (query, expected-source-doc) pairs in `tests/test_retrieval.py` and assert the expected doc shows up in top-k. This is a regression guard, not a correctness proof — say so if you write it.

---

## 8. Module contracts

These are the function signatures other modules should be able to rely on. An agent implementing one of these should match the signature even if the internal implementation changes.

```python
# ingestion/loaders/*.py
def load(file_path: str) -> list[dict]:
    """Returns list of {"text": str, "source": str, "page": int | None}."""

# ingestion/chunker.py
def create_chunks(documents: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """Returns list of {"text": str, "source": str, "chunk_id": str, "metadata": dict}."""

# embeddings/model.py
def embed_texts(texts: list[str]) -> list[list[float]]: ...
def embed_query(query: str) -> list[float]: ...

# vectorstore/qdrant_store.py
def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None: ...
def query(embedding: list[float], top_k: int) -> list[dict]:
    """Returns list of {"text": str, "score": float, "metadata": dict}."""

# retrieval/hybrid_search.py
def hybrid_search(query: str, top_k: int) -> list[dict]: ...

# retrieval/reranker.py
def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]: ...

# generation/llm.py
def generate(prompt: str, system: str | None = None) -> str:
    """Tries LLM_PROVIDER_PRIMARY, falls back to LLM_PROVIDER_FALLBACK on rate-limit/failure."""

# generation/answer_generator.py
def generate_answer(query: str, chunks: list[dict]) -> dict:
    """Returns {"answer": str, "citations": list[{"source": str, "page": int | None}]}."""

# query/query_router.py
def route(query: str) -> str:
    """Returns 'local', 'web', or 'both'."""

# research/web_search.py
def search(query: str, max_results: int = 5) -> list[dict]:
    """Returns list of {"title": str, "url": str, "snippet": str}."""
```

If a real implementation needs to deviate from one of these signatures, update this file in the same change — don't let the contract drift silently.

---

## 9. Provider integration details

**LLM fallback pattern (`generation/llm.py`):**
```python
def generate(prompt, system=None):
    try:
        return _call_groq(prompt, system)
    except RateLimitError:
        return _call_gemini(prompt, system)
    except ProviderError:
        raise LLMProviderError("Both providers failed")
```
Both `_call_groq` and `_call_gemini` should normalize their SDK-specific exceptions into `RateLimitError`/`ProviderError` at the point of the call, not leak raw SDK exceptions upward.

**Search (`research/web_search.py`):** Tavily's client already returns structured results — don't re-parse HTML on top of it. Cache identical queries within a session to avoid burning the ~1,000/month free credits on repeated lookups.

**Known volatility:** free-tier limits for both Groq and Gemini have changed before (Google cut Gemini free limits 50–80% in Dec 2025) and will likely change again. Don't hardcode rate-limit numbers into logic — read them from config, and note in code comments that these are current-as-of-writing, not guaranteed.

---

## 10. Before marking any phase complete, verify:

- [ ] Runs from a clean checkout with only `requirements.txt` installed
- [ ] No secrets in git history or current files (`grep -r "AIza\|gsk_" .` as a rough check)
- [ ] Tested against at least one real (not synthetic) input
- [ ] Matches the module contract in §8, or §8 has been updated
- [ ] Errors from external calls are caught and typed, not left as raw tracebacks

## 11. Common anti-patterns to avoid

- Writing the reranker and hybrid search together "since they're related" — test hybrid search alone first, it's where most retrieval bugs actually live.
- Loading the embedding model fresh on every function call instead of once at module import / cached resource — this is a common Streamlit performance bug.
- Putting Qdrant connection logic inline in `streamlit_app.py` "just to get it working" — it belongs in `vectorstore/qdrant_store.py` from the start.
- Skipping the fallback-provider test because "it probably works" — this is exactly the code path that only gets exercised in production when it's too late to debug calmly.
