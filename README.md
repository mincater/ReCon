# research-rag

A modular Retrieval-Augmented Generation (RAG) application: ingest PDF, DOCX, TXT, CSV, XLSX, and web documents → clean and chunk → embed locally → store in Qdrant → hybrid retrieve (dense vector + BM25 keyword search) → rerank → generate responses with citations via free-tier LLM APIs (Groq primary with Google Gemini Flash-Lite fallback), optionally augmented with live Tavily web search.

Developed for interactive exploration in Google Colab (importing directly from `app/`) and production deployment on Streamlit Community Cloud.

---

## Architecture Overview

```
                        +----------------------+
                        |   User Query / Web   |
                        +----------+-----------+
                                   |
                         +---------v---------+
                         |   Query Router    |
                         +----+---------+----+
                              |         |
                  +-----------+         +-----------+
                  |                                 |
         +--------v--------+               +--------v--------+
         |  Hybrid Search  |               | Live Web Search |
         | (Vector + BM25) |               |    (Tavily)     |
         +--------+--------+               +--------+--------+
                  |                                 |
         +--------v--------+                        |
         | Cross-Encoder   |                        |
         |    Reranker     |                        |
         +--------+--------+                        |
                  |                                 |
                  +----------------+----------------+
                                   |
                         +---------v---------+
                         | LLM Generation    |
                         | Groq -> Gemini    |
                         +---------+---------+
                                   |
                         +---------v---------+
                         | Answer + Citation |
                         +-------------------+
```

---

## Core Stack

- **Embeddings**: Local Sentence Transformers (`all-MiniLM-L6-v2`)
- **Reranker**: Local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, behind config flag)
- **Vector Store**: Qdrant (local disk/memory for dev, hosted Qdrant Cloud for deployment)
- **Keyword Search**: BM25 (`rank-bm25`)
- **Primary LLM**: Groq (`llama-4-maverick` / `gpt-oss-120b`)
- **Fallback LLM**: Google AI Studio Gemini (`gemini-2.5-flash-lite`)
- **Web Search**: Tavily (`tavily-python`)
- **Frontend**: Streamlit

---

## Project Structure

```
research-rag/
├── app/
│   ├── streamlit_app.py
│   ├── config.py
│   ├── ingestion/
│   │   ├── pipeline.py, cleaner.py, chunker.py, metadata.py
│   │   └── loaders/{pdf,docx,txt,csv,xlsx,web}_loader.py
│   ├── embeddings/model.py
│   ├── vectorstore/qdrant_store.py
│   ├── retrieval/{vector_search,keyword_search,hybrid_search,reranker}.py
│   ├── generation/{llm,prompts,answer_generator}.py
│   ├── query/{query_rewriter,query_router}.py
│   ├── research/{web_search,source_manager,research_pipeline}.py
│   └── ui/{sidebar,upload,chat,citations}.py
├── notebooks/01_document_ingestion.ipynb … 07_llm_testing.ipynb
├── tests/test_{scaffolding,ingestion,chunking,retrieval,generation}.py
├── data/.gitkeep
├── .streamlit/config.toml
├── .gitignore, requirements.txt, README.md, LICENSE, AGENTS.md
```

---

## Getting Started

### Local Setup

1. **Clone the repository and enter the folder**:
   ```bash
   git clone <repo-url>
   cd research-rag
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (GROQ_API_KEY, GEMINI_API_KEY, TAVILY_API_KEY, etc.)
   ```

5. **Run tests**:
   ```bash
   pytest tests/
   ```

6. **Launch the Streamlit app**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

---

## Google Colab Usage

In Google Colab, mount or clone the repository and add the root to `sys.path`:

```python
import sys
sys.path.append("/content/research-rag")

from app.config import get_settings
from app.ingestion.pipeline import ingest_document
```

---

## Development Guidelines

Please refer to [`AGENTS.md`](AGENTS.md) for non-negotiable rules, strict build order, memory budget constraints (~1GB RAM budget for Streamlit Community Cloud), and module interface contracts.
