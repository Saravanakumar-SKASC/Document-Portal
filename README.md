# 📄 Document Portal

**LLM-powered document intelligence: analyze, chat with, and compare documents.**

Upload a PDF and get a structured briefing in seconds, ask questions across several documents with cited answers, or find page-level differences between two versions of a file. Built with LangChain, FAISS, FastAPI and Streamlit, with provider-agnostic LLMs (Groq or Google Gemini) switched by config.

![CI](https://github.com/Saravanakumar-SKASC/Document-Portal/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.x-green)

---

## Features

| Module | What it does |
|---|---|
| **Document Analyzer** | Extracts title, author, dates, page count, language, sentiment and a bullet summary as schema-validated JSON. Malformed LLM output is repaired automatically by a self-correcting parser. |
| **Single-document chat** | History-aware RAG: follow-up questions are rewritten into standalone ones, answered only from retrieved context, with `[file, p.N]` citations. Sessions persist to disk. |
| **Multi-document chat** | One FAISS index across many files with three retrieval strategies: similarity, **MMR** (diverse results across files) and **MMR + contextual compression** (drops redundant and off-topic chunks, with a fallback so context is never empty). |
| **Document compare** | Page-by-page diff of a reference vs. an actual PDF, returned as validated `{Page, Changes}` rows. |
| **Evaluation** | Hit rate@k and MRR for retrieval, keyword coverage and LLM-as-judge faithfulness for answers, plus a benchmark that compares retrieval strategies on the same labelled questions. |

## Architecture

```mermaid
flowchart LR
    U[User] --> UI[Streamlit UI]
    U --> API[FastAPI REST API]
    UI & API --> ING[Ingestion<br/>PyMuPDF · safe uploads · chunking]
    ING --> EMB[Gemini embeddings]
    EMB --> VS[(FAISS index<br/>per session)]

    subgraph RAG[Conversational RAG]
        RW[Rewrite follow-up<br/>into standalone question] --> RET{Retriever}
        RET -->|similarity| CTX[Context]
        RET -->|MMR| CTX
        RET -->|MMR + compression| CTX
        CTX --> ANS[Answer with citations]
    end

    VS --> RET
    ING --> AN[Analyzer<br/>Pydantic schema + OutputFixingParser]
    ING --> CMP[Comparator<br/>page-level diff]
    ANS & AN & CMP --> LLM[LLM via config<br/>Groq · Gemini]
```

## Tech stack

- **LLM orchestration:** LangChain 1.x (LCEL), prompt library, Pydantic output schemas
- **Models:** Groq `llama-3.3-70b-versatile` or Google `gemini-2.5-flash`, `gemini-embedding-001` embeddings (all set in `config/config.yaml`)
- **Retrieval:** FAISS, MMR, contextual compression (redundancy + similarity filters)
- **Serving:** FastAPI (OpenAPI docs at `/docs`), Streamlit
- **Engineering:** structured JSON logging (structlog), custom exceptions with file/line tracing, dependency injection for testability, pytest, ruff, GitHub Actions, Docker

## Quick start

```bash
git clone https://github.com/Saravanakumar-SKASC/Document-Portal.git
cd Document-Portal
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then add GOOGLE_API_KEY (and GROQ_API_KEY if LLM_PROVIDER=groq)
```

**Web UI**

```bash
streamlit run streamlit_ui.py
```

**REST API**

```bash
uvicorn app:app --reload    # interactive docs at http://127.0.0.1:8000/docs
```

**Docker**

```bash
docker build -t document-portal .
docker run -p 8000:8000 --env-file .env document-portal
```

## API

| Method | Endpoint | Body | Returns |
|---|---|---|---|
| `GET` | `/health` | – | `{"status": "ok"}` |
| `POST` | `/analyze` | `file` (PDF) | structured metadata |
| `POST` | `/chat/index` | `files` (PDF/TXT/MD, one or more) | `session_id`, chunk count |
| `POST` | `/chat/query` | `{session_id, question, strategy, chat_history}` | answer + sources |
| `POST` | `/compare` | `reference`, `actual` (PDFs) | list of `{Page, Changes}` |

```bash
curl -F "files=@policy.pdf" -F "files=@handbook.pdf" localhost:8000/chat/index
curl -X POST localhost:8000/chat/query -H "Content-Type: application/json" \
  -d '{"session_id": "multi_...", "question": "How much annual leave do staff get?", "strategy": "mmr+compression"}'
```

## Evaluating retrieval

```python
from src.multidoc_chat.data_ingestion import MultiDocIngestor
from src.multidoc_chat.evaluation import EvalExample, compare_strategies

store = MultiDocIngestor().ingest(["policy.pdf", "handbook.pdf"])
examples = [
    EvalExample("How much annual leave?", expected_source="policy.pdf", expected_page=2),
    EvalExample("What is the remote work policy?", expected_keywords=["remote", "days"]),
]
print(compare_strategies(store, examples))
# {'similarity': {'hit_rate': ..., 'mrr': ...}, 'mmr': {...}, 'mmr+compression': {...}}
```

## Tests

The suite runs fully offline (no API keys): a scripted fake LLM plus a small bag-of-words embedding model, so retrieval tests check genuinely relevant results.

```bash
pip install -r requirements-dev.txt
pytest -q && ruff check .
```

## Project structure

```
├── app.py                     # FastAPI app
├── streamlit_ui.py            # Streamlit UI
├── config/config.yaml         # models, chunking, retrieval settings
├── src/
│   ├── document_analyzer/     # PDF ingestion + metadata extraction
│   ├── singledoc_chat/        # ingestion, conversational RAG, evaluation
│   ├── multidoc_chat/         # ingestion, MMR, contextual compression, strategy benchmark
│   └── doc_compare/           # page-level comparison
├── utils/                     # config, model loader, document ops, RAG core
├── prompt/prompt_library.py   # all prompts in one place
├── model/models.py            # Pydantic schemas
├── logger/ · exception/       # structured logging, custom exceptions
└── tests/                     # offline pytest suite
```

## Roadmap

- OCR for scanned PDFs
- Streaming responses in the UI
- Hybrid search (BM25 + vectors) and a reranker
- Deployment to AWS (ECS/EC2) with a persistent vector store

## Author

**Saravanakumar N** · MSc Artificial Intelligence, National College of Ireland
[LinkedIn](https://linkedin.com/in/saravanakumarn3899) · [GitHub](https://github.com/Saravanakumar-SKASC) · [Portfolio](https://datascienceportfol.io/saravanakumar3899)
