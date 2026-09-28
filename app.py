"""
Document Portal REST API.

Run:  uvicorn app:app --reload        Docs: http://127.0.0.1:8000/docs
Auth: send your key in the X-API-Key header (see utils/security.py and .env.example).
"""

import re
import time
import uuid
from functools import lru_cache
from pathlib import Path

import structlog
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from src.doc_compare.data_ingestion import DocumentIngestion
from src.doc_compare.retrieval import DocumentComparatorLLM
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_analyzer.data_ingestion import DocumentHandler
from src.multidoc_chat.data_ingestion import MultiDocIngestor
from src.multidoc_chat.retrieval import STRATEGIES, MultiDocChat
from utils.rag_core import load_faiss_index, load_session_info
from utils.security import Principal, authenticate, can_access

log = CustomLogger().get_logger(__name__)
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
ROLE_PATTERN = re.compile(r"^[a-z0-9_-]{1,40}$")
FAISS_DIR = Path("faiss_index")

app = FastAPI(
    title="Document Portal API",
    description="LLM document intelligence: analyze, chat (RAG) and compare documents.",
    version="1.1.0",
)


# ---------- Observability: request id + latency on every request ----------
@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    log.info("request completed", method=request.method, path=request.url.path,
             status=response.status_code, latency_ms=latency_ms)
    return response


# ---------- Dependencies (overridable in tests) ----------
@lru_cache(maxsize=1)
def _model_loader():
    from utils.model_loader import ModelLoader

    return ModelLoader()


def get_llm():
    return _model_loader().load_llm()


def get_embeddings():
    return _model_loader().load_embeddings()


def get_principal(x_api_key: str | None = Header(default=None)) -> Principal:
    principal = authenticate(x_api_key)
    if principal is None:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.",
                            headers={"WWW-Authenticate": "API-Key"})
    structlog.contextvars.bind_contextvars(caller=principal.name, roles=sorted(principal.roles))
    return principal


@app.exception_handler(DocumentPortalException)
async def portal_error_handler(_, exc: DocumentPortalException):
    log.error("Request failed", error=exc.error_message)
    return JSONResponse(status_code=422, content={"detail": exc.error_message})


# ---------- Schemas ----------
class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=2000)
    strategy: str = "hybrid"
    chat_history: list[ChatTurn] = Field(default_factory=list, max_length=20)


def _parse_roles(raw: str | None) -> list[str]:
    roles = [r.strip().lower() for r in (raw or "").split(",") if r.strip()]
    bad = [r for r in roles if not ROLE_PATTERN.match(r)]
    if bad:
        raise HTTPException(status_code=400, detail=f"Invalid role name(s): {bad}")
    return roles


# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(file: UploadFile = File(...), llm=Depends(get_llm), principal: Principal = Depends(get_principal)):
    handler = DocumentHandler()
    text = handler.read_pdf(handler.save_pdf(file))
    return {"session_id": handler.session_id, "metadata": DocumentAnalyzer(llm=llm).analyze_document(text)}


@app.post("/chat/index")
def chat_index(
    files: list[UploadFile] = File(...),
    allowed_roles: str | None = Form(default=None, description="Comma-separated roles, e.g. engineering,admin"),
    revision: str | None = Form(default=None, max_length=40, description="Document revision, e.g. Rev 13"),
    embeddings=Depends(get_embeddings),
    principal: Principal = Depends(get_principal),
):
    roles = _parse_roles(allowed_roles)
    if not principal.is_admin and roles and not set(roles) <= principal.roles:
        raise HTTPException(status_code=403, detail="You can only share a collection with roles you hold.")
    # Default: a new collection is visible only to the uploader's own roles
    roles = roles or sorted(principal.roles)
    ingestor = MultiDocIngestor(embeddings=embeddings, faiss_dir=FAISS_DIR)
    store = ingestor.ingest(files, allowed_roles=roles, revision=revision)
    return {"session_id": ingestor.session_id, "files": [f.filename for f in files],
            "chunks": store.index.ntotal, "allowed_roles": roles, "revision": revision}


@app.post("/chat/query")
def chat_query(req: ChatRequest, llm=Depends(get_llm), embeddings=Depends(get_embeddings),
               principal: Principal = Depends(get_principal)):
    if not SESSION_ID_PATTERN.match(req.session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id.")
    if req.strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"strategy must be one of {list(STRATEGIES)}")
    index_dir = FAISS_DIR / req.session_id
    if not (index_dir / "index.faiss").exists():
        raise HTTPException(status_code=404, detail="Collection not found.")
    info = load_session_info(index_dir)
    if not can_access(principal, info.get("allowed_roles")):
        # 404 rather than 403 so callers can't probe which collections exist
        raise HTTPException(status_code=404, detail="Collection not found.")
    store = load_faiss_index(index_dir, embeddings)
    chat = MultiDocChat(store, strategy=req.strategy, llm=llm, embeddings=embeddings, session_id=req.session_id)
    return chat.invoke(req.question, chat_history=[t.model_dump() for t in req.chat_history])


@app.post("/compare")
def compare(reference: UploadFile = File(...), actual: UploadFile = File(...), llm=Depends(get_llm),
            principal: Principal = Depends(get_principal)):
    ingestion = DocumentIngestion()
    ref_path, act_path = ingestion.save_uploaded_files(reference, actual)
    rows = DocumentComparatorLLM(llm=llm).compare_documents(ingestion.combine_documents(ref_path, act_path))
    return {"session_id": ingestion.session_id, "changes": rows}
