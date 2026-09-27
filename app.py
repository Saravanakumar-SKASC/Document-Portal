"""
Document Portal REST API.

Run:  uvicorn app:app --reload        Docs: http://127.0.0.1:8000/docs
"""

import re
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
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
from utils.rag_core import load_faiss_index

log = CustomLogger().get_logger(__name__)
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
FAISS_DIR = Path("faiss_index")

app = FastAPI(
    title="Document Portal API",
    description="LLM document intelligence: analyze, chat (RAG) and compare documents.",
    version="1.0.0",
)


# ---------- Dependencies (overridable in tests) ----------
@lru_cache(maxsize=1)
def _model_loader():
    from utils.model_loader import ModelLoader

    return ModelLoader()


def get_llm():
    return _model_loader().load_llm()


def get_embeddings():
    return _model_loader().load_embeddings()


@app.exception_handler(DocumentPortalException)
async def portal_error_handler(_, exc: DocumentPortalException):
    log.error("Request failed", error=exc.error_message)
    return JSONResponse(status_code=422, content={"detail": exc.error_message})


# ---------- Schemas ----------
class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1)
    strategy: str = "mmr"
    chat_history: list[ChatTurn] = Field(default_factory=list)


# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(file: UploadFile = File(...), llm=Depends(get_llm)):
    handler = DocumentHandler()
    text = handler.read_pdf(handler.save_pdf(file))
    return {"session_id": handler.session_id, "metadata": DocumentAnalyzer(llm=llm).analyze_document(text)}


@app.post("/chat/index")
def chat_index(files: list[UploadFile] = File(...), embeddings=Depends(get_embeddings)):
    ingestor = MultiDocIngestor(embeddings=embeddings, faiss_dir=FAISS_DIR)
    store = ingestor.ingest(files)
    return {"session_id": ingestor.session_id, "files": [f.filename for f in files],
            "chunks": store.index.ntotal}


@app.post("/chat/query")
def chat_query(req: ChatRequest, llm=Depends(get_llm), embeddings=Depends(get_embeddings)):
    if not SESSION_ID_PATTERN.match(req.session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id.")
    if req.strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"strategy must be one of {list(STRATEGIES)}")
    store = load_faiss_index(FAISS_DIR / req.session_id, embeddings)
    chat = MultiDocChat(store, strategy=req.strategy, llm=llm, embeddings=embeddings, session_id=req.session_id)
    return chat.invoke(req.question, chat_history=[t.model_dump() for t in req.chat_history])


@app.post("/compare")
def compare(reference: UploadFile = File(...), actual: UploadFile = File(...), llm=Depends(get_llm)):
    ingestion = DocumentIngestion()
    ref_path, act_path = ingestion.save_uploaded_files(reference, actual)
    rows = DocumentComparatorLLM(llm=llm).compare_documents(ingestion.combine_documents(ref_path, act_path))
    return {"session_id": ingestion.session_id, "changes": rows}
