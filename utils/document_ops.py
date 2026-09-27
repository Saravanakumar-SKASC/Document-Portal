"""Shared helpers for saving uploads, loading documents and chunking them."""

import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from exception.custom_exception import DocumentPortalException

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def new_session_id(prefix: str = "session") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def safe_filename(name: str) -> str:
    """Strip directories and unsafe characters so uploads can't escape their folder."""
    base = os.path.basename(str(name).replace("\\", "/"))
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    if not base:
        raise DocumentPortalException("Invalid file name.")
    return base


def save_uploaded_file(uploaded_file, target_dir: str | os.PathLike, allowed_extensions=SUPPORTED_EXTENSIONS) -> Path:
    """
    Save an upload to ``target_dir`` and return the saved path.

    Accepts a Streamlit UploadedFile (``.name`` + ``.getbuffer()``), a FastAPI
    UploadFile (``.filename`` + ``.file``), or a local path (str / Path).
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(uploaded_file, (str, os.PathLike)):
        original_name = Path(uploaded_file).name
    else:
        original_name = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "filename", None)
    if not original_name:
        raise DocumentPortalException("Uploaded file has no name.")

    filename = safe_filename(original_name)
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise DocumentPortalException(f"Unsupported file type '{ext or 'none'}'. Allowed: {allowed}")

    save_path = target_dir / filename
    if isinstance(uploaded_file, (str, os.PathLike)):
        shutil.copyfile(uploaded_file, save_path)
    elif hasattr(uploaded_file, "getbuffer"):
        save_path.write_bytes(bytes(uploaded_file.getbuffer()))
    elif hasattr(uploaded_file, "file"):
        uploaded_file.file.seek(0)
        with open(save_path, "wb") as out:
            shutil.copyfileobj(uploaded_file.file, out)
    elif hasattr(uploaded_file, "read"):
        save_path.write_bytes(uploaded_file.read())
    else:
        raise DocumentPortalException("Unsupported upload object.")
    return save_path


def load_documents(paths) -> list[Document]:
    """Load files into LangChain Documents: one Document per PDF page, one per text file."""
    import fitz  # PyMuPDF

    documents: list[Document] = []
    for path in map(Path, paths):
        ext = path.suffix.lower()
        if ext == ".pdf":
            with fitz.open(path) as pdf:
                for page_num, page in enumerate(pdf, start=1):
                    text = page.get_text().strip()
                    if text:
                        documents.append(Document(page_content=text, metadata={"source": path.name, "page": page_num}))
        elif ext in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                documents.append(Document(page_content=text, metadata={"source": path.name, "page": 1}))
        else:
            raise DocumentPortalException(f"Unsupported file type: {ext}")
    if not documents:
        raise DocumentPortalException("No extractable text found. Scanned PDFs need OCR first.")
    return documents


def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks with source + page tags the LLM can cite."""
    return "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}, p.{d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs
    )
