"""Shared helpers for saving uploads, loading documents and chunking them."""

import os
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from exception.custom_exception import DocumentPortalException
from utils.config_loader import load_config
from utils.pii import redact_text

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def new_session_id(prefix: str = "session") -> str:
    return f"{prefix}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


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

    if isinstance(uploaded_file, str | os.PathLike):
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
    if isinstance(uploaded_file, str | os.PathLike):
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

    max_mb = _limits().get("max_file_mb")
    if max_mb and save_path.stat().st_size > max_mb * 1024 * 1024:
        save_path.unlink(missing_ok=True)
        raise DocumentPortalException(f"File '{filename}' is larger than the {max_mb} MB limit.")
    return save_path


def _limits() -> dict:
    return load_config().get("limits", {})


def pii_redaction_enabled() -> bool:
    return bool(load_config().get("privacy", {}).get("redact_pii", False))


def maybe_redact(text: str, enabled: bool | None = None) -> str:
    """Redact PII when enabled (defaults to config privacy.redact_pii)."""
    if enabled is None:
        enabled = pii_redaction_enabled()
    return redact_text(text) if enabled else text


def check_page_limit(page_count: int, name: str):
    max_pages = _limits().get("max_pages")
    if max_pages and page_count > max_pages:
        raise DocumentPortalException(f"'{name}' has {page_count} pages; the limit is {max_pages}.")


def load_documents(paths, redact_pii: bool | None = None, extra_metadata: dict | None = None) -> list[Document]:
    """
    Load files into LangChain Documents: one Document per PDF page, one per text file.

    PII is redacted before the text is embedded or stored (config privacy.redact_pii).
    ``extra_metadata`` (e.g. {"revision": "Rev 13"}) is attached to every page so
    answers can cite the exact document revision.
    """
    import pymupdf

    if redact_pii is None:
        redact_pii = pii_redaction_enabled()
    extra_metadata = {k: v for k, v in (extra_metadata or {}).items() if v not in (None, "")}

    documents: list[Document] = []
    for path in map(Path, paths):
        ext = path.suffix.lower()
        pages: list[tuple[int, str]] = []
        if ext == ".pdf":
            with pymupdf.open(path) as pdf:
                if pdf.is_encrypted:
                    raise DocumentPortalException(f"PDF is encrypted: {path.name}")
                check_page_limit(pdf.page_count, path.name)
                pages = [(n, page.get_text()) for n, page in enumerate(pdf, start=1)]
        elif ext in {".txt", ".md"}:
            pages = [(1, path.read_text(encoding="utf-8", errors="ignore"))]
        else:
            raise DocumentPortalException(f"Unsupported file type: {ext}")

        for page_num, text in pages:
            text = maybe_redact(text.strip(), redact_pii)
            if text:
                metadata = {"source": path.name, "page": page_num, **extra_metadata}
                documents.append(Document(page_content=text, metadata=metadata))
    if not documents:
        raise DocumentPortalException("No extractable text found. Scanned PDFs need OCR first.")
    return documents


def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def citation_tag(doc: Document) -> str:
    tag = f"{doc.metadata.get('source', 'unknown')}, p.{doc.metadata.get('page', '?')}"
    return f"[{tag}]"


def format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks with source + page tags the LLM can cite (plus revision when known)."""
    blocks = []
    for d in docs:
        header = citation_tag(d)
        if d.metadata.get("revision"):
            header += f" (revision: {d.metadata['revision']})"
        blocks.append(f"{header}\n{d.page_content}")
    return "\n\n".join(blocks)
