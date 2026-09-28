import os
from pathlib import Path

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from utils.config_loader import load_config
from utils.document_ops import load_documents, new_session_id, save_uploaded_file, split_documents
from utils.rag_core import build_faiss_index


class MultiDocIngestor:
    """
    Ingest several documents into one FAISS index. Every chunk keeps its source
    file name and page number, so answers can cite exactly where they came from.
    """

    def __init__(self, data_dir=None, faiss_dir=None, embeddings=None, session_id=None):
        self.log = CustomLogger().get_logger(__name__)
        self.config = load_config()
        base = Path(os.getenv("DATA_STORAGE_PATH", "data"))
        self.session_id = session_id or new_session_id("multi")
        self.data_dir = Path(data_dir or base / "multi_document_chat") / self.session_id
        self.faiss_dir = Path(faiss_dir or "faiss_index") / self.session_id
        if embeddings is None:
            from utils.model_loader import ModelLoader

            embeddings = ModelLoader().load_embeddings()
        self.embeddings = embeddings

    def ingest(self, uploaded_files, allowed_roles: list[str] | None = None, revision: str | None = None):
        """
        Returns the FAISS vector store; pick a retrieval strategy with ``build_retriever``.

        ``allowed_roles`` restricts who may query this collection (e.g. ["engineering"]);
        ``revision`` (e.g. "Rev 13") is stored on every chunk and shown in citations.
        """
        try:
            if not uploaded_files:
                raise DocumentPortalException("No files provided.")
            max_files = self.config.get("limits", {}).get("max_files_per_request")
            if max_files and len(uploaded_files) > max_files:
                raise DocumentPortalException(f"Too many files: {len(uploaded_files)} (limit {max_files}).")
            paths = [save_uploaded_file(f, self.data_dir) for f in uploaded_files]
            documents = load_documents(paths, extra_metadata={"revision": revision})
            splitter_cfg = self.config.get("text_splitter", {})
            chunks = split_documents(documents, splitter_cfg.get("chunk_size", 1000),
                                     splitter_cfg.get("chunk_overlap", 150))
            session_info = {"session_id": self.session_id, "files": [p.name for p in paths],
                            "allowed_roles": sorted(set(allowed_roles or [])), "revision": revision,
                            "chunks": len(chunks)}
            self.vectorstore = build_faiss_index(chunks, self.embeddings, self.faiss_dir, session_info)
            self.log.info("Multiple documents ingested", files=[p.name for p in paths], chunks=len(chunks),
                          allowed_roles=session_info["allowed_roles"], session_id=self.session_id)
            return self.vectorstore
        except DocumentPortalException:
            raise
        except Exception as e:
            self.log.error("Multi-document ingestion failed", error=str(e))
            raise DocumentPortalException("Failed to ingest documents", e) from e
