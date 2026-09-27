import os
from pathlib import Path

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from utils.config_loader import load_config
from utils.document_ops import load_documents, new_session_id, save_uploaded_file, split_documents
from utils.rag_core import build_faiss_index


class SingleDocIngestor:
    """
    Ingest one document for chat: save it, split into chunks, embed, and persist a
    FAISS index under ``faiss_index/<session_id>`` so the session can be reloaded later.
    """

    def __init__(self, data_dir=None, faiss_dir=None, embeddings=None, session_id=None):
        self.log = CustomLogger().get_logger(__name__)
        self.config = load_config()
        base = Path(os.getenv("DATA_STORAGE_PATH", "data"))
        self.session_id = session_id or new_session_id("single")
        self.data_dir = Path(data_dir or base / "single_document_chat") / self.session_id
        self.faiss_dir = Path(faiss_dir or "faiss_index") / self.session_id
        if embeddings is None:
            from utils.model_loader import ModelLoader

            embeddings = ModelLoader().load_embeddings()
        self.embeddings = embeddings

    def ingest(self, uploaded_file, k: int | None = None):
        """Returns a similarity retriever over the new index."""
        try:
            path = save_uploaded_file(uploaded_file, self.data_dir)
            documents = load_documents([path])
            splitter_cfg = self.config.get("text_splitter", {})
            chunks = split_documents(documents, splitter_cfg.get("chunk_size", 1000),
                                     splitter_cfg.get("chunk_overlap", 150))
            self.vectorstore = build_faiss_index(chunks, self.embeddings, self.faiss_dir)
            self.log.info("Single document ingested", file=path.name, pages=len(documents), chunks=len(chunks),
                          session_id=self.session_id)
            top_k = k or self.config["retriever"]["top_k"]
            return self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        except DocumentPortalException:
            raise
        except Exception as e:
            self.log.error("Single document ingestion failed", error=str(e))
            raise DocumentPortalException("Failed to ingest document", e) from e
