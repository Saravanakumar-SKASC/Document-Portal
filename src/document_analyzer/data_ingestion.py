import os
import sys
from pathlib import Path

import pymupdf

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from utils.document_ops import check_page_limit, maybe_redact, new_session_id, save_uploaded_file


class DocumentHandler:
    """
    Handles PDF saving and reading operations.
    Automatically logs all actions and supports session-based organization.
    """

    def __init__(self, data_dir=None, session_id=None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.data_dir = data_dir or os.getenv(
                "DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data", "document_analysis")
            )
            self.session_id = session_id or new_session_id()
            self.session_path = os.path.join(self.data_dir, self.session_id)
            os.makedirs(self.session_path, exist_ok=True)
            self.log.info("PDFHandler initialized", session_id=self.session_id, session_path=self.session_path)
        except Exception as e:
            raise DocumentPortalException("Error initializing DocumentHandler", e) from e

    def save_pdf(self, uploaded_file) -> str:
        """Save a PDF upload (Streamlit, FastAPI or a local path) into this session's folder."""
        try:
            save_path = save_uploaded_file(uploaded_file, self.session_path, allowed_extensions={".pdf"})
            self.log.info("PDF saved successfully", file=save_path.name, save_path=str(save_path),
                          session_id=self.session_id)
            return str(save_path)
        except DocumentPortalException:
            raise
        except Exception as e:
            self.log.error("Error saving PDF", error=str(e))
            raise DocumentPortalException("Error saving PDF", e) from e

    def read_pdf(self, pdf_path: str) -> str:
        try:
            text_chunks = []
            with pymupdf.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise DocumentPortalException(f"PDF is encrypted: {Path(pdf_path).name}")
                check_page_limit(doc.page_count, Path(pdf_path).name)
                for page_num, page in enumerate(doc, start=1):
                    # PII is masked before the text ever reaches the LLM (config privacy.redact_pii)
                    text_chunks.append(f"\n--- Page {page_num} ---\n{maybe_redact(page.get_text())}")
            text = "\n".join(text_chunks)
            self.log.info("PDF read successfully", pdf_path=pdf_path, session_id=self.session_id,
                          pages=len(text_chunks))
            return text
        except DocumentPortalException:
            raise
        except Exception as e:
            self.log.error("Error reading PDF", error=str(e))
            raise DocumentPortalException("Error reading PDF", e) from e


if __name__ == "__main__":
    # Usage: python -m src.document_analyzer.data_ingestion path/to/file.pdf
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m src.document_analyzer.data_ingestion <path-to-pdf>")
    handler = DocumentHandler()
    saved_path = handler.save_pdf(Path(sys.argv[1]))
    print(f"Saved to: {saved_path}")
    print(handler.read_pdf(saved_path)[:500])
