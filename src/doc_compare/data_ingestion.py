import os
from pathlib import Path

import fitz

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from utils.document_ops import new_session_id, save_uploaded_file


class DocumentIngestion:
    """Save a reference and an actual PDF for one comparison session and extract page-marked text."""

    def __init__(self, base_dir=None, session_id=None):
        self.log = CustomLogger().get_logger(__name__)
        base = Path(base_dir or Path(os.getenv("DATA_STORAGE_PATH", "data")) / "document_compare")
        self.session_id = session_id or new_session_id("compare")
        self.session_path = base / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)

    def save_uploaded_files(self, reference_file, actual_file) -> tuple[Path, Path]:
        try:
            ref_path = save_uploaded_file(reference_file, self.session_path / "reference", {".pdf"})
            act_path = save_uploaded_file(actual_file, self.session_path / "actual", {".pdf"})
            self.log.info("Comparison files saved", reference=ref_path.name, actual=act_path.name,
                          session_id=self.session_id)
            return ref_path, act_path
        except DocumentPortalException:
            raise
        except Exception as e:
            raise DocumentPortalException("Error saving files for comparison", e) from e

    def read_pdf(self, pdf_path: Path) -> str:
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise DocumentPortalException(f"PDF is encrypted: {Path(pdf_path).name}")
                pages = [f"\n--- Page {i} ---\n{page.get_text()}" for i, page in enumerate(doc, start=1)]
            return "\n".join(pages)
        except DocumentPortalException:
            raise
        except Exception as e:
            raise DocumentPortalException("Error reading PDF", e) from e

    def combine_documents(self, reference_path: Path, actual_path: Path) -> str:
        """Label each document so the LLM knows which text is which."""
        return (
            f"Document: REFERENCE ({Path(reference_path).name})\n{self.read_pdf(reference_path)}\n\n"
            f"Document: ACTUAL ({Path(actual_path).name})\n{self.read_pdf(actual_path)}"
        )
