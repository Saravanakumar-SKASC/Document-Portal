import pytest

from exception.custom_exception import DocumentPortalException
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_analyzer.data_ingestion import DocumentHandler
from tests.conftest import METADATA_JSON


def test_handler_saves_and_reads_pdf(cloud_pdf):
    handler = DocumentHandler(session_id="test")
    text = handler.read_pdf(handler.save_pdf(cloud_pdf))
    assert "--- Page 3 ---" in text and "Kubernetes" in text


def test_handler_rejects_non_pdf(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    with pytest.raises(DocumentPortalException):
        DocumentHandler().save_pdf(txt)


def test_analyzer_returns_structured_metadata(fake_llm):
    result = DocumentAnalyzer(llm=fake_llm(METADATA_JSON)).analyze_document("Kubernetes guide")
    assert result["Title"] == "Cloud Guide"
    assert result["PageCount"] == 3


def test_analyzer_repairs_malformed_json(fake_llm):
    # First reply is broken; OutputFixingParser asks the LLM again and gets valid JSON.
    analyzer = DocumentAnalyzer(llm=fake_llm("Sure! Title: Cloud Guide, no JSON here", METADATA_JSON))
    assert analyzer.analyze_document("Kubernetes guide")["Title"] == "Cloud Guide"


def test_analyzer_rejects_empty_text(fake_llm):
    with pytest.raises(DocumentPortalException, match="empty"):
        DocumentAnalyzer(llm=fake_llm(METADATA_JSON)).analyze_document("   ")
