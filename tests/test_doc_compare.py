import pytest

from exception.custom_exception import DocumentPortalException
from src.doc_compare.data_ingestion import DocumentIngestion
from src.doc_compare.retrieval import DocumentComparatorLLM
from tests.conftest import make_pdf

CHANGES = '[{"Page": "1", "Changes": "Leave increased from 20 to 25 days"}, {"Page": "2", "Changes": "NO CHANGE"}]'


@pytest.fixture
def pdf_pair(tmp_path):
    ref = make_pdf(tmp_path / "policy_v1.pdf", ["Annual leave is 20 days.", "Remote work 3 days."])
    act = make_pdf(tmp_path / "policy_v2.pdf", ["Annual leave is 25 days.", "Remote work 3 days."])
    return ref, act


def test_combined_text_labels_both_documents(pdf_pair):
    ingestion = DocumentIngestion()
    ref, act = ingestion.save_uploaded_files(*pdf_pair)
    combined = ingestion.combine_documents(ref, act)
    assert "REFERENCE (policy_v1.pdf)" in combined and "ACTUAL (policy_v2.pdf)" in combined
    assert "20 days" in combined and "25 days" in combined


def test_comparator_returns_validated_rows(pdf_pair, fake_llm):
    rows = DocumentComparatorLLM(llm=fake_llm(CHANGES)).compare_documents("combined text")
    assert rows == [
        {"Page": "1", "Changes": "Leave increased from 20 to 25 days"},
        {"Page": "2", "Changes": "NO CHANGE"},
    ]


def test_comparator_unwraps_dict_and_repairs_json(fake_llm):
    wrapped = '{"changes": ' + CHANGES + "}"
    rows = DocumentComparatorLLM(llm=fake_llm("oops not json", wrapped)).compare_documents("combined")
    assert len(rows) == 2


def test_comparator_rejects_rows_missing_fields(fake_llm):
    with pytest.raises(DocumentPortalException):
        DocumentComparatorLLM(llm=fake_llm('[{"Page": "1"}]')).compare_documents("combined")
