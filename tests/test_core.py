import pytest

from exception.custom_exception import DocumentPortalException
from utils.config_loader import load_config
from utils.document_ops import load_documents, safe_filename, save_uploaded_file, split_documents


def test_exception_outside_except_block_does_not_crash():
    err = DocumentPortalException("Invalid file type")
    assert "Invalid file type" in str(err)
    assert err.lineno > 0


def test_exception_wraps_cause_with_traceback():
    try:
        1 / 0
    except ZeroDivisionError as e:
        err = DocumentPortalException("Division failed", e)
    assert "ZeroDivisionError" in err.traceback_str
    assert err.file_name.endswith("test_core.py")


def test_config_loads_from_any_working_directory():
    config = load_config()
    assert config["retriever"]["top_k"] > 0
    assert {"groq", "google"} <= set(config["llm"])


def test_safe_filename_blocks_path_traversal():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("C:\\Users\\me\\My Report.pdf") == "My_Report.pdf"


def test_save_rejects_unsupported_extension(tmp_path):
    bad = tmp_path / "malware.exe"
    bad.write_bytes(b"x")
    with pytest.raises(DocumentPortalException, match="Unsupported file type"):
        save_uploaded_file(bad, tmp_path / "out")


def test_load_and_split_keep_source_and_page(cloud_pdf):
    docs = load_documents([cloud_pdf])
    assert [d.metadata["page"] for d in docs] == [1, 2, 3]
    chunks = split_documents(docs, chunk_size=60, chunk_overlap=10)
    assert len(chunks) > len(docs)
    assert all(c.metadata["source"] == "cloud_guide.pdf" for c in chunks)


def test_model_loader_requires_only_needed_keys(monkeypatch):
    from utils.model_loader import ModelLoader

    monkeypatch.setattr("utils.model_loader.load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert ModelLoader().api_keys == {"GOOGLE_API_KEY": "test-key"}

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    with pytest.raises(DocumentPortalException, match="GROQ_API_KEY"):
        ModelLoader()
