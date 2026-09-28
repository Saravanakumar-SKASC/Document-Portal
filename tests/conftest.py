"""
Shared fixtures. Everything runs offline: no API keys, no network.

- ``fake_llm`` scripts LLM replies (LangChain's FakeListChatModel)
- ``bow_embeddings`` is a tiny hashed bag-of-words embedding, so similarity search
  returns genuinely relevant chunks (unlike random fake embeddings)
"""

import os
import tempfile

import pymupdf
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from utils.config_loader import load_config
from utils.offline_models import BagOfWordsEmbeddings  # noqa: F401  (re-exported for tests)

os.environ.setdefault("LOG_DIR", os.path.join(tempfile.gettempdir(), "document_portal_test_logs"))


def make_pdf(path, pages: list[str]):
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture(autouse=True)
def isolated_workdir(tmp_path, monkeypatch):
    """Run each test in its own folder so data/ and faiss_index/ never touch the repo."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_STORAGE_PATH", str(tmp_path / "data"))
    monkeypatch.delenv("API_KEYS", raising=False)
    # Bag-of-words similarities run lower than real embeddings, so use a lower guardrail threshold
    monkeypatch.setitem(load_config()["guardrails"], "min_relevance", 0.1)
    return tmp_path


@pytest.fixture
def bow_embeddings():
    return BagOfWordsEmbeddings()


@pytest.fixture
def fake_llm():
    def _make(*responses):
        return FakeListChatModel(responses=list(responses))
    return _make


@pytest.fixture
def cloud_pdf(tmp_path):
    return make_pdf(tmp_path / "cloud_guide.pdf", [
        "Kubernetes orchestrates containers. Pods are the smallest deployable units in Kubernetes clusters.",
        "Terraform provisions infrastructure as code. State files track the resources Terraform manages.",
        "Prometheus scrapes metrics and Grafana dashboards visualise latency and error rates.",
    ])


@pytest.fixture
def hr_pdf(tmp_path):
    return make_pdf(tmp_path / "hr_policy.pdf", [
        "Employees receive twenty five days of annual leave each year.",
        "Remote work is allowed three days per week with manager approval.",
    ])


METADATA_JSON = (
    '{"Summary": ["Covers Kubernetes, Terraform and monitoring."], "Title": "Cloud Guide", '
    '"Author": "Not Available", "DateCreated": "Not Available", "LastModifiedDate": "Not Available", '
    '"Publisher": "Not Available", "Language": "English", "PageCount": 3, "SentimentTone": "Informative"}'
)
