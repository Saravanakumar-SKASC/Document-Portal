"""
Shared fixtures. Everything runs offline: no API keys, no network.

- ``fake_llm`` scripts LLM replies (LangChain's FakeListChatModel)
- ``bow_embeddings`` is a tiny hashed bag-of-words embedding, so similarity search
  returns genuinely relevant chunks (unlike random fake embeddings)
"""

import hashlib
import math
import os
import re
import tempfile

import fitz
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

os.environ.setdefault("LOG_DIR", os.path.join(tempfile.gettempdir(), "document_portal_test_logs"))


class BagOfWordsEmbeddings(Embeddings):
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def make_pdf(path, pages: list[str]):
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 800), text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture(autouse=True)
def isolated_workdir(tmp_path, monkeypatch):
    """Run each test in its own folder so data/ and faiss_index/ never touch the repo."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_STORAGE_PATH", str(tmp_path / "data"))
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
