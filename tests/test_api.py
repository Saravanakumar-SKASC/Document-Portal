import pytest
from fastapi.testclient import TestClient

import app as api
from tests.conftest import METADATA_JSON, make_pdf


@pytest.fixture
def client(bow_embeddings, fake_llm, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "FAISS_DIR", tmp_path / "faiss_index")
    responses = {"llm": fake_llm(METADATA_JSON)}
    api.app.dependency_overrides[api.get_llm] = lambda: responses["llm"]
    api.app.dependency_overrides[api.get_embeddings] = lambda: bow_embeddings
    yield TestClient(api.app), responses
    api.app.dependency_overrides.clear()


def test_health(client):
    assert client[0].get("/health").json() == {"status": "ok"}


def test_analyze_endpoint(client, cloud_pdf):
    http, _ = client
    with open(cloud_pdf, "rb") as f:
        response = http.post("/analyze", files={"file": ("cloud_guide.pdf", f, "application/pdf")})
    assert response.status_code == 200
    assert response.json()["metadata"]["Title"] == "Cloud Guide"


def test_analyze_rejects_non_pdf(client):
    response = client[0].post("/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 422
    assert "Unsupported file type" in response.json()["detail"]


def test_index_then_query(client, cloud_pdf, hr_pdf, fake_llm):
    http, responses = client
    with open(cloud_pdf, "rb") as a, open(hr_pdf, "rb") as b:
        indexed = http.post("/chat/index", files=[("files", ("cloud_guide.pdf", a)), ("files", ("hr_policy.pdf", b))])
    assert indexed.status_code == 200
    session_id = indexed.json()["session_id"]

    responses["llm"] = fake_llm("25 days [hr_policy.pdf, p.1]")
    answer = http.post("/chat/query", json={"session_id": session_id, "question": "How much annual leave?",
                                             "strategy": "mmr+compression"})
    assert answer.status_code == 200
    assert answer.json()["sources"][0]["source"] == "hr_policy.pdf"


def test_query_rejects_path_traversal_session(client):
    response = client[0].post("/chat/query", json={"session_id": "../../etc", "question": "hi"})
    assert response.status_code == 400


def test_compare_endpoint(client, tmp_path, fake_llm):
    http, responses = client
    responses["llm"] = fake_llm('[{"Page": "1", "Changes": "20 -> 25 days"}]')
    ref = make_pdf(tmp_path / "v1.pdf", ["Leave is 20 days."])
    act = make_pdf(tmp_path / "v2.pdf", ["Leave is 25 days."])
    with open(ref, "rb") as r, open(act, "rb") as a:
        response = http.post("/compare", files={"reference": ("v1.pdf", r), "actual": ("v2.pdf", a)})
    assert response.json()["changes"] == [{"Page": "1", "Changes": "20 -> 25 days"}]
