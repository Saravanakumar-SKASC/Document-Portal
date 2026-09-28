"""Production-hardening features: PII, limits, hybrid search, guardrails, citations, fallback, auth."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import app as api
from exception.custom_exception import DocumentPortalException
from src.doc_compare.data_ingestion import DocumentIngestion
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.multidoc_chat.data_ingestion import MultiDocIngestor
from src.multidoc_chat.hybrid import tokenize
from src.multidoc_chat.retrieval import MultiDocChat, build_retriever
from tests.conftest import METADATA_JSON
from utils.config_loader import load_config
from utils.document_ops import load_documents, save_uploaded_file
from utils.model_loader import ModelLoader, with_fallback_llm
from utils.pii import redact
from utils.rag_core import check_citations
from utils.security import authenticate, can_access, parse_api_keys

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
TASK_CARD = SAMPLES / "demoair_task_card_ata32_brakes.pdf"
CREW_SOP = SAMPLES / "demoair_cabin_crew_sop.pdf"
BAGGAGE_12 = SAMPLES / "demoair_baggage_policy_rev12.pdf"
BAGGAGE_13 = SAMPLES / "demoair_baggage_policy_rev13.pdf"


# ---------- PII ----------
@pytest.mark.parametrize("text, kind", [
    ("mail j.murphy@demoair.example now", "email"),
    ("call +353 1 555 0147", "phone"),
    ("Phone: 089 206 2924", "phone"),
    ("card 4111 1111 1111 1111", "card"),
    ("HKID A123456(7)", "hkid"),
    ("Passport No: K12345678", "passport"),
])
def test_pii_is_redacted(text, kind):
    result = redact(text)
    assert result.counts.get(kind) == 1, result


def test_aviation_identifiers_are_not_redacted():
    text = "Replace P/N 2315-0045-001 per task 32-41-00-210-801 on flight DA123, 2026-09-01, 3000 psi"
    assert redact(text).text == text


def test_non_luhn_number_is_not_treated_as_card():
    assert redact("ref 4111 1111 1111 1112").text == "ref 4111 1111 1111 1112"


def test_documents_are_redacted_and_tagged_with_revision():
    pages = load_documents([CREW_SOP], extra_metadata={"revision": "Rev 4"})
    text = " ".join(p.page_content for p in pages)
    assert "K12345678" not in text and "[PASSPORT]" in text and "[EMAIL]" in text
    assert all(p.metadata["revision"] == "Rev 4" for p in pages)


def test_compare_input_is_redacted():
    ingestion = DocumentIngestion()
    assert "j.murphy@demoair.example" not in ingestion.read_pdf(TASK_CARD)


# ---------- Limits ----------
def test_oversized_upload_is_rejected_and_deleted(tmp_path, monkeypatch):
    monkeypatch.setitem(load_config()["limits"], "max_file_mb", 0.0001)  # ~100 bytes
    with pytest.raises(DocumentPortalException, match="larger than"):
        save_uploaded_file(TASK_CARD, tmp_path / "out")
    assert not (tmp_path / "out" / TASK_CARD.name).exists()


def test_page_limit(monkeypatch):
    monkeypatch.setitem(load_config()["limits"], "max_pages", 1)
    with pytest.raises(DocumentPortalException, match="limit is 1"):
        load_documents([TASK_CARD])


def test_too_many_files(monkeypatch, bow_embeddings):
    monkeypatch.setitem(load_config()["limits"], "max_files_per_request", 1)
    with pytest.raises(DocumentPortalException, match="Too many files"):
        MultiDocIngestor(embeddings=bow_embeddings).ingest([TASK_CARD, CREW_SOP])


# ---------- Hybrid search ----------
def test_tokenizer_keeps_identifiers_whole_and_split():
    tokens = tokenize("Replace P/N 2315-0045-001 (ATA 32-41-00)")
    assert "2315-0045-001" in tokens and "32-41-00" in tokens and "0045" in tokens


@pytest.fixture
def airline_store(bow_embeddings):
    return MultiDocIngestor(embeddings=bow_embeddings).ingest([TASK_CARD, CREW_SOP, BAGGAGE_13], revision="demo")


def test_hybrid_finds_exact_identifier(airline_store):
    docs = build_retriever(airline_store, "hybrid", k=3).invoke("What is tool T-3241-06?")
    assert docs[0].metadata["source"] == TASK_CARD.name and docs[0].metadata["page"] == 2
    assert len(docs) <= 3


# ---------- Guardrails and citations ----------
def test_off_topic_question_is_refused_without_calling_llm(airline_store):
    never_called = FakeListChatModel(responses=[])  # raises if invoked
    chat = MultiDocChat(airline_store, strategy="hybrid", llm=never_called)
    result = chat.invoke("Who won the football world cup in 1998?")
    assert result["refused"] is True and result["sources"] == []


def test_grounded_answer_with_valid_citation(airline_store, fake_llm):
    answer = f"Replace it when the wear pin is flush [{TASK_CARD.name}, p.1]."
    chat = MultiDocChat(airline_store, strategy="hybrid", llm=fake_llm(answer))
    result = chat.invoke("When must brake assembly P/N 2315-0045-001 be replaced?")
    assert result["grounded"] is True and result["refused"] is False
    assert result["sources"][0]["revision"] == "demo"


def test_invented_citation_is_flagged(airline_store, fake_llm):
    chat = MultiDocChat(airline_store, strategy="hybrid", llm=fake_llm("It is 42 [made_up.pdf, p.9]."))
    result = chat.invoke("When must brake assembly P/N 2315-0045-001 be replaced?")
    assert result["grounded"] is False and result["unverified_citations"] == ["made_up.pdf, p.9"]


def test_answer_without_citation_is_not_grounded():
    assert check_citations("No citations here.", [])["grounded"] is False


# ---------- LLM resilience ----------
class FailingLLM(FakeListChatModel):
    def _call(self, *args, **kwargs):
        raise TimeoutError("provider timed out")


def test_fallback_llm_takes_over_when_primary_fails(fake_llm):
    llm = with_fallback_llm(FailingLLM(responses=["unused"]), fake_llm(METADATA_JSON))
    assert DocumentAnalyzer(llm=llm).analyze_document("text")["Title"] == "Cloud Guide"


def test_model_loader_enables_fallback_when_both_keys_exist(monkeypatch):
    monkeypatch.setattr("utils.model_loader.load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    monkeypatch.setenv("GROQ_API_KEY", "q-test")
    llm = ModelLoader().load_llm()
    assert type(llm).__name__ == "RunnableWithFallbacks"
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.setenv("LLM_PROVIDER", "google")
    assert type(ModelLoader().load_llm()).__name__ == "ChatGoogleGenerativeAI"


# ---------- Security ----------
KEYS = "eng-key-1111:engineering,crew-key-2222:cabin_crew|flight_ops,admin-key-3333:admin"


def test_api_key_parsing_and_access_rules(monkeypatch):
    assert parse_api_keys(KEYS)["crew-key-2222"] == {"cabin_crew", "flight_ops"}
    monkeypatch.setenv("API_KEYS", KEYS)
    engineer = authenticate("eng-key-1111")
    assert engineer.roles == {"engineering"} and authenticate("wrong") is None and authenticate(None) is None
    assert can_access(engineer, ["engineering"]) and not can_access(engineer, ["cabin_crew"])
    assert can_access(authenticate("admin-key-3333"), ["cabin_crew"])
    assert can_access(engineer, [])  # open collection


@pytest.fixture
def secured_client(bow_embeddings, fake_llm, monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEYS", KEYS)
    monkeypatch.setattr(api, "FAISS_DIR", tmp_path / "faiss_index")
    api.app.dependency_overrides[api.get_llm] = lambda: fake_llm(f"Flush means replace [{TASK_CARD.name}, p.1].")
    api.app.dependency_overrides[api.get_embeddings] = lambda: bow_embeddings
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def _index(client, key, roles=None):
    with open(TASK_CARD, "rb") as f:
        data = {"revision": "Rev 7", **({"allowed_roles": roles} if roles else {})}
        return client.post("/chat/index", files=[("files", (TASK_CARD.name, f))], data=data,
                           headers={"X-API-Key": key})


def test_requests_without_valid_key_are_rejected(secured_client):
    assert secured_client.post("/chat/query", json={"session_id": "x", "question": "hi"}).status_code == 401
    assert secured_client.post("/chat/query", json={"session_id": "x", "question": "hi"},
                               headers={"X-API-Key": "nope"}).status_code == 401
    assert secured_client.get("/health").status_code == 200  # health stays public


def test_unknown_collection_returns_404_without_leaking_paths(secured_client):
    response = secured_client.post("/chat/query", json={"session_id": "missing_1", "question": "hi"},
                                   headers={"X-API-Key": "eng-key-1111"})
    assert response.status_code == 404 and "faiss" not in response.text


def test_role_based_access_to_collections(secured_client):
    indexed = _index(secured_client, "eng-key-1111")
    assert indexed.status_code == 200 and indexed.json()["allowed_roles"] == ["engineering"]
    body = {"session_id": indexed.json()["session_id"], "question": "When must P/N 2315-0045-001 be replaced?"}

    ok = secured_client.post("/chat/query", json=body, headers={"X-API-Key": "eng-key-1111"})
    assert ok.status_code == 200 and ok.json()["grounded"] is True
    assert ok.json()["sources"][0]["revision"] == "Rev 7"

    denied = secured_client.post("/chat/query", json=body, headers={"X-API-Key": "crew-key-2222"})
    assert denied.status_code == 404  # hidden, not just forbidden

    admin = secured_client.post("/chat/query", json=body, headers={"X-API-Key": "admin-key-3333"})
    assert admin.status_code == 200


def test_cannot_share_collection_with_roles_you_do_not_hold(secured_client):
    assert _index(secured_client, "eng-key-1111", roles="cabin_crew").status_code == 403
    assert _index(secured_client, "admin-key-3333", roles="cabin_crew").status_code == 200


def test_request_id_is_returned_and_propagated(secured_client):
    response = secured_client.get("/health", headers={"X-Request-ID": "trace-123"})
    assert response.headers["X-Request-ID"] == "trace-123"
    assert len(secured_client.get("/health").headers["X-Request-ID"]) == 32


# ---------- Evaluation script ----------
def test_evaluation_script_runs_offline(capsys):
    from scripts.evaluate import main

    assert main(["--offline", "--min-hit-rate", "0.8"]) == 0
    output = capsys.readouterr().out
    assert "hybrid" in output and "min_relevance" in output


def test_revision_compare_input_contains_both_versions():
    ingestion = DocumentIngestion()
    ref, act = ingestion.save_uploaded_files(BAGGAGE_12, BAGGAGE_13)
    combined = ingestion.combine_documents(ref, act)
    assert "maximum 23 kg" in combined and "maximum 25 kg" in combined
