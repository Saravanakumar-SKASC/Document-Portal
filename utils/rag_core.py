"""FAISS index helpers and the conversational RAG core shared by single- and multi-document chat."""

import json
import re
from collections.abc import Callable
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from prompt.prompt_library import context_qa_prompt, contextualize_question_prompt
from utils.config_loader import load_config
from utils.document_ops import format_docs
from utils.pii import redact_text

log = CustomLogger().get_logger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_CITATION = re.compile(r"\[([^\[\]]+?),\s*p\.?\s*(\d+)\]")
SESSION_FILE = "session.json"

NOT_FOUND_ANSWER = (
    "I couldn't find this in the approved documents, so I won't guess. "
    "Try rephrasing, or check that the right manual has been uploaded."
)


# ---------- Vector store ----------
def build_faiss_index(chunks: list[Document], embeddings, save_dir: str | Path | None = None,
                      session_info: dict | None = None) -> FAISS:
    """
    Build a FAISS index with L2-normalised vectors (so distances map cleanly to cosine
    similarity) and optionally persist it together with a small session.json
    (allowed roles, revision, file names) used for access control.
    """
    if not chunks:
        raise DocumentPortalException("No chunks to index.")
    vectorstore = FAISS.from_documents(chunks, embeddings, normalize_L2=True)
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(save_dir))
        (save_dir / SESSION_FILE).write_text(json.dumps(session_info or {}, indent=2), encoding="utf-8")
        log.info("FAISS index saved", path=str(save_dir), chunks=len(chunks))
    return vectorstore


def load_faiss_index(index_dir: str | Path, embeddings) -> FAISS:
    index_dir = Path(index_dir)
    if not (index_dir / "index.faiss").exists():
        raise DocumentPortalException(f"No FAISS index found at {index_dir}. Ingest documents first.")
    # The index files are created by this app only; never load indexes from untrusted sources.
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True, normalize_L2=True)


def load_session_info(index_dir: str | Path) -> dict:
    path = Path(index_dir) / SESSION_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def all_chunks(vectorstore: FAISS) -> list[Document]:
    """Every chunk stored in the index (used to build the BM25 keyword index)."""
    return [vectorstore.docstore.search(doc_id) for doc_id in vectorstore.index_to_docstore_id.values()]


def top_cosine_similarity(vectorstore: FAISS, query: str) -> float:
    """
    Cosine similarity between the question and its closest chunk.
    With normalised vectors FAISS returns squared L2 distance d, and cos = 1 - d / 2.
    """
    results = vectorstore.similarity_search_with_score(query, k=1)
    if not results:
        return 0.0
    return 1.0 - float(results[0][1]) / 2.0


def relevance_gate_for(vectorstore: FAISS, threshold: float | None = "config") -> Callable[[str], bool] | None:
    """Return a function that says whether a question is answerable from this index, or None if disabled."""
    if threshold == "config":
        threshold = load_config().get("guardrails", {}).get("min_relevance")
    if threshold is None:
        return None
    return lambda query: top_cosine_similarity(vectorstore, query) >= threshold


# ---------- Chat helpers ----------
def to_messages(chat_history) -> list[BaseMessage]:
    """Accept LangChain messages, (role, content) tuples or {'role', 'content'} dicts."""
    messages: list[BaseMessage] = []
    for item in chat_history or []:
        if isinstance(item, BaseMessage):
            messages.append(item)
            continue
        role, content = (item["role"], item["content"]) if isinstance(item, dict) else item
        messages.append(HumanMessage(content) if role in {"user", "human"} else AIMessage(content))
    return messages


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    return _THINK_BLOCK.sub("", text).strip()


def sources_from_docs(docs: list[Document], snippet_chars: int = 200) -> list[dict]:
    sources = []
    for d in docs:
        source = {"source": d.metadata.get("source", "unknown"), "page": d.metadata.get("page"),
                  "snippet": d.page_content[:snippet_chars]}
        if d.metadata.get("revision"):
            source["revision"] = d.metadata["revision"]
        sources.append(source)
    return sources


def check_citations(answer: str, docs: list[Document]) -> dict:
    """
    Parse [file, p.N] citations from the answer and check each one points at a chunk
    that was actually retrieved. ``grounded`` is True only if there is at least one
    citation and none of them are invented.
    """
    retrieved = {(d.metadata.get("source"), d.metadata.get("page")) for d in docs}
    cited = [(src.strip(), int(page)) for src, page in _CITATION.findall(answer)]
    unknown = [f"{s}, p.{p}" for s, p in cited if (s, p) not in retrieved]
    return {"citations": [f"{s}, p.{p}" for s, p in cited], "unverified_citations": unknown,
            "grounded": bool(cited) and not unknown}


class ConversationalRAG:
    """
    History-aware RAG:
      1. rewrite the follow-up question into a standalone one
      2. relevance gate - if nothing in the index is close enough, refuse without calling the LLM
      3. retrieve chunks and answer strictly from them with [file, p.N] citations
      4. verify the citations point at retrieved chunks
    """

    def __init__(self, retriever, llm=None, session_id: str | None = None,
                 relevance_gate: Callable[[str], bool] | None = None):
        if llm is None:
            from utils.model_loader import ModelLoader

            llm = ModelLoader().load_llm()
        self.llm = llm
        self.retriever = retriever
        self.session_id = session_id
        self.relevance_gate = relevance_gate
        self._rewrite_chain = contextualize_question_prompt | self.llm | StrOutputParser()
        self._answer_chain = context_qa_prompt | self.llm | StrOutputParser()

    def _refusal(self, standalone: str) -> dict:
        log.info("Question refused by relevance gate", session_id=self.session_id,
                 standalone_question=redact_text(standalone))
        return {"answer": NOT_FOUND_ANSWER, "sources": [], "standalone_question": standalone,
                "citations": [], "unverified_citations": [], "grounded": False, "refused": True}

    def invoke(self, question: str, chat_history=None) -> dict:
        try:
            if not question or not question.strip():
                raise DocumentPortalException("Question is empty.")
            history = to_messages(chat_history)
            standalone = question
            if history:
                standalone = strip_reasoning(self._rewrite_chain.invoke({"input": question, "chat_history": history}))

            if self.relevance_gate is not None and not self.relevance_gate(standalone):
                return self._refusal(standalone)

            docs = self.retriever.invoke(standalone)
            if not docs:
                return self._refusal(standalone)

            answer = strip_reasoning(self._answer_chain.invoke({
                "input": question,
                "chat_history": history,
                "context": format_docs(docs),
            }))
            citation_report = check_citations(answer, docs)
            log.info("RAG answer generated", session_id=self.session_id, retrieved=len(docs),
                     grounded=citation_report["grounded"], standalone_question=redact_text(standalone))
            return {"answer": answer, "sources": sources_from_docs(docs), "standalone_question": standalone,
                    **citation_report, "refused": False}
        except DocumentPortalException:
            raise
        except Exception as e:
            log.error("RAG invocation failed", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to answer the question", e) from e
