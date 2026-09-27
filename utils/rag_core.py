"""FAISS index helpers and the conversational RAG core shared by single- and multi-document chat."""

import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from prompt.prompt_library import context_qa_prompt, contextualize_question_prompt
from utils.document_ops import format_docs

log = CustomLogger().get_logger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


# ---------- Vector store ----------
def build_faiss_index(chunks: list[Document], embeddings, save_dir: str | Path | None = None) -> FAISS:
    if not chunks:
        raise DocumentPortalException("No chunks to index.")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(save_dir))
        log.info("FAISS index saved", path=str(save_dir), chunks=len(chunks))
    return vectorstore


def load_faiss_index(index_dir: str | Path, embeddings) -> FAISS:
    index_dir = Path(index_dir)
    if not (index_dir / "index.faiss").exists():
        raise DocumentPortalException(f"No FAISS index found at {index_dir}. Ingest documents first.")
    # The index files are created by this app only; never load indexes from untrusted sources.
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)


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
    return [
        {
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page"),
            "snippet": d.page_content[:snippet_chars],
        }
        for d in docs
    ]


class ConversationalRAG:
    """
    History-aware RAG: rewrite the follow-up question into a standalone one,
    retrieve chunks, then answer strictly from that context with [file, p.N] citations.
    """

    def __init__(self, retriever, llm=None, session_id: str | None = None):
        if llm is None:
            from utils.model_loader import ModelLoader

            llm = ModelLoader().load_llm()
        self.llm = llm
        self.retriever = retriever
        self.session_id = session_id
        self._rewrite_chain = contextualize_question_prompt | self.llm | StrOutputParser()
        self._answer_chain = context_qa_prompt | self.llm | StrOutputParser()

    def invoke(self, question: str, chat_history=None) -> dict:
        try:
            if not question or not question.strip():
                raise DocumentPortalException("Question is empty.")
            history = to_messages(chat_history)
            standalone = question
            if history:
                standalone = strip_reasoning(self._rewrite_chain.invoke({"input": question, "chat_history": history}))
            docs = self.retriever.invoke(standalone)
            answer = self._answer_chain.invoke({
                "input": question,
                "chat_history": history,
                "context": format_docs(docs) if docs else "(no relevant context found)",
            })
            log.info("RAG answer generated", session_id=self.session_id, retrieved=len(docs),
                     standalone_question=standalone)
            return {"answer": strip_reasoning(answer), "sources": sources_from_docs(docs),
                    "standalone_question": standalone}
        except DocumentPortalException:
            raise
        except Exception as e:
            log.error("RAG invocation failed", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to answer the question", e) from e
