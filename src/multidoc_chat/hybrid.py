"""
Hybrid retrieval: BM25 keyword search + vector search, merged with Reciprocal Rank Fusion.

Why: embeddings capture meaning ("landing gear won't retract" ~ "gear retraction fault")
but blur exact identifiers. Airline documents are full of them - part numbers
(2315-0045-001), ATA chapters (32-41-00), MEL items, flight numbers (CX888).
BM25 matches those tokens exactly; RRF combines both ranked lists without having
to compare their incompatible scores.
"""

import re

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from utils.config_loader import load_config
from utils.rag_core import all_chunks

# Keep identifiers like 2315-0045-001, 32-41-00, CX888 and A350-900 as single tokens
_TOKEN = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    # Also index the parts of compound identifiers so "32-41" still matches "32-41-00"
    parts = [p for t in tokens if "-" in t for p in t.split("-")]
    return tokens + parts


def build_bm25_retriever(vectorstore, k: int):
    retriever = BM25Retriever.from_documents(all_chunks(vectorstore), preprocess_func=tokenize)
    retriever.k = k
    return retriever


def build_hybrid_retriever(vectorstore, k: int | None = None, bm25_weight: float | None = None,
                           vector_weight: float | None = None):
    cfg = load_config()["retriever"]
    hybrid_cfg = cfg.get("hybrid", {})
    k = k or cfg["top_k"]
    bm25 = build_bm25_retriever(vectorstore, k)
    vector = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    weights = [
        bm25_weight if bm25_weight is not None else hybrid_cfg.get("bm25_weight", 0.5),
        vector_weight if vector_weight is not None else hybrid_cfg.get("vector_weight", 0.5),
    ]
    fused = EnsembleRetriever(retrievers=[bm25, vector], weights=weights, id_key="chunk_id")
    return TopKRetriever(base=fused, k=k)


class TopKRetriever(BaseRetriever):
    """RRF returns the union of both lists (up to 2k); keep only the best k."""

    base: BaseRetriever
    k: int

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.base.invoke(query)[: self.k]
