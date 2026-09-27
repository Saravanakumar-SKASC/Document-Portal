"""Contextual compression: drop redundant and off-topic chunks before they reach the LLM."""

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from utils.config_loader import load_config


class FallbackRetriever(BaseRetriever):
    """Use ``primary``; if it filters everything out, fall back to ``fallback`` so the LLM never gets empty context."""

    primary: BaseRetriever
    fallback: BaseRetriever

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        docs = self.primary.invoke(query)
        return docs if docs else self.fallback.invoke(query)


def build_compression_retriever(base_retriever, embeddings, similarity_threshold: float | None = None):
    """
    Wrap a retriever with a two-stage filter pipeline:
      1. EmbeddingsRedundantFilter - removes near-duplicate chunks
      2. EmbeddingsFilter          - removes chunks below a query-similarity threshold
    Fewer, cleaner chunks means a shorter prompt, lower cost and fewer distractions for the LLM.
    If the threshold removes every chunk, the uncompressed results are used instead.
    """
    threshold = similarity_threshold
    if threshold is None:
        threshold = load_config()["retriever"].get("compression", {}).get("similarity_threshold", 0.35)
    pipeline = DocumentCompressorPipeline(transformers=[
        EmbeddingsRedundantFilter(embeddings=embeddings),
        EmbeddingsFilter(embeddings=embeddings, similarity_threshold=threshold),
    ])
    compressed = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=base_retriever)
    return FallbackRetriever(primary=compressed, fallback=base_retriever)
