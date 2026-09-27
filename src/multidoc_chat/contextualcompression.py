"""Contextual compression: drop redundant and off-topic chunks before they reach the LLM."""

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter

from utils.config_loader import load_config


def build_compression_retriever(base_retriever, embeddings, similarity_threshold: float | None = None):
    """
    Wrap a retriever with a two-stage filter pipeline:
      1. EmbeddingsRedundantFilter - removes near-duplicate chunks
      2. EmbeddingsFilter          - removes chunks below a query-similarity threshold
    Fewer, cleaner chunks means a shorter prompt, lower cost and fewer distractions for the LLM.
    """
    threshold = similarity_threshold
    if threshold is None:
        threshold = load_config()["retriever"].get("compression", {}).get("similarity_threshold", 0.35)
    pipeline = DocumentCompressorPipeline(transformers=[
        EmbeddingsRedundantFilter(embeddings=embeddings),
        EmbeddingsFilter(embeddings=embeddings, similarity_threshold=threshold),
    ])
    return ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=base_retriever)
