"""Benchmark retrieval strategies (similarity vs MMR vs MMR + compression) on the same labelled questions."""

from src.multidoc_chat.retrieval import STRATEGIES, build_retriever
from src.singledoc_chat.evaluation import EvalExample, evaluate_rag, evaluate_retriever

__all__ = ["EvalExample", "compare_strategies", "evaluate_rag", "evaluate_retriever"]


def compare_strategies(vectorstore, examples: list[EvalExample], embeddings=None, strategies=STRATEGIES) -> dict:
    """Returns {strategy: {"hit_rate", "mrr", "n"}} so you can pick the best retriever with evidence."""
    results = {}
    for strategy in strategies:
        retriever = build_retriever(vectorstore, strategy, embeddings)
        metrics = evaluate_retriever(retriever, examples)
        results[strategy] = {k: metrics[k] for k in ("hit_rate", "mrr", "n")}
    return results
