"""
Lightweight RAG evaluation.

- Retrieval metrics (no LLM needed): hit rate@k and MRR against a small labelled set.
- Answer metrics: keyword coverage and optional LLM-as-judge faithfulness (0-1).
"""

from dataclasses import dataclass, field

from langchain_core.output_parsers import JsonOutputParser

from prompt.prompt_library import faithfulness_judge_prompt
from utils.document_ops import format_docs


@dataclass
class EvalExample:
    question: str
    expected_source: str | None = None
    expected_page: int | None = None
    expected_keywords: list[str] = field(default_factory=list)


def is_relevant(doc, example: EvalExample) -> bool:
    """A chunk is relevant if it matches the expected source/page, or (if none given) contains all keywords."""
    if example.expected_source or example.expected_page is not None:
        if example.expected_source and doc.metadata.get("source") != example.expected_source:
            return False
        if example.expected_page is not None and doc.metadata.get("page") != example.expected_page:
            return False
        return True
    text = doc.page_content.lower()
    return bool(example.expected_keywords) and all(k.lower() in text for k in example.expected_keywords)


def evaluate_retriever(retriever, examples: list[EvalExample]) -> dict:
    hits, reciprocal_ranks, rows = 0, [], []
    for ex in examples:
        docs = retriever.invoke(ex.question)
        rank = next((i for i, d in enumerate(docs, start=1) if is_relevant(d, ex)), None)
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        rows.append({"question": ex.question, "rank": rank, "retrieved": len(docs)})
    n = len(examples) or 1
    return {"hit_rate": hits / n, "mrr": sum(reciprocal_ranks) / n, "n": len(examples), "rows": rows}


def keyword_coverage(answer: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    answer_l = answer.lower()
    return sum(k.lower() in answer_l for k in keywords) / len(keywords)


def judge_faithfulness(llm, question: str, answer: str, contexts) -> dict:
    """LLM-as-judge: is every claim in the answer supported by the retrieved context?"""
    chain = faithfulness_judge_prompt | llm | JsonOutputParser()
    context = contexts if isinstance(contexts, str) else format_docs(contexts)
    result = chain.invoke({"question": question, "answer": answer, "context": context})
    return {"score": float(result.get("score", 0)), "reason": result.get("reason", "")}


def evaluate_rag(rag, examples: list[EvalExample], judge_llm=None) -> dict:
    """Run a ConversationalRAG over the examples and score its answers."""
    rows = []
    for ex in examples:
        out = rag.invoke(ex.question)
        row = {"question": ex.question, "answer": out["answer"],
               "keyword_coverage": keyword_coverage(out["answer"], ex.expected_keywords)}
        if judge_llm is not None:
            contexts = "\n\n".join(s["snippet"] for s in out["sources"])
            row["faithfulness"] = judge_faithfulness(judge_llm, ex.question, out["answer"], contexts)["score"]
        rows.append(row)

    def mean(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {"keyword_coverage": mean("keyword_coverage"), "faithfulness": mean("faithfulness"),
            "n": len(rows), "rows": rows}
