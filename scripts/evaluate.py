"""
Benchmark retrieval strategies on a labelled question set.

    python scripts/evaluate.py                    # real Gemini embeddings (needs GOOGLE_API_KEY)
    python scripts/evaluate.py --offline          # offline bag-of-words embeddings, no keys
    python scripts/evaluate.py --docs a.pdf b.pdf --eval-set my_questions.jsonl

Each line of the eval set is JSON:
    {"question": "...", "expected_source": "file.pdf", "expected_page": 2, "expected_keywords": ["..."]}

Prints hit rate@k and MRR per strategy, and the cosine similarity of every question's
best chunk, which is what you use to calibrate guardrails.min_relevance.
Run it in CI with --min-hit-rate to block releases that make retrieval worse.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.multidoc_chat.data_ingestion import MultiDocIngestor  # noqa: E402
from src.multidoc_chat.evaluation import compare_strategies  # noqa: E402
from src.singledoc_chat.evaluation import EvalExample  # noqa: E402
from utils.rag_core import top_cosine_similarity  # noqa: E402

DEFAULT_DOCS = [
    ROOT / "samples" / "demoair_baggage_policy_rev13.pdf",
    ROOT / "samples" / "demoair_task_card_ata32_brakes.pdf",
    ROOT / "samples" / "demoair_cabin_crew_sop.pdf",
]


def load_examples(path: Path) -> list[EvalExample]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            examples.append(EvalExample(row["question"], row.get("expected_source"), row.get("expected_page"),
                                        row.get("expected_keywords", [])))
    return examples


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", nargs="+", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--eval-set", type=Path, default=ROOT / "samples" / "eval_set.jsonl")
    parser.add_argument("--offline", action="store_true", help="use offline bag-of-words embeddings")
    parser.add_argument("--min-hit-rate", type=float, default=None, help="exit 1 if any strategy scores below this")
    args = parser.parse_args(argv)

    if args.offline:
        from utils.offline_models import BagOfWordsEmbeddings

        embeddings = BagOfWordsEmbeddings()
    else:
        from utils.model_loader import ModelLoader

        embeddings = ModelLoader().load_embeddings()

    examples = load_examples(args.eval_set)
    workdir = Path(tempfile.mkdtemp(prefix="docportal_eval_"))
    ingestor = MultiDocIngestor(embeddings=embeddings, data_dir=workdir / "data", faiss_dir=workdir / "faiss_index")
    store = ingestor.ingest(args.docs)
    results = compare_strategies(store, examples, embeddings=embeddings)

    print(f"\n{len(examples)} questions, {store.index.ntotal} chunks\n")
    print(f"{'strategy':<18}{'hit rate':>10}{'MRR':>8}")
    for strategy, m in results.items():
        print(f"{strategy:<18}{m['hit_rate']:>10.2f}{m['mrr']:>8.2f}")

    print("\nBest-chunk similarity per question (use this to set guardrails.min_relevance):")
    for ex in examples:
        print(f"  {top_cosine_similarity(store, ex.question):.3f}  {ex.question}")

    if args.min_hit_rate is not None and any(m["hit_rate"] < args.min_hit_rate for m in results.values()):
        print(f"\nFAIL: a strategy scored below the minimum hit rate {args.min_hit_rate}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
