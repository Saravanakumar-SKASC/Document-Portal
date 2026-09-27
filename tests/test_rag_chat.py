from src.multidoc_chat.data_ingestion import MultiDocIngestor
from src.multidoc_chat.evaluation import compare_strategies
from src.multidoc_chat.retrieval import STRATEGIES, MultiDocChat, build_retriever
from src.singledoc_chat.data_ingestion import SingleDocIngestor
from src.singledoc_chat.evaluation import EvalExample, evaluate_rag, evaluate_retriever, judge_faithfulness
from src.singledoc_chat.retrieval import SingleDocChat


def test_single_doc_chat_answers_with_sources(cloud_pdf, bow_embeddings, fake_llm):
    ingestor = SingleDocIngestor(embeddings=bow_embeddings)
    retriever = ingestor.ingest(cloud_pdf, k=2)
    chat = SingleDocChat(retriever=retriever, llm=fake_llm("Pods are the smallest units [cloud_guide.pdf, p.1]."))

    result = chat.invoke("What are pods in Kubernetes?")

    assert "Pods" in result["answer"]
    assert result["sources"][0]["page"] == 1  # the Kubernetes page is retrieved first


def test_follow_up_question_is_rewritten_using_history(cloud_pdf, bow_embeddings, fake_llm):
    retriever = SingleDocIngestor(embeddings=bow_embeddings).ingest(cloud_pdf, k=1)
    llm = fake_llm("What does Terraform state track?", "<think>hmm</think>It tracks managed resources.")
    chat = SingleDocChat(retriever=retriever, llm=llm)

    result = chat.invoke("And what does its state track?",
                         chat_history=[("user", "Tell me about Terraform"), ("assistant", "It is IaC.")])

    assert result["standalone_question"] == "What does Terraform state track?"
    assert result["answer"] == "It tracks managed resources."  # reasoning tags stripped
    assert result["sources"][0]["page"] == 2


def test_single_doc_session_reloads_from_disk(cloud_pdf, bow_embeddings, fake_llm):
    ingestor = SingleDocIngestor(embeddings=bow_embeddings)
    ingestor.ingest(cloud_pdf)
    chat = SingleDocChat.from_session(ingestor.session_id, embeddings=bow_embeddings, llm=fake_llm("ok"))
    assert chat.invoke("Prometheus metrics")["sources"]


def test_multi_doc_chat_cites_both_files(cloud_pdf, hr_pdf, bow_embeddings, fake_llm):
    store = MultiDocIngestor(embeddings=bow_embeddings).ingest([cloud_pdf, hr_pdf])
    for strategy in STRATEGIES:
        retriever = build_retriever(store, strategy, bow_embeddings, k=4)
        docs = retriever.invoke("annual leave for employees")
        assert docs and docs[0].metadata["source"] == "hr_policy.pdf", strategy

    chat = MultiDocChat(store, strategy="mmr", llm=fake_llm("25 days [hr_policy.pdf, p.1]"))
    assert "25 days" in chat.invoke("How much annual leave?")["answer"]


def test_compression_filters_off_topic_chunks(cloud_pdf, hr_pdf, bow_embeddings):
    store = MultiDocIngestor(embeddings=bow_embeddings).ingest([cloud_pdf, hr_pdf])
    plain = build_retriever(store, "mmr", k=5).invoke("remote work days per week")
    compressed = build_retriever(store, "mmr+compression", bow_embeddings, k=5).invoke("remote work days per week")
    assert 0 < len(compressed) <= len(plain)
    assert any("Remote work" in d.page_content for d in compressed)


def test_compression_falls_back_when_everything_is_filtered(cloud_pdf, bow_embeddings):
    from src.multidoc_chat.contextualcompression import build_compression_retriever
    from src.multidoc_chat.mmr import build_mmr_retriever

    store = MultiDocIngestor(embeddings=bow_embeddings).ingest([cloud_pdf])
    strict = build_compression_retriever(build_mmr_retriever(store, k=2), bow_embeddings, similarity_threshold=0.99)
    assert len(strict.invoke("Kubernetes pods")) == 2  # nothing passes 0.99, so MMR results are used


def test_retrieval_metrics_and_strategy_comparison(cloud_pdf, hr_pdf, bow_embeddings):
    store = MultiDocIngestor(embeddings=bow_embeddings).ingest([cloud_pdf, hr_pdf])
    examples = [
        EvalExample("How do pods work in Kubernetes?", expected_source="cloud_guide.pdf", expected_page=1),
        EvalExample("How many days of annual leave?", expected_source="hr_policy.pdf", expected_page=1),
        EvalExample("Grafana dashboards", expected_keywords=["grafana"]),
    ]
    metrics = evaluate_retriever(build_retriever(store, "similarity", k=3), examples)
    assert metrics["hit_rate"] == 1.0 and metrics["mrr"] > 0.5

    table = compare_strategies(store, examples, embeddings=bow_embeddings)
    assert set(table) == set(STRATEGIES)


def test_answer_evaluation_with_judge(cloud_pdf, bow_embeddings, fake_llm):
    retriever = SingleDocIngestor(embeddings=bow_embeddings).ingest(cloud_pdf, k=2)
    rag = SingleDocChat(retriever=retriever, llm=fake_llm("Terraform manages infrastructure as code."))
    judge = fake_llm('{"score": 1, "reason": "Supported by page 2."}')

    report = evaluate_rag(rag, [EvalExample("What is Terraform?", expected_keywords=["terraform", "code"])],
                          judge_llm=judge)

    assert report["keyword_coverage"] == 1.0
    assert report["faithfulness"] == 1.0
    assert judge_faithfulness(judge, "q", "a", "ctx")["score"] == 1.0
