from pathlib import Path

from exception.custom_exception import DocumentPortalException
from src.multidoc_chat.contextualcompression import build_compression_retriever
from src.multidoc_chat.mmr import build_mmr_retriever
from utils.config_loader import load_config
from utils.rag_core import ConversationalRAG, load_faiss_index

STRATEGIES = ("similarity", "mmr", "mmr+compression")


def build_retriever(vectorstore, strategy: str = "mmr", embeddings=None, k: int | None = None):
    """Create a retriever for one of: similarity | mmr | mmr+compression."""
    if strategy not in STRATEGIES:
        raise DocumentPortalException(f"Unknown strategy '{strategy}'. Options: {', '.join(STRATEGIES)}")
    if strategy == "similarity":
        top_k = k or load_config()["retriever"]["top_k"]
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    mmr = build_mmr_retriever(vectorstore, k=k)
    if strategy == "mmr":
        return mmr
    return build_compression_retriever(mmr, embeddings or vectorstore.embeddings)


class MultiDocChat(ConversationalRAG):
    """Chat across several documents with a selectable retrieval strategy."""

    def __init__(self, vectorstore, strategy: str = "mmr", llm=None, embeddings=None, session_id=None):
        self.strategy = strategy
        self.vectorstore = vectorstore
        super().__init__(retriever=build_retriever(vectorstore, strategy, embeddings), llm=llm, session_id=session_id)

    @classmethod
    def from_session(cls, session_id: str, strategy: str = "mmr", embeddings=None, llm=None,
                     faiss_dir="faiss_index"):
        if embeddings is None:
            from utils.model_loader import ModelLoader

            embeddings = ModelLoader().load_embeddings()
        vectorstore = load_faiss_index(Path(faiss_dir) / session_id, embeddings)
        return cls(vectorstore, strategy=strategy, llm=llm, embeddings=embeddings, session_id=session_id)


if __name__ == "__main__":
    # Usage: python -m src.multidoc_chat.retrieval "your question" file1.pdf file2.pdf ...
    import sys

    from src.multidoc_chat.data_ingestion import MultiDocIngestor

    if len(sys.argv) < 3:
        sys.exit('Usage: python -m src.multidoc_chat.retrieval "<question>" <file> [<file> ...]')
    ingestor = MultiDocIngestor()
    store = ingestor.ingest([Path(p) for p in sys.argv[2:]])
    chat = MultiDocChat(store, strategy="mmr+compression", embeddings=ingestor.embeddings,
                        session_id=ingestor.session_id)
    result = chat.invoke(sys.argv[1])
    print(result["answer"])
    for s in result["sources"]:
        print(f"- {s['source']} p.{s['page']}")
