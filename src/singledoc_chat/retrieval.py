from pathlib import Path

from utils.config_loader import load_config
from utils.rag_core import ConversationalRAG, load_faiss_index, relevance_gate_for


class SingleDocChat(ConversationalRAG):
    """Chat with one document. Build from a retriever, or reload a saved session index."""

    @classmethod
    def from_session(cls, session_id: str, embeddings=None, llm=None, faiss_dir="faiss_index", k: int | None = None):
        if embeddings is None:
            from utils.model_loader import ModelLoader

            embeddings = ModelLoader().load_embeddings()
        vectorstore = load_faiss_index(Path(faiss_dir) / session_id, embeddings)
        top_k = k or load_config()["retriever"]["top_k"]
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        return cls(retriever=retriever, llm=llm, session_id=session_id,
                   relevance_gate=relevance_gate_for(vectorstore))


if __name__ == "__main__":
    # Usage: python -m src.singledoc_chat.retrieval path/to/file.pdf "your question"
    import sys

    from src.singledoc_chat.data_ingestion import SingleDocIngestor

    if len(sys.argv) != 3:
        sys.exit('Usage: python -m src.singledoc_chat.retrieval <file> "<question>"')
    ingestor = SingleDocIngestor()
    chat = SingleDocChat(retriever=ingestor.ingest(Path(sys.argv[1])), session_id=ingestor.session_id)
    result = chat.invoke(sys.argv[2])
    print(result["answer"])
    for s in result["sources"]:
        print(f"- {s['source']} p.{s['page']}")
