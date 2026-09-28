"""
Document Portal web UI.

Run:  streamlit run streamlit_ui.py
"""

import os

import streamlit as st

from exception.custom_exception import DocumentPortalException
from src.doc_compare.data_ingestion import DocumentIngestion
from src.doc_compare.retrieval import DocumentComparatorLLM
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_analyzer.data_ingestion import DocumentHandler
from src.multidoc_chat.data_ingestion import MultiDocIngestor
from src.multidoc_chat.retrieval import STRATEGIES, MultiDocChat
from utils.config_loader import load_config

st.set_page_config(page_title="Document Portal", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def get_models():
    from utils.model_loader import ModelLoader

    loader = ModelLoader()
    return loader.load_llm(), loader.load_embeddings()


def show_error(e: Exception):
    st.error(e.error_message if isinstance(e, DocumentPortalException) else str(e))


# ---------- Sidebar ----------
config = load_config()
provider = os.getenv("LLM_PROVIDER", "groq")
with st.sidebar:
    st.title("📄 Document Portal")
    st.caption("LLM document intelligence with LangChain + FAISS")
    st.markdown(f"**LLM:** `{config['llm'].get(provider, {}).get('model_name', provider)}` ({provider})")
    st.markdown(f"**Embeddings:** `{config['embedding_model']['model_name']}`")

try:
    llm, embeddings = get_models()
except DocumentPortalException as e:
    st.error(f"{e.error_message}. Copy .env.example to .env and add your API keys.")
    st.stop()

analyze_tab, chat_tab, compare_tab = st.tabs(["🔎 Analyze", "💬 Chat with documents", "🆚 Compare"])

# ---------- Analyze ----------
with analyze_tab:
    st.subheader("Extract metadata and a summary from a PDF")
    pdf = st.file_uploader("Upload a PDF", type=["pdf"], key="analyze_file")
    if pdf and st.button("Analyze", type="primary"):
        with st.spinner("Reading and analyzing..."):
            try:
                handler = DocumentHandler()
                result = DocumentAnalyzer(llm=llm).analyze_document(handler.read_pdf(handler.save_pdf(pdf)))
                st.markdown(f"### {result.get('Title', 'Untitled')}")
                cols = st.columns(4)
                cols[0].metric("Pages", result.get("PageCount", "–"))
                cols[1].metric("Language", result.get("Language", "–"))
                cols[2].metric("Tone", result.get("SentimentTone", "–"))
                cols[3].metric("Author", result.get("Author", "–"))
                st.markdown("\n".join(f"- {point}" for point in result.get("Summary", [])))
                with st.expander("Raw JSON"):
                    st.json(result)
            except Exception as e:
                show_error(e)

# ---------- Chat ----------
with chat_tab:
    st.subheader("Ask questions across one or more documents")
    files = st.file_uploader("Upload PDFs or text files", type=["pdf", "txt", "md"], accept_multiple_files=True,
                             key="chat_files")
    col_strategy, col_revision = st.columns([2, 1])
    strategy = col_strategy.selectbox(
        "Retrieval strategy", STRATEGIES, index=STRATEGIES.index("hybrid"),
        help="hybrid = keywords + meaning (best for part numbers and codes); MMR diversifies results "
             "across files; compression drops redundant/off-topic chunks.")
    revision = col_revision.text_input("Document revision (optional)", placeholder="e.g. Rev 13")
    if files and st.button("Build index", type="primary"):
        with st.spinner("Chunking and embedding..."):
            try:
                ingestor = MultiDocIngestor(embeddings=embeddings)
                st.session_state.vectorstore = ingestor.ingest(files, revision=revision or None)
                st.session_state.session_id = ingestor.session_id
                st.session_state.history = []
                st.success(f"Indexed {len(files)} file(s) · {st.session_state.vectorstore.index.ntotal} chunks")
            except Exception as e:
                show_error(e)

    if "vectorstore" in st.session_state:
        for turn in st.session_state.history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
        if question := st.chat_input("Ask about your documents"):
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"), st.spinner("Thinking..."):
                try:
                    chat = MultiDocChat(st.session_state.vectorstore, strategy=strategy, llm=llm,
                                        embeddings=embeddings, session_id=st.session_state.session_id)
                    result = chat.invoke(question, chat_history=st.session_state.history)
                    st.markdown(result["answer"])
                    if result.get("refused"):
                        st.info("Not found in the uploaded documents, so no answer was generated.")
                    elif result.get("grounded"):
                        st.caption("✅ Every citation points to a retrieved passage.")
                    else:
                        st.warning("⚠️ This answer has missing or unverifiable citations. Check the sources.")
                    with st.expander(f"Sources ({len(result['sources'])})"):
                        for s in result["sources"]:
                            rev = f" · {s['revision']}" if s.get("revision") else ""
                            st.markdown(f"**{s['source']}**, p.{s['page']}{rev} — {s['snippet']}…")
                    st.session_state.history += [{"role": "user", "content": question},
                                                 {"role": "assistant", "content": result["answer"]}]
                except Exception as e:
                    show_error(e)

# ---------- Compare ----------
with compare_tab:
    st.subheader("Find page-level differences between two PDFs")
    left, right = st.columns(2)
    reference = left.file_uploader("Reference (original)", type=["pdf"], key="ref")
    actual = right.file_uploader("Actual (new version)", type=["pdf"], key="act")
    if reference and actual and st.button("Compare", type="primary"):
        with st.spinner("Comparing..."):
            try:
                ingestion = DocumentIngestion()
                ref_path, act_path = ingestion.save_uploaded_files(reference, actual)
                rows = DocumentComparatorLLM(llm=llm).compare_documents(
                    ingestion.combine_documents(ref_path, act_path))
                st.dataframe(rows, use_container_width=True, hide_index=True)
            except Exception as e:
                show_error(e)
