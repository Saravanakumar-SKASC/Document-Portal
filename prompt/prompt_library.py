from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------- Document Analyzer ----------
document_analysis_prompt = ChatPromptTemplate.from_template("""
You are a highly capable assistant trained to analyze and summarize documents.
Return ONLY valid JSON matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
""")

# Backwards-compatible name used by earlier versions of the analyzer
prompt = document_analysis_prompt

# ---------- Conversational RAG ----------
contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question, rewrite the question so it can be "
     "understood without the chat history. Do NOT answer it. If it is already standalone, return it unchanged."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise assistant answering questions about the user's documents. "
     "Use ONLY the context below. Cite sources inline as [file, p.N]. "
     "If the answer is not in the context, say \"I don't know based on the provided documents.\"\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ---------- Document Comparator ----------
document_comparison_prompt = ChatPromptTemplate.from_template("""
You will be given the text of two documents, a reference and an actual version, each marked page by page.
Compare them page by page and identify the differences.
- Only describe real content differences; ignore whitespace and formatting.
- For pages with no difference, use "NO CHANGE" as the Changes value.

{format_instruction}

Input documents:
{combined_docs}
""")

# ---------- Evaluation (LLM-as-judge) ----------
faithfulness_judge_prompt = ChatPromptTemplate.from_template("""
You are grading whether an answer is fully supported by the given context.
Return ONLY a JSON object: {{"score": <number from 0 to 1>, "reason": "<one sentence>"}}
1 = every claim is supported by the context, 0 = the answer is unsupported or contradicts it.

Question: {question}
Context:
{context}
Answer: {answer}
""")

PROMPT_REGISTRY = {
    "document_analysis": document_analysis_prompt,
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
    "document_comparison": document_comparison_prompt,
    "faithfulness_judge": faithfulness_judge_prompt,
}
