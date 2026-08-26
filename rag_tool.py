"""
rag_tool.py
-----------

Retrieval-augmented generation over a small local knowledge source:
loads data/forda_dataset_notes.md, splits it into chunks, embeds the
chunks locally with Ollama's nomic-embed-text model, stores them in a
local FAISS vector index, and answers questions by retrieving the most
relevant chunks and composing them with the LLM via LCEL (LangChain
Expression Language) -- the retriever, prompt, and model are chained
together explicitly rather than hidden behind a one-line helper class.

Building the vector store requires a live Ollama instance (for
embeddings) -- it is built lazily on first use, not at import time, so
this module can be imported (e.g. for tool registration checks) without
Ollama running.
"""
from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool

DATA_PATH = Path(__file__).parent / "data" / "forda_dataset_notes.md"

_vectorstore = None
_retrieval_chain = None


def _chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> list[str]:
    """Simple paragraph-aware chunking: splits on blank lines first (so a
    chunk doesn't straddle two unrelated sections), then falls back to a
    fixed-size sliding window with overlap for any paragraph longer than
    chunk_size, so no single chunk is too large for a focused retrieval
    match."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
            continue
        start = 0
        while start < len(para):
            end = start + chunk_size
            chunks.append(para[start:end])
            start = end - chunk_overlap
    return chunks


def _build_vectorstore():
    from langchain_community.vectorstores import FAISS
    from langchain_ollama import OllamaEmbeddings

    text = DATA_PATH.read_text()
    chunks = _chunk_text(text)
    embeddings = OllamaEmbeddings(model=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    return FAISS.from_texts(chunks, embeddings)


def _get_retrieval_chain():
    global _vectorstore, _retrieval_chain
    if _retrieval_chain is not None:
        return _retrieval_chain

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_ollama import ChatOllama

    if _vectorstore is None:
        _vectorstore = _build_vectorstore()
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the following context. "
        "If the context doesn't contain the answer, say so.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    llm = ChatOllama(model=os.environ.get("OLLAMA_MODEL", "llama3.2"), temperature=0)

    def _format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    _retrieval_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return _retrieval_chain, retriever


@tool
def search_notes(query: str) -> str:
    """Answer a question about the FordA dataset by retrieving relevant
    passages from a local reference-notes document and grounding the
    answer in them. Use this for any question about what FordA is, where
    it came from, its structure, or how it's typically used."""
    chain, retriever = _get_retrieval_chain()
    retrieved_docs = retriever.invoke(query)
    answer = chain.invoke(query)
    return f"{answer}\n\n[Grounded in {len(retrieved_docs)} retrieved chunk(s) from forda_dataset_notes.md]"
