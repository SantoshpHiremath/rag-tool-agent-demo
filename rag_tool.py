"""
rag_tool.py
-----------
Builds a retrieval-augmented generation (RAG) pipeline over a local document
(data/forda_dataset_notes.md) and exposes it as a single callable function
(and a LangChain @tool) that the agent can invoke when it decides a question
needs to be answered by looking something up, rather than computed or
answered directly from general knowledge.

Pipeline steps:
1. Load the source document.
2. Split it into overlapping chunks (so context isn't lost at chunk boundaries).
3. Embed each chunk with a local embedding model (via Ollama).
4. Store the embeddings in a local FAISS vector index.
5. At query time: embed the question, retrieve the most similar chunks,
   and pass them to the LLM as context so it answers grounded in the
   source document instead of guessing.

This version runs fully locally via Ollama (https://ollama.com) - no API
key, no cloud calls, no cost. It uses the `nomic-embed-text` model for
embeddings and `llama3.2` for generation; both must be pulled first with
`ollama pull nomic-embed-text` and `ollama pull llama3.2`.

Note on LangChain versions: this file targets LangChain 1.x, where text
splitters live in the separate `langchain_text_splitters` package and the
old `RetrievalQA` chain class has been replaced by composing the retriever
and LLM directly with LCEL (LangChain Expression Language). This is more
explicit and, if anything, easier to explain in an interview than the old
one-line `RetrievalQA.from_chain_type` shortcut.
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool

DATA_PATH = Path(__file__).parent / "data" / "forda_dataset_notes.md"
INDEX_PATH = Path(__file__).parent / "faiss_index"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"

RAG_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the context below. If the context
doesn't contain the answer, say you don't know rather than guessing.

Context:
{context}

Question: {question}

Answer:"""
)


def build_vectorstore() -> FAISS:
    """Load the source doc, chunk it, embed it, and build (or load) a FAISS index."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if INDEX_PATH.exists():
        return FAISS.load_local(
            str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True
        )

    loader = TextLoader(str(DATA_PATH), encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(INDEX_PATH))
    return vectorstore


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


_retriever = None
_rag_chain = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        vectorstore = build_vectorstore()
        _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return _retriever


def _get_rag_chain():
    """Compose retriever + prompt + LLM into a single runnable chain (LCEL)."""
    global _rag_chain
    if _rag_chain is None:
        retriever = _get_retriever()
        llm = ChatOllama(model=CHAT_MODEL, temperature=0)
        _rag_chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )
    return _rag_chain


def query_knowledge_base(question: str) -> str:
    """Answer a question using retrieval-augmented generation over the FordA dataset notes."""
    chain = _get_rag_chain()
    answer = chain.invoke(question)

    retriever = _get_retriever()
    retrieved_docs = retriever.invoke(question)
    source_note = f"\n\n[Grounded in {len(retrieved_docs)} retrieved chunk(s) from forda_dataset_notes.md]"
    return answer + source_note


@tool
def search_notes(question: str) -> str:
    """Search the FordA dataset reference notes to answer questions about the dataset,
    its structure, preprocessing steps, or known challenges. Use this tool whenever the
    question is about dataset facts, definitions, or documented details rather than a
    calculation."""
    return query_knowledge_base(question)
