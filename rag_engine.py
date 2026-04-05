"""
DocMind_AI — RAG Engine
Handles document loading, chunking, embedding, vector storage, and retrieval.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# ── LangChain Document Loaders ────────────────────────────────────────────────
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)

# ── LangChain Text Splitters ──────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── LangChain Embeddings & Chat Model (Ollama) ────────────────────────────────
from langchain_ollama import OllamaEmbeddings, ChatOllama

# ── LangChain Vector Store ────────────────────────────────────────────────────
from langchain_chroma import Chroma

# ── LangChain Core ────────────────────────────────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


# ── Prompt Template ───────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are DocMind AI, an expert assistant that answers questions strictly based
on the provided document context. Be precise, thorough, and cite relevant
details from the context.

If the answer cannot be found in the context, say:
"I couldn't find relevant information in the document to answer this question."

Context:
{context}

Question: {question}

Answer:""")


# ── Loader Registry ───────────────────────────────────────────────────────────

LOADERS: dict[str, tuple] = {
    ".pdf":  (PyPDFLoader,               {}),
    ".txt":  (TextLoader,                {"encoding": "utf-8"}),
    ".csv":  (CSVLoader,                 {}),
    ".docx": (Docx2txtLoader,            {}),
    ".md":   (UnstructuredMarkdownLoader, {}),
}


class RAGEngine:
    """
    Core RAG pipeline:
      load  →  split  →  embed  →  store (Chroma)  →  retrieve  →  generate
    """

    CHAT_MODELS: dict[str, str] = {
        "minimax": "minimax-m2.7:cloud",
        "gemma":   "gemma4:e4b",
    }
    # Human-readable label → Ollama model string (used by the GUI combobox)
    MODEL_DISPLAY_NAMES: dict[str, str] = {v: k.capitalize() for k, v in CHAT_MODELS.items()}

    DEFAULT_CHAT_MODEL  = CHAT_MODELS["minimax"]   # ← change key here to switch default
    EMBED_MODEL         = "qwen3-embedding:0.6b"

    def __init__(self, model_name: str | None = None,
                 embed_model: str = EMBED_MODEL,
                 persist_dir: str = "./chroma_db"):
        # Use the class-level default if none supplied; evaluated at call time
        self.model_name  = model_name if model_name is not None else self.DEFAULT_CHAT_MODEL
        self.embed_model = embed_model
        self.persist_dir = persist_dir
        self._vectorstore: Chroma | None = None
        self._chain: Any = None
        self._doc_info: dict = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def set_model(self, model_name: str) -> None:
        """Hot-swap the Ollama model without re-indexing."""
        self.model_name = model_name
        # Rebuild chain only if a document is already indexed
        if self._vectorstore is not None:
            self._build_chain(self._vectorstore, self._doc_info.get("top_k", 4))

    def is_ready(self) -> bool:
        return self._chain is not None

    def load_and_index(self, file_path: str, chunk_size: int = 800,
                       chunk_overlap: int = 150, model_name: str | None = None,
                       top_k: int = 4) -> dict:
        """
        Full ingestion pipeline for a document file.
        Returns metadata dict for the GUI.
        """
        if model_name:
            self.model_name = model_name   # chat model override from GUI

        path = Path(file_path)
        ext  = path.suffix.lower()

        if ext not in LOADERS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {list(LOADERS.keys())}")

        # 1. Load ──────────────────────────────────────────────────────────────
        loader_cls, loader_kwargs = LOADERS[ext]
        loader    = loader_cls(str(path), **loader_kwargs)
        documents = loader.load()

        if not documents:
            raise RuntimeError("The document appears to be empty or unreadable.")

        # 2. Split ─────────────────────────────────────────────────────────────
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise RuntimeError("No text chunks were produced. "
                               "Check the document content.")

        # 3. Embed + Store ─────────────────────────────────────────────────────
        embeddings = OllamaEmbeddings(model=self.embed_model)

        # Clear previous collection if re-indexing
        if self._vectorstore is not None:
            try:
                self._vectorstore.delete_collection()
            except Exception:
                pass

        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_dir,
            collection_name="docmind_collection",
        )

        # 4. Build RAG chain ───────────────────────────────────────────────────
        self._build_chain(self._vectorstore, top_k)

        # 5. Collect metadata ──────────────────────────────────────────────────
        pages = len(documents)
        self._doc_info = {
            "pages":         pages,
            "chunks":        len(chunks),
            "chunk_size":    chunk_size,
            "chunk_overlap": chunk_overlap,
            "loader":        loader_cls.__name__,
            "chat_model":    self.model_name,
            "embed_model":   self.embed_model,
            "top_k":         top_k,
        }
        return self._doc_info

    def query(self, question: str, top_k: int = 4) -> dict:
        """
        Run a RAG query. Returns {answer, sources}.
        """
        if not self.is_ready():
            raise RuntimeError("No document has been indexed yet.")

        # Rebuild retriever with requested top_k if changed
        if top_k != self._doc_info.get("top_k"):
            self._build_chain(self._vectorstore, top_k)
            self._doc_info["top_k"] = top_k

        answer = self._chain.invoke(question)

        # Fetch source metadata for attribution
        retriever = self._vectorstore.as_retriever(
            search_kwargs={"k": top_k})
        docs = retriever.invoke(question)
        sources = list({self._extract_source(d) for d in docs})

        return {"answer": answer, "sources": sources}

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _build_chain(self, vectorstore: Chroma, top_k: int) -> None:
        """Assemble the LangChain LCEL RAG chain."""
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        llm = ChatOllama(model=self.model_name, temperature=0.1)

        def format_docs(docs):
            return "\n\n".join(
                f"[Source {i+1}]\n{d.page_content}"
                for i, d in enumerate(docs)
            )

        self._chain = (
            RunnableParallel({
                "context":  retriever | format_docs,
                "question": RunnablePassthrough(),
            })
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )

    def load_existing(self, top_k: int = 4) -> dict | None:
        """
        Reload a previously persisted Chroma collection from disk.
        Returns metadata dict if found, None if no collection exists.
        """
        try:
            embeddings = OllamaEmbeddings(model=self.embed_model)
            vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=embeddings,
                collection_name="docmind_collection",
            )
            # Check if the collection actually has documents
            count = vectorstore._collection.count()
            if count == 0:
                return None

            self._vectorstore = vectorstore
            self._build_chain(self._vectorstore, top_k)

            self._doc_info = {
                "pages":         "N/A (restored)",
                "chunks":        count,
                "chunk_size":    "N/A (restored)",
                "chunk_overlap": "N/A (restored)",
                "loader":        "N/A (restored)",
                "chat_model":    self.model_name,
                "embed_model":   self.embed_model,
                "top_k":         top_k,
            }
            return self._doc_info

        except Exception:
            return None

    @staticmethod
    def _extract_source(doc) -> str:
        meta = doc.metadata or {}
        src  = meta.get("source", "")
        page = meta.get("page")
        if src:
            name = Path(src).name
            return f"{name} p.{page + 1}" if page is not None else name
        return "Document"