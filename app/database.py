"""
DocuMind AI - Database Management

Handles ChromaDB vector store and SQLite metadata storage.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings

from app.models import get_embedding_model


class DatabaseManager:
    """Manages ChromaDB vector store and SQLite metadata."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # SQLite setup
        self.db_path = self.data_dir / "documind.db"
        self._init_sqlite()

        # ChromaDB setup
        self.chroma_path = str(self.data_dir / "chroma_db")
        self._init_chroma()

    def _init_sqlite(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0
                )
            """)

            # Quiz results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quiz_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER,
                    score REAL,
                    total_questions INTEGER,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(id)
                )
            """)

            conn.commit()

    def _init_chroma(self):
        """Initialize ChromaDB client and collection."""
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    def add_document(self, filename: str, filepath: str, chunks: List[str]) -> int:
        """Add a document and its chunks to the database."""
        # Get embedding model
        embedding_model = get_embedding_model()

        # Add to SQLite
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO documents (filename, filepath, created_at, chunk_count) VALUES (?, ?, ?, ?)",
                (filename, filepath, datetime.now().isoformat(), len(chunks))
            )
            doc_id = cursor.lastrowid
            conn.commit()

        # Add chunks to ChromaDB
        embeddings = embedding_model.embed_documents(chunks)
        ids = [f"doc_{doc_id}_chunk_{i}" for i in range(len(chunks))]

        metadatas = [
            {
                "doc_id": doc_id,
                "chunk_index": i,
                "source_file": filename
            }
            for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return doc_id

    def get_relevant_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, dict]]:
        """Retrieve relevant document chunks for a query."""
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"]
        )

        chunks_with_metadata = []
        if results["documents"] and len(results["documents"]) > 0:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                chunks_with_metadata.append((doc, metadata))

        return chunks_with_metadata

    def get_all_documents(self) -> List[Tuple[int, str, str, str, int]]:
        """Get all documents from SQLite database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, filename, filepath, created_at, chunk_count FROM documents")
            return cursor.fetchall()

    def get_document_chunks(self, doc_id: int) -> List[str]:
        """Get all chunks for a specific document."""
        results = self.collection.get(
            where={"doc_id": doc_id},
            include=["documents"]
        )
        return results.get("documents", [])

    def delete_document(self, doc_id: int):
        """Delete a document and its chunks."""
        # Delete from SQLite
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM quiz_results WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()

        # Delete from ChromaDB
        self.collection.delete(where={"doc_id": doc_id})

    def clear_all_data(self):
        """Clear all documents and chunks from both databases."""
        # Clear SQLite
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM quiz_results")
            cursor.execute("DELETE FROM documents")
            conn.commit()

        # Clear ChromaDB
        self.chroma_client.delete_collection("document_chunks")
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    def save_quiz_result(self, doc_id: int, score: float, total_questions: int):
        """Save a quiz result to the database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO quiz_results (doc_id, score, total_questions, timestamp) VALUES (?, ?, ?, ?)",
                (doc_id, score, total_questions, datetime.now().isoformat())
            )
            conn.commit()

    def get_quiz_history(self, doc_id: Optional[int] = None) -> List[Tuple]:
        """Get quiz history, optionally filtered by document."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            if doc_id:
                cursor.execute(
                    "SELECT id, doc_id, score, total_questions, timestamp FROM quiz_results WHERE doc_id = ? ORDER BY timestamp DESC",
                    (doc_id,)
                )
            else:
                cursor.execute(
                    "SELECT id, doc_id, score, total_questions, timestamp FROM quiz_results ORDER BY timestamp DESC"
                )
            return cursor.fetchall()

    def get_document_count(self) -> int:
        """Get the number of documents in the database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            return cursor.fetchone()[0]


# Global database instance
_db = None


def get_database() -> DatabaseManager:
    """Get or create the global database instance."""
    global _db
    if _db is None:
        _db = DatabaseManager()
    return _db
