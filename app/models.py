"""
DocuMind AI - Ollama Model Initialization

This module initializes the Ollama chat and embedding models
using LangChain's Ollama integration.
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Model configuration
CHAT_MODEL = "minimax-m2.7:cloud"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
TEMPERATURE = 0.7
TIMEOUT = 120


class OllamaModels:
    """Manages Ollama chat and embedding models."""

    def __init__(self):
        self._chat_model = None
        self._embedding_model = None

    def get_chat_model(self) -> ChatOllama:
        """Get or create the chat model instance."""
        if self._chat_model is None:
            self._chat_model = ChatOllama(
                model=CHAT_MODEL,
                temperature=TEMPERATURE,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                timeout=TIMEOUT,
            )
        return self._chat_model

    def get_embedding_model(self) -> OllamaEmbeddings:
        """Get or create the embedding model instance."""
        if self._embedding_model is None:
            self._embedding_model = OllamaEmbeddings(
                model=EMBEDDING_MODEL,
            )
        return self._embedding_model

    def test_connection(self) -> bool:
        """Test if Ollama is running and models are available."""
        try:
            self.get_embedding_model().embed_query("test")
            self.get_chat_model().invoke("test")
            return True
        except Exception as e:
            print(f"Ollama connection test failed: {e}")
            return False


# Global model instance
_models = OllamaModels()


def get_chat_model() -> ChatOllama:
    """Convenience function to get the chat model."""
    return _models.get_chat_model()


def get_embedding_model() -> OllamaEmbeddings:
    """Convenience function to get the embedding model."""
    return _models.get_embedding_model()


def test_ollama_connection() -> bool:
    """Test if Ollama is accessible."""
    return _models.test_connection()
