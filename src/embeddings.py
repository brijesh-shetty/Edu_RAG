"""
embeddings.py — Wrapper around Ollama's embedding model.
Provides a LangChain-compatible embedding class.
"""

import logging
from langchain_ollama import OllamaEmbeddings
from src.config import embeddings_model

logger = logging.getLogger(__name__)


def get_embedding_model(model_name: str = None) -> OllamaEmbeddings:
    """
    Returns an Ollama embedding model instance compatible with LangChain.

    Args:
        model_name: Ollama model to use for embeddings.
                    Default read from config.toml (mxbai-embed-large).
    """
    import os
    name = model_name or embeddings_model()
    logger.info("Using embedding model: %s", name)
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return OllamaEmbeddings(base_url=ollama_host, model=name)
