"""
config.py — Centralized configuration loader.
Reads config.toml and exposes typed accessors with sane defaults.
"""

import os

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.toml")

def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}

_config = _load_config()


# --- Models ---

def llm_model() -> str:
    return _config.get("models", {}).get("llm", "llama3.1:8b")

def embeddings_model() -> str:
    return _config.get("models", {}).get("embeddings", "mxbai-embed-large")

def vision_model() -> str:
    return _config.get("models", {}).get("vision", "llava:7b")

def llm_temperature() -> float:
    return _config.get("models", {}).get("llm_temperature", 0.1)


# --- Retrieval ---

def child_chunk_size() -> int:
    return _config.get("retrieval", {}).get("child_chunk_size", 400)

def child_chunk_overlap() -> int:
    return _config.get("retrieval", {}).get("child_chunk_overlap", 100)

def retriever_k() -> int:
    return _config.get("retrieval", {}).get("retriever_k", 5)

def bm25_weight() -> float:
    return _config.get("retrieval", {}).get("bm25_weight", 0.5)

def vector_weight() -> float:
    return _config.get("retrieval", {}).get("vector_weight", 0.5)

def reranker_top_n() -> int:
    return _config.get("retrieval", {}).get("reranker_top_n", 5)


# --- Memory ---

def memory_max_token_limit() -> int:
    return _config.get("memory", {}).get("max_token_limit", 800)

def memory_enabled() -> bool:
    return _config.get("memory", {}).get("enabled", True)


# --- Verification ---

def verification_enabled() -> bool:
    return _config.get("verification", {}).get("enabled", True)

def coverage_threshold() -> float:
    return _config.get("verification", {}).get("coverage_threshold", 0.8)


# --- Quiz ---

def quiz_default_num_questions() -> int:
    return _config.get("quiz", {}).get("default_num_questions", 3)

def quiz_default_type() -> str:
    return _config.get("quiz", {}).get("default_type", "mcq")


# --- Tesseract ---

def tesseract_path() -> str:
    import sys
    import shutil
    raw = _config.get("tesseract", {}).get("path", "")
    if raw:
        return raw
    # Auto-detect: on Windows try default install location
    if sys.platform == "win32":
        default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_path):
            return default_path
    # On Unix / other platforms, check PATH
    found = shutil.which("tesseract")
    return found or "tesseract"
