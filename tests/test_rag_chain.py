"""Unit tests for RAG chain helpers — no LLM calls required."""
import pytest
import streamlit as st
from src.rag_chain import format_docs, verify_answer_against_context, detect_comparison_intent
from langchain_core.documents import Document

# Mock streamlit session state if needed for testing
if "user_id" not in st.session_state:
    st.session_state["user_id"] = "test_user"


def make_doc(content, source="test.pdf", page=1):
    return Document(page_content=content, metadata={"source": source, "page": page})


def test_format_docs_numbered():
    docs = [
        make_doc("Photosynthesis converts light to energy."),
        make_doc("Chlorophyll absorbs red and blue light.", source="bio.pdf", page=3)
    ]
    result = format_docs(docs)
    assert "[1]" in result
    assert "[2]" in result
    assert "bio.pdf" in result


def test_format_docs_empty():
    assert format_docs([]) == ""


def test_verify_grounded():
    context = "the mitochondria is the powerhouse of the cell"
    answer = "The mitochondria is the powerhouse of the cell [1]."
    result = verify_answer_against_context(answer, context)
    assert result["verdict"] in ("GROUNDED", "PARTIAL")


def test_verify_unrelated():
    context = "photosynthesis occurs in chloroplasts"
    answer = "The moon orbits the earth every 27 days."
    result = verify_answer_against_context(answer, context)
    assert result["verdict"] in ("PARTIAL", "UNCERTAIN")


def test_detect_comparison_intent_positive():
    sources = ["chapter1.pdf", "chapter2.pdf"]
    is_cmp, s1, s2 = detect_comparison_intent(
        "Compare chapter1 and chapter2 on energy storage", sources)
    assert is_cmp is True
    assert "chapter1.pdf" in (s1, s2)
    assert "chapter2.pdf" in (s1, s2)


def test_detect_comparison_intent_negative():
    sources = ["notes.pdf"]
    is_cmp, _, _ = detect_comparison_intent("What is photosynthesis?", sources)
    assert is_cmp is False
