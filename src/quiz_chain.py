"""
quiz_chain.py — Quiz / Self-Test generation chain.
Generates MCQ or short-answer questions from course notes.
"""

import logging
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.vector_store import (
    get_parent_document_retriever, get_filtered_vector_retriever,
    ADMIN_COLLECTION, STUDENT_COLLECTION
)
from src.rag_chain import format_docs
from src.config import llm_model, llm_temperature

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _get_quiz_llm():
    import os
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return OllamaLLM(base_url=ollama_host, model=llm_model(), temperature=llm_temperature() + 0.1, streaming=True)

MCQ_PROMPT_TEMPLATE = """You are a quiz generator for university course notes. Generate exactly {num_questions} multiple-choice questions based on the following course material.

Rules:
- Each question should test understanding, not just recall.
- Provide 4 options (A, B, C, D) for each question.
- Mark the correct answer clearly.
- After all questions, provide an "Answer Key" section.

Format each question as:

**Q1.** [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]

Topic focus: {topic}

Course Material:
{context}

Generate the quiz:"""

SHORT_ANSWER_PROMPT_TEMPLATE = """You are a quiz generator for university course notes. Generate exactly {num_questions} short-answer questions based on the following course material.

Rules:
- Each question should require a brief explanation (2-3 sentences).
- Questions should test conceptual understanding.
- After all questions, provide a "Model Answers" section with concise answers.

Format each question as:

**Q1.** [Question text]

Topic focus: {topic}

Course Material:
{context}

Generate the quiz:"""


MCQ_PROMPT = ChatPromptTemplate.from_template(MCQ_PROMPT_TEMPLATE)
SHORT_ANSWER_PROMPT = ChatPromptTemplate.from_template(SHORT_ANSWER_PROMPT_TEMPLATE)


def generate_quiz(topic: str, num_questions: int, question_type: str,
                  collection_name: str, source_filter: str = None):
    """
    Generate a quiz from course notes.

    Args:
        topic: Topic to focus questions on
        num_questions: Number of questions to generate
        question_type: "mcq" or "short_answer"
        collection_name: Which collection to retrieve from
        source_filter: Optional filename to restrict to a specific document

    Returns:
        Generator stream of the quiz text
    """
    llm = _get_quiz_llm()

    if source_filter:
        retriever = get_filtered_vector_retriever(collection_name, source_filter)
    else:
        retriever = get_parent_document_retriever(collection_name)

    docs = retriever.invoke(topic)
    context_str = format_docs(docs)

    prompt = MCQ_PROMPT if question_type == "mcq" else SHORT_ANSWER_PROMPT

    chain = (
        {
            "context": lambda x: context_str,
            "topic": lambda x: topic,
            "num_questions": lambda x: str(num_questions),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.stream(topic)


def generate_quiz_dual(topic: str, num_questions: int, question_type: str,
                       source_filter: str = None):
    """
    Generate a quiz from both admin and student collections (Upgrade 8).
    Merges and deduplicates results, consistent with dual-collection chat mode.

    Args:
        topic: Topic to focus questions on
        num_questions: Number of questions to generate
        question_type: "mcq" or "short_answer"
        source_filter: Optional filename to restrict to a specific document

    Returns:
        Generator stream of the quiz text
    """
    llm = _get_quiz_llm()

    # Retrieve from both collections
    all_docs = []
    student_user = st.session_state.get("user_id", "student")
    student_coll = f"{STUDENT_COLLECTION}_{student_user}"
    for collection in [ADMIN_COLLECTION, student_coll]:
        try:
            if source_filter:
                retriever = get_filtered_vector_retriever(collection, source_filter)
            else:
                retriever = get_parent_document_retriever(collection)
            docs = retriever.invoke(topic)
            all_docs.extend(docs)
        except Exception:
            continue

    # Deduplicate by doc_id
    seen_ids = set()
    merged_docs = []
    for doc in all_docs:
        doc_id = doc.metadata.get("doc_id", id(doc))
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            merged_docs.append(doc)

    context_str = format_docs(merged_docs)

    prompt = MCQ_PROMPT if question_type == "mcq" else SHORT_ANSWER_PROMPT

    chain = (
        {
            "context": lambda x: context_str,
            "topic": lambda x: topic,
            "num_questions": lambda x: str(num_questions),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.stream(topic)
