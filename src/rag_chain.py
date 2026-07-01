"""
rag_chain.py — RAG pipeline using LangChain Expression Language (LCEL).
Ties retrieval (ChromaDB) + generation (Ollama) together.
Supports single-collection and dual-collection retrieval, re-ranking, and citations.
"""

import os
import re
import pickle
import hashlib
import logging
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from src.vector_store import (
    get_vector_store, get_all_documents,
    get_parent_document_retriever, get_filtered_vector_retriever,
    ADMIN_COLLECTION, STUDENT_COLLECTION
)
from src.config import (
    llm_model, llm_temperature, bm25_weight, vector_weight,
    reranker_top_n, verification_enabled, coverage_threshold
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _get_llm():
    """Singleton LLM — shared across all requests in this process."""
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return OllamaLLM(base_url=ollama_host, model=llm_model(), temperature=llm_temperature(), streaming=True)


BM25_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "bm25_cache")

def _bm25_cache_path(collection_name: str, content_hash: str) -> str:
    os.makedirs(BM25_CACHE_DIR, exist_ok=True)
    return os.path.join(BM25_CACHE_DIR, f"{collection_name}_{content_hash}.pkl")


def clear_bm25_cache(collection_name: str = None):
    """Delete BM25 cached .pkl files for a collection, or all collections if None."""
    import shutil
    if os.path.exists(BM25_CACHE_DIR):
        if collection_name:
            for fname in os.listdir(BM25_CACHE_DIR):
                if fname.startswith(f"{collection_name}_"):
                    try:
                        os.remove(os.path.join(BM25_CACHE_DIR, fname))
                    except Exception as e:
                        logger.error(f"Error removing cached BM25 file {fname}: {e}")
        else:
            try:
                shutil.rmtree(BM25_CACHE_DIR)
            except Exception as e:
                logger.error(f"Error removing BM25 cache dir: {e}")


@st.cache_resource(show_spinner=False)
def get_bm25_retriever(_cache_key):
    """
    Cache the BM25 retriever. Pass (collection_name, content_hash) as cache key.
    Dumps/loads from disk pkl cache if app restarts.
    """
    if isinstance(_cache_key, tuple):
        collection_name, content_hash = _cache_key
    else:
        collection_name, content_hash = ADMIN_COLLECTION, "default"

    cache_path = _bm25_cache_path(collection_name, content_hash)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading BM25 cache from disk: {e}")

    docs = get_all_documents(collection_name)
    if not docs:
        return None
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 10  # Fetch more candidates for re-ranking

    try:
        with open(cache_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
    except Exception as e:
        logger.error(f"Error saving BM25 cache to disk: {e}")

    return bm25_retriever


def _compute_content_hash(collection_name: str) -> str:
    """Compute a hash based on document IDs for cache invalidation."""
    docs = get_all_documents(collection_name)
    doc_ids = sorted([d.metadata.get("doc_id", str(id(d))) for d in docs])
    return hashlib.md5("|".join(doc_ids).encode()).hexdigest()


# Prompt template with numbered source citations
RAG_PROMPT_TEMPLATE = """You are a helpful teaching assistant. Answer the student's question
using ONLY the following context from their course notes.

Rules:
- Answer accurately based on the context provided.
- After each factual claim, place a citation marker [N] matching the source number from the context.
- If the answer is not in the context, say "I couldn't find this information in your uploaded notes."
- Keep your answer clear, well-structured, and student-friendly.
- IMPORTANT FOR CODE: If the context contains programming code (like Python, Java, R, SQL, etc.), you MUST re-format it cleanly in your response. Wrap any multiline code blocks in standard Markdown backticks (```python) and preserve all indentation and spacing exactly as it appears in the notes!
- If the student provides numerical values for a math or physics problem:
  1. Extract the relevant formula(s) from the notes.
  2. Clearly state the formula.
  3. Substitute the given values step-by-step.
  4. Perform the computation carefully and state the final result clearly. Double-check your arithmetic!

Context from course notes:
{context}

Chat History:
{chat_history}

Student's Question: {question}

Helpful Answer:"""

RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


COMPARE_PROMPT_TEMPLATE = """You are a helpful teaching assistant performing a comparative analysis.
Compare the following information from two different sources.

Source 1 ({source1_name}):
{source1_context}

Source 2 ({source2_name}):
{source2_context}

Chat History:
{chat_history}

Student's Question: {question}

Structure your response as:
## What {source1_name} Says
<summary of source 1>

## What {source2_name} Says
<summary of source 2>

## Similarities
<common points>

## Differences
<key differences>

Helpful Answer:"""

COMPARE_PROMPT = ChatPromptTemplate.from_template(COMPARE_PROMPT_TEMPLATE)


def format_docs(docs):
    """Format retrieved documents into a numbered context string with citation markers."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")

        # Strip contextual chunk header if present (metadata already carries source/page)
        content = doc.page_content
        if doc.metadata.get("has_context_header"):
            # Remove the header line "[Document: ... | Page: ...]\n\n"
            content = re.sub(r'^\[Document:.*?\| Page:.*?\]\n\n', '', content)

        formatted.append(f"[{i}] [Source: {source}, Page/Slide: {page}]\n{content}")
    return "\n\n---\n\n".join(formatted)


def _build_ensemble_retriever(collection_name: str):
    """Build an EnsembleRetriever (vector + BM25) for a given collection, with optional re-ranking."""
    vector_retriever = get_parent_document_retriever(collection_name)

    content_hash = _compute_content_hash(collection_name)
    cache_key = (collection_name, content_hash)
    bm25_retriever = get_bm25_retriever(cache_key)

    if bm25_retriever:
        base_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[vector_weight(), bm25_weight()]
        )
    else:
        base_retriever = vector_retriever

    # Try to apply cross-encoder re-ranking
    try:
        from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
        try:
            from langchain.retrievers import ContextualCompressionRetriever
        except ImportError:
            from langchain_classic.retrievers import ContextualCompressionRetriever

        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=reranker_top_n())
        reranked_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        logger.info("Using FlashRank re-ranking for collection '%s'", collection_name)
        return reranked_retriever
    except ImportError:
        logger.info("FlashRank not installed, skipping re-ranking for '%s'", collection_name)
        return base_retriever


def _jaccard_verify(answer: str, context: str) -> dict:
    """Fallback Jaccard word overlap check."""
    import string

    def tokenize(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return set(text.split())

    # Split answer into sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    sentences = [s for s in sentences if len(s.split()) > 3]  # Skip very short fragments

    if not sentences:
        return {"coverage_score": 1.0, "verdict": "GROUNDED", "unverified_sentences": []}

    context_tokens = tokenize(context)
    verified = 0
    unverified = []

    for sentence in sentences:
        # Skip citation markers and meta-text
        clean = re.sub(r'\[\d+\]', '', sentence)
        if clean.lower().startswith("i couldn't find"):
            verified += 1
            continue

        sent_tokens = tokenize(clean)
        if not sent_tokens:
            verified += 1
            continue

        overlap = len(sent_tokens & context_tokens)
        jaccard = overlap / len(sent_tokens) if sent_tokens else 0

        if jaccard >= 0.3:
            verified += 1
        else:
            unverified.append(sentence)

    score = verified / len(sentences) if sentences else 1.0
    threshold = coverage_threshold()

    if score >= threshold:
        verdict = "GROUNDED"
    elif score >= 0.5:
        verdict = "PARTIAL"
    else:
        verdict = "UNCERTAIN"

    return {
        "coverage_score": score,
        "verdict": verdict,
        "unverified_sentences": unverified,
    }


@st.cache_resource(show_spinner=False)
def _get_sentence_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")   # ~80 MB, fast CPU model


def verify_answer_against_context(answer: str, context: str) -> dict:
    """
    Sentence-level verification utilizing semantic similarity (SentenceTransformers).
    Falls back to lexical Jaccard check if sentence-transformers is not available.
    """
    try:
        from sentence_transformers import util
        model = _get_sentence_model()
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        sentences = [s for s in sentences if len(s.split()) > 3]
        if not sentences:
            return {"coverage_score": 1.0, "verdict": "GROUNDED", "unverified_sentences": []}

        # Encode sentences and context
        sentence_embs = model.encode(sentences, convert_to_tensor=True)
        context_emb = model.encode([context], convert_to_tensor=True)
        
        # Calculate cosine similarity
        scores = util.cos_sim(sentence_embs, context_emb).squeeze(1).tolist()
        if not isinstance(scores, list):
            scores = [scores]

        threshold = coverage_threshold()
        # Threshold 0.4 indicates semantic alignment
        unverified = [s for s, sc in zip(sentences, scores) if sc < 0.4]
        grounded = sum(1 for sc in scores if sc >= 0.4)
        score = grounded / len(sentences)

        verdict = "GROUNDED" if score >= threshold else ("PARTIAL" if score >= 0.5 else "UNCERTAIN")
        return {"coverage_score": score, "verdict": verdict, "unverified_sentences": unverified}
    except Exception as e:
        logger.warning(f"Error in semantic verification, falling back to Jaccard: {e}")
        return _jaccard_verify(answer, context)


def stream_rag_answer(question: str, collection_name: str = ADMIN_COLLECTION,
                      chat_history: str = "", extra_context: str = ""):
    """
    Ask a question and stream an answer grounded in uploaded course notes.
    Single-collection retrieval (used by admin).

    Args:
        question: The user's question
        collection_name: Which collection to retrieve from
        chat_history: Formatted conversation history
        extra_context: Optional extra context to prepend (e.g. formula docs)

    Returns:
        Tuple of (generator_stream, sources, retrieved_docs)
    """
    llm = _get_llm()

    retriever = _build_ensemble_retriever(collection_name)
    retrieved_docs = retriever.invoke(question)
    context_str = extra_context + format_docs(retrieved_docs)

    chain = (
        {
            "context": lambda x: context_str,
            "chat_history": lambda x: chat_history,
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    sources = []
    seen = set()
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}_p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": source,
                "page": page,
                "text": doc.page_content
            })

    return chain.stream(question), sources, retrieved_docs


def stream_rag_answer_dual(question: str, chat_history: str = "", extra_context: str = ""):
    """
    Dual-collection retrieval (used by student).
    Queries BOTH admin and student collections, merges results by doc_id dedup.

    Returns:
        Tuple of (generator_stream, sources, retrieved_docs)
    """
    llm = _get_llm()

    # Retrieve from both collections
    student_user = st.session_state.get("user_id", "student")
    student_coll = f"{STUDENT_COLLECTION}_{student_user}"
    admin_retriever = _build_ensemble_retriever(ADMIN_COLLECTION)
    student_retriever = _build_ensemble_retriever(student_coll)

    admin_docs = admin_retriever.invoke(question)
    student_docs = student_retriever.invoke(question)

    # Merge and deduplicate by doc_id
    seen_ids = set()
    merged_docs = []
    for doc in admin_docs + student_docs:
        doc_id = doc.metadata.get("doc_id", id(doc))
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            merged_docs.append(doc)

    context_str = extra_context + format_docs(merged_docs)

    chain = (
        {
            "context": lambda x: context_str,
            "chat_history": lambda x: chat_history,
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    sources = []
    seen = set()
    for doc in merged_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}_p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": source,
                "page": page,
                "text": doc.page_content
            })

    return chain.stream(question), sources, merged_docs


def stream_comparative_answer(question: str, source1: str, source2: str,
                              collection_name: str = ADMIN_COLLECTION, chat_history: str = ""):
    """
    Comparative analysis: retrieves from two specific sources and generates structured comparison.

    Returns:
        Tuple of (generator_stream, sources, retrieved_docs)
    """
    llm = _get_llm()

    ret1 = get_filtered_vector_retriever(collection_name, source1)
    ret2 = get_filtered_vector_retriever(collection_name, source2)

    docs1 = ret1.invoke(question)
    docs2 = ret2.invoke(question)

    context1 = format_docs(docs1)
    context2 = format_docs(docs2)

    chain = (
        {
            "source1_name": lambda x: source1,
            "source2_name": lambda x: source2,
            "source1_context": lambda x: context1,
            "source2_context": lambda x: context2,
            "chat_history": lambda x: chat_history,
            "question": RunnablePassthrough()
        }
        | COMPARE_PROMPT
        | llm
        | StrOutputParser()
    )

    all_docs = docs1 + docs2
    sources = []
    seen = set()
    for doc in all_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}_p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({"source": source, "page": page, "text": doc.page_content})

    return chain.stream(question), sources, all_docs


def detect_comparison_intent(question: str, available_sources: list[str]) -> tuple:
    """
    Detect if a question is asking for comparison between two sources.
    Returns (is_comparison, source1, source2) or (False, None, None).
    """
    comparison_keywords = ["compare", "vs", "versus", "difference between", "differences between",
                           "contrast", "similar", "similarities between"]

    question_lower = question.lower()
    is_comparison = any(kw in question_lower for kw in comparison_keywords)

    if not is_comparison:
        return False, None, None

    matched = []
    for src in available_sources:
        name_lower = src.lower().replace(".pdf", "").replace(".docx", "").replace(".pptx", "")
        # Check if any significant part of the filename appears in the question
        parts = re.split(r'[_\s\-]+', name_lower)
        for part in parts:
            if len(part) > 2 and part in question_lower:
                matched.append(src)
                break

    if len(matched) >= 2:
        return True, matched[0], matched[1]

    return False, None, None
