"""
ui_components.py — Shared UI helpers for Admin & Student pages.
Eliminates ~70% code duplication between the two pages.
"""

import os
import re
import streamlit as st
from src.file_parser import parse_file, parse_file_streaming
from src.vector_store import (
    add_documents_to_store, get_all_documents, FORMULA_COLLECTION
)
from src.rag_chain import verify_answer_against_context, format_docs
from src.ingestion_tracker import (
    is_already_processed, mark_as_processed, get_all_processed_filenames
)
from src.formula_extractor import extract_formulas, is_formula_query
from src.quiz_chain import generate_quiz
from streamlit_pdf_viewer import pdf_viewer
from src.pdf_utils import get_pdf_annotations
from src.config import (
    llm_model, vision_model, embeddings_model,
    verification_enabled, memory_enabled, memory_max_token_limit,
    quiz_default_num_questions, quiz_default_type
)


# ──────────────────────────────────────────────
#  Citation rendering
# ──────────────────────────────────────────────

def render_citations(text: str) -> str:
    """Wrap [N] citation markers in styled badge spans."""
    return re.sub(r'\[(\d+)\]', r'<span class="citation-badge">[\1]</span>', text)


# ──────────────────────────────────────────────
#  Memory helpers
# ──────────────────────────────────────────────

def init_memory(session_key: str):
    """Initialize ConversationSummaryBufferMemory in session state if enabled."""
    if memory_enabled() and session_key not in st.session_state:
        try:
            from langchain.memory import ConversationSummaryBufferMemory
            from langchain_ollama import OllamaLLM
            memory_llm = OllamaLLM(model=llm_model(), temperature=0)
            st.session_state[session_key] = ConversationSummaryBufferMemory(
                llm=memory_llm,
                max_token_limit=memory_max_token_limit(),
                return_messages=False,
            )
        except Exception:
            st.session_state[session_key] = None


def build_chat_history(chat_history_list: list, memory_key: str) -> str:
    """Build formatted chat history string from memory or manual history."""
    if memory_enabled() and st.session_state.get(memory_key):
        memory_vars = st.session_state[memory_key].load_memory_variables({})
        return memory_vars.get("history", "")

    formatted_history = ""
    if len(chat_history_list) > 1:
        for msg in chat_history_list[:-1]:
            role_name = "Student" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role_name}: {msg['content']}\n"
    return formatted_history


def save_to_memory(memory_key: str, prompt: str, response: str):
    """Save a conversation turn to memory if enabled."""
    if memory_enabled() and st.session_state.get(memory_key):
        st.session_state[memory_key].save_context(
            {"input": prompt}, {"output": response}
        )


# ──────────────────────────────────────────────
#  File upload & ingestion
# ──────────────────────────────────────────────

import concurrent.futures

def _ingest_single_file(file_path: str, images_dir: str, collection_name: str) -> int:
    """Worker: parse file, extract formulas, and add to vector store."""
    from src.file_parser import parse_file
    from src.formula_extractor import extract_formulas
    
    docs = parse_file(file_path, images_dir)
    if docs:
        add_documents_to_store(docs, collection_name)
        # Extract and store formulas
        for doc in docs:
            formulas = extract_formulas(doc["text"], doc["source"], doc["page"])
            if formulas:
                formula_docs = [{
                    "text": f"Formula: {f['formula']}\nContext: {f['context']}",
                    "source": f["source"],
                    "page": f["page"],
                    "type": "formula",
                } for f in formulas]
                add_documents_to_store(formula_docs, FORMULA_COLLECTION)
    return len(docs) if docs else 0


def handle_file_upload(uploaded_files, upload_dir: str, images_dir: str,
                       collection_name: str, files_processed_key: str):
    """
    Process uploaded files in background threads with st.status tracker.
    """
    if not uploaded_files:
        return

    new_files = []
    for uf in uploaded_files:
        file_path = os.path.join(upload_dir, uf.name)
        # Check if already processed
        if is_already_processed(file_path, collection_name):
            if uf.name not in st.session_state[files_processed_key]:
                st.session_state[files_processed_key].add(uf.name)
            st.info(f"'{uf.name}' is already indexed.")
            continue
        new_files.append(uf)

    if not new_files:
        return

    with st.status(f"Processing {len(new_files)} file(s)...", expanded=True) as status:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}
            for uf in new_files:
                dest = os.path.join(upload_dir, uf.name)
                # Write to disk
                with open(dest, "wb") as f:
                    f.write(uf.getbuffer())
                
                # Submit to worker pool
                fut = pool.submit(_ingest_single_file, dest, images_dir, collection_name)
                futures[fut] = uf.name

            for fut in concurrent.futures.as_completed(futures):
                name = futures[fut]
                try:
                    num_chunks = fut.result()
                    dest = os.path.join(upload_dir, name)
                    mark_as_processed(dest, collection_name)
                    st.session_state[files_processed_key].add(name)
                    st.write(f"✅ **{name}** — {num_chunks} parent docs indexed")
                except Exception as e:
                    st.write(f"❌ **{name}** — Failed to process: {e}")
        status.update(label="Upload and indexing complete", state="complete")


# ──────────────────────────────────────────────
#  Image display
# ──────────────────────────────────────────────

def find_and_display_image(sources: list, image_dirs: list):
    """
    Find and display the first matching relevant diagram image.

    Args:
        sources: List of source dicts with 'source' and 'page' keys
        image_dirs: List of image directory paths to search
    """
    if not sources:
        return

    displayed = False
    for s in sources:
        if displayed:
            break
        safe_filename = "".join([c if c.isalnum() else "_" for c in s["source"]])
        page_img_prefix = f"{safe_filename}_page{s['page']}_img"
        for img_dir in image_dirs:
            if displayed:
                break
            if os.path.exists(img_dir):
                for img_file in os.listdir(img_dir):
                    if img_file.startswith(page_img_prefix):
                        st.image(os.path.join(img_dir, img_file),
                                 caption=f"Relevant Diagram from {s['source']} (Page/Slide {s['page']})",
                                 use_container_width=True)
                        displayed = True
                        break


# ──────────────────────────────────────────────
#  Source documents viewer
# ──────────────────────────────────────────────

def render_sources(sources: list, upload_dirs: list):
    """
    Render source documents expander with PDF viewer integration.

    Args:
        sources: List of source dicts with 'source', 'page', 'text' keys
        upload_dirs: List of upload directory paths to search for PDF files
    """
    if not sources:
        return

    with st.expander("🔍 View Source Documents"):
        for s in sources:
            st.markdown(f"**📄 {s['source']} (Page {s['page']})**")
            if s["source"].lower().endswith(".pdf"):
                pdf_path = _find_file_in_dirs(s["source"], upload_dirs)
                if pdf_path:
                    page_num = int(s["page"])
                    try:
                        annotations = get_pdf_annotations(pdf_path, page_num, s.get("text", ""))
                        pdf_viewer(input=pdf_path, width=700, annotations=annotations,
                                   pages_to_render=[page_num])
                    except Exception as e:
                        st.error(f"Error viewing PDF: {e}")
                        st.info(s.get("text", "Text unavailable."))
                else:
                    st.info(s.get("text", "Text unavailable."))
            else:
                st.info(s.get("text", "Text unavailable."))


def _find_file_in_dirs(filename: str, dirs: list) -> str | None:
    """Search for a file across multiple directories, return first match."""
    for d in dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None


# ──────────────────────────────────────────────
#  Chat history rendering
# ──────────────────────────────────────────────

def render_chat_history(chat_history: list, image_dirs: list, upload_dirs: list):
    """
    Render full chat history with verification badges, images, and sources.

    Args:
        chat_history: List of message dicts
        image_dirs: List of image directory paths
        upload_dirs: List of upload directory paths
    """
    for msg in chat_history:
        with st.chat_message(msg["role"]):
            render_with_latex(msg["content"])

            # Verification badge
            if msg.get("verification"):
                v = msg["verification"]
                css_class = f"confidence-{v['verdict'].lower()}"
                score_pct = int(v["coverage_score"] * 100)
                st.markdown(
                    f'<p class="{css_class}">Confidence: {v["verdict"]} '
                    f'({score_pct}% of claims found in notes)</p>',
                    unsafe_allow_html=True
                )

            # Image
            find_and_display_image(msg.get("sources", []), image_dirs)

            # Sources
            render_sources(msg.get("sources", []), upload_dirs)


# ──────────────────────────────────────────────
#  Verification
# ──────────────────────────────────────────────

def run_verification(full_response: str, retrieved_docs: list) -> dict | None:
    """Run answer verification against context if enabled. Returns verification dict or None."""
    if not verification_enabled() or not retrieved_docs:
        return None

    context_str = format_docs(retrieved_docs)
    verification = verify_answer_against_context(full_response, context_str)
    if verification["verdict"] != "GROUNDED":
        css_class = f"confidence-{verification['verdict'].lower()}"
        score_pct = int(verification["coverage_score"] * 100)
        st.markdown(
            f'<p class="{css_class}">Confidence: {verification["verdict"]} '
            f'({score_pct}% of claims found in notes)</p>',
            unsafe_allow_html=True
        )
    return verification


# ──────────────────────────────────────────────
#  Formula context enrichment (Upgrade 3 fix)
# ──────────────────────────────────────────────

def get_formula_context(prompt: str) -> str:
    """
    If the prompt is a formula query, retrieve formula docs and return
    a context string to prepend to the RAG context. Returns empty string otherwise.
    """
    if not is_formula_query(prompt):
        return ""

    formula_docs = get_all_documents(FORMULA_COLLECTION)
    if not formula_docs:
        return ""

    # Take up to 5 most relevant formula docs
    formula_texts = [d.page_content for d in formula_docs[:5]]
    return "\n\nRelevant Formulas:\n" + "\n".join(formula_texts) + "\n\n"


# ──────────────────────────────────────────────
#  Quiz UI
# ──────────────────────────────────────────────

def render_quiz_ui(stats: dict, files_processed: set, collection_name: str,
                   key_prefix: str, extra_stats: dict = None,
                   extra_files: set = None, dual_collection: bool = False):
    """
    Render the quiz generator UI.

    Args:
        stats: Collection stats dict
        files_processed: Set of processed filenames
        collection_name: Primary collection name
        key_prefix: Prefix for Streamlit widget keys (e.g. 'admin', 'student')
        extra_stats: Optional second collection stats (for student dual-collection)
        extra_files: Optional extra filenames from other collection
        dual_collection: If True, merge results from both collections for quiz
    """
    st.markdown("### 🧪 Quiz Generator")

    quiz_topic = st.text_input(
        "Topic (e.g., 'sorting algorithms', 'neural networks')",
        key=f"{key_prefix}_quiz_topic"
    )

    qcol1, qcol2 = st.columns(2)
    with qcol1:
        q_type = st.selectbox(
            "Question Type", ["MCQ", "Short Answer"],
            index=0 if quiz_default_type() == "mcq" else 1,
            key=f"{key_prefix}_q_type"
        )
    with qcol2:
        num_q = st.slider(
            "Number of Questions", 1, 10,
            quiz_default_num_questions(),
            key=f"{key_prefix}_num_q"
        )

    # Source filter
    source_filter = None
    all_files = sorted(files_processed | (extra_files or set()))
    if all_files:
        selected_source = st.selectbox(
            "Filter by source (optional)",
            ["All Sources"] + all_files,
            key=f"{key_prefix}_quiz_source"
        )
        if selected_source != "All Sources":
            source_filter = selected_source

    # Check if any docs available
    total = stats["total_chunks"]
    if extra_stats:
        total += extra_stats["total_chunks"]

    if st.button("Generate Quiz", use_container_width=True, type="primary",
                 key=f"{key_prefix}_gen_quiz"):
        if not quiz_topic:
            st.warning("Please enter a topic.")
        elif total == 0:
            st.warning("Please upload some course notes first!")
        else:
            with st.spinner("Generating quiz..."):
                q_type_key = "mcq" if q_type == "MCQ" else "short_answer"

                if dual_collection:
                    # Upgrade 8: Use dual-collection quiz generation
                    from src.quiz_chain import generate_quiz_dual
                    quiz_stream = generate_quiz_dual(
                        quiz_topic, num_q, q_type_key, source_filter
                    )
                else:
                    quiz_stream = generate_quiz(
                        quiz_topic, num_q, q_type_key,
                        collection_name, source_filter
                    )
                st.write_stream(quiz_stream)


# ──────────────────────────────────────────────
#  Sidebar helpers
# ──────────────────────────────────────────────

def render_sidebar_stats(stats: dict, files_processed: set,
                         label_docs: str = "Parent Docs Indexed",
                         label_files: str = "Files Uploaded",
                         extra_stats: dict = None):
    """Render knowledge base stats cards in sidebar."""
    st.markdown("---")
    st.markdown("### 📊 Knowledge Base")

    if extra_stats:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="stats-card"><h3>{stats.get("parent_docs", stats["total_chunks"])}</h3>'
                f'<p>Admin Docs</p></div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f'<div class="stats-card"><h3>{extra_stats.get("parent_docs", extra_stats["total_chunks"])}</h3>'
                f'<p>Your Docs</p></div>',
                unsafe_allow_html=True
            )
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="stats-card"><h3>{stats.get("parent_docs", stats["total_chunks"])}</h3>'
                f'<p>{label_docs}</p></div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f'<div class="stats-card"><h3>{len(files_processed)}</h3>'
                f'<p>{label_files}</p></div>',
                unsafe_allow_html=True
            )


def render_file_list(files_processed: set, title: str = "📄 Uploaded Files"):
    """Render the uploaded files list in sidebar."""
    if not files_processed:
        return
    st.markdown("---")
    st.markdown(f"### {title}")
    for fname in sorted(files_processed):
        ext = os.path.splitext(fname)[1].lower()
        icon = {"pdf": "📕", "docx": "📘", "pptx": "📙"}.get(ext.strip("."), "📄")
        st.caption(f"{icon} {fname}")


def render_model_info():
    """Render model info section in sidebar."""
    st.markdown("---")
    st.markdown("### ⚙️ Models")
    st.caption(f"🧠 LLM: {llm_model()}")
    st.caption(f"👁️ Vision: {vision_model()}")
    st.caption(f"📐 Embeddings: {embeddings_model()}")
    st.caption("💾 Vector DB: ChromaDB")


def render_export_buttons(chat_history: list, prefix: str = "chat"):
    """Render JSON and Markdown export buttons for chat history (Improvement 6)."""
    if not chat_history:
        return
    import json
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    col1, col2 = st.columns(2)

    # JSON export
    with col1:
        blob = json.dumps(chat_history, indent=2, ensure_ascii=False)
        st.download_button(
            "⬇️ Export JSON",
            data=blob.encode("utf-8"),
            file_name=f"{prefix}_{ts}.json",
            mime="application/json",
            use_container_width=True,
            key=f"{prefix}_export_json_{ts}"
        )

    # Markdown export
    with col2:
        lines = [f"# Chat Export — {ts}\n"]
        for msg in chat_history:
            role = "**You**" if msg["role"] == "user" else "**Assistant**"
            lines.append(f"\n{role}:\n{msg['content']}\n")
            if msg.get("sources"):
                lines.append("\n*Sources:* " +
                             ", ".join(f"{s['source']} p.{s['page']}" for s in msg["sources"]))
        md = "\n".join(lines)
        st.download_button(
            "⬇️ Export Markdown",
            data=md.encode("utf-8"),
            file_name=f"{prefix}_{ts}.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"{prefix}_export_md_{ts}"
        )


def render_with_latex(response: str):
    """
    Split response into text and LaTeX blocks, render LaTeX with st.latex() (Improvement 12).
    """
    parts = re.split(r'\$\$(.+?)\$\$', response, flags=re.DOTALL)
    for i, part in enumerate(parts):
        if i % 2 == 1:  # LaTeX block
            st.latex(part.strip())
        else:
            # Handle inline $...$
            inline_parts = re.split(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', part)
            for j, ip in enumerate(inline_parts):
                if j % 2 == 1:
                    st.latex(ip.strip())
                elif ip.strip():
                    st.markdown(ip, unsafe_allow_html=True)
