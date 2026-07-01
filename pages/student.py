"""
NoteGo RAG — Student Portal
Upload PYQs/supplementary notes and chat with combined admin + student knowledge base.
"""

import os
import streamlit as st
from src.styles import inject_css
from src.vector_store import (
    get_collection_stats, clear_store,
    ADMIN_COLLECTION, STUDENT_COLLECTION
)
from src.rag_chain import stream_rag_answer_dual
from src.ingestion_tracker import get_all_processed_filenames
from src.ui_components import (
    init_memory, build_chat_history, save_to_memory,
    handle_file_upload, find_and_display_image, render_sources,
    render_chat_history, run_verification,
    render_quiz_ui, render_sidebar_stats, render_file_list,
    render_model_info, render_citations, render_export_buttons,
    render_with_latex, get_formula_context
)
from src.progress_tracker import get_summary, record_question

inject_css()

# --- Segment by user_id ---
student_user = st.session_state.get("user_id", "student")
STUDENT_USER_COLLECTION = f"{STUDENT_COLLECTION}_{student_user}"

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", f"uploads_student_{student_user}")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", f"images_student_{student_user}")
ADMIN_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images_admin")
ADMIN_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads_admin")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- Session state ---
if "student_chat_history" not in st.session_state:
    st.session_state.student_chat_history = []
if "student_files_processed" not in st.session_state:
    st.session_state.student_files_processed = set(get_all_processed_filenames(STUDENT_USER_COLLECTION))

# --- Conversation Memory ---
init_memory("student_memory")

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📁 Upload Your Notes / PYQs")
    st.caption("Supported: PDF, DOCX, PPTX")

    mode = st.radio("Mode", ["Chat", "Quiz"], horizontal=True, key="student_mode")

    uploaded_files = st.file_uploader(
        "Drag & drop your files",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="student_uploader",
    )

    handle_file_upload(uploaded_files, UPLOAD_DIR, IMAGES_DIR,
                       STUDENT_USER_COLLECTION, "student_files_processed")

    # Stats — show both collections
    admin_stats = get_collection_stats(ADMIN_COLLECTION)
    student_stats = get_collection_stats(STUDENT_USER_COLLECTION)
    render_sidebar_stats(admin_stats, st.session_state.student_files_processed,
                          extra_stats=student_stats)
    render_file_list(st.session_state.student_files_processed, title="📄 Your Uploaded Files")
    render_model_info()

    # Database Management — student can only clear their own data
    st.markdown("---")
    st.markdown("### 🛠️ Manage")
    if st.button("🗑️ Clear My Uploads", use_container_width=True, key="student_clear"):
        with st.spinner("Deleting your data..."):
            clear_store(STUDENT_USER_COLLECTION)
            st.session_state.student_files_processed.clear()
            st.session_state.student_chat_history.clear()
            if "student_memory" in st.session_state and st.session_state.student_memory:
                st.session_state.student_memory.clear()
            st.success("Your uploads cleared successfully!")
            st.rerun()

# --- Main Area ---
st.markdown('<h1 class="main-header">📚 NoteGo RAG — Student Portal</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your PYQs & notes, then ask questions from all course material</p>', unsafe_allow_html=True)

# --- Progress Stats Dashboard ---
summary = get_summary(student_user)
if summary["total_questions"] > 0 or summary["quiz_attempts"] > 0:
    st.markdown("### 📈 Your Learning Progress")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Questions Asked", summary["total_questions"])
    col_m2.metric("Groundedness Rate", f"{int(summary['grounded_rate'] * 100)}%")
    col_m3.metric("Quizzes Attempted", summary["quiz_attempts"])
    st.markdown("---")

if mode == "Quiz":
    admin_filenames = set(get_all_processed_filenames(ADMIN_COLLECTION))
    render_quiz_ui(admin_stats, st.session_state.student_files_processed,
                   STUDENT_USER_COLLECTION, "student",
                   extra_stats=student_stats, extra_files=admin_filenames,
                   dual_collection=True)
else:
    # --- Chat Mode ---
    render_chat_history(
        st.session_state.student_chat_history,
        image_dirs=[IMAGES_DIR, ADMIN_IMAGES_DIR],
        upload_dirs=[UPLOAD_DIR, ADMIN_UPLOAD_DIR]
    )

    # Export buttons
    render_export_buttons(st.session_state.student_chat_history, "student")

    # Chat input
    total_chunks = admin_stats["total_chunks"] + student_stats["total_chunks"]
    if prompt := st.chat_input("Ask a question about your course notes...", key="student_chat_input"):
        if total_chunks == 0:
            st.warning("⚠️ No course notes available. Ask your admin to upload notes or upload your own!")
        else:
            st.session_state.student_chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            formatted_history = build_chat_history(
                st.session_state.student_chat_history, "student_memory"
            )

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching notes & generating answer..."):
                    formula_ctx = get_formula_context(prompt)
                    generator, sources, retrieved_docs = stream_rag_answer_dual(
                        prompt, formatted_history, extra_context=formula_ctx)

                full_response = st.write_stream(generator)

                # Apply citation badge rendering
                full_response = render_citations(full_response)

                # Verification
                verification = run_verification(full_response, retrieved_docs)

                # Display relevant images (check both dirs)
                find_and_display_image(sources, [IMAGES_DIR, ADMIN_IMAGES_DIR])

                # Sources
                render_sources(sources, [UPLOAD_DIR, ADMIN_UPLOAD_DIR])

            # Save to progress tracker
            verdict = verification["verdict"] if verification else "UNKNOWN"
            record_question(student_user, prompt, verdict)

            # Save to history
            st.session_state.student_chat_history.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
                "verification": verification,
            })

            # Save to memory
            save_to_memory("student_memory", prompt, full_response)
            st.rerun()
