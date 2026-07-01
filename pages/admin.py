"""
NoteGo RAG — Admin Panel
Upload course notes, manage knowledge base, and chat with RAG.
"""

import os
import streamlit as st
from src.styles import inject_css
from src.vector_store import (
    get_collection_stats, clear_store, ADMIN_COLLECTION
)
from src.rag_chain import (
    stream_rag_answer, detect_comparison_intent, stream_comparative_answer
)
from src.ingestion_tracker import get_all_processed_filenames
from src.ui_components import (
    init_memory, build_chat_history, save_to_memory,
    handle_file_upload, find_and_display_image, render_sources,
    render_chat_history, run_verification, get_formula_context,
    render_quiz_ui, render_sidebar_stats, render_file_list,
    render_model_info, render_citations, render_export_buttons,
    render_with_latex
)

inject_css()

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads_admin")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images_admin")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- Session state ---
if "admin_chat_history" not in st.session_state:
    st.session_state.admin_chat_history = []
if "admin_files_processed" not in st.session_state:
    st.session_state.admin_files_processed = set(get_all_processed_filenames(ADMIN_COLLECTION))

# --- Conversation Memory ---
init_memory("admin_memory")

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📁 Upload Course Notes")
    st.caption("Supported: PDF, DOCX, PPTX")

    mode = st.radio("Mode", ["Chat", "Quiz"], horizontal=True, key="admin_mode")

    uploaded_files = st.file_uploader(
        "Drag & drop your files",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="admin_uploader",
    )

    handle_file_upload(uploaded_files, UPLOAD_DIR, IMAGES_DIR,
                       ADMIN_COLLECTION, "admin_files_processed")

    # Stats
    stats = get_collection_stats(ADMIN_COLLECTION)
    render_sidebar_stats(stats, st.session_state.admin_files_processed)
    render_file_list(st.session_state.admin_files_processed)
    render_model_info()

    st.info("💡 **Tip for Images:** Before uploading PDFs/PPTXs with complex diagrams, ensure you've run `ollama pull llava:7b` in your terminal.")

    # Database Management
    st.markdown("---")
    st.markdown("### 🛠️ Manage")
    if st.button("🗑️ Clear Admin Database", use_container_width=True, key="admin_clear"):
        with st.spinner("Deleting admin database..."):
            clear_store(ADMIN_COLLECTION)
            st.session_state.admin_files_processed.clear()
            st.session_state.admin_chat_history.clear()
            if "admin_memory" in st.session_state and st.session_state.admin_memory:
                st.session_state.admin_memory.clear()
            st.success("Admin database cleared successfully!")
            st.rerun()

# --- Main Area ---
st.markdown('<h1 class="main-header">📚 NoteGo RAG — Admin Panel</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload course notes, then ask any question about them</p>', unsafe_allow_html=True)

if mode == "Quiz":
    render_quiz_ui(stats, st.session_state.admin_files_processed,
                   ADMIN_COLLECTION, "admin")
else:
    # --- Chat Mode ---
    render_chat_history(
        st.session_state.admin_chat_history,
        image_dirs=[IMAGES_DIR],
        upload_dirs=[UPLOAD_DIR]
    )

    # Export buttons
    render_export_buttons(st.session_state.admin_chat_history, "admin")

    # Chat input
    if prompt := st.chat_input("Ask a question about your course notes...", key="admin_chat_input"):
        if stats["total_chunks"] == 0:
            st.warning("⚠️ Please upload some course notes first using the sidebar!")
        else:
            st.session_state.admin_chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            formatted_history = build_chat_history(
                st.session_state.admin_chat_history, "admin_memory"
            )

            # Check for comparison intent
            available_sources = list(st.session_state.admin_files_processed)
            is_compare, src1, src2 = detect_comparison_intent(prompt, available_sources)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching notes & generating answer..."):
                    if is_compare and src1 and src2:
                        generator, sources, retrieved_docs = stream_comparative_answer(
                            prompt, src1, src2, ADMIN_COLLECTION, formatted_history)
                    else:
                        # Upgrade 3 fix: enrich with formula context if applicable
                        formula_ctx = get_formula_context(prompt)
                        generator, sources, retrieved_docs = stream_rag_answer(
                            prompt, ADMIN_COLLECTION, formatted_history,
                            extra_context=formula_ctx)

                full_response = st.write_stream(generator)

                # Upgrade 4 fix: apply citation badge rendering
                full_response = render_citations(full_response)

                # Verification
                verification = run_verification(full_response, retrieved_docs)

                # Display relevant images
                find_and_display_image(sources, [IMAGES_DIR])

                # Sources
                render_sources(sources, [UPLOAD_DIR])

            # Save to history
            st.session_state.admin_chat_history.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
                "verification": verification,
            })

            # Save to memory
            save_to_memory("admin_memory", prompt, full_response)
