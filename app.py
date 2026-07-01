"""
NoteGo RAG — Course Notes Q&A System
Auth router: Login page + st.navigation to Admin/Student pages.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st
from dotenv import load_dotenv


def load_environment():
    """Load the project .env file using its actual text encoding."""
    dotenv_path = Path(__file__).with_name(".env")
    if not dotenv_path.exists():
        return

    byte_order_mark = dotenv_path.read_bytes()[:2]
    encoding = "utf-16" if byte_order_mark in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    load_dotenv(dotenv_path=dotenv_path, encoding=encoding)


load_environment()

# --- Page Config (MUST be first Streamlit call) ---
st.set_page_config(
    page_title="NoteGo RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Ollama Health Check ---
try:
    import ollama
    ollama.list()
except Exception:
    st.error("⚠️ **Ollama is not running.** Please start it with `ollama serve` in your terminal.")
    st.stop()


# --- Session Expiry (Improvement 13) ---
SESSION_TIMEOUT_MINUTES = 60


def check_session_timeout():
    last_active = st.session_state.get("last_active")
    if last_active:
        if isinstance(last_active, str):
            try:
                last_active = datetime.fromisoformat(last_active)
            except Exception:
                last_active = datetime.now()
        elapsed = datetime.now() - last_active
        if elapsed > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            for key in ["authenticated", "role", "user_id", "last_active"]:
                st.session_state.pop(key, None)
            st.warning("Session expired. Please log in again.")
            st.rerun()
    st.session_state.last_active = datetime.now()


def logout():
    """Clear auth state and rerun."""
    for key in ["authenticated", "role", "user_id", "last_active"]:
        st.session_state.pop(key, None)
    st.rerun()


def login_page():
    """Render the login page (Improvement 7)."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .login-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0;
        }
        .login-sub {
            text-align: center;
            color: #888;
            font-size: 1.1rem;
            margin-top: -5px;
            margin-bottom: 40px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="login-header">📚 NoteGo RAG</h1>', unsafe_allow_html=True)
    st.markdown('<p class="login-sub">Course Notes Q&A System</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", use_container_width=True, type="primary"):
            from src.auth import authenticate
            role, user_id = authenticate(username, password)
            if role:
                st.session_state.authenticated = True
                st.session_state.role = role
                st.session_state.user_id = user_id
                st.session_state.last_active = datetime.now()
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")


# --- Main routing logic ---
if not st.session_state.get("authenticated"):
    login_page()
else:
    check_session_timeout()
    role = st.session_state.get("role", "student")

    logout_page = st.Page(logout, title="Logout", icon="🚪")

    if role == "admin":
        admin_page = st.Page("pages/admin.py", title="Admin Panel", icon="🔧", default=True)
        nav = st.navigation([admin_page, logout_page])
    else:
        student_page = st.Page("pages/student.py", title="Student Portal", icon="🎓", default=True)
        nav = st.navigation([student_page, logout_page])

    nav.run()
