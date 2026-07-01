"""
styles.py — Shared CSS for all pages.
"""

import streamlit as st


def inject_css():
    """Inject premium CSS styling into the Streamlit page."""
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0;
    }

    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    .source-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea44;
        border-radius: 8px;
        padding: 4px 10px;
        margin: 3px 4px 3px 0;
        font-size: 0.8rem;
        color: #667eea;
    }

    .citation-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea33, #764ba233);
        border: 1px solid #667eea55;
        border-radius: 6px;
        padding: 1px 6px;
        font-size: 0.75rem;
        color: #667eea;
        font-weight: 600;
        vertical-align: super;
    }

    .stats-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        border: 1px solid #334155;
    }

    .stats-card h3 {
        color: #667eea;
        font-size: 1.8rem;
        margin: 0;
    }

    .stats-card p {
        color: #94a3b8;
        font-size: 0.85rem;
        margin: 5px 0 0 0;
    }

    .upload-success {
        background: linear-gradient(135deg, #065f4622, #10b98122);
        border: 1px solid #10b98144;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
    }

    .confidence-grounded {
        color: #10b981;
        font-size: 0.85rem;
    }

    .confidence-partial {
        color: #f59e0b;
        font-size: 0.85rem;
    }

    .confidence-uncertain {
        color: #ef4444;
        font-size: 0.85rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e293b);
    }

    .stChatMessage {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)
