import streamlit as st

NEURO_CSS = """
<style>
    :root {
        --bg: #F4F3FF;
        --surface: #F8F7FF;
        --surface2: #F1EFFF;
        --text: #2B2A33;
        --muted: #6B6A77;
        --primary: #7C3AED;
        --primary-light: #A78BFA;
        --accent: #EC4899;
        --shadow-light: #ffffff;
        --shadow-dark: rgba(32,24,72,.12);
    }

    /* 背景 */
    .stApp {
        background: radial-gradient(1200px 800px at 20% 10%, #fff 0%, var(--bg) 55%, #EEF0FF 100%);
    }

    /* ニューモーフィズムカード */
    .neu-card {
        background: var(--surface);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 10px 10px 24px var(--shadow-dark), -10px -10px 24px var(--shadow-light);
        border: 1px solid rgba(124, 58, 237, 0.06);
        margin: 20px 0;
    }

    /* タイトルスタイル */
    .main-title {
        font-size: 3em;
        font-weight: bold;
        color: var(--primary);
        text-align: center;
        margin: 40px 0 20px 0;
        text-shadow: 2px 2px 4px rgba(124, 58, 237, 0.1);
    }

    .subtitle {
        text-align: center;
        color: var(--muted);
        font-size: 1.1em;
        margin-bottom: 40px;
    }

    /* ボタンスタイル */
    .stButton > button {
        background: var(--surface);
        border: none;
        padding: 15px 30px;
        border-radius: 50px;
        color: var(--primary);
        font-weight: bold;
        box-shadow: 5px 5px 10px var(--shadow-dark), -5px -5px 10px var(--shadow-light);
        transition: all 0.2s ease;
        width: 100%;
        font-size: 1.1em;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 7px 7px 14px var(--shadow-dark), -7px -7px 14px var(--shadow-light);
    }

    .stButton > button:active {
        box-shadow: inset 5px 5px 10px var(--shadow-dark), inset -5px -5px 10px var(--shadow-light);
        transform: translateY(1px);
    }

    /* バッジ */
    .badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 12px;
        font-size: 0.9em;
        font-weight: bold;
        background: #DCFCE7;
        color: #166534;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""

def apply_styles():
    """Call once near the top of app.py."""
    st.markdown(NEURO_CSS, unsafe_allow_html=True)