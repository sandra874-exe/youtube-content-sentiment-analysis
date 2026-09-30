import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="YouTube Audience Analytics",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp { background: #faf7f2; }
    .block-container { padding-top: 1.8rem; max-width: 1450px; }

    section[data-testid="stSidebar"] {
        background: #fffaf5;
        border-right: 1px solid #eee4dc;
    }

    .sidebar-title {
        font-size: 24px;
        font-weight: 800;
        color: #252238;
        margin-bottom: 2px;
    }

    .sidebar-subtitle {
        color: #817b88;
        font-size: 13px;
        margin-bottom: 18px;
    }

    .hero {
        background: linear-gradient(135deg, #eee5ff, #ffeaf1, #e7f8f0);
        border: 1px solid #eadff4;
        border-radius: 28px;
        padding: 42px 48px;
        margin-bottom: 28px;
    }

    .hero-kicker {
        color: #81758e;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.7px;
        text-transform: uppercase;
    }

    .hero-title {
        color: #27243b;
        font-size: 44px;
        font-weight: 850;
        line-height: 1.08;
        margin-top: 10px;
    }

    .hero-text {
        color: #696372;
        font-size: 16px;
        line-height: 1.7;
        max-width: 900px;
        margin-top: 14px;
    }

    .info-card {
        background: white;
        border: 1px solid #eee7df;
        border-radius: 20px;
        padding: 24px;
        min-height: 170px;
        box-shadow: 0 6px 24px rgba(45, 37, 60, 0.03);
    }

    .mini-card {
        background: #ffffff;
        border: 1px solid #eee7df;
        border-radius: 18px;
        padding: 18px 20px;
    }

    .pipeline-step {
        text-align: center;
        background: white;
        border: 1px solid #eee7df;
        border-radius: 18px;
        padding: 18px 12px;
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
    }

    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-title">▶️ YouTube Analytics</div>
        <div class="sidebar-subtitle">Audience • Sentiment • Models • Insights</div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### Research pipeline")
    st.caption("YouTube API → Comments")
    st.caption("Comments → Preprocessing")
    st.caption("Labels → TF-IDF + ML")
    st.caption("Comments → VADER + RoBERTa")
    st.caption("Outputs → Plotly Insights")
    st.divider()
    st.caption("MSc BDA • Computing for Data Science")

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">MSC-LEVEL NLP + DATA ANALYTICS</div>
        <div class="hero-title">YouTube Audience Intelligence</div>
        <div class="hero-text">
            A modular pipeline for collecting YouTube audience reactions,
            cleaning text, comparing sentiment-analysis approaches,
            evaluating supervised machine-learning models, and exploring
            creator, domain, video and topic-level audience patterns.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Project overview")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="info-card">
            <h3>🎥 Collect</h3>
            <p>Use the YouTube Data API to collect creators, videos, comments and engagement metadata.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="info-card">
            <h3>🧠 Model</h3>
            <p>Evaluate VADER, pretrained RoBERTa, TF-IDF + Logistic Regression and TF-IDF + Linear SVM.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="info-card">
            <h3>📊 Explain</h3>
            <p>Use interactive Plotly visualisations to study sentiment, engagement, creators, domains and topics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### How the ML component works")
flow = st.columns(6)
labels = [
    ("01", "Labelled data", "Human ground truth"),
    ("02", "Preprocess", "Clean text"),
    ("03", "TF-IDF", "Numerical features"),
    ("04", "LR + SVM", "Supervised ML"),
    ("05", "VADER + RoBERTa", "Comparison"),
    ("06", "Metrics", "F1 / precision / recall"),
]
for col, (num, title, desc) in zip(flow, labels):
    with col:
        st.markdown(
            f"<div class='pipeline-step'><b>{num}</b><br><strong>{title}</strong><br><small>{desc}</small></div>",
            unsafe_allow_html=True,
        )

st.markdown("### Important methodology note")
st.info(
    "The application does not allow users to upload the ML evaluation CSV. "
    "The labelled dataset is bundled inside the project. The 150-row file is used "
    "when present; the distributed ZIP includes the 120 labelled rows that were actually "
    "available in the uploaded materials as a fallback."
)
