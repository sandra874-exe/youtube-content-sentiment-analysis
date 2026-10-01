from pathlib import Path
import math
import re

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix

from src.youtube_api import (
    identify_youtube_url,
    get_video_info,
    get_channel_info,
    get_channel_videos,
    get_video_comments,
)

from src.sentiment_analyzer import (
    run_complete_analysis,
    load_transformer,
    get_vader_sentiment,
)

from src.creator_comparison import compare_creators



# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="YouTube Pulse",
    page_icon="YT",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --canvas: #F7F4EF;
        --card: #FFFFFF;
        --coral: #D95F43;
        --coral-dark: #A83D28;
        --peach: #FFD9CF;
        --mint: #C7EAE4;
        --sage: #A7C4B5;
        --lavender: #D6D2E8;
        --blue: #BDD4E7;
        --rose: #F4B8C1;
        --ink: #202638;
        --muted: #465064;
        --faint: #697386;
        --border: #DED8D0;
    }

    html {
        scroll-behavior: smooth;
    }

    body,
    .stApp,
    .main,
    [data-testid="stAppViewContainer"] {
        background: var(--canvas) !important;
        color: var(--ink) !important;
    }

    /* Hide Streamlit's native header so it cannot cover the custom navigation. */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    [data-testid="stToolbar"] {
        display: none !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 4rem;
        max-width: 1500px;
    }

    /* Custom one-page navigation */
    .topbar {
        position: sticky;
        top: .75rem;
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: .75rem 1rem;
        margin-bottom: 1.6rem;
        background: #FFFFFF;
        border: 1px solid #D8D2C9;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(32,38,56,.12);
        backdrop-filter: blur(14px);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: .65rem;
        color: #202638 !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 1.05rem;
        white-space: nowrap;
    }

    .brand-mark {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        display: grid;
        place-items: center;
        background: #FFD3C8;
        color: #8F321E;
        font-size: 1.05rem;
    }

    .topnav {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: .35rem;
        overflow-x: auto;
        scrollbar-width: none;
    }

    .topnav::-webkit-scrollbar {
        display: none;
    }

    .topnav a {
        display: inline-block;
        padding: .55rem .85rem;
        border-radius: 999px;
        color: #374151 !important;
        text-decoration: none !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: .82rem;
        font-weight: 700;
        white-space: nowrap;
        transition: all .18s ease;
    }

    .topnav a:hover {
        background: #FFE0D8;
        color: #8F321E !important;
    }

    .anchor-section {
        scroll-margin-top: 105px;
    }

    /* Main hero */
    .hero {
        padding: 2.35rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #FFEDE6 0%, #F8E6E1 52%, #E8E5F2 100%);
        border: 1px solid #DCD5D0;
        color: var(--ink);
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 34px rgba(32,38,56,.09);
    }

    .hero h1 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: clamp(2.1rem, 4vw, 3.15rem);
        line-height: 1.08;
        font-weight: 800;
        margin: .75rem 0 .7rem 0;
        color: #1E2537 !important;
        letter-spacing: -.025em;
    }

    .hero p {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1rem;
        line-height: 1.7;
        color: #465064 !important;
        max-width: 900px;
        margin: 0;
        font-weight: 500;
    }

    .badge {
        display: inline-block;
        padding: .4rem .75rem;
        border-radius: 999px;
        background: #FFD2C7;
        color: #8F321E !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .05em;
    }

    .section-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        color: #202638 !important;
        margin-top: 1.7rem;
        margin-bottom: .65rem;
        scroll-margin-top: 105px;
    }

    .section-kicker {
        color: #596579 !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: .78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .09em;
        margin-bottom: .35rem;
    }

    .metric-card {
        background: #FFFFFF;
        padding: 1.15rem;
        border-radius: 16px;
        border: 1px solid #DED8D0;
        box-shadow: 0 5px 22px rgba(32,38,56,.07);
        min-height: 115px;
    }

    .metric-header {
        display: flex;
        align-items: center;
        gap: .65rem;
        margin-bottom: .15rem;
    }

    .metric-icon {
        width: 38px;
        height: 38px;
        border-radius: 12px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #F7F4EF;
        flex: 0 0 38px;
    }

    .metric-icon svg,
    .section-title-icon svg,
    .insight-icon svg {
        display: block;
    }

    .section-title {
        display: flex;
        align-items: center;
        gap: .65rem;
    }

    .section-title-icon {
        width: 32px;
        height: 32px;
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #FFF0EC;
        flex: 0 0 32px;
    }

    .insight-icon {
        display: inline-flex;
        vertical-align: middle;
        margin-right: .45rem;
    }

    .metric-label {
        color: #596579 !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: .72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .05em;
    }

    .metric-value {
        color: #202638 !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.9rem;
        font-weight: 800;
        margin-top: .45rem;
    }

    .insight-card,
    .video-card {
        background: #FFFFFF;
        border: 1px solid #DED8D0;
        border-radius: 16px;
        box-shadow: 0 5px 22px rgba(32,38,56,.07);
    }

    .insight-card {
        padding: 1rem 1.1rem;
        border-left: 5px solid var(--coral);
        margin-bottom: .7rem;
        color: #30394C !important;
    }

    .video-card {
        padding: 1rem;
        margin-bottom: .7rem;
    }

    .small-muted {
        color: #596579 !important;
        font-size: .8rem;
    }

    /* Streamlit text and controls */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] label {
        color: #30394C;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        background: #FFFFFF !important;
        color: #202638 !important;
        border: 1px solid #CFC8BF !important;
        border-radius: 10px !important;
    }

    div[data-testid="stTextInput"] input::placeholder,
    div[data-testid="stTextArea"] textarea::placeholder {
        color: #7A8495 !important;
        opacity: 1 !important;
    }

    div[data-testid="stButton"] > button {
        border-radius: 10px;
        border: 1px solid var(--coral);
        background: var(--coral);
        color: #FFFFFF !important;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
    }

    div[data-testid="stButton"] > button:hover {
        background: #C94F35;
        border-color: #C94F35;
        box-shadow: 0 6px 18px rgba(217,95,67,.28);
    }

    div[data-testid="stSlider"] [role="slider"] {
        background: var(--coral);
    }

    .section-divider {
        height: 1px;
        background: #D9D3CA;
        margin: 2.5rem 0;
    }

    .footer-note {
        text-align: center;
        color: #697386 !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: .75rem;
        padding: 2rem 0 .5rem;
    }

    @media (max-width: 800px) {
        .topbar {
            align-items: flex-start;
            flex-direction: column;
        }

        .topnav {
            width: 100%;
            justify-content: flex-start;
        }

        .hero {
            padding: 1.4rem;
        }
    }
    </style>""",
    unsafe_allow_html=True,
)


# ============================================================
# COLORS
# ============================================================

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "neutral": "#94a3b8",
    "negative": "#ef4444",
}


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "content_info" not in st.session_state:
    st.session_state.content_info = None

if "content_type" not in st.session_state:
    st.session_state.content_type = None

if "channel_videos" not in st.session_state:
    st.session_state.channel_videos = []

if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None


# ============================================================
# HELPERS
# ============================================================

# ------------------------------------------------------------
# SVG ICON SYSTEM
# ------------------------------------------------------------

def svg_icon(name, size=22, color="currentColor"):
    icons = {
        "positive": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color:{color};"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2"/><path d="M8 14C9 15.2 10.3 16 12 16C13.7 16 15 15.2 16 14" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="9" cy="10" r="1" fill="currentColor"/><circle cx="15" cy="10" r="1" fill="currentColor"/></svg>''',
        "neutral": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color:{color};"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2"/><path d="M8.5 15H15.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="9" cy="10" r="1" fill="currentColor"/><circle cx="15" cy="10" r="1" fill="currentColor"/></svg>''',
        "negative": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color:{color};"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2"/><path d="M8 16C9 14.8 10.3 14 12 14C13.7 14 15 14.8 16 16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="9" cy="10" r="1" fill="currentColor"/><circle cx="15" cy="10" r="1" fill="currentColor"/></svg>''',
        "eye": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color:{color};"><path d="M2.5 12C4.8 8.5 8 6.5 12 6.5C16 6.5 19.2 8.5 21.5 12C19.2 15.5 16 17.5 12 17.5C8 17.5 4.8 15.5 2.5 12Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="2"/></svg>''',
        "like": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color:{color};"><path d="M7 10V20H4C3.45 20 3 19.55 3 19V11C3 10.45 3.45 10 4 10H7Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M7 20H17.2C18.5 20 19.5 19.1 19.7 17.8L20.6 12.2C20.8 11 19.9 10 18.7 10H14L15 6.8C15.2 5.3 14.1 4 12.6 4C12 4 11.5 4.3 11.2 4.8L7 10V20Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>''',
        "comment": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color:{color};"><path d="M20 11.5C20 16.2 16.4 20 12 20C10.5 20 9.1 19.6 7.9 18.9L4 20L5.1 16.4C4.4 15.1 4 13.6 4 12C4 7.3 7.6 4 12 4C16.4 4 20 7.3 20 11.5Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>''',
        "users": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color:{color};"><circle cx="9" cy="8" r="3" stroke="currentColor" stroke-width="2"/><path d="M3.5 19C3.8 15.7 5.7 14 9 14C12.3 14 14.2 15.7 14.5 19" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M15 5.5C17.2 5.8 18.5 7.1 18.5 9C18.5 10.7 17.5 12 16 12.5M17 15C19.2 15.5 20.3 16.8 20.5 19" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>''',
        "video": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color:{color};"><rect x="3" y="6" width="13" height="12" rx="2" stroke="currentColor" stroke-width="2"/><path d="M16 10L21 7.5V16.5L16 14V10Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>''',
        "heart": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color:{color};"><path d="M20.8 8.8C20.8 13.5 12 19 12 19C12 19 3.2 13.5 3.2 8.8C3.2 6.2 5.1 4.5 7.4 4.5C9.2 4.5 10.7 5.5 12 7C13.3 5.5 14.8 4.5 16.6 4.5C18.9 4.5 20.8 6.2 20.8 8.8Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>''',
        "chart": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color:{color};"><path d="M4 19V5M4 19H20" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M7 15L10 11L13 13L18 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>''',
        "lightbulb": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color:{color};"><path d="M9 18H15M9.5 21H14.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M8.5 14.5C7.3 13.5 6.5 11.9 6.5 10C6.5 6.9 8.9 4.5 12 4.5C15.1 4.5 17.5 6.9 17.5 10C17.5 11.9 16.7 13.5 15.5 14.5C14.7 15.2 14.5 16 14.5 17H9.5C9.5 16 9.3 15.2 8.5 14.5Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>''',
        "brain": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color:{color};"><path d="M9.5 4.5C8 3.2 5.5 4.1 5.5 6.2C3.6 6.5 3 8.8 4.2 10.1C2.9 11.8 4 14.1 5.8 14.3C5.2 16.3 6.9 18 8.7 17.7C9.1 19.8 11.7 20.2 12 18V6C11.8 4.7 10.8 4.2 9.5 4.5Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M14.5 4.5C16 3.2 18.5 4.1 18.5 6.2C20.4 6.5 21 8.8 19.8 10.1C21.1 11.8 20 14.1 18.2 14.3C18.8 16.3 17.1 18 15.3 17.7C14.9 19.8 12.3 20.2 12 18V6C12.2 4.7 13.2 4.2 14.5 4.5Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>''',
        "search": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color:{color};"><circle cx="10.8" cy="10.8" r="6.8" stroke="currentColor" stroke-width="2"/><path d="M16 16L21 21" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>''',
        "download": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color:{color};"><path d="M12 3V15M7 11L12 16L17 11M4 20H20" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>''',
        "shield": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color:{color};"><path d="M12 3L20 6V11C20 16 16.8 19.5 12 21C7.2 19.5 4 16 4 11V6L12 3Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M9 12L11 14L15 10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>''',
        "sparkles": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="color:{color};"><path d="M12 3L13.2 7.8L18 9L13.2 10.2L12 15L10.8 10.2L6 9L10.8 7.8L12 3Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M19 14L19.6 16.4L22 17L19.6 17.6L19 20L18.4 17.6L16 17L18.4 16.4L19 14Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>''',
    }
    return icons.get(name, "")


METRIC_ICON_MAP = {
    "views": "eye",
    "channel views": "eye",
    "likes": "like",
    "average likes": "heart",
    "avg likes": "heart",
    "comments": "comment",
    "comments available": "comment",
    "comments analysed": "brain",
    "analysed comments": "brain",
    "comments checked": "comment",
    "subscribers": "users",
    "creators": "users",
    "videos": "video",
    "domains": "chart",
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    "avg. positive %": "positive",
    "avg. negative %": "negative",
    "sources compared": "users",
    "compound": "chart",
}


SECTION_ICON_MAP = {
    "comparison overview": "chart",
    "audience mood": "chart",
    "what the audience is saying": "lightbulb",
    "topics & language": "search",
    "comment explorer": "comment",
    "vader vs transformer": "brain",
    "channel video breakdown": "video",
    "dataset insights": "lightbulb",
    "overall dataset sentiment": "chart",
    "video-level sentiment": "video",
    "distinctive audience vocabulary": "search",
}


def _clean_label(text):
    text = str(text)
    text = re.sub(r"^[^\w]+", "", text, flags=re.UNICODE)
    return text.strip()


def _find_icon(label, mapping, default="chart"):
    clean = _clean_label(label).lower()
    if clean in mapping:
        return mapping[clean]
    for key, icon in mapping.items():
        if key in clean:
            return icon
    return default


def format_number(value):

    if value is None:
        return "0"

    try:
        value = float(value)
    except:
        return str(value)

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if value >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{int(value):,}"


def metric_card(label, value, icon=None, icon_color=None):

    icon = icon or _find_icon(label, METRIC_ICON_MAP)

    if icon_color is None:
        clean = _clean_label(label).lower()
        if "positive" in clean:
            icon_color = SENTIMENT_COLORS["positive"]
        elif "negative" in clean:
            icon_color = SENTIMENT_COLORS["negative"]
        elif "neutral" in clean:
            icon_color = SENTIMENT_COLORS["neutral"]
        else:
            icon_color = "#D95F43"

    clean_label = _clean_label(label)

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-header">
                <div class="metric-icon">
                    {svg_icon(icon, size=21, color=icon_color)}
                </div>
                <div class="metric-label">{clean_label}</div>
            </div>
            <div class="metric-value">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title, icon=None):

    icon = icon or _find_icon(title, SECTION_ICON_MAP)
    clean_title = _clean_label(title)

    st.markdown(
        f"""
        <div class="section-title">
            <span class="section-title-icon">
                {svg_icon(icon, size=18, color="#D95F43")}
            </span>
            <span>{clean_title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_percentage(summary, sentiment):

    if summary is None or summary.empty:
        return 0

    row = summary[
        summary["sentiment"] == sentiment
    ]

    if row.empty:
        return 0

    return float(
        row.iloc[0]["percentage"]
    )


def sentiment_donut(summary, title="Audience Mood"):

    if summary.empty:
        st.info("No sentiment data available.")
        return

    fig = go.Figure(
        data=[
            go.Pie(
                labels=summary["sentiment"].str.title(),
                values=summary["count"],
                hole=0.62,
                marker=dict(
                    colors=[
                        SENTIMENT_COLORS.get(
                            x,
                            "#64748b",
                        )
                        for x in summary["sentiment"]
                    ]
                ),
                textinfo="percent",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Comments: %{value}<br>"
                    "Share: %{percent}<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        height=420,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
        showlegend=True,
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )


def sentiment_bar(summary):

    if summary.empty:
        return

    fig = px.bar(
        summary,
        x="sentiment",
        y="percentage",
        text="percentage",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Distribution",
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    fig.update_layout(
        yaxis_title="Percentage",
        xaxis_title="Sentiment",
        showlegend=False,
        height=400,
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )


# ============================================================
# DEMO DATA
# ============================================================

@st.cache_data
def load_demo_data():

    path = (
        DATA_DIR
        / "youtube_comments_vader.csv"
    )

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_csv(filename):

    path = DATA_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()


@st.cache_resource
def get_transformer():

    return load_transformer()




# ============================================================
# YOUTUBE ANALYSIS
# ============================================================

def analyze_youtube_url(
    url,
    comment_limit,
    progress=None,
):

    detected = identify_youtube_url(
        url
    )

    if not detected:
        raise ValueError(
            "The YouTube URL could not be identified."
        )

    content_type = detected.get(
        "type"
    )

    content_id = detected.get(
        "id"
    )

    if content_type not in [
        "video",
        "channel",
    ]:
        raise ValueError(
            "Please enter a valid YouTube video "
            "or channel URL."
        )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    if content_type == "video":

        if progress:
            progress.update(
                label="🎬 Fetching video information...",
                state="running",
                expanded=False,
            )

        info = get_video_info(
            content_id
        )

        if not info:
            raise ValueError(
                "Could not retrieve video information."
            )

        if progress:
            progress.update(
                label=f"💬 Collecting up to {comment_limit:,} comments...",
                state="running",
                expanded=False,
            )

        comments = get_video_comments(
            content_id,
            max_comments=comment_limit,
        )

        if not comments:
            raise ValueError(
                "No comments were available "
                "for this video."
            )

        if progress:
            progress.update(
                label=f"📝 {len(comments):,} comments collected · Ready for analysis",
                state="running",
                expanded=False,
            )

        return (
            content_type,
            info,
            comments,
            [],
        )

    # --------------------------------------------------------
    # CHANNEL
    # --------------------------------------------------------

    if progress:
        progress.update(
            label="📺 Fetching channel information...",
            state="running",
            expanded=False,
        )

    channel_info = get_channel_info(
        content_id
    )

    if not channel_info:
        raise ValueError(
            "Could not retrieve channel information."
        )

    if progress:
        progress.update(
            label="🎥 Finding recent videos...",
            state="running",
            expanded=False,
        )

    videos = get_channel_videos(
        content_id,
        max_videos=10,
    )

    if not videos:
        raise ValueError(
            "No recent videos were found "
            "for this channel."
        )

    per_video = max(
        1,
        math.ceil(
            comment_limit / len(videos)
        ),
    )

    comments = []

    for index, video in enumerate(videos):

        if len(comments) >= comment_limit:
            break

        video_id = video.get(
            "video_id"
        )

        video_title = video.get(
            "title",
            "Untitled video",
        )

        if progress:
            progress.update(
                label=f"🎥 Collecting comments · Video {index + 1}/{len(videos)}",
                state="running",
                expanded=False,
            )

        remaining = (
            comment_limit
            - len(comments)
        )

        amount = min(
            per_video,
            remaining,
        )

        video_comments = get_video_comments(
            video_id,
            max_comments=amount,
        )

        for comment in video_comments:

            comment["video_id"] = video_id

            comment["video_title"] = (
                video_title
            )

        comments.extend(
            video_comments
        )

    comments = comments[
        :comment_limit
    ]

    if not comments:
        raise ValueError(
            "No comments were available "
            "from the recent channel videos."
        )

    if progress:
        progress.update(
            label=f"💬 {len(comments):,} channel comments collected · Ready for analysis",
            state="running",
            expanded=False,
        )

    return (
        content_type,
        channel_info,
        comments,
        videos,
    )

# ============================================================
# MULTI-URL ANALYSIS / COMPARISON
# ============================================================

def parse_youtube_urls(raw_text, minimum=1, maximum=5):
    """Parse and validate a batch of YouTube URLs without silently dropping entries."""
    text = (raw_text or "").strip()

    if not text:
        raise ValueError(f"Please enter at least {minimum} YouTube URL(s).")

    # Accept one URL per line, comma-separated URLs, or pasted URLs with bullets.
    candidates = re.findall(
        r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/[^\s,]+",
        text,
        flags=re.IGNORECASE,
    )

    if not candidates:
        candidates = [line.strip().strip("-•,") for line in text.splitlines() if line.strip()]

    urls = []
    for raw_url in candidates:
        url = raw_url.strip().strip("\"'(),[]{}<>.,")
        if url and url not in urls:
            urls.append(url)

    if len(urls) < minimum:
        raise ValueError(
            f"Only {len(urls)} YouTube URL(s) were detected. "
            f"Please enter exactly {minimum if minimum == maximum else f'{minimum}–{maximum}'} valid URL(s)."
        )

    if len(urls) > maximum:
        raise ValueError(f"Please enter no more than {maximum} YouTube URLs at a time.")

    return urls


def analyze_source(url, comment_limit, transformer, progress=None):
    content_type, info, comments, videos = analyze_youtube_url(
        url,
        comment_limit,
        progress,
    )

    # Main sentiment pipeline: NO sarcasm arguments here.
    result = run_complete_analysis(
        comments,
        transformer_classifier=transformer,
        use_transformer=True,
    )

    result["source_url"] = url
    result["content_type"] = content_type
    result["content_info"] = info
    result["channel_videos"] = videos
    return result


def build_comparison_row(result, index):
    info = result.get("content_info", {}) or {}
    comments = result.get("comments", pd.DataFrame())
    summary = result.get("summary", pd.DataFrame())

    if result.get("content_type") == "video":
        title = info.get("title", "YouTube Video")
    else:
        title = info.get("title", info.get("channel_title", "YouTube Channel"))

    def pct(sentiment):
        if summary is None or summary.empty or "sentiment" not in summary.columns:
            return 0.0
        row = summary[summary["sentiment"] == sentiment]
        return float(row.iloc[0]["percentage"]) if not row.empty else 0.0

    stats = result.get("statistics", {}) or {}
    return {
        "Source": f"{index + 1}. {title or 'Source'}",
        "Type": str(result.get("content_type", "")).title(),
        "Comments Analysed": len(comments),
        "Positive %": round(pct("positive"), 1),
        "Neutral %": round(pct("neutral"), 1),
        "Negative %": round(pct("negative"), 1),
        "Average Likes": round(float(stats.get("average_likes", 0) or 0), 1),
        "URL": result.get("source_url", ""),
    }


def show_comparison_results(results, comparison_type):
    if not results:
        return

    section_title(" Comparison Overview")
    st.write(
        "Each source is analysed independently, then compared side-by-side. "
        "Sentiment percentages are calculated from the comments collected for that source."
    )

    rows = [build_comparison_row(r, i) for i, r in enumerate(results)]
    comparison_df = pd.DataFrame(rows)

    total_comments = int(comparison_df["Comments Analysed"].sum())
    avg_positive = float(comparison_df["Positive %"].mean())
    avg_negative = float(comparison_df["Negative %"].mean())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Sources Compared", len(results))
    with c2:
        metric_card("Comments Analysed", format_number(total_comments))
    with c3:
        metric_card("Avg. Positive %", f"{avg_positive:.1f}%")
    with c4:
        metric_card("Avg. Negative %", f"{avg_negative:.1f}%")

    chart_df = comparison_df.melt(
        id_vars=["Source"],
        value_vars=["Positive %", "Neutral %", "Negative %"],
        var_name="Sentiment",
        value_name="Percentage",
    )
    chart_df["Sentiment"] = chart_df["Sentiment"].str.replace(" %", "", regex=False)

    fig = px.bar(
        chart_df,
        x="Source",
        y="Percentage",
        color="Sentiment",
        barmode="group",
        text="Percentage",
        color_discrete_map={
            "Positive": SENTIMENT_COLORS["positive"],
            "Neutral": SENTIMENT_COLORS["neutral"],
            "Negative": SENTIMENT_COLORS["negative"],
        },
        title=("Creator" if comparison_type == "channel" else "Video") + " Sentiment Comparison",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        yaxis_title="Percentage of analysed comments",
        xaxis_title="",
        height=500,
        legend_title="Sentiment",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Side-by-side results")
    st.dataframe(
        comparison_df.drop(columns=["URL"]),
        use_container_width=True,
        hide_index=True,
    )

    engagement_df = comparison_df[["Source", "Average Likes"]].copy()
    fig2 = px.bar(
        engagement_df,
        x="Source",
        y="Average Likes",
        text="Average Likes",
        title="Average Comment Likes by Source",
    )
    fig2.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig2.update_layout(height=420, xaxis_title="", yaxis_title="Average likes")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Individual source reports")
    tab_labels = []
    for i, result in enumerate(results):
        label = build_comparison_row(result, i)["Source"]
        tab_labels.append(label[:40])

    tabs = st.tabs(tab_labels)
    for idx, (tab, result) in enumerate(zip(tabs, results)):
        with tab:
            info = result.get("content_info", {}) or {}
            content_type = result.get("content_type")
            show_overview(result, info, content_type)
            if content_type == "channel":
                show_channel_breakdown(result, result.get("channel_videos", []))
            show_topics(result, key_suffix=f"_comparison_{idx}")
            show_engagement(result)
            show_comments(result, key_suffix=f"_comparison_{idx}")
            show_models(result)



# ============================================================
# CREATOR COMPARISON
# ============================================================

def comparison_sentiment_chart(metrics):
    if metrics.empty:
        return

    plot_df = metrics[
        ["creator", "positive_pct", "neutral_pct", "negative_pct"]
    ].melt(
        id_vars="creator",
        var_name="sentiment",
        value_name="percentage",
    )

    plot_df["sentiment"] = plot_df["sentiment"].str.replace(
        "_pct", "", regex=False
    )

    fig = px.bar(
        plot_df,
        x="creator",
        y="percentage",
        color="sentiment",
        barmode="group",
        text="percentage",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={
            "sentiment": ["positive", "neutral", "negative"]
        },
        title="Sentiment composition by creator",
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )
    fig.update_layout(
        yaxis_title="Comments (%)",
        xaxis_title="",
        height=470,
    )
    st.plotly_chart(fig, width="stretch")


def comparison_engagement_chart(metrics, metric):
    if metrics.empty:
        return

    fig = px.bar(
        metrics,
        x="creator",
        y=metric,
        text=metric,
        title=metric.replace("_", " ").title(),
    )

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
    )
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        height=430,
    )
    st.plotly_chart(fig, width="stretch")


def show_creator_comparison(result):
    metrics = result["metrics"]
    video_sentiment = result["video_sentiment"]
    keywords = result["keywords"]

    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-kicker">Creator benchmark</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">How the audiences compare</div>',
        unsafe_allow_html=True,
    )

    if metrics.empty:
        st.info("No comparison results are available.")
        return

    st.dataframe(
        metrics[
            [
                "creator",
                "comments_analyzed",
                "videos_analyzed",
                "positive_pct",
                "neutral_pct",
                "negative_pct",
                "average_comment_likes",
                "model_agreement_pct",
            ]
        ].rename(
            columns={
                "creator": "Creator",
                "comments_analyzed": "Comments",
                "videos_analyzed": "Videos",
                "positive_pct": "Positive %",
                "neutral_pct": "Neutral %",
                "negative_pct": "Negative %",
                "average_comment_likes": "Avg. Comment Likes",
                "model_agreement_pct": "Model Agreement %",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    left, right = st.columns(2)

    with left:
        comparison_sentiment_chart(metrics)

    with right:
        metric = st.selectbox(
            "Engagement metric",
            [
                "average_comment_likes",
                "average_video_views",
                "average_video_likes",
            ],
            format_func=lambda value: value.replace("_", " ").title(),
            key="comparison_engagement_metric",
        )
        comparison_engagement_chart(metrics, metric)

    section_title("Video-level sentiment", "video")

    if not video_sentiment.empty:
        video_display = video_sentiment[
            [
                "creator",
                "video_title",
                "positive_pct",
                "neutral_pct",
                "negative_pct",
                "total",
            ]
        ].rename(
            columns={
                "creator": "Creator",
                "video_title": "Video",
                "positive_pct": "Positive %",
                "neutral_pct": "Neutral %",
                "negative_pct": "Negative %",
                "total": "Comments",
            }
        )

        st.dataframe(
            video_display,
            width="stretch",
            hide_index=True,
            height=420,
        )

    section_title("Distinctive audience vocabulary", "search")

    if not keywords.empty:
        keyword_display = keywords.pivot(
            index="rank",
            columns="creator",
            values="keyword",
        ).reset_index(drop=True)

        st.dataframe(
            keyword_display,
            width="stretch",
            hide_index=False,
        )

    st.caption(
        "Comparison results are descriptive. Each creator is sampled with the same number "
        "of videos and comments per video, so raw counts should be interpreted alongside "
        "percentages and engagement statistics."
    )



# ============================================================
# ANALYSIS DISPLAY
# ============================================================

def show_overview(
    result,
    info,
    content_type,
):

    comments = result["comments"]
    summary = result["summary"]
    stats = result["statistics"]

    positive_pct = safe_percentage(
        summary,
        "positive",
    )

    neutral_pct = safe_percentage(
        summary,
        "neutral",
    )

    negative_pct = safe_percentage(
        summary,
        "negative",
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    title = info.get(
        "title",
        info.get(
            "channel_title",
            "YouTube Analysis",
        ),
    )

    st.markdown(
        f"""
        <div class="hero">
            <div class="badge">
                {"VIDEO ANALYSIS" if content_type == "video"
                 else "CHANNEL ANALYSIS"}
            </div>
            <h1>{title}</h1>
            <p>
                Audience intelligence based on
                {len(comments):,} analysed comments.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # VIDEO / CHANNEL INFO
    # --------------------------------------------------------

    if content_type == "video":

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "👁 Views",
                format_number(
                    info.get("views", 0)
                ),
            )

        with c2:
            metric_card(
                "👍 Likes",
                format_number(
                    info.get("likes", 0)
                ),
            )

        with c3:
            metric_card(
                "💬 Comments Available",
                format_number(
                    info.get(
                        "comment_count",
                        len(comments),
                    )
                ),
            )

        with c4:
            metric_card(
                "🧠 Comments Analysed",
                format_number(
                    len(comments)
                ),
            )

    else:

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "👥 Subscribers",
                format_number(
                    info.get(
                        "subscribers",
                        0,
                    )
                ),
            )

        with c2:
            metric_card(
                "👁 Channel Views",
                format_number(
                    info.get(
                        "views",
                        0,
                    )
                ),
            )

        with c3:
            metric_card(
                "🎥 Videos",
                format_number(
                    info.get(
                        "videos",
                        0,
                    )
                ),
            )

        with c4:
            metric_card(
                "💬 Analysed Comments",
                format_number(
                    len(comments)
                ),
            )

    st.write("")

    # --------------------------------------------------------
    # SENTIMENT KPI
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "😊 Positive",
            f"{positive_pct:.1f}%",
        )

    with c2:
        metric_card(
            "😐 Neutral",
            f"{neutral_pct:.1f}%",
        )

    with c3:
        metric_card(
            "😞 Negative",
            f"{negative_pct:.1f}%",
        )

    with c4:
        metric_card(
            "❤️ Average Likes",
            format_number(
                stats.get(
                    "average_likes",
                    0,
                )
            ),
        )

    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    section_title(
        "📊 Audience Mood"
    )

    left, right = st.columns(
        [1, 1]
    )

    with left:
        sentiment_donut(
            summary
        )

    with right:
        sentiment_bar(
            summary
        )

    # --------------------------------------------------------
    # INSIGHTS
    # --------------------------------------------------------

    section_title(
        "💡 What the Audience Is Saying"
    )

    for insight in result["insights"]:

        st.markdown(
            f"""
            <div class="insight-card">
                <span class="insight-icon">{svg_icon("lightbulb", size=18, color="#D95F43")}</span>{insight}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TOPICS / KEYWORDS
# ============================================================

def show_topics(result, key_suffix=""):

    section_title(
        "🔑 Topics & Language"
    )

    sentiment = st.selectbox(
        "Analyse keywords for",
        [
            "all",
            "positive",
            "neutral",
            "negative",
        ],
        format_func=lambda x: x.title(),
        key=f"keyword_sentiment{key_suffix}",
    )

    keywords = result[
        "keywords"
    ].get(
        sentiment,
        pd.DataFrame(),
    )

    tfidf = result[
        "tfidf"
    ].get(
        sentiment,
        pd.DataFrame(),
    )

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Most Frequent Keywords"
        )

        if not keywords.empty:

            fig = px.bar(
                keywords.sort_values(
                    "frequency"
                ),
                x="frequency",
                y="word",
                orientation="h",
                title="Keyword Frequency",
            )

            fig.update_layout(
                height=500,
                yaxis_title="",
                xaxis_title="Frequency",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

            st.dataframe(
                keywords,
                width="stretch",
                hide_index=True,
            )

        else:
            st.info(
                "No keywords available."
            )

    with right:

        st.subheader(
            "TF-IDF Terms"
        )

        if not tfidf.empty:

            fig = px.bar(
                tfidf.sort_values(
                    "tfidf"
                ),
                x="tfidf",
                y="term",
                orientation="h",
                title="TF-IDF Importance",
            )

            fig.update_layout(
                height=500,
                yaxis_title="",
                xaxis_title="TF-IDF Score",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

            st.dataframe(
                tfidf,
                width="stretch",
                hide_index=True,
            )

        else:
            st.info(
                "Not enough text for TF-IDF."
            )


# ============================================================
# ENGAGEMENT
# ============================================================

def show_engagement(result):

    st.markdown(
        """
        <div class="section-header">
            <div class="section-title">Engagement Intelligence</div>
            <div class="section-subtitle">
                Understand how audience sentiment relates to comment engagement.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    engagement = result.get(
        "engagement",
        pd.DataFrame()
    )

    statistics = result.get(
        "statistics",
        {}
    )

    if engagement.empty:
        st.info("No engagement data available.")
        return

    engagement = engagement.copy()

    # Make sure numeric columns exist
    for column in [
        "comments",
        "average_likes",
        "total_likes"
    ]:

        if column not in engagement.columns:
            engagement[column] = 0

        engagement[column] = pd.to_numeric(
            engagement[column],
            errors="coerce"
        ).fillna(0)

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    total_likes = statistics.get(
        "total_likes",
        int(engagement["total_likes"].sum())
    )

    average_likes = statistics.get(
        "average_likes",
        float(
            engagement["average_likes"].mean()
        )
    )

    total_comments = statistics.get(
        "total_comments",
        int(engagement["comments"].sum())
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Likes",
            format_number(total_likes)
        )

    with col2:
        st.metric(
            "Average Likes / Comment",
            f"{average_likes:,.1f}"
        )

    with col3:
        st.metric(
            "Comments Analyzed",
            format_number(total_comments)
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # CHART 1 — TOTAL LIKES
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            engagement,
            x="sentiment",
            y="total_likes",
            text="total_likes",
            title="Total Likes by Sentiment",
            labels={
                "sentiment": "Sentiment",
                "total_likes": "Total Likes"
            },
            color="sentiment",
            color_discrete_map={
                "positive": "#22c55e",
                "neutral": "#94a3b8",
                "negative": "#ef4444"
            }
        )

        fig.update_traces(
            texttemplate="%{text:,}",
            textposition="outside"
        )

        fig.update_layout(
            height=430,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    # --------------------------------------------------------
    # CHART 2 — AVERAGE LIKES
    # --------------------------------------------------------

    with col2:

        fig = px.bar(
            engagement,
            x="sentiment",
            y="average_likes",
            text="average_likes",
            title="Average Likes per Comment",
            labels={
                "sentiment": "Sentiment",
                "average_likes": "Average Likes"
            },
            color="sentiment",
            color_discrete_map={
                "positive": "#22c55e",
                "neutral": "#94a3b8",
                "negative": "#ef4444"
            }
        )

        fig.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside"
        )

        fig.update_layout(
            height=430,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    # --------------------------------------------------------
    # CHART 3 — COMMENT VOLUME
    # --------------------------------------------------------

    fig = px.bar(
        engagement,
        x="sentiment",
        y="comments",
        text="comments",
        title="Comment Volume by Sentiment",
        labels={
            "sentiment": "Sentiment",
            "comments": "Number of Comments"
        },
        color="sentiment",
        color_discrete_map={
            "positive": "#22c55e",
            "neutral": "#94a3b8",
            "negative": "#ef4444"
        }
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # --------------------------------------------------------
    # DATA TABLE
    # --------------------------------------------------------

    st.markdown(
        "### Engagement Breakdown"
    )

    display_df = engagement.copy()

    display_df["sentiment"] = (
        display_df["sentiment"]
        .str.title()
    )

    display_df["average_likes"] = (
        display_df["average_likes"]
        .round(2)
    )

    display_df["total_likes"] = (
        display_df["total_likes"]
        .astype(int)
    )

    display_df.columns = [
        "Sentiment",
        "Comments",
        "Average Likes",
        "Total Likes"
    ]

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True
    )

# ============================================================
# COMMENTS
# ============================================================

def show_comments(result, key_suffix=""):

    section_title(
        "💬 Comment Explorer"
    )

    data = result[
        "comments"
    ].copy()

    if data.empty:
        st.info(
            "No comments available."
        )
        return

    sentiment_filter = st.multiselect(
        "Sentiment",
        [
            "positive",
            "neutral",
            "negative",
        ],
        default=[
            "positive",
            "neutral",
            "negative",
        ],
        key=f"comments_sentiment_filter{key_suffix}",
    )

    search = st.text_input(
        "Search comments",
        key=f"comments_search{key_suffix}",
    )

    min_likes = st.slider(
        "Minimum likes",
        min_value=0,
        max_value=int(
            max(
                1,
                data["like_count"]
                .fillna(0)
                .max(),
            )
        ),
        value=0,
        key=f"comments_min_likes{key_suffix}",
    )

    filtered = data[
        data["vader_sentiment"]
        .isin(sentiment_filter)
    ].copy()

    filtered["like_count"] = pd.to_numeric(
        filtered["like_count"],
        errors="coerce",
    ).fillna(0)

    filtered = filtered[
        filtered["like_count"]
        >= min_likes
    ]

    if search:

        filtered = filtered[
            filtered["comment_text"]
            .fillna("")
            .str.contains(
                search,
                case=False,
                na=False,
            )
        ]

    display_columns = [
        "comment_text",
        "vader_sentiment",
        "like_count",
    ]

    if "transformer_sentiment" in filtered.columns:
        display_columns += [
            "transformer_sentiment",
            "transformer_score",
        ]

    if "video_title" in filtered.columns:
        display_columns.insert(
            1,
            "video_title",
        )

    display_columns = [
        col
        for col in display_columns
        if col in filtered.columns
    ]

    st.write(
        f"Showing **{len(filtered):,}** comments"
    )

    st.dataframe(
        filtered[
            display_columns
        ],
        width="stretch",
        hide_index=True,
        height=550,
    )

    csv = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Filtered Comments",
        data=csv,
        file_name="youtube_pulse_comments.csv",
        mime="text/csv",
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

def show_models(result):

    section_title(
        "🧠 VADER vs Transformer"
    )

    data = result[
        "comments"
    ]

    if (
        "transformer_sentiment"
        not in data.columns
    ):
        st.info(
            "Transformer results are unavailable."
        )
        return

    agreement = result[
        "model_agreement"
    ]

    c1, c2 = st.columns(2)

    with c1:
        metric_card(
            "Model Agreement",
            f"{agreement:.1f}%"
            if agreement is not None
            else "N/A",
        )

    with c2:
        metric_card(
            "Comments Compared",
            format_number(
                len(data)
            ),
        )

    comparison = pd.DataFrame(
        {
            "VADER": data[
                "vader_sentiment"
            ].value_counts(),

            "Transformer": data[
                "transformer_sentiment"
            ].value_counts(),
        }
    ).fillna(0)

    comparison = comparison.reindex(
        [
            "positive",
            "neutral",
            "negative",
        ]
    ).fillna(0)

    comparison = (
        comparison
        .reset_index()
        .rename(
            columns={
                "index": "sentiment"
            }
        )
    )

    melted = comparison.melt(
        id_vars="sentiment",
        var_name="model",
        value_name="count",
    )

    fig = px.bar(
        melted,
        x="sentiment",
        y="count",
        color="model",
        barmode="group",
        title="Sentiment Predictions by Model",
    )

    fig.update_layout(
        height=420,
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.subheader(
        "Prediction Comparison"
    )

    st.dataframe(
        data[
            [
                "comment_text",
                "vader_sentiment",
                "transformer_sentiment",
                "transformer_score",
            ]
        ],
        width="stretch",
        hide_index=True,
        height=500,
    )


# ============================================================
# CHANNEL BREAKDOWN
# ============================================================

def show_channel_breakdown(
    result,
    videos,
):

    section_title(
        "📺 Channel Video Breakdown"
    )

    comments = result[
        "comments"
    ].copy()

    if (
        not videos
        or "video_id" not in comments.columns
    ):
        st.info(
            "Video-level channel information "
            "is not available."
        )
        return

    rows = []

    for video in videos:

        video_id = video.get(
            "video_id"
        )

        subset = comments[
            comments["video_id"]
            == video_id
        ]

        if subset.empty:
            continue

        counts = (
            subset["vader_sentiment"]
            .value_counts()
        )

        total = len(subset)

        rows.append(
            {
                "video_id": video_id,
                "video_title": video.get(
                    "title",
                    "Untitled",
                ),
                "comments": total,
                "positive": round(
                    counts.get(
                        "positive",
                        0,
                    )
                    / total
                    * 100,
                    1,
                ),
                "neutral": round(
                    counts.get(
                        "neutral",
                        0,
                    )
                    / total
                    * 100,
                    1,
                ),
                "negative": round(
                    counts.get(
                        "negative",
                        0,
                    )
                    / total
                    * 100,
                    1,
                ),
            }
        )

    video_df = pd.DataFrame(
        rows
    )

    if video_df.empty:
        st.info(
            "No video-level data available."
        )
        return

    chart_data = video_df.melt(
        id_vars=[
            "video_title",
            "comments",
        ],
        value_vars=[
            "positive",
            "neutral",
            "negative",
        ],
        var_name="sentiment",
        value_name="percentage",
    )

    fig = px.bar(
        chart_data,
        x="percentage",
        y="video_title",
        color="sentiment",
        orientation="h",
        barmode="stack",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Across Recent Videos",
    )

    fig.update_layout(
        height=max(
            450,
            len(video_df) * 55,
        ),
        xaxis_title="Percentage",
        yaxis_title="",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    display = video_df.drop(
        columns=["video_id"]
    )

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# COMPLETE RESULTS
# ============================================================

def show_analysis_results():

    result = (
        st.session_state
        .analysis_result
    )

    info = (
        st.session_state
        .content_info
    )

    content_type = (
        st.session_state
        .content_type
    )

    if result is None:
        return

    show_overview(
        result,
        info,
        content_type,
    )

    tabs = st.tabs(
        [
            "Overview",
            "Topics",
            "Engagement",
            "Comments",
            "Models",
        ]
    )

    with tabs[0]:

        if content_type == "channel":

            show_channel_breakdown(
                result,
                st.session_state.channel_videos,
            )

    with tabs[1]:

        show_topics(
            result
        )

    with tabs[2]:

        show_engagement(
            result
        )

    with tabs[3]:

        show_comments(
            result
        )

    with tabs[4]:

        show_models(
            result
        )


# ============================================================
# DEMO DATASET DASHBOARD
# ============================================================

def show_demo_dashboard():

    st.markdown(
        """
        <div class="hero">
            <div class="badge">
                RESEARCH DATASET
            </div>
            <h1>Demo Dataset Intelligence</h1>
            <p>
                Explore the project's 750-comment
                sentiment-analysis dataset,
                domains, creators, videos,
                keywords and model validation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data = load_demo_data()

    if data.empty:
        st.error(
            "Demo dataset not found."
        )
        return

    # --------------------------------------------------------
    # DATA PREPARATION
    # --------------------------------------------------------

    if "like_count" in data.columns:

        data["like_count"] = pd.to_numeric(
            data["like_count"],
            errors="coerce",
        ).fillna(0)

    total_comments = len(data)

    creators = (
        data["creator"].nunique()
        if "creator" in data.columns
        else 0
    )

    videos = (
        data["video_id"].nunique()
        if "video_id" in data.columns
        else 0
    )

    domains = (
        data["domain"].nunique()
        if "domain" in data.columns
        else 0
    )

    avg_likes = (
        data["like_count"].mean()
        if "like_count" in data.columns
        else 0
    )

    summary = (
        data["vader_sentiment"]
        .value_counts()
        .reindex(
            [
                "positive",
                "neutral",
                "negative",
            ],
            fill_value=0,
        )
        .reset_index()
    )

    summary.columns = [
        "sentiment",
        "count",
    ]

    summary["percentage"] = (
        summary["count"]
        / total_comments
        * 100
    ).round(2)

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        metric_card(
            "💬 Comments",
            format_number(
                total_comments
            ),
        )

    with c2:
        metric_card(
            "👤 Creators",
            format_number(
                creators
            ),
        )

    with c3:
        metric_card(
            "🎥 Videos",
            format_number(
                videos
            ),
        )

    with c4:
        metric_card(
            "🌐 Domains",
            format_number(
                domains
            ),
        )

    with c5:
        metric_card(
            "❤️ Avg Likes",
            format_number(
                avg_likes
            ),
        )

    st.write("")

    # --------------------------------------------------------
    # DEMO TABS
    # --------------------------------------------------------

    tabs = st.tabs(
        [
            "Overview",
            "Domains",
            "Creators",
            "Videos",
            "Keywords",
            "Comments",
            "Model Validation",
            "Error Analysis",
        ]
    )

    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

    with tabs[0]:

        left, right = st.columns(2)

        with left:

            sentiment_donut(
                summary,
                "Overall Dataset Sentiment",
            )

        with right:

            sentiment_bar(
                summary
            )

        section_title(
            "Dataset Insights"
        )

        positive = safe_percentage(
            summary,
            "positive",
        )

        neutral = safe_percentage(
            summary,
            "neutral",
        )

        negative = safe_percentage(
            summary,
            "negative",
        )

        insights = [
            f"The dataset contains {total_comments:,} analysed YouTube comments.",
            f"{positive:.1f}% of comments are classified as positive.",
            f"{neutral:.1f}% of comments are classified as neutral.",
            f"{negative:.1f}% of comments are classified as negative.",
            f"The dataset covers {domains} content domains, {creators} creators and {videos} videos.",
        ]

        for insight in insights:

            st.markdown(
                f"""
                <div class="insight-card">
                    <span class="insight-icon">{svg_icon("lightbulb", size=18, color="#D95F43")}</span>{insight}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # DOMAINS
    # --------------------------------------------------------

    with tabs[1]:

        if "domain" not in data.columns:

            st.info(
                "Domain information unavailable."
            )

        else:

            domain_summary = (
                data
                .groupby(
                    [
                        "domain",
                        "vader_sentiment",
                    ]
                )
                .size()
                .reset_index(
                    name="count"
                )
            )

            domain_total = (
                domain_summary
                .groupby("domain")
                ["count"]
                .transform("sum")
            )

            domain_summary[
                "percentage"
            ] = (
                domain_summary["count"]
                / domain_total
                * 100
            ).round(2)

            fig = px.bar(
                domain_summary,
                x="domain",
                y="percentage",
                color="vader_sentiment",
                barmode="stack",
                color_discrete_map=SENTIMENT_COLORS,
                title="Sentiment Distribution by Domain",
            )

            fig.update_layout(
                height=500,
                yaxis_title="Percentage",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

            st.dataframe(
                domain_summary,
                width="stretch",
                hide_index=True,
            )

    # --------------------------------------------------------
    # CREATORS
    # --------------------------------------------------------

    with tabs[2]:

        if "creator" not in data.columns:

            st.info(
                "Creator information unavailable."
            )

        else:

            creator_summary = (
                data
                .groupby(
                    [
                        "creator",
                        "vader_sentiment",
                    ]
                )
                .size()
                .reset_index(
                    name="comments"
                )
            )

            fig = px.bar(
                creator_summary,
                x="creator",
                y="comments",
                color="vader_sentiment",
                barmode="group",
                color_discrete_map=SENTIMENT_COLORS,
                title="Creator Sentiment Distribution",
            )

            fig.update_layout(
                height=550,
                xaxis_tickangle=-35,
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

            st.dataframe(
                creator_summary,
                width="stretch",
                hide_index=True,
            )

    # --------------------------------------------------------
    # VIDEOS
    # --------------------------------------------------------

    with tabs[3]:

        if "video_title" not in data.columns:

            st.info(
                "Video information unavailable."
            )

        else:

            video_summary = (
                data
                .groupby(
                    [
                        "video_title",
                        "vader_sentiment",
                    ]
                )
                .size()
                .reset_index(
                    name="comments"
                )
            )

            fig = px.bar(
                video_summary,
                x="comments",
                y="video_title",
                color="vader_sentiment",
                orientation="h",
                barmode="stack",
                color_discrete_map=SENTIMENT_COLORS,
                title="Sentiment Across Videos",
            )

            fig.update_layout(
                height=max(
                    500,
                    len(video_summary)
                    * 30,
                ),
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

            st.dataframe(
                video_summary,
                width="stretch",
                hide_index=True,
            )

    # --------------------------------------------------------
    # KEYWORDS
    # --------------------------------------------------------

    with tabs[4]:

        from src.sentiment_analyzer import (
            extract_keywords,
            extract_tfidf,
        )

        sentiment_choice = st.selectbox(
            "Sentiment",
            [
                "All",
                "Positive",
                "Neutral",
                "Negative",
            ],
            key="demo_keyword_sentiment",
        )

        selected = (
            None
            if sentiment_choice == "All"
            else sentiment_choice.lower()
        )

        keywords = extract_keywords(
            data,
            sentiment=selected,
            top_n=20,
        )

        tfidf = extract_tfidf(
            data,
            sentiment=selected,
            top_n=20,
        )

        left, right = st.columns(2)

        with left:

            st.subheader(
                "Top Keywords"
            )

            if not keywords.empty:

                fig = px.bar(
                    keywords.sort_values(
                        "frequency"
                    ),
                    x="frequency",
                    y="word",
                    orientation="h",
                    title="Most Frequent Terms",
                )

                fig.update_layout(
                    height=600
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

        with right:

            st.subheader(
                "TF-IDF"
            )

            if not tfidf.empty:

                fig = px.bar(
                    tfidf.sort_values(
                        "tfidf"
                    ),
                    x="tfidf",
                    y="term",
                    orientation="h",
                    title="Most Important Terms",
                )

                fig.update_layout(
                    height=600
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

    # --------------------------------------------------------
    # COMMENTS
    # --------------------------------------------------------

    with tabs[5]:

        sentiment_filter = st.multiselect(
            "Sentiment",
            [
                "positive",
                "neutral",
                "negative",
            ],
            default=[
                "positive",
                "neutral",
                "negative",
            ],
            key="demo_sentiment_filter",
        )

        search = st.text_input(
            "Search dataset comments",
            key="demo_search",
        )

        filtered = data[
            data["vader_sentiment"]
            .isin(sentiment_filter)
        ].copy()

        if search:

            filtered = filtered[
                filtered["comment_text"]
                .fillna("")
                .str.contains(
                    search,
                    case=False,
                    na=False,
                )
            ]

        st.write(
            f"Showing {len(filtered):,} comments"
        )

        columns = [
            "domain",
            "creator",
            "video_title",
            "comment_text",
            "like_count",
            "vader_sentiment",
        ]

        columns = [
            col
            for col in columns
            if col in filtered.columns
        ]

        st.dataframe(
            filtered[columns],
            width="stretch",
            hide_index=True,
            height=600,
        )

        csv = filtered.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Dataset Selection",
            data=csv,
            file_name="youtube_pulse_demo_filtered.csv",
            mime="text/csv",
        )

    # --------------------------------------------------------
    # MODEL VALIDATION
    # --------------------------------------------------------

    with tabs[6]:

        model_comparison = load_csv(
            "model_comparison.csv"
        )

        vader_validation = load_csv(
            "vader_validation_results.csv"
        )

        transformer_validation = load_csv(
            "transformer_validation_results.csv"
        )

        if not model_comparison.empty:

            st.subheader(
                "Model Performance"
            )

            st.dataframe(
                model_comparison,
                width="stretch",
                hide_index=True,
            )

            metric_columns = [
                col
                for col in [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                ]
                if col in model_comparison.columns
            ]

            if metric_columns:

                numeric = model_comparison[
                    metric_columns
                ].copy()

                if (
                    numeric.max().max()
                    <= 1.0
                ):
                    numeric = numeric * 100

                numeric["model"] = (
                    model_comparison
                    .iloc[:, 0]
                    .astype(str)
                )

                melted = numeric.melt(
                    id_vars="model",
                    var_name="metric",
                    value_name="score",
                )

                fig = px.bar(
                    melted,
                    x="metric",
                    y="score",
                    color="model",
                    barmode="group",
                    title="Model Performance Comparison",
                )

                fig.update_layout(
                    height=500,
                    yaxis_title="Score (%)",
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

        # VADER confusion matrix
        if not vader_validation.empty:

            human_col = (
                "human-sentiment"
                if "human-sentiment"
                in vader_validation.columns
                else "human"
            )

            if (
                human_col in vader_validation.columns
                and "vader_sentiment"
                in vader_validation.columns
            ):

                cm = confusion_matrix(
                    vader_validation[
                        human_col
                    ],
                    vader_validation[
                        "vader_sentiment"
                    ],
                    labels=[
                        "positive",
                        "neutral",
                        "negative",
                    ],
                )

                fig = px.imshow(
                    cm,
                    x=[
                        "positive",
                        "neutral",
                        "negative",
                    ],
                    y=[
                        "positive",
                        "neutral",
                        "negative",
                    ],
                    text_auto=True,
                    title="VADER Confusion Matrix",
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

        # Transformer confusion matrix
        if not transformer_validation.empty:

            human_col = (
                "human-sentiment"
                if "human-sentiment"
                in transformer_validation.columns
                else "human"
            )

            if (
                human_col
                in transformer_validation.columns
                and "transformer_sentiment"
                in transformer_validation.columns
            ):

                cm = confusion_matrix(
                    transformer_validation[
                        human_col
                    ],
                    transformer_validation[
                        "transformer_sentiment"
                    ],
                    labels=[
                        "positive",
                        "neutral",
                        "negative",
                    ],
                )

                fig = px.imshow(
                    cm,
                    x=[
                        "positive",
                        "neutral",
                        "negative",
                    ],
                    y=[
                        "positive",
                        "neutral",
                        "negative",
                    ],
                    text_auto=True,
                    title="Transformer Confusion Matrix",
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

    # --------------------------------------------------------
    # ERROR ANALYSIS
    # --------------------------------------------------------

    with tabs[7]:

        error_data = load_csv(
            "error_analysis.csv"
        )

        if error_data.empty:

            st.info(
                "Error analysis file not found."
            )

        else:

            if "error_category" in error_data.columns:

                counts = (
                    error_data[
                        "error_category"
                    ]
                    .value_counts()
                    .reset_index()
                )

                counts.columns = [
                    "category",
                    "count",
                ]

                fig = px.bar(
                    counts,
                    x="count",
                    y="category",
                    orientation="h",
                    title="Model Error Analysis",
                )

                fig.update_layout(
                    height=500
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

                fig = px.pie(
                    counts,
                    names="category",
                    values="count",
                    hole=0.55,
                    title="Error Category Distribution",
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

            st.dataframe(
                error_data,
                width="stretch",
                hide_index=True,
                height=600,
            )




# ============================================================
# ONE-PAGE DASHBOARD NAVIGATION
# ============================================================

st.markdown(
    """<div class="topbar">
    <div class="brand"><span class="brand-mark">▶</span><span>YouTube Pulse</span></div>
    <nav class="topnav">
        <a href="#overview">Overview</a>
        <a href="#analyze">Analyze</a>
        <a href="#compare">Compare Creators</a>
        <a href="#results">Results</a>
        <a href="#demo">Demo Dataset</a>
        <a href="#comment-analyzer">Comment Analyzer</a>
        <a href="#about">About</a>
    </nav>
</div>""",
    unsafe_allow_html=True,
)

st.markdown('<div id="overview" class="anchor-section"></div>', unsafe_allow_html=True)
st.markdown(
    """<div class="hero">
    <div class="badge">YOUTUBE AUDIENCE INTELLIGENCE</div>
    <h1>YouTube Pulse</h1>
    <p>Understand what audiences really think about YouTube content through sentiment, engagement, topics, keywords and contextual NLP analysis. Everything is now available on one scrollable dashboard.</p>
</div>""",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">Quick overview</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Explore the dashboard</div>', unsafe_allow_html=True)
demo = load_demo_data()
if not demo.empty:
    c1,c2,c3,c4=st.columns(4)
    with c1: metric_card("Comments", format_number(len(demo)))
    with c2: metric_card("Creators", format_number(demo["creator"].nunique()) if "creator" in demo.columns else "0")
    with c3: metric_card("Videos", format_number(demo["video_id"].nunique()) if "video_id" in demo.columns else "0")
    with c4: metric_card("Domains", format_number(demo["domain"].nunique()) if "domain" in demo.columns else "0")

st.markdown('<div id="analyze" class="anchor-section"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-kicker">Live YouTube analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Analyze any YouTube video or channel</div>', unsafe_allow_html=True)
st.write("Paste a YouTube URL and discover audience sentiment, engagement, topics, keywords and model-based insights.")

url=st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=... or https://www.youtube.com/@channel", key="one_page_url")
comment_limit=st.slider("Comments to analyze",100,5000,1000,100,help="Maximum number of comments collected for this analysis.",key="one_page_limit")

if st.button("Analyze Now",type="primary",width="stretch",key="one_page_analyze"):
    if not url.strip():
        st.warning("Please enter a YouTube URL.")
    else:
        try:
            with st.status("🔄 Analyzing YouTube content...", expanded=False) as status:
                content_type, info, comments, videos = analyze_youtube_url(
                    url, comment_limit, status
                )

                status.update(
                    label=f"🧠 Analyzing {len(comments):,} comments with VADER + Transformer...",
                    state="running",
                    expanded=False,
                )

                transformer = get_transformer()
                result = run_complete_analysis(
                    comments,
                    transformer_classifier=transformer,
                    use_transformer=True,
                )

                st.session_state.analysis_result = result
                st.session_state.content_info = info
                st.session_state.content_type = content_type
                st.session_state.channel_videos = videos

                status.update(
                    label=f"✨ Analysis complete · {len(comments):,} comments analyzed",
                    state="complete",
                    expanded=False,
                )
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

st.markdown('<div id="results" class="anchor-section"></div>', unsafe_allow_html=True)
if st.session_state.analysis_result is not None:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Your analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Audience results</div>', unsafe_allow_html=True)
    show_analysis_results()
else:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Results appear here</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Run an analysis to unlock the full report</div>', unsafe_allow_html=True)
    st.info("Your sentiment breakdown, model comparison, keywords, engagement, comments and insights will appear in this section after you analyze a YouTube URL.")


st.markdown('<div id="compare" class="anchor-section"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-kicker">Multi-creator analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Compare YouTube Creators</div>', unsafe_allow_html=True)
st.write(
    "Enter 2–3 YouTube creator links to compare audience sentiment, comment engagement, "
    "video-level sentiment and distinctive audience vocabulary."
)

cc1, cc2, cc3 = st.columns(3)

with cc1:
    compare_url_1 = st.text_input(
        "Creator 1",
        placeholder="https://www.youtube.com/@creator",
        key="compare_url_1",
    )

with cc2:
    compare_url_2 = st.text_input(
        "Creator 2",
        placeholder="https://www.youtube.com/@creator",
        key="compare_url_2",
    )

with cc3:
    compare_url_3 = st.text_input(
        "Creator 3 (optional)",
        placeholder="https://www.youtube.com/@creator",
        key="compare_url_3",
    )

cmp1, cmp2 = st.columns(2)

with cmp1:
    comparison_videos = st.slider(
        "Videos per creator",
        2,
        10,
        5,
        1,
        key="comparison_videos",
        help="The same number of videos is sampled for every creator.",
    )

with cmp2:
    comparison_comments = st.slider(
        "Comments per video",
        25,
        200,
        50,
        25,
        key="comparison_comments",
        help="The same number of comments is collected from every sampled video.",
    )

if st.button(
    "Compare Creators",
    type="primary",
    width="stretch",
    key="compare_creators_button",
):
    compare_urls = [
        compare_url_1,
        compare_url_2,
        compare_url_3,
    ]
    compare_urls = [
        value.strip()
        for value in compare_urls
        if value and value.strip()
    ]

    if len(compare_urls) < 2:
        st.warning("Enter at least two YouTube creator links.")
    elif len(compare_urls) > 3:
        st.warning("Compare two or three creators at a time.")
    else:
        try:
            with st.status(
                "🔄 Collecting and comparing creators...",
                expanded=True,
            ) as status:
                transformer = get_transformer()

                def comparison_progress(index, message):
                    st.write(f"**{message}**")

                comparison_result = compare_creators(
                    compare_urls,
                    max_videos=comparison_videos,
                    comments_per_video=comparison_comments,
                    transformer_classifier=transformer,
                    progress_callback=comparison_progress,
                )

                st.session_state.comparison_result = comparison_result

                status.update(
                    label="✅ Creator comparison complete",
                    state="complete",
                    expanded=False,
                )
        except Exception as error:
            st.error(f"Creator comparison failed: {error}")

if st.session_state.comparison_result is not None:
    show_creator_comparison(
        st.session_state.comparison_result
    )
else:
    st.info(
        "Use the controls above to build a balanced 2–3 creator comparison."
    )

st.markdown('<div id="demo" class="anchor-section"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-kicker">Research dataset</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Demo Dataset</div>', unsafe_allow_html=True)
st.write("Explore the 750-comment research dataset used for sentiment validation, keyword analysis, TF-IDF, engagement and model comparison.")
show_demo_dashboard()

st.markdown('<div id="comment-analyzer" class="anchor-section"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Comment Analyzer</div>', unsafe_allow_html=True)
st.write("Analyze the sentiment behind an individual YouTube comment.")

comment=st.text_area("Enter a YouTube comment",height=150,placeholder="Example: This video was absolutely amazing!",key="one_page_comment")
if st.button("Analyze Comment",type="primary",key="one_page_comment_analyze"):
    if not comment.strip():
        st.warning("Enter a comment first.")
    else:
        # Use the validated transformer model for the primary prediction.
        transformer = get_transformer()
        transformer_result = transformer(comment, truncation=True, max_length=512)[0]

        transformer_label = str(transformer_result.get("label", "neutral")).lower()
        transformer_score = float(transformer_result.get("score", 0.0))
        sentiment = transformer_label

        st.subheader(f"Detected sentiment: {sentiment.title()}")
        st.caption(f"Transformer prediction · confidence {transformer_score:.1%}")

        if sentiment == "positive":
            context = "The language carries a positive emotional signal, suggesting approval, enjoyment or appreciation."
        elif sentiment == "negative":
            context = "The language carries a negative emotional signal, suggesting criticism, frustration or dissatisfaction."
        else:
            context = "The language is relatively neutral, with limited positive or negative emotional intensity."

        st.markdown(
            f'<div class="insight-card"><span class="insight-icon">{svg_icon("lightbulb", size=18, color="#D95F43")}</span><strong>Context</strong><br>{context}</div>',
            unsafe_allow_html=True,
        )

st.markdown('<div id="about" class="anchor-section"></div>', unsafe_allow_html=True)
st.markdown('<div id="about" class="anchor-section"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-kicker">About the project</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">YouTube Pulse</div>', unsafe_allow_html=True)
st.write("An interactive YouTube audience sentiment intelligence platform for exploring sentiment, engagement, language, topics and model behaviour.")
c1,c2=st.columns(2)
with c1:
    st.subheader("Analysis Features")
    st.markdown("""
    - Video sentiment analysis
    - Channel sentiment analysis
    - VADER sentiment
    - Transformer sentiment
    - Keyword extraction
    - TF-IDF analysis
    - Engagement analysis
    - Comment exploration
    - Automatic insights
    - Video-level channel comparison
    """)
with c2:
    st.subheader("NLP Models")
    st.markdown("""
    **VADER**

    A lexicon and rule-based sentiment analysis approach.

    **Transformer**

    CardiffNLP's Twitter-RoBERTa sentiment model provides contextual sentiment predictions.

    The application compares the predictions produced by both models.
    """)

st.markdown('<div class="footer-note">YouTube Pulse • One-page audience intelligence dashboard</div>',unsafe_allow_html=True)