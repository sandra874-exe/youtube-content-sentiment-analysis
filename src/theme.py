"""
Pastel Data Canvas — design system for YouTube Pulse.

Everything visual lives here so app.py only has to call a few helpers:

    inject_theme()                      # once, right after st.set_page_config
    page_header(kicker, title, sub)     # replaces the old .hero banner
    metric_card(label, value, ...)      # KPI card with optional delta pill
    section_title(title, subtitle)      # section heading
    callout(text, kind)                 # soft insight box
    sidebar_brand(name, tagline)        # logo block at the top of the sidebar
    themed_chart(fig)                   # st.plotly_chart without Streamlit's own re-theming
    sentiment_split_bar(p, n, neg)      # thin 3-part sentiment bar (HTML)
    creator_cards(df)                   # "Target Creator Profiles" cards from the demo CSV

Tokens come from the Stitch DESIGN.md ("Pastel Data Canvas").
"""

from __future__ import annotations

import html
import re
from typing import Iterable, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ============================================================
# DESIGN TOKENS
# ============================================================

CANVAS = "#FAF8F5"
CARD = "#FFFFFF"
CARD_TINT = "#F7F5F0"
BORDER = "#EBE6DF"
TEXT = "#2D3142"
TEXT_2 = "#636978"
MUTED = "#9197A6"

CORAL = "#E07A5F"
CORAL_SOFT = "#F7A399"
PEACH = "#FCD5CE"
MINT = "#C7EAE4"
SAGE = "#6CA68A"
PERIWINKLE = "#BDD4E7"
SLATE_BLUE = "#7692B0"
LAVENDER = "#D6D2E8"
ROSE = "#F4B8C1"
BUTTER = "#E9C46A"

# Sentiment colours: pastel family, but dark enough to read on white.
SENTIMENT_COLORS = {
    "positive": SAGE,
    "neutral": "#9DB9D4",
    "negative": CORAL,
}

COLORWAY = [CORAL, SLATE_BLUE, SAGE, "#A9A3CF", BUTTER, ROSE]


# ============================================================
# CSS
# ============================================================

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=swap');

:root {{
    --canvas: {CANVAS};
    --card: {CARD};
    --card-tint: {CARD_TINT};
    --border: {BORDER};
    --text: {TEXT};
    --text-2: {TEXT_2};
    --muted: {MUTED};
    --coral: {CORAL};
    --coral-soft: {CORAL_SOFT};
    --peach: {PEACH};
    --mint: {MINT};
    --sage: {SAGE};
    --periwinkle: {PERIWINKLE};
    --lavender: {LAVENDER};
    --rose: {ROSE};
    --shadow-1: 0 4px 20px -2px rgba(118, 146, 176, 0.08);
    --shadow-2: 0 10px 28px -4px rgba(224, 122, 95, 0.12);
    --font-sans: "Plus Jakarta Sans", "Segoe UI", system-ui, sans-serif;
    --font-mono: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
}}

/* ---------- Base ---------- */
.stApp {{
    background: var(--canvas);
    color: var(--text);
}}
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5, .stApp button, .stApp input, .stApp textarea,
.stApp [data-baseweb="tab"], .stApp [data-baseweb="select"] {{
    font-family: var(--font-sans);
}}
.stApp h1, .stApp h2, .stApp h3 {{
    color: var(--text);
    letter-spacing: -0.015em;
    font-weight: 700;
}}
[data-testid="stHeader"] {{
    background: transparent;
}}
.block-container {{
    padding: 2.25rem 3rem 4rem;
    max-width: 1450px;
}}
hr {{
    border-color: var(--border) !important;
}}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #FDF1EB 0%, #FAF8F5 100%);
    border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0.5rem;
}}
.yp-brand {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0.25rem 0.25rem;
}}
.yp-brand-mark {{
    width: 40px; height: 40px;
    border-radius: 12px;
    background: var(--coral);
    color: #fff;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 6px 16px -4px rgba(224,122,95,0.5);
}}
.yp-brand-name {{
    font-weight: 700; font-size: 18px; line-height: 1.1; color: var(--text);
}}
.yp-brand-tag {{
    font-family: var(--font-mono);
    font-size: 10px; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--muted);
    margin-top: 2px;
}}
.yp-nav-label {{
    font-family: var(--font-mono);
    font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--muted);
    margin: 1rem 0 0.25rem 0.25rem;
}}

/* Sidebar nav: turn the radio group into a vertical menu */
[data-testid="stSidebar"] [role="radiogroup"] {{
    gap: 0.25rem;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label {{
    width: 100%;
    margin: 0;
    padding: 0.65rem 0.9rem;
    border-radius: 12px;
    cursor: pointer;
    align-items: center;
    transition: background .15s ease, box-shadow .15s ease;
}}
/* hide the radio circle (first child that is not the text container) */
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child:not(:has([data-testid="stMarkdownContainer"])) {{
    display: none;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {{
    background: rgba(224, 122, 95, 0.09);
}}
[data-testid="stSidebar"] [role="radiogroup"] > label p {{
    font-weight: 600;
    font-size: 15px;
    color: var(--text-2);
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
    background: var(--coral);
    box-shadow: 0 8px 20px -6px rgba(224, 122, 95, 0.55);
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p,
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked)::before {{
    color: #fff;
}}
/* Nav icons (Material Symbols ligatures) — order matches the NAV list in app.py */
[data-testid="stSidebar"] [role="radiogroup"] > label::before {{
    font-family: "Material Symbols Rounded";
    font-size: 21px;
    line-height: 1;
    font-weight: 400;
    margin-right: 0.7rem;
    color: var(--text-2);
    display: inline-block;
    white-space: nowrap;
    letter-spacing: normal;
    text-transform: none;
    -webkit-font-feature-settings: "liga";
    font-feature-settings: "liga";
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(1)::before {{ content: "home"; }}
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(2)::before {{ content: "travel_explore"; }}
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(3)::before {{ content: "dataset"; }}
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(4)::before {{ content: "chat"; }}
[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child(5)::before {{ content: "info"; }}
/* the widget's own "Navigation" label is replaced by .yp-nav-label */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
    display: none;
}}

/* ---------- Page header ---------- */
.yp-header {{
    margin: 0 0 1.75rem 0;
}}
.yp-kicker {{
    display: inline-flex; align-items: center; gap: .4rem;
    font-family: var(--font-mono);
    font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #9A442D;
    background: var(--peach);
    padding: 5px 12px;
    border-radius: 9999px;
}}
.yp-title {{
    font-size: 38px; line-height: 1.2; font-weight: 700;
    letter-spacing: -0.02em; color: var(--text);
    margin: 0.7rem 0 0.4rem 0;
}}
.yp-sub {{
    font-size: 16px; line-height: 1.6; color: var(--text-2);
    max-width: 760px; margin: 0;
}}
.yp-chips {{ display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; }}

/* ---------- Cards ---------- */
.metric-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    min-height: 132px;
    box-shadow: var(--shadow-1);
    transition: box-shadow .2s ease, transform .2s ease;
}}
.metric-card:hover {{
    box-shadow: var(--shadow-2);
    transform: translateY(-1px);
}}
.metric-top {{
    display: flex; align-items: center; justify-content: space-between; gap: .5rem;
}}
.metric-label {{
    font-family: var(--font-mono);
    font-size: 12px; font-weight: 500; letter-spacing: 0.04em;
    text-transform: uppercase; color: var(--text-2);
}}
.metric-icon {{
    width: 30px; height: 30px; border-radius: 10px;
    background: var(--card-tint);
    display: flex; align-items: center; justify-content: center;
    flex: none;
}}
.metric-icon .ms {{ font-size: 18px; }}
.metric-value {{
    font-size: 32px; line-height: 1.15; font-weight: 700;
    color: var(--text);
    margin-top: 10px;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.015em;
}}
.metric-foot {{
    display: flex; align-items: center; gap: .6rem;
    margin-top: 10px; min-height: 22px;
}}
.metric-hint {{
    font-size: 12px; color: var(--muted);
}}
.ms {{
    font-family: "Material Symbols Rounded";
    font-weight: 400; font-style: normal;
    line-height: 1; letter-spacing: normal; text-transform: none;
    display: inline-block; white-space: nowrap;
    -webkit-font-feature-settings: "liga"; font-feature-settings: "liga";
}}

.section-title {{
    font-size: 22px; font-weight: 700; color: var(--text);
    letter-spacing: -0.01em;
    margin: 2.25rem 0 0.25rem 0;
}}
.section-sub {{
    font-size: 14px; color: var(--text-2); margin: 0 0 0.9rem 0;
}}

.insight-card, .yp-callout {{
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--coral);
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 12px;
    color: var(--text);
    line-height: 1.6;
    box-shadow: var(--shadow-1);
}}
.yp-callout.info    {{ background: #F1F6FB; border-left-color: var(--periwinkle); }}
.yp-callout.success {{ background: #EEF8F4; border-left-color: var(--sage); }}
.yp-callout.warn    {{ background: #FFF4EF; border-left-color: var(--coral); }}
.yp-callout-title {{
    font-family: var(--font-mono);
    font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--text-2);
    margin-bottom: 4px;
}}

.video-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 12px;
    box-shadow: var(--shadow-1);
}}
.small-muted {{ color: var(--muted); font-size: 13px; }}

/* ---------- Pills / badges ---------- */
.badge, .yp-chip {{
    display: inline-block;
    padding: 5px 12px;
    border-radius: 9999px;
    background: var(--lavender);
    color: #322A4A;
    font-family: var(--font-mono);
    font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
}}
.yp-pill {{
    display: inline-flex; align-items: center; gap: 3px;
    padding: 3px 10px; border-radius: 9999px;
    font-family: var(--font-mono);
    font-size: 11px; font-weight: 600; letter-spacing: 0.03em;
}}
.yp-pill.pos {{ background: var(--mint); color: #2A5944; }}
.yp-pill.neg {{ background: var(--rose); color: #7A2533; }}
.yp-pill.neu {{ background: var(--periwinkle); color: #2D4964; }}
.yp-pill .ms {{ font-size: 14px; }}

/* ---------- Creator profile cards ---------- */
.yp-creator {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: var(--shadow-1);
    overflow: hidden;
    height: 100%;
}}
.yp-creator-body {{ padding: 18px 20px 14px; }}
.yp-creator-head {{
    display: flex; justify-content: space-between; align-items: flex-start; gap: .75rem;
}}
.yp-creator-name {{ font-weight: 700; font-size: 17px; color: var(--text); }}
.yp-creator-domain {{ font-size: 13px; color: var(--text-2); margin-top: 2px; }}
.yp-split-label {{
    display: flex; justify-content: space-between;
    font-family: var(--font-mono); font-size: 11px; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-2);
    margin: 16px 0 6px;
}}
.yp-split {{
    display: flex; height: 10px; border-radius: 9999px; overflow: hidden;
    background: var(--card-tint);
}}
.yp-split > span {{ display: block; height: 100%; }}
.yp-split .p {{ background: {SAGE}; }}
.yp-split .n {{ background: {PEACH}; }}
.yp-split .g {{ background: {CORAL}; }}
.yp-split-legend {{
    display: flex; justify-content: space-between;
    font-family: var(--font-mono); font-size: 10px; color: var(--muted);
    margin-top: 6px;
}}
.yp-creator-foot {{
    display: flex; justify-content: space-between;
    padding: 10px 20px;
    background: var(--card-tint);
    border-top: 1px solid var(--border);
    font-family: var(--font-mono); font-size: 11px; color: var(--text-2);
}}
.yp-creator-foot b {{ color: var(--text); }}

/* ---------- Tabs -> pill segmented control ---------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.25rem;
    background: var(--card-tint);
    border: 1px solid var(--border);
    padding: 0.3rem;
    border-radius: 9999px;
    width: fit-content;
    max-width: 100%;
    overflow-x: auto;
}}
.stTabs [data-baseweb="tab"] {{
    height: auto;
    padding: 0.5rem 1.1rem;
    border-radius: 9999px;
    background: transparent;
}}
.stTabs [data-baseweb="tab"] p {{
    font-weight: 600; font-size: 14px; color: var(--text-2);
}}
.stTabs [aria-selected="true"] {{
    background: var(--coral);
    box-shadow: 0 6px 16px -6px rgba(224, 122, 95, 0.6);
}}
.stTabs [aria-selected="true"] p {{ color: #fff; }}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ display: none; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.5rem; }}

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {{
    border-radius: 8px;
    font-weight: 600;
    border: 1px solid rgba(189, 212, 231, 0.6);
    background: rgba(189, 212, 231, 0.25);
    color: #495B70;
    transition: all .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background: rgba(189, 212, 231, 0.45);
    border-color: rgba(189, 212, 231, 0.9);
    color: #2D4964;
}}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
    background: var(--coral);
    border: 1px solid var(--coral);
    color: #fff;
}}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {{
    background: #d66c50;
    border-color: #d66c50;
    color: #fff;
    box-shadow: 0 4px 14px rgba(224, 122, 95, 0.3);
}}

/* ---------- Inputs ---------- */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"],
[data-baseweb="select"] > div {{
    background: #fff !important;
    border-radius: 8px !important;
    border-color: var(--border) !important;
}}
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within,
[data-baseweb="select"] > div:focus-within {{
    border-color: var(--coral) !important;
    box-shadow: 0 0 0 3px rgba(247, 163, 153, 0.35) !important;
}}
[data-baseweb="tag"] {{
    background: var(--lavender) !important;
    color: #322A4A !important;
    border-radius: 9999px !important;
}}
[data-baseweb="slider"] [role="slider"] {{
    background: #fff !important;
    border: 2px solid var(--coral) !important;
    box-shadow: 0 2px 8px rgba(224, 122, 95, 0.35);
}}
[data-testid="stSliderThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {{
    font-family: var(--font-mono);
}}

/* ---------- Native metric ---------- */
[data-testid="stMetric"] {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: var(--shadow-1);
}}
[data-testid="stMetricLabel"] p {{
    font-family: var(--font-mono);
    font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase;
    color: var(--text-2);
}}
[data-testid="stMetricValue"] {{
    font-weight: 700; font-variant-numeric: tabular-nums;
}}

/* ---------- Charts, tables, alerts ---------- */
[data-testid="stPlotlyChart"] {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.5rem;
    box-shadow: var(--shadow-1);
}}
[data-testid="stDataFrame"] {{
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: var(--shadow-1);
}}
[data-testid="stAlert"] {{
    border-radius: 16px;
}}
[data-testid="stAlert"] [data-baseweb="notification"],
[data-testid="stAlertContainer"] {{
    border-radius: 16px;
    border: 1px solid var(--border);
    box-shadow: none;
}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{ background: #F1F6FB; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{ background: #EEF8F4; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{ background: #FFF4EF; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{ background: #FDECEE; }}
[data-testid="stExpander"] details {{
    border: 1px solid var(--border);
    border-radius: 16px;
    background: var(--card);
}}
[data-testid="stStatusWidget"], [data-testid="stStatus"] {{
    border-radius: 16px;
}}

@media (max-width: 900px) {{
    .block-container {{ padding: 1.25rem 1rem 3rem; }}
    .yp-title {{ font-size: 28px; }}
}}
</style>
"""


def inject_theme() -> None:
    """Inject the design-system CSS and register the Plotly template. Call once per run."""
    st.markdown(_CSS, unsafe_allow_html=True)
    _register_plotly_template()


# ============================================================
# PLOTLY
# ============================================================

def _register_plotly_template() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="Plus Jakarta Sans, Segoe UI, sans-serif", size=13, color=TEXT_2),
        title=dict(font=dict(size=17, color=TEXT), x=0.02, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        margin=dict(l=24, r=24, t=64, b=24),
        xaxis=dict(
            gridcolor="#F0ECE4", linecolor=BORDER, zerolinecolor=BORDER,
            tickfont=dict(family="Space Grotesk, sans-serif", size=11, color=MUTED),
            title=dict(font=dict(size=12, color=TEXT_2)),
        ),
        yaxis=dict(
            gridcolor="#F0ECE4", linecolor=BORDER, zerolinecolor=BORDER,
            tickfont=dict(family="Space Grotesk, sans-serif", size=11, color=MUTED),
            title=dict(font=dict(size=12, color=TEXT_2)),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
            font=dict(size=12, color=TEXT_2), title=dict(text=""),
        ),
        hoverlabel=dict(
            bgcolor=TEXT, font=dict(family="Space Grotesk, sans-serif", color="#fff", size=12),
            bordercolor=TEXT,
        ),
    )
    pio.templates["pastel"] = template
    pio.templates.default = "pastel"


def themed_chart(fig, **kwargs) -> None:
    """st.plotly_chart with Streamlit's own re-theming switched off, so our template wins."""
    kwargs.pop("use_container_width", None)
    kwargs.setdefault("theme", None)
    kwargs.setdefault("config", {"displayModeBar": False})
    try:
        st.plotly_chart(fig, width="stretch", **kwargs)  # Streamlit >= 1.50
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kwargs)  # older versions


# ============================================================
# COMPONENT HELPERS
# ============================================================

# leading emoji -> Material Symbol name (used by metric_card so old labels keep working)
_ICON_MAP = {
    "👁": "visibility", "👍": "thumb_up", "💬": "forum", "🧠": "psychology",
    "👤": "person", "👥": "groups", "🎥": "movie", "🌐": "public",
    "❤": "favorite", "😊": "sentiment_satisfied", "😐": "sentiment_neutral",
    "😞": "sentiment_dissatisfied", "📊": "bar_chart", "🔑": "key",
    "💡": "lightbulb", "⚠": "warning",
}
_ICON_COLORS = {
    "sentiment_satisfied": SAGE, "sentiment_dissatisfied": CORAL,
    "sentiment_neutral": SLATE_BLUE, "favorite": CORAL, "thumb_up": SAGE,
}
_LEADING_JUNK = re.compile(r"^[^\w$%#(]+", flags=re.UNICODE)


def _split_label(label: str, icon: Optional[str]):
    """'👁 Views' -> ('Views', 'visibility'). Explicit icon wins."""
    label = str(label).strip()
    found = None
    for emoji, name in _ICON_MAP.items():
        if label.startswith(emoji):
            found = name
            break
    clean = _LEADING_JUNK.sub("", label).strip() or label
    return clean, (icon or found or "analytics")


def strip_emoji(text: str) -> str:
    return _LEADING_JUNK.sub("", str(text)).strip() or str(text)


def page_header(
    kicker: str,
    title: str,
    subtitle: Optional[str] = None,
    chips: Optional[Iterable[str]] = None,
) -> None:
    """Replacement for the old purple .hero banner."""
    sub = f'<p class="yp-sub">{subtitle}</p>' if subtitle else ""
    chip_html = ""
    if chips:
        chip_html = '<div class="yp-chips">' + "".join(
            f'<span class="yp-chip">{html.escape(str(c))}</span>' for c in chips
        ) + "</div>"
    st.markdown(
        f"""
<div class="yp-header">
  <span class="yp-kicker">{kicker}</span>
  <div class="yp-title">{title}</div>
  {sub}
  {chip_html}
</div>
""",
        unsafe_allow_html=True,
    )


def metric_card(
    label: str,
    value,
    delta: Optional[str] = None,
    delta_kind: str = "pos",
    hint: Optional[str] = None,
    icon: Optional[str] = None,
) -> None:
    """
    KPI card. Old calls  metric_card("👁 Views", "1.2M")  still work.
    delta_kind: "pos" (mint), "neg" (rose) or "neu" (periwinkle).
    """
    clean, icon_name = _split_label(label, icon)
    color = _ICON_COLORS.get(icon_name, CORAL)
    arrow = {"pos": "trending_up", "neg": "trending_down", "neu": "trending_flat"}.get(delta_kind, "trending_flat")
    delta_html = (
        f'<span class="yp-pill {delta_kind}"><span class="ms">{arrow}</span>{delta}</span>'
        if delta else ""
    )
    hint_html = f'<span class="metric-hint">{hint}</span>' if hint else ""
    foot = f'<div class="metric-foot">{delta_html}{hint_html}</div>' if (delta or hint) else ""
    st.markdown(
        f"""
<div class="metric-card">
  <div class="metric-top">
    <div class="metric-label">{clean}</div>
    <div class="metric-icon"><span class="ms" style="color:{color}">{icon_name}</span></div>
  </div>
  <div class="metric-value">{value}</div>
  {foot}
</div>
""",
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: Optional[str] = None) -> None:
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="section-title">{strip_emoji(title)}</div>{sub}',
        unsafe_allow_html=True,
    )


def callout(text: str, kind: str = "info", title: Optional[str] = None) -> None:
    """Soft insight box. kind: info | success | warn."""
    t = f'<div class="yp-callout-title">{title}</div>' if title else ""
    st.markdown(
        f'<div class="yp-callout {kind}">{t}{text}</div>',
        unsafe_allow_html=True,
    )


def sidebar_brand(name: str, tagline: str, nav_label: str = "Navigation") -> None:
    st.markdown(
        f"""
<div class="yp-brand">
  <div class="yp-brand-mark"><span class="ms" style="font-size:24px">play_arrow</span></div>
  <div>
    <div class="yp-brand-name">{name}</div>
    <div class="yp-brand-tag">{tagline}</div>
  </div>
</div>
<div class="yp-nav-label">{nav_label}</div>
""",
        unsafe_allow_html=True,
    )


def sentiment_split_bar(pos: float, neu: float, neg: float, show_legend: bool = True) -> str:
    """Return HTML for a 3-part sentiment bar. Values are percentages (0-100)."""
    legend = (
        f'<div class="yp-split-legend"><span>Pos {pos:.0f}%</span>'
        f'<span>Neu {neu:.0f}%</span><span>Neg {neg:.0f}%</span></div>'
        if show_legend else ""
    )
    return (
        f'<div class="yp-split">'
        f'<span class="p" style="width:{pos:.2f}%"></span>'
        f'<span class="n" style="width:{neu:.2f}%"></span>'
        f'<span class="g" style="width:{neg:.2f}%"></span>'
        f"</div>{legend}"
    )


def creator_cards(
    data: pd.DataFrame,
    creator_col: str = "creator",
    domain_col: str = "domain",
    video_col: str = "video_id",
    sentiment_col: str = "vader_sentiment",
    max_cards: int = 3,
) -> None:
    """
    'Target Creator Domain Profiles' row from the mockup, built from the demo CSV.
    Shows the first `max_cards` creators by comment volume, in rows of three.
    """
    if data is None or data.empty or creator_col not in data.columns or sentiment_col not in data.columns:
        return

    top = data[creator_col].value_counts().head(max_cards).index.tolist()
    cols = st.columns(3)
    for i, creator in enumerate(top):
        sub = data[data[creator_col] == creator]
        total = len(sub)
        shares = (
            sub[sentiment_col].value_counts(normalize=True)
            .reindex(["positive", "neutral", "negative"], fill_value=0) * 100
        )
        pos, neu, neg = float(shares["positive"]), float(shares["neutral"]), float(shares["negative"])
        domain = str(sub[domain_col].mode().iloc[0]) if domain_col in sub.columns and not sub[domain_col].dropna().empty else ""
        videos = int(sub[video_col].nunique()) if video_col in sub.columns else 0
        net = (pos - neg) / 100

        with cols[i % 3]:
            st.markdown(
                f"""
<div class="yp-creator">
  <div class="yp-creator-body">
    <div class="yp-creator-head">
      <div>
        <div class="yp-creator-name">{html.escape(str(creator))}</div>
        <div class="yp-creator-domain">{html.escape(domain)}</div>
      </div>
      <span class="yp-chip">{html.escape(domain.title()) if domain else "Creator"}</span>
    </div>
    <div class="yp-split-label"><span>Sentiment split</span><span>{pos:.0f}% / {neu:.0f}% / {neg:.0f}%</span></div>
    {sentiment_split_bar(pos, neu, neg)}
  </div>
  <div class="yp-creator-foot">
    <span>{videos} videos</span><span>{total:,} comments</span><span>Net <b>{net:+.2f}</b></span>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )