"""
Design tokens, CSS styling, and visual component helpers for Model Benchmark Studio.
"""

from __future__ import annotations
import streamlit as st


def inject_custom_css():
    """Inject modern CSS for sleek styling and responsive cards."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(34, 211, 238, 0.4);
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 9px;
        font-size: 0.74rem;
        font-weight: 600;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .badge-classical {
        background: rgba(96, 165, 250, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(96, 165, 250, 0.3);
    }
    .badge-ensemble {
        background: rgba(34, 211, 238, 0.15);
        color: #22D3EE;
        border: 1px solid rgba(34, 211, 238, 0.3);
    }
    .badge-winner {
        background: rgba(251, 191, 36, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }

    /* Header styling */
    .studio-header {
        margin-bottom: 24px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    }
    .studio-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #22D3EE 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .studio-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 4px;
    }

    /* Model card container */
    .card-box {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header(title: str, subtitle: str):
    """Render top header banner."""
    st.markdown(
        f"""
        <div class="studio-header">
            <h1 class="studio-title">{title}</h1>
            <div class="studio-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(title: str, value: str, sub: str = "", card_type: str = "normal"):
    """Render a KPI metric card."""
    border_style = ""
    if card_type == "winner":
        border_style = "border-color: rgba(251, 191, 36, 0.4);"

    st.markdown(
        f"""
        <div class="metric-card" style="{border_style}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-sub">{sub}</div>' if sub else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

