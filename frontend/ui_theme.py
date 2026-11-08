"""
Design system for AutoEDA Studio
================================
Central place for the visual identity: theme tokens, CSS, hero/KPI/console HTML
cards, and the small JavaScript widgets (quality gauge, copy button, menu
toggle). CSS is injected once per rerun; JS lives inside isolated Streamlit
components so it cannot break the app when Streamlit upgrades.
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import re
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, List, Optional, Sequence

import streamlit as st
import streamlit.components.v1 as components

FONT_STACK = "'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif"
DISPLAY_STACK = "'Sora', 'Inter', system-ui, sans-serif"
MONO_STACK = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

THEME_LABELS = {"System": None, "Light": "light", "Night": "dark"}
THEME_OPTIONS = ("System", "Light", "Night")


# ------------------------------------------------------------------------------
# Assets (logo, background) embedded as data URIs so they always load
# ------------------------------------------------------------------------------
@lru_cache(maxsize=8)
def asset_data_uri(name: str) -> str:
    """Return ``data:<mime>;base64,…`` for a file in ``assets/`` ('' if missing)."""
    path = ASSETS_DIR / name
    try:
        payload = path.read_bytes()
    except OSError:
        return ""
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


# ------------------------------------------------------------------------------
# Theme tokens
# ------------------------------------------------------------------------------
THEMES: Dict[str, Dict[str, str]] = {
    "dark": {
        "bg": "#08090B",
        "bg_soft": "#0B0D10",
        "surface": "#101216",
        "surface_2": "#14171C",
        "surface_3": "#191D24",
        "line": "rgba(255, 255, 255, 0.075)",
        "line_strong": "rgba(255, 255, 255, 0.16)",
        "text": "#F4F6F8",
        "body": "#C3CAD5",
        "muted": "#7C8798",
        "cyan": "#22D3EE",
        "cyan_soft": "rgba(34, 211, 238, 0.12)",
        "pink": "#F472B6",
        "pink_soft": "rgba(244, 114, 182, 0.12)",
        "violet": "#A78BFA",
        "violet_soft": "rgba(167, 139, 250, 0.12)",
        "green": "#86EFAC",
        "green_soft": "rgba(134, 239, 172, 0.12)",
        "amber": "#FBBF24",
        "amber_soft": "rgba(251, 191, 36, 0.12)",
        "orange": "#FB923C",
        "orange_soft": "rgba(251, 146, 60, 0.12)",
        "rose": "#FB7185",
        "rose_soft": "rgba(251, 113, 133, 0.12)",
        "blue": "#60A5FA",
        "blue_soft": "rgba(96, 165, 250, 0.12)",
        "input_bg": "rgba(255, 255, 255, 0.04)",
        "hover": "rgba(255, 255, 255, 0.06)",
        "sidebar_bg": "#0A0C0F",
        "sidebar_text": "#D5DBE4",
        "on_accent": "#04121A",
        "shadow": "0 26px 60px -42px rgba(0, 0, 0, 0.95)",
        "shadow_soft": "0 14px 36px -30px rgba(0, 0, 0, 0.9)",
        "scrollbar": "rgba(124, 135, 152, 0.45)",
        "switch_track": "rgba(255, 255, 255, 0.06)",
        "switch_knob": "#FFFFFF",
        "switch_sheen_1": "rgba(255, 255, 255, 0.16)",
        "switch_sheen_2": "rgba(255, 255, 255, 0.03)",
        "switch_sheen_3": "rgba(255, 255, 255, 0.10)",
        "switch_glass_border": "rgba(255, 255, 255, 0.22)",
        "switch_rim": "rgba(255, 255, 255, 0.45)",
        "switch_glow": "rgba(96, 190, 255, 0.22)",
        "switch_knob_1": "rgba(255, 255, 255, 0.34)",
        "switch_knob_2": "rgba(255, 255, 255, 0.10)",
        "switch_knob_border": "rgba(255, 255, 255, 0.55)",
        "switch_knob_shine": "rgba(255, 255, 255, 0.90)",
        "switch_icon_on": "#0A1220",
        "bg_veil": "rgba(6, 10, 20, 0.62)",
        "bg_veil_2": "rgba(7, 11, 22, 0.80)",
        "sidebar_glass": "rgba(8, 12, 22, 0.86)",
        "particle_a": "34, 211, 238",
        "particle_b": "167, 139, 250",
        "glass_bg": "rgba(20, 24, 32, 0.38)",
        "glass_border": "rgba(255, 255, 255, 0.30)",
        "glass_highlight": "rgba(255, 255, 255, 0.55)",
        "glass_text": "#F4F6F8",
    },
    "light": {
        "bg": "#F3F5F9",
        "bg_soft": "#EDF1F7",
        "surface": "#FFFFFF",
        "surface_2": "#F8FAFC",
        "surface_3": "#F1F5F9",
        "line": "#E2E8F0",
        "line_strong": "#CBD5E1",
        "text": "#0B1220",
        "body": "#42526B",
        "muted": "#64748B",
        "cyan": "#0891B2",
        "cyan_soft": "rgba(8, 145, 178, 0.10)",
        "pink": "#DB2777",
        "pink_soft": "rgba(219, 39, 119, 0.09)",
        "violet": "#7C3AED",
        "violet_soft": "rgba(124, 58, 237, 0.09)",
        "green": "#059669",
        "green_soft": "rgba(5, 150, 105, 0.10)",
        "amber": "#D97706",
        "amber_soft": "rgba(217, 119, 6, 0.10)",
        "orange": "#EA580C",
        "orange_soft": "rgba(234, 88, 12, 0.10)",
        "rose": "#E11D48",
        "rose_soft": "rgba(225, 29, 72, 0.09)",
        "blue": "#2563EB",
        "blue_soft": "rgba(37, 99, 235, 0.09)",
        "input_bg": "#FFFFFF",
        "hover": "rgba(15, 23, 42, 0.045)",
        "sidebar_bg": "#FFFFFF",
        "sidebar_text": "#334155",
        "on_accent": "#FFFFFF",
        "shadow": "0 20px 44px -36px rgba(15, 23, 42, 0.40)",
        "shadow_soft": "0 12px 28px -26px rgba(15, 23, 42, 0.35)",
        "scrollbar": "rgba(100, 116, 139, 0.35)",
        "switch_track": "rgba(255, 255, 255, 0.55)",
        "switch_knob": "#FFFFFF",
        "switch_sheen_1": "rgba(255, 255, 255, 0.95)",
        "switch_sheen_2": "rgba(255, 255, 255, 0.45)",
        "switch_sheen_3": "rgba(255, 255, 255, 0.80)",
        "switch_glass_border": "rgba(15, 23, 42, 0.14)",
        "switch_rim": "rgba(255, 255, 255, 0.95)",
        "switch_glow": "rgba(120, 170, 255, 0.16)",
        "switch_knob_1": "rgba(255, 255, 255, 1)",
        "switch_knob_2": "rgba(240, 245, 252, 0.92)",
        "switch_knob_border": "rgba(15, 23, 42, 0.10)",
        "switch_knob_shine": "rgba(255, 255, 255, 1)",
        "switch_icon_on": "#0B1220",
        "bg_veil": "rgba(244, 247, 252, 0.90)",
        "bg_veil_2": "rgba(240, 244, 250, 0.96)",
        "sidebar_glass": "rgba(255, 255, 255, 0.90)",
        "particle_a": "8, 145, 178",
        "particle_b": "124, 58, 237",
        "glass_bg": "rgba(255, 255, 255, 0.55)",
        "glass_border": "rgba(15, 23, 42, 0.20)",
        "glass_highlight": "rgba(255, 255, 255, 0.95)",
        "glass_text": "#0B1220",
    },
}


def theme_tokens(mode: Optional[str] = None) -> Dict[str, str]:
    text = str(mode).lower()
    if text == "system":
        text = system_theme()
    return THEMES["light" if text == "light" else "dark"]


def system_theme() -> str:
    """The theme the browser / operating system reports right now."""
    try:
        context_type = getattr(getattr(st.context, "theme", None), "type", None)
        if context_type in ("light", "dark"):
            return str(context_type)
    except Exception:
        pass
    return "dark"


def requested_mode() -> str:
    """The user's choice in the sidebar: ``System``, ``Light`` or ``Night``."""
    value = st.session_state.get("theme_mode")
    if value in THEME_OPTIONS:
        return str(value)
    return "System"


def resolve_theme() -> str:
    """Concrete rendering theme: ``light`` or ``dark``."""
    mode = requested_mode()
    if mode == "Light":
        return "light"
    if mode == "Night":
        return "dark"
    return system_theme()


# ------------------------------------------------------------------------------
# Stylesheet
# ------------------------------------------------------------------------------
_CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: {{bg}};
    --bg-soft: {{bg_soft}};
    --surface: {{surface}};
    --surface-2: {{surface_2}};
    --surface-3: {{surface_3}};
    --line: {{line}};
    --line-strong: {{line_strong}};
    --text: {{text}};
    --body: {{body}};
    --muted: {{muted}};
    --cyan: {{cyan}};
    --cyan-soft: {{cyan_soft}};
    --pink: {{pink}};
    --pink-soft: {{pink_soft}};
    --violet: {{violet}};
    --violet-soft: {{violet_soft}};
    --green: {{green}};
    --green-soft: {{green_soft}};
    --amber: {{amber}};
    --amber-soft: {{amber_soft}};
    --orange: {{orange}};
    --orange-soft: {{orange_soft}};
    --rose: {{rose}};
    --rose-soft: {{rose_soft}};
    --blue: {{blue}};
    --blue-soft: {{blue_soft}};
    --input-bg: {{input_bg}};
    --hover: {{hover}};
    --sidebar-bg: {{sidebar_bg}};
    --sidebar-text: {{sidebar_text}};
    --on-accent: {{on_accent}};
    --shadow: {{shadow}};
    --shadow-soft: {{shadow_soft}};
    --scrollbar: {{scrollbar}};
    --switch-track: {{switch_track}};
    --switch-knob: {{switch_knob}};
    --switch-sheen-1: {{switch_sheen_1}};
    --switch-sheen-2: {{switch_sheen_2}};
    --switch-sheen-3: {{switch_sheen_3}};
    --switch-glass-border: {{switch_glass_border}};
    --switch-rim: {{switch_rim}};
    --switch-glow: {{switch_glow}};
    --switch-knob-1: {{switch_knob_1}};
    --switch-knob-2: {{switch_knob_2}};
    --switch-knob-border: {{switch_knob_border}};
    --switch-knob-shine: {{switch_knob_shine}};
    --switch-icon-on: {{switch_icon_on}};
    --bg-veil: {{bg_veil}};
    --bg-veil-2: {{bg_veil_2}};
    --sidebar-glass: {{sidebar_glass}};
    --font: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    --display: 'Sora', 'Inter', sans-serif;
    --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}

html, body, [class*="css"] {
    font-family: var(--font);
}

/* The artwork lives on the root element: a translucent veil over the photo,
   so text keeps its contrast. The particle canvas is pushed to z-index -1,
   which paints above that root background but below all app content.
   NOTE: never change position/height on .stApp — Streamlit's layout for it
   is absolute, overriding it collapses the whole app to 0px. */
html {
    background-color: var(--bg) !important;
    background-image:
        linear-gradient(180deg, var(--bg-veil) 0%, var(--bg-veil-2) 100%),
        url("{{bg_image}}") !important;
    background-position: center center !important;
    background-size: cover !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
}
body {
    background: transparent !important;
    min-height: 100vh;
}
#aes-particles {
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
}
.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    color: var(--text);
    background: transparent !important;
}
/* The scrolling panes must stay see-through or they hide the artwork. */
.stMain,
[data-testid="stMain"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    max-width: 1440px;
    /* Leaves the top-right toolbar (Deploy) its own band: the hero title can
       never slide underneath it. */
    padding-top: 3.1rem;
    padding-bottom: 3.4rem;
}

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background: transparent;
    height: 2.7rem;
    z-index: 20;
}
header[data-testid="stHeader"] > div { background: transparent; }
/* Shift the Streamlit toolbar up into the band reserved above the content. */
[data-testid="stToolbar"],
div[data-testid="stToolbar"] {
    top: -0.45rem !important;
    right: 0.65rem !important;
}
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] a {
    color: var(--muted) !important;
}
[data-testid="stToolbar"] button:hover,
[data-testid="stToolbar"] a:hover {
    color: var(--cyan) !important;
}

h1, h2, h3, h4, h5 {
    font-family: var(--display);
    color: var(--text);
    letter-spacing: -0.015em;
}
p, li, label { color: var(--body); }
h1 span, h2 span, h3 span, h4 span, h5 span { color: inherit; }
a { color: var(--cyan); }
strong { color: var(--text); }
code {
    font-family: var(--mono);
    background: var(--surface-3);
    color: var(--cyan);
    border-radius: 5px;
    padding: 1px 5px;
    font-size: 0.82em;
}

/* ---------------- Console header (hero) ---------------- */
.hero {
    position: relative;
    overflow: hidden;
    border-radius: 16px;
    padding: 20px 26px 19px;
    background:
        linear-gradient(120deg, var(--surface) 0%, var(--surface-2) 52%, var(--surface) 100%);
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
    margin-bottom: 1.05rem;
    animation: rise 0.5s ease both;
}
.hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(620px 220px at 8% -60%, var(--cyan-soft), transparent 70%),
        radial-gradient(520px 220px at 100% 140%, var(--violet-soft), transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: "";
    position: absolute;
    left: 0; right: 0; bottom: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--cyan), var(--pink) 55%, var(--violet));
    opacity: 0.9;
}
.hero .eyebrow {
    position: relative;
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--mono);
    font-size: 0.67rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--cyan);
    margin-bottom: 8px;
}
.hero .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 10px var(--green);
    animation: pulse 2.2s ease-in-out infinite;
}
.hero h1,
.hero h1 span {
    position: relative;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text);
    font-family: var(--display);
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.005em;
    font-size: clamp(1.4rem, 2.6vw, 2rem);
    margin: 0;
    line-height: 1.16;
}
.hero .subtitle {
    position: relative;
    color: var(--body) !important;
    margin: 0.42rem 0 0;
    font-size: 0.94rem;
    max-width: 90ch;
    line-height: 1.55;
}
.hero .meta {
    position: relative;
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 14px;
}
.pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    background: var(--surface-3);
    border: 1px solid var(--line);
    color: var(--body);
    font-family: var(--mono);
    font-size: 0.71rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}
.pill strong { color: var(--text); }
.pill code { background: transparent; padding: 0; color: var(--cyan); }

/* ---------------- KPI cards ---------------- */
.kpi-grid {
    display: grid;
    gap: 12px;
    margin: 4px 0 16px;
}
.kpi {
    position: relative;
    overflow: hidden;
    --tone: var(--cyan);
    background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface) 100%);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 16px 17px 14px;
    box-shadow: var(--shadow-soft);
    transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
    animation: rise 0.45s ease both;
}
.kpi::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--tone);
    opacity: 0.95;
    box-shadow: 0 0 12px var(--tone);
}
/* coloured bloom just under the light line, for the 3D card edge */
.kpi::after {
    content: "";
    position: absolute;
    top: 0; left: 6%; right: 6%;
    height: 44px;
    background: radial-gradient(60% 100% at 50% 0%, var(--tone), transparent 72%);
    opacity: 0.13;
    pointer-events: none;
}
.kpi > * {
    position: relative;
    z-index: 1;
}
.kpi:hover {
    transform: translateY(-2px);
    border-color: var(--line-strong);
    box-shadow: 0 22px 44px -34px var(--tone);
}
.kpi .label {
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
}
.kpi .value {
    font-family: var(--display);
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--tone);
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin-top: 6px;
    overflow-wrap: anywhere;
    word-break: break-word;
}
.kpi .value.value-sm {
    font-size: 1.35rem;
    letter-spacing: -0.01em;
}
.kpi .hint {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--muted);
    margin-top: 5px;
    letter-spacing: 0.02em;
}
.kpi.tone-danger { --tone: var(--rose); }
.kpi.tone-warn { --tone: var(--amber); }
.kpi.tone-good { --tone: var(--green); }
.kpi.tone-info { --tone: var(--cyan); }
.kpi.tone-neutral { --tone: var(--violet); }

/* ---------------- Chips & callouts ---------------- */
.chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 999px;
    font-family: var(--mono);
    font-size: 0.71rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    border: 1px solid transparent;
    white-space: nowrap;
}
.chip-balanced { background: var(--green-soft); color: var(--green); border-color: var(--green-soft); }
.chip-mild { background: var(--amber-soft); color: var(--amber); border-color: var(--amber-soft); }
.chip-moderate { background: var(--orange-soft); color: var(--orange); border-color: var(--orange-soft); }
.chip-severe { background: var(--rose-soft); color: var(--rose); border-color: var(--rose-soft); }
.chip-info { background: var(--cyan-soft); color: var(--cyan); border-color: var(--cyan-soft); }
.chip-neutral { background: var(--surface-3); color: var(--muted); border-color: var(--line); }

.callout {
    border-radius: 12px;
    padding: 13px 16px;
    margin: 0.55rem 0 0.95rem;
    border: 1px solid var(--line);
    border-left: 3px solid var(--cyan);
    background: var(--surface-2);
    color: var(--body);
    font-size: 0.9rem;
    line-height: 1.55;
}
.callout.success { border-left-color: var(--green); background: var(--green-soft); }
.callout.warning { border-left-color: var(--amber); background: var(--amber-soft); }
.callout.danger { border-left-color: var(--rose); background: var(--rose-soft); }
.callout.info { border-left-color: var(--cyan); background: var(--surface-2); }
.callout strong { color: var(--text); }

.formula-chip {
    display: inline-block;
    background: var(--surface-3);
    border: 1px solid var(--line);
    color: var(--cyan);
    border-radius: 6px;
    padding: 2px 8px;
    font-family: var(--mono);
    font-size: 0.8rem;
}

/* ---------------- Section headers ---------------- */
.section-title {
    font-family: var(--mono);
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--body);
    margin: 0.55rem 0 0.1rem;
}
.section-sub {
    color: var(--muted);
    font-size: 0.83rem;
    margin-bottom: 0.7rem;
}

.chart-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin: 0.95rem 0 0.35rem;
    padding-bottom: 7px;
    border-bottom: 1px solid var(--line);
}
.chart-title {
    font-family: var(--mono);
    font-size: 0.71rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--body);
}
.chart-meta {
    font-family: var(--mono);
    font-size: 0.63rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 2px 9px;
    background: var(--surface-2);
    white-space: nowrap;
}

/* ---------------- Sidebar ---------------- */
section[data-testid="stSidebar"] {
    background: var(--sidebar-glass);
    border-right: 1px solid var(--line);
    backdrop-filter: blur(18px) saturate(1.05);
    -webkit-backdrop-filter: blur(18px) saturate(1.05);
    position: relative;
    z-index: 3;
}
section[data-testid="stSidebar"] > div {
    height: 100vh;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0.55rem;
    overflow-y: auto;
    overscroll-behavior: contain;
    scrollbar-width: thin;
    scrollbar-color: var(--scrollbar) transparent;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
    display: none;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {
    color: var(--sidebar-text) !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--sidebar-text) !important;
}
section[data-testid="stSidebar"] .stMarkdown a { color: var(--cyan); }
section[data-testid="stSidebar"] hr {
    border-color: var(--line);
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: var(--surface-2);
    border: 1px dashed var(--line-strong);
    border-radius: 12px;
}
/* Browse button inside the dropzone. Streamlit paints it with the theme's dark
   secondary surface even when our panel is light, which leaves dark-on-dark
   text; give it the palette surface so the frame follows the active theme. */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 9px !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
    background: var(--cyan-soft) !important;
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
}
/* Uploaded-file chip: same light surface as the dropzone, so the whole uploader
   reads as one light frame instead of a dark block. Scoped to the sidebar so
   these rules outrank the dropzone-colour rule above (same specificity,
   declared later). */
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
section[data-testid="stSidebar"] [data-testid="stUploadedFile"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] *,
section[data-testid="stSidebar"] [data-testid="stUploadedFile"] * {
    color: var(--text) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] small,
section[data-testid="stSidebar"] [data-testid="stUploadedFile"] small,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileData"],
section[data-testid="stSidebar"] [data-testid="stUploadedFile"] [data-testid="stUploadedFileData"] {
    color: var(--muted) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] svg,
section[data-testid="stSidebar"] [data-testid="stUploadedFile"] svg {
    fill: var(--body) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] button svg,
section[data-testid="stSidebar"] [data-testid="stUploadedFile"] button svg {
    fill: var(--muted) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: var(--body) !important;
}

/* Uploader label + its description in a colour that stands out from the
   body copy, plus the help tooltip (rendered in a body-level portal). */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] label p {
    color: var(--violet) !important;
    font-weight: 600;
    letter-spacing: 0.01em;
}
.upload-hint {
    margin: 6px 0 12px;
    padding: 7px 10px;
    border-radius: 9px;
    border: 1px dashed var(--violet);
    background: var(--violet-soft);
    color: var(--amber);
    font-family: var(--mono);
    font-size: 0.68rem;
    line-height: 1.5;
    letter-spacing: 0.02em;
}
.upload-hint strong { color: var(--violet); }
div[data-baseweb="tooltip"],
div[data-baseweb="tooltip"] > div,
[data-testid="stTooltipContent"] {
    background: var(--surface-3) !important;
    border: 1px solid var(--violet) !important;
    border-radius: 10px !important;
    box-shadow: 0 16px 34px -24px rgba(0, 0, 0, 0.9) !important;
}
div[data-baseweb="tooltip"] *,
div[data-baseweb="tooltip"] p,
div[data-baseweb="tooltip"] span,
[data-testid="stTooltipContent"] * {
    color: var(--amber) !important;
    font-family: var(--mono) !important;
    font-size: 0.74rem !important;
    line-height: 1.45 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: var(--surface-2);
    color: var(--text);
    border: 1px solid var(--line);
    border-radius: 9px;
    font-weight: 600;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--cyan);
    color: var(--cyan);
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
    border-radius: 8px;
    transition: background 0.14s ease, color 0.14s ease;
    margin: 1px 0;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
    background: var(--hover);
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] span {
    color: var(--sidebar-text) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
    background: var(--cyan-soft);
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] span {
    color: var(--cyan) !important;
}

/* Sidebar pipeline navigation (custom page links) */
.nav-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 15px 0 5px 6px;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] {
    margin: 0;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a,
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"],
section[data-testid="stSidebar"] .st-key-pipeline-nav a {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 8px 11px;
    margin: 2px 0;
    border-radius: 10px;
    border: 1px solid transparent;
    background: transparent;
    color: var(--sidebar-text) !important;
    font-size: 0.87rem;
    font-weight: 500;
    line-height: 1.25;
    text-decoration: none !important;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover,
section[data-testid="stSidebar"] .st-key-pipeline-nav a:hover {
    background: var(--hover);
    color: var(--text) !important;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"],
section[data-testid="stSidebar"] .st-key-pipeline-nav a[aria-current="page"] {
    background: var(--cyan-soft);
    border-color: var(--line);
    color: var(--cyan) !important;
    box-shadow: inset 2px 0 0 var(--cyan);
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a p,
section[data-testid="stSidebar"] .st-key-pipeline-nav a p {
    color: inherit !important;
    font-size: inherit;
    margin: 0;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a span,
section[data-testid="stSidebar"] .st-key-pipeline-nav a span {
    color: inherit !important;
}

/* Sidebar radio navigation — rendered as pill links */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 3px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    width: 100%;
    padding: 8px 10px;
    margin: 1px 0;
    border-radius: 10px;
    border: 1px solid transparent;
    transition: background 0.15s ease, border-color 0.15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: var(--sidebar-text) !important;
    font-size: 0.87rem;
    font-weight: 500;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: var(--hover); }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: var(--cyan-soft);
    border-color: var(--line);
    box-shadow: inset 2px 0 0 var(--cyan);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: var(--cyan) !important;
}

/* Brand block */
.brand {
    display: flex;
    align-items: center;
    gap: 11px;
    margin: 2px 0 12px;
}
.brand-mark {
    width: 38px;
    height: 38px;
    border-radius: 11px;
    background: linear-gradient(135deg, var(--cyan), var(--violet));
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--on-accent);
    font-family: var(--display);
    font-weight: 800;
    font-size: 18px;
    box-shadow: 0 0 22px -6px var(--cyan);
}
.brand-name {
    font-family: var(--display);
    font-weight: 800;
    color: var(--text);
    font-size: 1.18rem;
    line-height: 1.1;
    letter-spacing: -0.01em;
}
.brand-sub {
    font-family: var(--mono);
    color: var(--muted);
    font-size: 0.62rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ---------------- Buttons / inputs ---------------- */
.stDownloadButton > button {
    background: linear-gradient(120deg, var(--cyan), var(--violet));
    color: var(--on-accent);
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    letter-spacing: 0.01em;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 30px -18px var(--cyan);
}
.stButton > button {
    border-radius: 9px;
    font-weight: 600;
    background: var(--surface-2);
    color: var(--text);
    border: 1px solid var(--line);
}
.stButton > button:hover {
    border-color: var(--cyan);
    color: var(--cyan);
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
    background: var(--input-bg) !important;
    border-color: var(--line) !important;
    border-radius: 9px;
    color: var(--text) !important;
}
div[data-baseweb="select"] input,
div[data-baseweb="input"] input,
.stTextInput input {
    color: var(--text) !important;
    background: transparent !important;
}
div[data-baseweb="select"] svg { fill: var(--muted) !important; }
[data-baseweb="popover"] > div,
[data-baseweb="popover"] ul,
ul[role="listbox"],
div[data-baseweb="menu"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--line) !important;
    color: var(--text) !important;
    border-radius: 10px;
}
li[role="option"] {
    color: var(--text) !important;
    background: transparent !important;
}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background: var(--hover) !important;
}
[data-testid="stSlider"] [role="slider"] { border-color: var(--cyan); }

/* ---------------- Theme switch — liquid glass ----------------
   Streamlit renders st.segmented_control as a stButtonGroup with role=radio
   buttons; the widget key ("theme_mode") gives us .st-key-theme_mode.
   The shell is a frosted pane: translucent fill, blurred backdrop, a bright
   inner rim, a slow-moving specular sheen and a glass bubble for the choice. */
.st-key-theme_mode [data-testid="stButtonGroup"],
[data-testid="stSegmentedControl"] {
    width: fit-content;
    display: flex;
    justify-content: flex-start;
    margin: 2px 0 6px;
}
.st-key-theme_mode [role="radiogroup"],
[data-testid="stSegmentedControl"] [role="radiogroup"] {
    position: relative;
    isolation: isolate;
    background:
        linear-gradient(135deg,
            var(--switch-sheen-1) 0%,
            var(--switch-sheen-2) 42%,
            var(--switch-sheen-3) 100%),
        var(--switch-track);
    background-size: 220% 220%;
    border: 1px solid var(--switch-glass-border);
    border-radius: 999px;
    padding: 5px;
    gap: 6px;
    display: inline-flex;
    align-items: center;
    width: fit-content;
    backdrop-filter: blur(22px) saturate(1.8);
    -webkit-backdrop-filter: blur(22px) saturate(1.8);
    box-shadow:
        inset 0 1px 0 var(--switch-rim),
        inset 0 -1px 0 rgba(0, 0, 0, 0.32),
        inset 0 10px 20px -14px rgba(255, 255, 255, 0.30),
        0 16px 34px -22px rgba(0, 0, 0, 0.95),
        0 2px 6px -3px rgba(0, 0, 0, 0.45);
    animation: liquidShift 16s ease-in-out infinite;
}
/* specular tint floating inside the glass */
.st-key-theme_mode [role="radiogroup"]::after,
[data-testid="stSegmentedControl"] [role="radiogroup"]::after {
    content: "";
    position: absolute;
    inset: 1px;
    border-radius: inherit;
    background:
        radial-gradient(95% 150% at 16% -35%, var(--switch-glow), transparent 60%),
        radial-gradient(70% 120% at 92% 145%, rgba(255, 255, 255, 0.10), transparent 62%);
    pointer-events: none;
    z-index: -1;
}
.st-key-theme_mode button[role="radio"],
[data-testid="stSegmentedControl"] button {
    position: relative;
    z-index: 1;
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    min-height: 44px !important;
    max-width: 44px !important;
    padding: 0 !important;
    margin: 0 !important;
    border: 1px solid transparent !important;
    border-radius: 50% !important;
    background: transparent !important;
    color: var(--body) !important;
    font-size: 19px !important;
    line-height: 1 !important;
    display: inline-flex !important;
    align-items: center;
    justify-content: center;
    opacity: 0.62;
    box-shadow: none !important;
    transition:
        background 0.24s ease,
        border-color 0.24s ease,
        transform 0.28s cubic-bezier(0.34, 1.56, 0.64, 1),
        box-shadow 0.24s ease,
        opacity 0.2s ease;
}
.st-key-theme_mode button[role="radio"] div,
.st-key-theme_mode button[role="radio"] span,
.st-key-theme_mode button[role="radio"] p,
[data-testid="stSegmentedControl"] button p {
    margin: 0 !important;
    padding: 0 !important;
    font-size: inherit !important;
    line-height: 1 !important;
    color: inherit !important;
    background: transparent !important;
}
/* Icons are drawn as masks, not emoji: they inherit the glass colour and look
   identical on every platform (the emoji sun renders as a brown blob). */
.st-key-theme_mode button[role="radio"],
[data-testid="stSegmentedControl"] button {
    font-size: 0 !important;
}
.st-key-theme_mode button[role="radio"]::before,
[data-testid="stSegmentedControl"] button::before {
    content: "";
    position: absolute;
    inset: 0;
    margin: auto;
    width: 20px;
    height: 20px;
    background-color: currentColor;
    -webkit-mask: var(--aes-theme-icon) center / 20px 20px no-repeat;
    mask: var(--aes-theme-icon) center / 20px 20px no-repeat;
    transition: background-color 0.24s ease;
}
/* Streamlit's label wrapper would push the mask icon aside; collapse it but
   leave it in the accessibility tree so the buttons keep their names. */
.st-key-theme_mode button[role="radio"] > div,
[data-testid="stSegmentedControl"] button > div {
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    opacity: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    font-size: 0 !important;
}
.st-key-theme_mode button[role="radio"]:nth-of-type(1),
[data-testid="stSegmentedControl"] button:nth-of-type(1) {
    --aes-theme-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='2' y='3' width='20' height='14' rx='2'/%3E%3Cpath d='M8 21h8M12 17v4'/%3E%3C/svg%3E");
}
.st-key-theme_mode button[role="radio"]:nth-of-type(2),
[data-testid="stSegmentedControl"] button:nth-of-type(2) {
    --aes-theme-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='4.2'/%3E%3Cpath d='M12 2.2v2.4M12 19.4v2.4M2.2 12h2.4M19.4 12h2.4M5 5l1.7 1.7M17.3 17.3L19 19M19 5l-1.7 1.7M6.7 17.3L5 19'/%3E%3C/svg%3E");
}
.st-key-theme_mode button[role="radio"]:nth-of-type(3),
[data-testid="stSegmentedControl"] button:nth-of-type(3) {
    --aes-theme-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20.5 13.2A8.6 8.6 0 1 1 10.8 3.5a6.9 6.9 0 0 0 9.7 9.7z'/%3E%3C/svg%3E");
}
.st-key-theme_mode button[role="radio"]:hover,
[data-testid="stSegmentedControl"] button:hover {
    opacity: 1;
    background: var(--hover) !important;
}
/* the selected lens: a polished glass bubble, not a flat white dot */
.st-key-theme_mode button[role="radio"][aria-checked="true"],
.st-key-theme_mode button[role="radio"][aria-pressed="true"],
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
    background:
        linear-gradient(160deg, var(--switch-knob-1) 0%, var(--switch-knob-2) 100%) !important;
    border: 1px solid var(--switch-knob-border) !important;
    color: var(--switch-icon-on) !important;
    backdrop-filter: blur(10px) saturate(1.7);
    -webkit-backdrop-filter: blur(10px) saturate(1.7);
    opacity: 1;
    transform: scale(1.04);
    box-shadow:
        inset 0 1px 0 var(--switch-knob-shine),
        inset 0 -3px 9px rgba(0, 0, 0, 0.20),
        inset 0 0 0 1px rgba(255, 255, 255, 0.06),
        0 12px 22px -12px rgba(0, 0, 0, 0.80),
        0 2px 6px -2px rgba(0, 0, 0, 0.35) !important;
}
.theme-note {
    margin: 2px 0 4px 2px;
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
}
.theme-note strong { color: var(--cyan); }

/* ---------------- Tabs, frames, data ---------------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
    background: var(--surface-2);
    padding: 5px;
    border-radius: 11px;
    border: 1px solid var(--line);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 6px 13px;
    font-family: var(--mono);
    font-size: 0.74rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    color: var(--muted);
}
.stTabs [aria-selected="true"] {
    background: var(--surface-3);
    color: var(--cyan);
    box-shadow: inset 0 0 0 1px var(--line);
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
    background: var(--surface);
    --gdg-bg-cell: {{surface}};
    --gdg-bg-cell-medium: {{surface_2}};
    --gdg-bg-header: {{surface_3}};
    --gdg-bg-header-hovered: {{surface_3}};
    --gdg-bg-header-pressed: {{surface_3}};
    --gdg-bg-bubble: {{surface_3}};
    --gdg-bg-bubble-selected: {{cyan_soft}};
    --gdg-bg-search-result: {{cyan_soft}};
    --gdg-border-color: {{line}};
    --gdg-horizontal-border-color: {{line}};
    --gdg-vertical-border-color: {{line}};
    --gdg-text-dark: {{text}};
    --gdg-text-medium: {{body}};
    --gdg-text-light: {{muted}};
    --gdg-text-bubble: {{text}};
    --gdg-accent-color: {{cyan}};
    --gdg-accent-light: {{cyan_soft}};
    --gdg-accent-fg: {{on_accent}};
}
[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--surface);
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    color: var(--text) !important;
    font-family: var(--mono);
    font-size: 0.76rem;
    letter-spacing: 0.03em;
}
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 13px;
    padding: 12px 15px;
}
[data-testid="stAlert"] {
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-radius: 11px;
    color: var(--body);
}
[data-testid="stAlert"] * { color: var(--body) !important; }
[data-testid="stPlotlyChart"] {
    background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface) 100%);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 10px 8px 4px;
    box-shadow: var(--shadow-soft);
}
[data-testid="stPlotlyChart"] .modebar {
    top: 4px !important;
    right: 6px !important;
    opacity: 0.55;
    transition: opacity 0.15s ease;
}
[data-testid="stPlotlyChart"]:hover .modebar { opacity: 1; }
[data-testid="stPlotlyChart"] .modebar-btn path { fill: var(--muted) !important; }
[data-testid="stPlotlyChart"] .modebar-btn:hover path { fill: var(--cyan) !important; }

/* ---------------- 3-D framed chart cards ---------------- */
/* Any st.container whose key starts with "frame3d-" becomes an extruded card;
   one characteristic per frame, as in the reference dashboard. */
div[class*="st-key-frame3d"] {
    position: relative;
    border-radius: 18px;
    padding: 12px 12px 8px;
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.055), rgba(0, 0, 0, 0.22) 78%),
        linear-gradient(160deg, var(--surface-2) 0%, var(--surface) 62%, var(--surface-3) 100%);
    border: 1px solid var(--line-strong);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.10),
        inset 0 -26px 42px -34px rgba(0, 0, 0, 0.95),
        0 26px 46px -34px rgba(0, 0, 0, 0.95),
        0 3px 0 -1px var(--surface-3),
        0 9px 18px -14px rgba(0, 0, 0, 0.85);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[class*="st-key-frame3d"]::before {
    content: "";
    position: absolute;
    left: 14%;
    right: 14%;
    top: 0;
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(90deg, transparent, var(--cyan), var(--violet), transparent);
    opacity: 0.85;
}
div[class*="st-key-frame3d"]::after {
    content: "";
    position: absolute;
    left: 8%;
    right: 8%;
    bottom: -9px;
    height: 14px;
    border-radius: 50%;
    background: radial-gradient(60% 100% at 50% 0%, rgba(0, 0, 0, 0.55), transparent 72%);
    pointer-events: none;
}
div[class*="st-key-frame3d"]:hover {
    transform: translateY(-2px);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.14),
        inset 0 -26px 42px -34px rgba(0, 0, 0, 0.95),
        0 32px 56px -32px var(--cyan-soft),
        0 32px 50px -34px rgba(0, 0, 0, 0.95);
}
div[class*="st-key-frame3d"] [data-testid="stPlotlyChart"] {
    background: transparent;
    border: 0;
    box-shadow: none;
    padding: 0;
    border-radius: 12px;
}

/* ---------------- Plot descriptions ---------------- */
.chart-desc {
    margin: -2px 0 18px;
    padding: 8px 13px;
    border-left: 2px solid var(--cyan);
    border-radius: 0 9px 9px 0;
    background: var(--surface-2);
    color: var(--muted);
    font-size: 0.81rem;
    line-height: 1.55;
}
.chart-desc strong { color: var(--body); }
.chart-desc .desc-icon {
    color: var(--cyan);
    font-family: var(--mono);
    font-size: 0.78rem;
    margin-right: 6px;
}

/* ---------------- Misc ---------------- */
hr {
    border-color: var(--line);
    margin: 1rem 0;
}
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--scrollbar);
    border-radius: 999px;
    border: 2px solid transparent;
    background-clip: content-box;
}
[data-testid="stCaptionContainer"] { color: var(--muted); }

@keyframes rise {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.55; transform: scale(0.85); }
}
/* The sheen drifts slowly across the glass, like light on a wet surface. */
@keyframes liquidShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

@media (max-width: 900px) {
    .hero { padding: 17px 18px 16px; border-radius: 14px; }
    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
    .kpi .value { font-size: 1.55rem; }
    .kpi .value.value-sm { font-size: 1.15rem; }
    .block-container { padding-top: 0.7rem; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
"""


# Light mode is printed, not projected: every neon halo / coloured bloom that
# reads well on black becomes a smear on white, so they are switched off here.
_LIGHT_OVERRIDES = """
/* ---------------- Light mode: no neon ---------------- */
.kpi::before { box-shadow: none; }
.kpi::after { display: none; }
.kpi:hover { box-shadow: 0 16px 34px -30px rgba(15, 23, 42, 0.28); }
.hero .dot { box-shadow: none; }
.brand-mark { box-shadow: none; }
.hero::before { opacity: 0.5; }
.stDownloadButton > button:hover {
    transform: none;
    box-shadow: 0 10px 22px -18px rgba(15, 23, 42, 0.35);
}
[data-testid="stPlotlyChart"] { box-shadow: none; }
div[class*="st-key-frame3d"] {
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.9), rgba(241, 245, 249, 0.85));
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.95),
        0 16px 30px -26px rgba(15, 23, 42, 0.42),
        0 2px 0 -1px #E2E8F0;
}
div[class*="st-key-frame3d"]::after { display: none; }
div[class*="st-key-frame3d"]:hover {
    transform: translateY(-2px);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.95),
        0 20px 34px -26px rgba(15, 23, 42, 0.45);
}
"""


_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
    "&family=Sora:wght@600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');"
)

_BG_IMAGE = ""  # filled lazily, see _css_core


def _css_core(mode: str) -> str:
    """Stylesheet body for one concrete theme (no <style> wrapper)."""
    global _BG_IMAGE
    if not _BG_IMAGE:
        _BG_IMAGE = asset_data_uri("background.jpg") or "none"

    css = _CSS_TEMPLATE
    values = dict(theme_tokens(mode))
    values["bg_image"] = _BG_IMAGE
    for key, value in values.items():
        css = css.replace("{{" + key + "}}", str(value))

    # Unwrap first, then append the light overrides — appending before the
    # unwrap would leave the template's closing </style> buried mid-sheet and
    # close the stylesheet early.
    css = css.strip()
    if css.startswith("<style>"):
        css = css[len("<style>"):]
    if css.endswith("</style>"):
        css = css[: -len("</style>")]

    if str(mode).lower() == "light":
        css += "\n" + _LIGHT_OVERRIDES
    # @import must be first in a sheet, so it is hoisted by build_css().
    # Drop it line-wise: the Google Fonts URL itself contains semicolons
    # (wght@400;500;600), so a regex cut at ";" would leave a broken tail that
    # swallows the rules after it.
    css = "\n".join(
        line for line in css.splitlines()
        if not line.strip().startswith("@import")
    )
    return css


def build_css(mode: str = "dark") -> str:
    """Wrap the stylesheet; ``system`` follows the OS with a media query."""
    if str(mode).lower() in ("system", "auto"):
        dark = _css_core("dark")
        light = _css_core("light")
        return (
            "<style>"
            + _FONT_IMPORT
            + "@media (prefers-color-scheme: dark) {" + dark + "}"
            + "@media (prefers-color-scheme: light) {" + light + "}"
            + "@media (prefers-color-scheme: no-preference) {" + dark + "}"
            + "</style>"
        )
    return "<style>" + _FONT_IMPORT + _css_core(mode) + "</style>"


def inject_css(mode: str = "dark") -> None:
    """Inject the global stylesheet (call once, right after page config)."""
    st.markdown(build_css(mode), unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# HTML building blocks
# ------------------------------------------------------------------------------
def hero(
    title: str,
    subtitle: str,
    meta: Optional[Sequence[str]] = None,
    badge: Optional[str] = None,
    eyebrow: str = "benchmark console",
) -> str:
    meta_html = "".join(f'<span class="pill">{item}</span>' for item in (meta or []))
    badge_html = f'<span class="pill">{badge}</span>' if badge else ""
    # One single line: an empty or indented line would make Markdown treat the
    # rest as a code block and print the tags instead of rendering them.
    return (
        '<div class="hero">'
        f'<div class="eyebrow"><span class="dot"></span>{eyebrow}</div>'
        f"<h1>{title}</h1>"
        f'<p class="subtitle">{subtitle}</p>'
        f'<div class="meta">{meta_html}{badge_html}</div>'
        "</div>"
    )


def kpi(
    label: str,
    value: Any,
    hint: str = "",
    tone: str = "default",
    delay: int = 0,
) -> str:
    hint_html = f'<div class="hint">{hint}</div>' if hint else ""
    value_text = str(value)
    size_class = " value-sm" if len(value_text) > 14 else ""
    return (
        f'<div class="kpi tone-{tone}" style="animation-delay:{delay}ms">'
        f'<div class="label">{label}</div>'
        f'<div class="value{size_class}">{value}</div>'
        f"{hint_html}</div>"
    )


def kpi_grid(items: Iterable[Dict[str, Any]], columns: int = 4) -> str:
    cards = []
    for index, item in enumerate(items):
        cards.append(
            kpi(
                label=str(item.get("label", "")),
                value=item.get("value", "—"),
                hint=str(item.get("hint", "")),
                tone=str(item.get("tone", "default")),
                delay=index * 45,
            )
        )
    return (
        f'<div class="kpi-grid" style="grid-template-columns:repeat({columns}, minmax(0,1fr));">'
        + "".join(cards)
        + "</div>"
    )


def severity_chip(severity: str) -> str:
    mapping = {
        "Balanced": "chip-balanced",
        "Mild Imbalance": "chip-mild",
        "Moderate Imbalance": "chip-moderate",
        "Severe Imbalance": "chip-severe",
    }
    cls = mapping.get(str(severity), "chip-neutral")
    return f'<span class="chip {cls}">{severity}</span>'


def chip(label: str, kind: str = "info") -> str:
    """Small mono chip used for category / status tags."""
    return f'<span class="chip chip-{kind}">{label}</span>'


def callout(text: str, kind: str = "info") -> str:
    return f'<div class="callout {kind}">{text}</div>'


def section(title: str, subtitle: str = "") -> str:
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f'<div class="section-title">{title}</div>{sub}'


def chart_header(title: str, meta: str = "OPTIONS") -> str:
    """Panel header in the reference style: mono uppercase title + ⚙ chip."""
    return (
        f'<div class="chart-head"><span class="chart-title">{title}</span>'
        f'<span class="chart-meta">⚙ {meta}</span></div>'
    )


def chart_description(text: str) -> str:
    """Small reading guide printed under every figure."""
    if not text:
        return ""
    return (
        '<div class="chart-desc"><span class="desc-icon">▸</span>'
        f"{text}</div>"
    )


def footer(text: str = "ML2 BENCHMARK STUDIO · COMPARE, TRAIN, DEPLOY") -> str:
    return (
        '<div style="margin-top:2.2rem;padding-top:1rem;border-top:1px solid var(--line);'
        'color:var(--muted);font-family:var(--mono);font-size:0.66rem;letter-spacing:0.12em;'
        f'text-align:center;">{text}</div>'
    )


# ------------------------------------------------------------------------------
# Theme switcher & collapsible sidebar
# ------------------------------------------------------------------------------
_THEME_ICONS = {"System": "🖥️", "Light": "☀️", "Night": "🌙"}


def theme_switcher() -> str:
    """Interactive System / Light / Night switch. Returns the concrete theme."""
    if st.session_state.get("theme_mode") not in THEME_OPTIONS:
        st.session_state["theme_mode"] = "System"

    st.segmented_control(
        "Theme",
        list(THEME_OPTIONS),
        key="theme_mode",
        label_visibility="collapsed",
        format_func=lambda value: _THEME_ICONS.get(str(value), "🌙"),
        help="System follows your operating system's light/dark setting.",
    )
    mode = requested_mode()
    st.markdown(
        f'<div class="theme-note">theme · <strong>{mode}</strong></div>',
        unsafe_allow_html=True,
    )
    return resolve_theme()


def sidebar_brand(
    name: str = "ML2 Benchmark",
    subtitle: str = "Studio · v3.0",
    mark: Optional[str] = None,
) -> None:
    """Product title pinned to the very top of the sidebar (text only)."""
    glyph = (mark or name.strip()[:1] or "M").upper()
    st.markdown(
        f'<div class="brand"><div class="brand-mark">{glyph}</div>'
        f'<div><div class="brand-name">{name}</div>'
        f'<div class="brand-sub">{subtitle}</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )


_MENU_TOGGLE_HTML = """
<script>
(function () {
    var pwin = window.parent, pdoc = pwin.document;
    if (!pdoc || !pdoc.body) return;

    function sidebarEl() { return pdoc.querySelector('[data-testid="stSidebar"]'); }
    function isCollapsed() {
        if (pdoc.querySelector('[data-testid="stSidebarCollapsedControl"]')) { return true; }
        var sidebar = sidebarEl();
        if (!sidebar) { return true; }
        if (sidebar.getAttribute('aria-expanded') === 'false') { return true; }
        // Streamlit 1.6x keeps the collapsed sidebar mounted at ~1px wide and
        // exposes no collapsed-control testid, so measure it instead.
        return sidebar.getBoundingClientRect().width <= 40;
    }
    function control() {
        return pdoc.querySelector('[data-testid="stSidebarCollapseButton"] button')
            || pdoc.querySelector('[data-testid="stSidebarCollapsedControl"] button')
            || pdoc.querySelector('[data-testid="collapsedControl"] button')
            || pdoc.querySelector('[data-testid="stSidebarCollapseButton"]');
    }

    // Colours follow the active theme, even when it changes after the button
    // was created: the render loop below reads these at paint time.
    pwin.__aesMenuColors = {
        line: '__LINE__', bg: '__BG__', text: '__TEXT__',
        accent: '__ACCENT__', highlight: '__HIGHLIGHT__'
    };

    var btn = pdoc.getElementById('aes-menu-toggle');
    function restyle() {
        var c = pwin.__aesMenuColors;
        btn.style.background = c.bg;
        btn.style.color = c.text;
        btn.style.borderColor = c.line;
    }
    if (!btn) {
        btn = pdoc.createElement('button');
        btn.id = 'aes-menu-toggle';
        btn.type = 'button';
        btn.setAttribute('aria-label', 'Toggle menu');
        // Frosted glass: translucent fill, blur of whatever is behind it and a
        // bright inner highlight along the top edge.
        btn.style.cssText = [
            'position:fixed', 'top:10px', 'left:12px', 'z-index:1000000',
            'width:38px', 'height:38px', 'border-radius:12px',
            'display:flex', 'align-items:center', 'justify-content:center',
            'border:1px solid transparent',
            'font-family:"JetBrains Mono", ui-monospace, monospace',
            'font-size:18px', 'font-weight:700', 'line-height:1',
            'padding:0 0 3px', 'cursor:pointer',
            'backdrop-filter:blur(16px) saturate(1.4)',
            '-webkit-backdrop-filter:blur(16px) saturate(1.4)',
            'box-shadow:0 14px 30px -18px rgba(0,0,0,.85), inset 0 1px 0 rgba(255,255,255,.28)',
            'transition:left .28s cubic-bezier(.4,0,.2,1),transform .15s ease,' +
            'color .15s ease,border-color .15s ease,background .3s ease'
        ].join(';');
        btn.addEventListener('mouseenter', function () {
            var c = pwin.__aesMenuColors;
            btn.style.color = c.accent;
            btn.style.borderColor = c.accent;
            btn.style.transform = 'scale(1.06)';
        });
        btn.addEventListener('mouseleave', function () {
            btn.style.transform = 'scale(1)';
            restyle();
        });
        btn.addEventListener('click', function () {
            var target = control();
            if (target) {
                target.click();
            } else {
                // Fallback: Streamlit's own control may be a div rather than a
                // button — dispatch a real MouseEvent on the first match.
                var fallback = pdoc.querySelector('[data-testid="stSidebarCollapseButton"]')
                    || pdoc.querySelector('[data-testid="collapsedControl"]');
                if (fallback) {
                    fallback.dispatchEvent(new pwin.MouseEvent('click', { bubbles: true, cancelable: true }));
                }
            }
            setTimeout(sync, 140);
            setTimeout(sync, 420);
            setTimeout(sync, 800);
        });
        pdoc.body.appendChild(btn);
    }

    // One button, two jobs: « collapses the open menu, » reopens it.
    // Every write is guarded: a blind textContent/style write inside a
    // subtree MutationObserver would re-trigger the observer forever.
    function sync() {
        var collapsed = isCollapsed();
        var glyph = collapsed ? '\\u00BB' : '\\u00AB';
        var title = collapsed ? 'Open menu' : 'Hide menu';

        if (btn.textContent !== glyph) { btn.textContent = glyph; }
        if (btn.title !== title) { btn.title = title; }
        if (btn.getAttribute('aria-expanded') !== (collapsed ? 'false' : 'true')) {
            btn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
        }

        var left = 12;
        if (!collapsed) {
            var sidebar = sidebarEl();
            var width = sidebar ? sidebar.getBoundingClientRect().width : 336;
            left = Math.max(10, Math.round(width - 46));
        }
        var leftPx = left + 'px';
        if (btn.style.left !== leftPx) { btn.style.left = leftPx; }
        restyle();
    }

    if (!pwin.__aesMenuObserver) {
        try {
            var pending = false;
            pwin.__aesMenuObserver = new pwin.MutationObserver(function () {
                if (pending) { return; }
                pending = true;
                pwin.requestAnimationFrame(function () {
                    pending = false;
                    sync();
                });
            });
            pwin.__aesMenuObserver.observe(pdoc.body, { childList: true, subtree: true });
        } catch (err) {}
    }
    sync();
    setInterval(sync, 700);
})();
</script>
"""


def sidebar_toggle(mode: str = "dark") -> None:
    """Frosted-glass double-chevron button: hides *and* reopens the menu."""
    colors = theme_tokens(mode)
    html = (
        _MENU_TOGGLE_HTML
        .replace("__LINE__", colors["glass_border"])
        .replace("__BG__", colors["glass_bg"])
        .replace("__TEXT__", colors["glass_text"])
        .replace("__ACCENT__", colors["cyan"])
        .replace("__HIGHLIGHT__", colors["glass_highlight"])
    )
    components.html(html, height=0, scrolling=False)


_PARTICLES_HTML = """
<script>
(function () {
    var pwin = window.parent, pdoc = pwin.document;
    if (!pdoc || !pdoc.body) return;

    // Colours may change with the theme: the running loop reads them live.
    pwin.__aesColors = ['__A__', '__B__'];

    if (pdoc.getElementById('aes-particles')) return;

    var canvas = pdoc.createElement('canvas');
    canvas.id = 'aes-particles';
    canvas.setAttribute('aria-hidden', 'true');
    pdoc.body.insertBefore(canvas, pdoc.body.firstChild);

    var ctx = canvas.getContext('2d');
    var dpr = Math.min(1.5, pwin.devicePixelRatio || 1);
    var w = 0, h = 0, dots = [];
    var LINK = 18000;   // squared link distance (~134 px)

    function seed() {
        var count = Math.max(24, Math.min(64, Math.round((w * h) / 38000)));
        dots = [];
        for (var i = 0; i < count; i++) {
            dots.push({
                x: Math.random() * w,
                y: Math.random() * h,
                vx: (Math.random() - 0.5) * 0.26,
                vy: (Math.random() - 0.5) * 0.26,
                r: 0.8 + Math.random() * 1.8,
                t: Math.random() * Math.PI * 2,
                s: 0.006 + Math.random() * 0.012
            });
        }
    }

    function resize() {
        w = pwin.innerWidth;
        h = pwin.innerHeight;
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        seed();
    }

    function draw() {
        var colors = pwin.__aesColors || ['34,211,238', '167,139,250'];
        ctx.clearRect(0, 0, w, h);

        for (var i = 0; i < dots.length; i++) {
            for (var j = i + 1; j < dots.length; j++) {
                var dx = dots[i].x - dots[j].x;
                var dy = dots[i].y - dots[j].y;
                var d2 = dx * dx + dy * dy;
                if (d2 < LINK) {
                    var alpha = (1 - d2 / LINK) * 0.17;
                    ctx.strokeStyle = 'rgba(' + colors[i % 2] + ',' + alpha.toFixed(3) + ')';
                    ctx.lineWidth = 0.7;
                    ctx.beginPath();
                    ctx.moveTo(dots[i].x, dots[i].y);
                    ctx.lineTo(dots[j].x, dots[j].y);
                    ctx.stroke();
                }
            }
        }

        for (var k = 0; k < dots.length; k++) {
            var p = dots[k];
            var pulse = 0.35 + 0.35 * Math.sin(p.t);
            var tone = colors[k % 2];
            // Cheap two-pass glow: a faint halo plus a bright core, no
            // per-dot shadow blur (that would cost a full repaint each frame).
            ctx.beginPath();
            ctx.fillStyle = 'rgba(' + tone + ',' + (0.05 + pulse * 0.10).toFixed(3) + ')';
            ctx.arc(p.x, p.y, p.r * 3.6, 0, Math.PI * 2);
            ctx.fill();
            ctx.beginPath();
            ctx.fillStyle = 'rgba(' + tone + ',' + (0.22 + pulse * 0.55).toFixed(3) + ')';
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function step() {
        for (var i = 0; i < dots.length; i++) {
            var p = dots[i];
            p.x += p.vx;
            p.y += p.vy;
            p.t += p.s;
            if (p.x < -24) { p.x = w + 24; }
            if (p.x > w + 24) { p.x = -24; }
            if (p.y < -24) { p.y = h + 24; }
            if (p.y > h + 24) { p.y = -24; }
        }
        draw();
        pwin.__aesRaf = pwin.requestAnimationFrame(step);
    }

    resize();
    pwin.addEventListener('resize', resize);

    if (pwin.matchMedia && pwin.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        draw();
    } else {
        step();
    }
})();
</script>
"""


def background_layer(mode: str = "dark") -> None:
    """Animated particle network painted between the artwork and the app."""
    colors = theme_tokens(mode)
    html = (
        _PARTICLES_HTML.replace("__A__", colors.get("particle_a", "34, 211, 238"))
        .replace("__B__", colors.get("particle_b", "167, 139, 250"))
    )
    components.html(html, height=0, scrolling=False)


_CHART_MOTION_HTML = """
<script>
(function () {
    var pwin = window.parent, pdoc = pwin.document;
    if (!pdoc || !pdoc.body || pwin.__aesChartMotion) { return; }
    pwin.__aesChartMotion = true;

    var reduced = pwin.matchMedia
        && pwin.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // 1. Lines draw themselves, like a pen tracing the series.
    function drawLines(gd) {
        var lines = gd.querySelectorAll('path.js-line');
        for (var i = 0; i < lines.length; i++) {
            (function (path) {
                var length = 0;
                try { length = path.getTotalLength(); } catch (e) { return; }
                if (!length || length > 60000) { return; }
                path.style.strokeDasharray = length + ' ' + length;
                path.style.strokeDashoffset = length;
                pwin.requestAnimationFrame(function () {
                    path.style.transition =
                        'stroke-dashoffset 1.3s cubic-bezier(.4,0,.2,1)';
                    path.style.strokeDashoffset = '0';
                    pwin.setTimeout(function () {
                        path.style.transition = '';
                        path.style.strokeDasharray = '';
                        path.style.strokeDashoffset = '';
                    }, 1500);
                });
            })(lines[i]);
        }
    }

    // 2. Bars grow from their baseline.
    function growBars(gd) {
        var horizontal = !!(gd.data && gd.data[0] && gd.data[0].orientation === 'h');
        var groups = gd.querySelectorAll('.barlayer .trace');
        for (var i = 0; i < groups.length; i++) {
            var el = groups[i];
            el.style.transformBox = 'fill-box';
            el.style.transformOrigin = horizontal ? 'left center' : 'center bottom';
            try {
                el.animate(
                    [{ transform: horizontal ? 'scaleX(0.002)' : 'scaleY(0.002)' },
                     { transform: 'scale(1)' }],
                    { duration: 750, easing: 'cubic-bezier(.22,1,.36,1)' }
                );
            } catch (e) {}
        }
    }

    // 3. Pies, heatmaps and boxes fade/scale into place.
    function fadeRest(gd) {
        var pies = gd.querySelectorAll('.pielayer');
        for (var i = 0; i < pies.length; i++) {
            try {
                pies[i].animate(
                    [{ opacity: 0, transform: 'scale(.82)' },
                     { opacity: 1, transform: 'scale(1)' }],
                    { duration: 700, easing: 'ease-out' }
                );
            } catch (e) {}
        }
        var layers = gd.querySelectorAll('.heatmaplayer, .boxlayer, .histogramlayer');
        for (var j = 0; j < layers.length; j++) {
            try { layers[j].animate([{ opacity: 0 }, { opacity: 1 }], { duration: 800, easing: 'ease-out' }); }
            catch (e) {}
        }
    }

    function animate(gd, attempt) {
        if (!gd || gd.dataset.aesMotion === 'done') { return; }
        var ready = gd.querySelector(
            'path.js-line, .barlayer .trace, .pielayer, .heatmaplayer, .boxlayer'
        );
        if (!ready && attempt < 10) {
            pwin.setTimeout(function () { animate(gd, attempt + 1); }, 220);
            return;
        }
        gd.dataset.aesMotion = 'done';
        if (reduced) { return; }
        try { drawLines(gd); } catch (e) {}
        try { growBars(gd); } catch (e) {}
        try { fadeRest(gd); } catch (e) {}
    }

    function scan() {
        var plots = pdoc.querySelectorAll('.js-plotly-plot');
        for (var i = 0; i < plots.length; i++) { animate(plots[i], 0); }
    }

    var queued = false;
    function schedule() {
        if (queued) { return; }
        queued = true;
        pwin.requestAnimationFrame(function () { queued = false; scan(); });
    }

    try {
        new pwin.MutationObserver(schedule)
            .observe(pdoc.body, { childList: true, subtree: true });
    } catch (e) {}
    scan();
    pwin.setInterval(schedule, 1500);
})();
</script>
"""


def chart_motion() -> None:
    """Give every Plotly figure a short entrance animation (draw / grow / fade)."""
    components.html(_CHART_MOTION_HTML, height=0, scrolling=False)
