"""
AI interpretation layer (Groq Cloud, OpenAI-compatible chat completions).

The client is provider-agnostic but defaults to Groq and
``llama-3.3-70b-versatile``. Credentials are resolved from, in order:

1. explicit in-app overrides (Streamlit session only),
2. environment variables (``GROQ_API_KEY`` / ``AI_API_KEY`` / ``AI_BASE_URL`` …),
3. an optional ``.env`` file (loaded with python-dotenv when installed),
4. Streamlit secrets.

Keys are never persisted or logged. The generated report is cached in SQLite by
the caller (``core.storage``) together with a hash of its context.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import (
    AI_BASE_URL,
    AI_MAX_CONTEXT_CHARS,
    AI_MODEL,
    AI_PROMPT_VERSION,
    AI_PROVIDER,
    AI_TIMEOUT_SECONDS,
)


def _find_env_path() -> Optional[Path]:
    """Locate the .env file across project directories or within a .env folder."""
    here = Path(__file__).resolve().parent  # core/
    dashboard_dir = here.parent             # ML-Benchmark-Dashboard-1/
    workspace_dir = dashboard_dir.parent    # TP/

    search_roots = [dashboard_dir, Path.cwd(), workspace_dir]
    seen = set()

    for root in search_roots:
        if not root.exists():
            continue
        real_root = root.resolve()
        if real_root in seen:
            continue
        seen.add(real_root)

        # 1. Standard .env file
        candidate = real_root / ".env"
        if candidate.is_file():
            return candidate

        # 2. Case where user created a folder named ".env"
        if candidate.is_dir():
            for name in (".env", "groq.env", "key.env", "api_key.env"):
                sub_candidate = candidate / name
                if sub_candidate.is_file():
                    return sub_candidate
            for sub_file in candidate.iterdir():
                if sub_file.is_file() and not sub_file.name.startswith(".git"):
                    return sub_file

    return None


def _load_env_file(path: Optional[Path] = None, override: bool = False) -> Dict[str, str]:
    """Parse .env key-value pairs without requiring external dependencies."""
    target = path or _find_env_path()
    loaded: Dict[str, str] = {}
    if not target or not target.is_file():
        return loaded

    try:
        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k:
                    loaded[k] = v
                    if override or k not in os.environ or not os.environ[k]:
                        os.environ[k] = v
    except Exception:
        pass

    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=target, override=override)
    except Exception:
        pass

    return loaded


# Initial load at import time
_load_env_file()

# ------------------------------------------------------------------------------
# Provider presets
# ------------------------------------------------------------------------------
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "groq": {
        "label": "Groq Cloud",
        "base_url": AI_BASE_URL,
        "default_model": AI_MODEL,
        "models": [
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ],
        "env_keys": ["GROQ_API_KEY", "AI_API_KEY"],
        "requires_key": True,
    },
    "xai": {
        "label": "xAI Grok",
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-2-latest",
        "models": ["grok-2-latest", "grok-beta"],
        "env_keys": ["XAI_API_KEY", "GROK_API_KEY", "AI_API_KEY"],
        "requires_key": True,
    },
    "ollama": {
        "label": "Local Ollama",
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.1:8b",
        "models": ["llama3.1:8b", "qwen2.5:7b", "mistral:7b"],
        "env_keys": ["OLLAMA_API_KEY", "AI_API_KEY"],
        "requires_key": False,
    },
    "custom": {
        "label": "Custom OpenAI-compatible",
        "base_url": AI_BASE_URL,
        "default_model": AI_MODEL,
        "models": [AI_MODEL],
        "env_keys": ["AI_API_KEY"],
        "requires_key": True,
    },
}


class AIInsightError(RuntimeError):
    """User-facing failure with an actionable message."""


@dataclass
class AISettings:
    provider: str
    base_url: str
    model: str
    api_key: Optional[str]
    key_source: str = "none"
    timeout: int = AI_TIMEOUT_SECONDS
    prompt_version: str = AI_PROMPT_VERSION
    requires_key: bool = True

    @property
    def is_configured(self) -> bool:
        if not self.requires_key:
            return bool(self.base_url)
        return bool(self.api_key)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @property
    def host(self) -> str:
        return self.base_url.split("://")[-1].split("/")[0]

    def masked_key(self) -> str:
        if not self.api_key:
            return "—"
        tail = self.api_key[-4:] if len(self.api_key) > 8 else ""
        return f"••••{tail}"

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "label": PROVIDERS.get(self.provider, {}).get("label", self.provider),
            "base_url": self.base_url,
            "host": self.host,
            "model": self.model,
            "key_source": self.key_source,
            "key": self.masked_key(),
            "configured": self.is_configured,
            "prompt_version": self.prompt_version,
        }


# ------------------------------------------------------------------------------
# Settings resolution
# ------------------------------------------------------------------------------
def _secret(name: str) -> Optional[str]:
    """Read a Streamlit secret if Streamlit is available and configured."""
    try:
        import streamlit as st  # local import: config must not depend on it

        value = st.secrets.get(name) if hasattr(st.secrets, "get") else None
        return str(value) if value else None
    except Exception:
        return None


def resolve_settings(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> AISettings:
    """Resolve provider/model/key from overrides, environment, .env file and secrets."""
    # Re-check .env file so any newly pasted keys are picked up dynamically
    env_file_vars = _load_env_file()
    env_path = _find_env_path()

    provider = (provider or os.getenv("AI_PROVIDER") or AI_PROVIDER).lower()
    preset = PROVIDERS.get(provider, PROVIDERS["custom"])
    provider = provider if provider in PROVIDERS else "custom"

    resolved_base = (
        base_url
        or os.getenv("AI_BASE_URL")
        or preset["base_url"]
    ).strip().rstrip("/")
    resolved_model = (
        model or os.getenv("AI_MODEL") or preset["default_model"]
    ).strip()

    key: Optional[str] = None
    source = "none"
    if api_key:
        key, source = api_key.strip(), "this session"
    else:
        for name in preset["env_keys"]:
            # Check if key is loaded from .env file directly
            if name in env_file_vars and env_file_vars[name].strip():
                key = env_file_vars[name].strip()
                source = f".env ({name})"
                break
            # Check os.environ
            value = os.getenv(name)
            if value and value.strip():
                key = value.strip()
                source = f"environment ({name})"
                break
        if not key:
            for name in preset["env_keys"]:
                value = _secret(name)
                if value and value.strip():
                    key, source = value.strip(), f"Streamlit secrets ({name})"
                    break

    return AISettings(
        provider=provider,
        base_url=resolved_base,
        model=resolved_model,
        api_key=key,
        key_source=source,
        requires_key=bool(preset.get("requires_key", True)),
    )


# ------------------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a senior applied-machine-learning analyst reviewing a benchmark "
    "experiment for a university ML project. You receive one JSON payload with "
    "the dataset metadata, data-quality diagnostics, model leaderboard(s), and "
    "the exact numbers behind every dashboard chart. Write precise, grounded "
    "English. Never invent numbers, models, or charts; quote the provided "
    "values to four decimals. Distinguish signal from noise, call out "
    "overfitting, class imbalance, and whether the preprocessing actually "
    "helped. Be critical and educational, not promotional."
)

REPORT_SCHEMA = {
    "summary": "string — 6 to 10 sentences summarising the experiment end to end",
    "key_findings": ["3 to 7 short, specific bullets"],
    "chart_explanations": {
        "<chart_key>": "one paragraph explaining what this chart shows and what it means here"
    },
    "model_analysis": {
        "best_model": "why the winner wins, with numbers",
        "overfitting": "which models overfit and the evidence",
        "tradeoffs": "accuracy vs latency vs complexity",
        "weak_models": "which models underperform and why",
    },
    "noise_impact": "effect of noise treatment / injection on performance, or 'Not available: no baseline-vs-treated comparison in this experiment.'",
    "outlier_impact": "effect of outlier handling, or 'Not available: no outlier statistics.'",
    "imputation_impact": "whether the chosen imputation plausibly matters, grounded in missingness stats",
    "recommendations": ["3 to 6 concrete next steps"],
    "limitations": ["2 to 5 caveats about this benchmark"],
}


def build_messages(context: Dict[str, Any]) -> List[Dict[str, str]]:
    schema = json.dumps(REPORT_SCHEMA, indent=2)
    user = (
        "Analyse this benchmark experiment.\n\n"
        "The payload contains `charts`: for every chart key you receive the "
        "chart title, a description of what it plots, and the data behind it. "
        "Explain every chart key present, and only those keys.\n\n"
        "Return ONLY one JSON object with exactly this shape "
        "(no markdown, no commentary outside the JSON):\n"
        f"{schema}\n\n"
        "Rules:\n"
        "- Use the numbers in the payload; do not estimate or invent.\n"
        "- If a section has no supporting data, say so explicitly.\n"
        "- `chart_explanations` must contain one entry per key in `charts`.\n"
        "- `noise_impact` must compare baseline and treated variants when both "
        "exist, citing the mean delta and the win/loss counts.\n"
        "- `outlier_impact` must cite the outlier percentages / removed rows "
        "when available.\n\n"
        "PAYLOAD:\n"
        f"{json.dumps(context, ensure_ascii=False, sort_keys=True)}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ------------------------------------------------------------------------------
# HTTP client
# ------------------------------------------------------------------------------
def _headers(settings: AISettings) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    return headers


def _post(settings: AISettings, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = requests.post(
            settings.endpoint,
            headers=_headers(settings),
            json=payload,
            timeout=(10, settings.timeout),
        )
    except requests.exceptions.ConnectionError as exc:
        hint = (
            " Is `ollama serve` running?" if settings.provider == "ollama" else ""
        )
        raise AIInsightError(
            f"Could not reach {settings.host}.{hint} Check the base URL and network."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise AIInsightError(
            f"{settings.host} did not answer within {settings.timeout}s. "
            "Try again or pick a faster model."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise AIInsightError(f"Request failed: {exc}") from exc

    if response.status_code in (429, 500, 502, 503, 504):
        # One retry for transient provider pressure.
        try:
            response = requests.post(
                settings.endpoint,
                headers=_headers(settings),
                json=payload,
                timeout=(10, settings.timeout),
            )
        except requests.exceptions.RequestException:
            pass

    if response.status_code == 401:
        raise AIInsightError(
            "Authentication failed (401). The API key is missing, revoked or for "
            "a different provider. Set GROQ_API_KEY in .env or paste it in the "
            "AI page."
        )
    if response.status_code == 404:
        try:
            err_msg = response.json().get("error", {}).get("message", "")
        except Exception:
            err_msg = response.text[:200]
        raise AIInsightError(
            f"HTTP 404 for model '{settings.model}' at {settings.host}: {err_msg}"
        )
    if response.status_code == 413:
        raise AIInsightError(
            f"Payload exceeded {settings.host} token limits (413). "
            "Please switch model to 'qwen/qwen3.8-27b' or wait a moment."
        )
    if response.status_code == 429:
        raise AIInsightError(
            f"Rate limit reached at {settings.host} (TPM/RPM exceeded). "
            "Groq free tier resets every 60 seconds — please wait a moment and click Generate again."
        )
    if response.status_code >= 400:
        detail = response.text[:400]
        raise AIInsightError(
            f"{settings.host} returned HTTP {response.status_code}: {detail}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise AIInsightError("The provider returned a non-JSON response.") from exc


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse a JSON object, tolerating code fences or leading prose."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Find the first balanced {...} block.
    start = cleaned.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
    raise AIInsightError(
        "The model replied, but its answer was not valid JSON. Press Regenerate."
    )


def _message_content(data: Dict[str, Any]) -> str:
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise AIInsightError(
            "Unexpected provider response: no message content. "
            f"Raw keys: {list(data.keys())[:6]}"
        ) from exc


def context_hash(context: Dict[str, Any]) -> str:
    payload = json.dumps(context, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def shrink_context(context: Dict[str, Any], limit: int = AI_MAX_CONTEXT_CHARS) -> Dict[str, Any]:
    """Drop optional and heavy payload sections until the prompt fits the budget."""
    working = dict(context)
    if len(json.dumps(working, ensure_ascii=False)) <= limit:
        return working

    # 1. Drop heavy metadata not needed for the core narrative
    working.pop("experiment_history", None)
    working.pop("chart_catalog", None)

    # 2. Compact data_quality if present (drop giant per-column & per-variant tables)
    if "data_quality" in working and isinstance(working["data_quality"], dict):
        dq = dict(working["data_quality"])
        dq.pop("variants", None)
        dq.pop("columns", None)
        working["data_quality"] = dq

    # 3. Drop heavy data-point chart dumps first
    charts = dict(working.get("charts") or {})
    for heavy_key in ("noise_profile", "score_vs_time", "heatmap", "radar", "residuals", "confusion_matrix"):
        if len(json.dumps(working, ensure_ascii=False)) <= limit:
            break
        charts.pop(heavy_key, None)
        working["charts"] = charts

    # 4. If still over limit, pop remaining charts until within budget
    while charts and len(json.dumps(working, ensure_ascii=False)) > limit:
        charts.popitem()
    working["charts"] = charts

    # 5. Last resort: keep core essentials only
    if len(json.dumps(working, ensure_ascii=False)) > limit:
        working = {
            key: value
            for key, value in working.items()
            if key in ("experiment", "task_type", "leaderboard", "comparison", "data_quality")
        }
        working["trimmed"] = True

    return working


def generate_report(
    context: Dict[str, Any],
    settings: Optional[AISettings] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Call the provider and return (normalised report, call metadata)."""
    settings = settings or resolve_settings()
    if not settings.is_configured:
        raise AIInsightError(
            "No API key configured. Add GROQ_API_KEY to a .env file, put it in "
            "Streamlit secrets, or paste it in the AI page."
        )

    compact = shrink_context(context)
    messages = build_messages(compact)
    payload: Dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }

    try:
        data = _post(settings, payload)
    except AIInsightError as exc:
        # Some OpenAI-compatible servers reject response_format; retry once
        # without it and rely on the parser.
        if "invalid" not in str(exc).lower() and "response_format" not in str(exc).lower():
            raise
        payload.pop("response_format", None)
        data = _post(settings, payload)

    content = _message_content(data)
    report = _extract_json(content)
    report = normalize_report(report)

    usage = data.get("usage") or {}
    meta = {
        "provider": settings.provider,
        "model": settings.model,
        "prompt_version": settings.prompt_version,
        "context_hash": context_hash(compact),
        "prompt_chars": len(json.dumps(compact, ensure_ascii=False)),
        "usage": usage,
        "trimmed": bool(compact.get("trimmed")),
    }
    return report, meta


def normalize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce the model's JSON into the expected shape with safe defaults."""
    normalized: Dict[str, Any] = {
        "summary": str(report.get("summary") or "").strip(),
        "key_findings": _as_list(report.get("key_findings")),
        "chart_explanations": {
            str(key): str(value)
            for key, value in (report.get("chart_explanations") or {}).items()
            if value
        },
        "model_analysis": report.get("model_analysis") or {},
        "noise_impact": str(report.get("noise_impact") or "").strip(),
        "outlier_impact": str(report.get("outlier_impact") or "").strip(),
        "imputation_impact": str(report.get("imputation_impact") or "").strip(),
        "recommendations": _as_list(report.get("recommendations")),
        "limitations": _as_list(report.get("limitations")),
    }
    if not isinstance(normalized["model_analysis"], dict):
        normalized["model_analysis"] = {"best_model": str(normalized["model_analysis"])}
    return normalized


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return [str(value)]


def test_connection(settings: Optional[AISettings] = None) -> Tuple[bool, str]:
    """Cheap round-trip used by the dashboard's Test connection button."""
    settings = settings or resolve_settings()
    if not settings.is_configured:
        return False, "No API key configured."
    payload = {
        "model": settings.model,
        "messages": [
            {"role": "user", "content": "Reply with the single word OK."}
        ],
        "temperature": 0.0,
        "max_tokens": 5,
    }
    try:
        data = _post(settings, payload)
        content = _message_content(data).strip()
        return True, f"Connected to {settings.host} · {settings.model} · reply: {content[:40]}"
    except AIInsightError as exc:
        return False, str(exc)


def report_to_markdown(report: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> str:
    """Render a cached report as a downloadable Markdown document."""
    lines: List[str] = ["# AI Benchmark Interpretation", ""]
    if meta:
        lines.append(
            f"_Generated with {meta.get('provider')} / {meta.get('model')} · "
            f"prompt {meta.get('prompt_version')}_"
        )
        lines.append("")

    lines.append("## Summary")
    lines.append(report.get("summary") or "_No summary._")
    lines.append("")

    findings = report.get("key_findings") or []
    if findings:
        lines.append("## Key findings")
        lines.extend(f"- {item}" for item in findings)
        lines.append("")

    analysis = report.get("model_analysis") or {}
    if analysis:
        lines.append("## Model analysis")
        for key, value in analysis.items():
            lines.append(f"**{str(key).replace('_', ' ').title()}**")
            lines.append(str(value))
            lines.append("")

    explanations = report.get("chart_explanations") or {}
    if explanations:
        lines.append("## Chart explanations")
        for key, value in explanations.items():
            lines.append(f"### {str(key).replace('_', ' ').title()}")
            lines.append(str(value))
            lines.append("")

    for section, title in (
        ("noise_impact", "Noise impact"),
        ("outlier_impact", "Outlier impact"),
        ("imputation_impact", "Imputation impact"),
    ):
        if report.get(section):
            lines.append(f"## {title}")
            lines.append(str(report[section]))
            lines.append("")

    for section, title in (
        ("recommendations", "Recommendations"),
        ("limitations", "Limitations"),
    ):
        items = report.get(section) or []
        if items:
            lines.append(f"## {title}")
            lines.extend(f"- {item}" for item in items)
            lines.append("")

    return "\n".join(lines)
