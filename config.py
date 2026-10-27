"""Central configuration: paths, vocabularies, limits and LLM settings.

Nothing secret is hard coded. The Groq key is read from the environment, which is filled
from a local .env file (see .env.example). All paths are resolved lazily so tests and the
evaluation sandbox can redirect the whole application with the RRT_HOME environment variable.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

try:  # python-dotenv is a declared dependency, but the app should still start without it
    from dotenv import load_dotenv

    load_dotenv(PROJECT_DIR / ".env")
except ImportError:  # pragma: no cover
    pass

APP_NAME = "Research Document Revision Tracker"
SCHEMA_VERSION = 1

# Paths


@dataclass(frozen=True)
class Paths:
    home: Path
    data: Path
    db: Path
    sources: Path
    exports: Path
    drafts: Path
    logs: Path
    failure_flag: Path


def get_paths() -> Paths:
    """Resolve application directories. RRT_HOME (if set) replaces the project directory."""
    home = Path(os.environ.get("RRT_HOME") or PROJECT_DIR).expanduser().resolve()
    data = home / "data"
    return Paths(home=home, data=data, db=data / "tracker.db", sources=data / "sources",
                 exports=home / "exports", drafts=home / "drafts", logs=home / "logs",
                 failure_flag=data / "failure_flag.json")


def ensure_dirs() -> Paths:
    p = get_paths()
    for d in (p.data, p.sources, p.exports, p.drafts, p.logs):
        d.mkdir(parents=True, exist_ok=True)
    return p


# Input limits and ID formats (validated in validators.py)
PAPER_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{2,39}$")
COMMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,19}$")
ALLOWED_SUFFIXES = (".pdf", ".tex", ".txt")
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_TEXT_FIELD = 4000
MAX_COMMENTS = 100
MAX_PASSAGE_CHARS = 1200      # passages longer than this are split at sentence boundaries
TOP_K_EVIDENCE = 6            # passages returned per evidence search
PROMPT_PASSAGE_CHARS = 1200   # characters of one passage sent to the LLM

# Vocabularies
VERDICTS = ("Addressed", "Partial", "Unverified", "Ask Author")
CLAIM_VALUES = ("resolved", "partial", "outstanding", "none")
FOLLOWUP_TOOLS = ("search_full_document", "verify_prior_evidence")

ACTION_DESCRIPTIONS = {
    "PRESENT_FOR_APPROVAL": "Evidence supports the claim: present the suggested verdict to the author for approval.",
    "RECHECK_NEXT_CYCLE": "Keep the comment open and re-check the missing requirement in the next revision.",
    "REQUEST_EVIDENCE": "Ask the author for specific supporting evidence (section, table, figure or result).",
    "ASK_AUTHOR": "A human decision or clarification is needed before this comment can be judged.",
    "MANUAL_INSPECTION": "Text comparison cannot settle this comment; inspect the revision manually.",
    "FLAG_REGRESSION": "A previously resolved comment is no longer supported: flag the regression for author review.",
    "CLOSE_COMMENT": "The author approved this comment as Addressed; nothing further is required.",
}

ALLOWED_ACTIONS_BY_VERDICT = {
    "Addressed": ("PRESENT_FOR_APPROVAL", "MANUAL_INSPECTION"),
    "Partial": ("RECHECK_NEXT_CYCLE", "REQUEST_EVIDENCE", "ASK_AUTHOR", "MANUAL_INSPECTION"),
    "Unverified": ("REQUEST_EVIDENCE", "RECHECK_NEXT_CYCLE", "ASK_AUTHOR", "MANUAL_INSPECTION"),
    "Ask Author": ("ASK_AUTHOR", "MANUAL_INSPECTION"),
}

DEFAULT_ACTION_BY_VERDICT = {
    "Addressed": "PRESENT_FOR_APPROVAL",
    "Partial": "RECHECK_NEXT_CYCLE",
    "Unverified": "RECHECK_NEXT_CYCLE",
    "Ask Author": "ASK_AUTHOR",
}

# Action chosen when the AUTHOR sets the final verdict by hand (author_review.py).
ACTION_FOR_AUTHOR_VERDICT = {
    "Addressed": "CLOSE_COMMENT",
    "Partial": "RECHECK_NEXT_CYCLE",
    "Unverified": "RECHECK_NEXT_CYCLE",
    "Ask Author": "ASK_AUTHOR",
}


# LLM settings
# llama-3.3-70b-versatile was retired by Groq on 2026-08-16 (see console.groq.com/docs/deprecations).
# openai/gpt-oss-120b is a production model and the replacement Groq recommends. openai/gpt-oss-20b is a
# cheaper production alternative. Change it with GROQ_MODEL in .env.
DEFAULT_MODEL = "openai/gpt-oss-120b"


@dataclass(frozen=True)
class LLMSettings:
    api_key: str | None
    model: str
    timeout: float
    max_retries: int
    max_completion_tokens: int


def get_llm_settings() -> LLMSettings:
    key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not key or key.lower().startswith("your_") or key.lower() in ("changeme", "xxx"):
        key = None
    return LLMSettings(
        api_key=key,
        model=(os.environ.get("GROQ_MODEL") or DEFAULT_MODEL).strip(),
        timeout=float(os.environ.get("GROQ_TIMEOUT", "60") or 60),
        max_retries=int(os.environ.get("GROQ_MAX_RETRIES", "2") or 2),
        max_completion_tokens=int(os.environ.get("GROQ_MAX_COMPLETION_TOKENS", "4096") or 4096),
    )
