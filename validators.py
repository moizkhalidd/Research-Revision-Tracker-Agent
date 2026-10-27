"""Input validation and untrusted-content handling.

Anything that comes from a document, a reviewer comment file or the LLM is
untrusted. This module holds the deterministic checks that keep it from
changing permissions, paper identity or approvals.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import config


class ValidationError(ValueError):
    """User-correctable input problem."""


def validate_paper_id(paper_id: str) -> str:
    if not isinstance(paper_id, str) or not config.PAPER_ID_RE.match(paper_id):
        raise ValidationError(
            "Invalid paper ID %r. Use 3-40 characters: a letter first, then letters, digits, '_' or '-'." % (paper_id,))
    return paper_id


def validate_comment_id(comment_id: str) -> str:
    if not isinstance(comment_id, str) or not config.COMMENT_ID_RE.match(comment_id):
        raise ValidationError(
            "Invalid comment ID %r. Use 1-20 characters: letters, digits, '.', '_' or '-'." % (comment_id,))
    return comment_id


def validate_input_file(path_str: str) -> Path:
    """Resolve and check a user-supplied document path (no directories, size and type limits)."""
    if not path_str or "\x00" in path_str:
        raise ValidationError("A file path is required.")
    p = Path(path_str).expanduser()
    try:
        p = p.resolve(strict=True)
    except (FileNotFoundError, OSError):
        raise ValidationError("File not found: %s" % path_str)
    if not p.is_file():
        raise ValidationError("Not a regular file: %s" % path_str)
    if p.suffix.lower() not in config.ALLOWED_SUFFIXES:
        raise ValidationError("Unsupported file type %r. Allowed: %s" % (p.suffix, ", ".join(config.ALLOWED_SUFFIXES)))
    size = p.stat().st_size
    if size == 0:
        raise ValidationError("File is empty: %s" % path_str)
    if size > config.MAX_FILE_BYTES:
        raise ValidationError("File is larger than %d MB: %s" % (config.MAX_FILE_BYTES // (1024 * 1024), path_str))
    return p


def clean_text(value, limit: int = config.MAX_TEXT_FIELD, field: str = "text") -> str:
    if not isinstance(value, str):
        raise ValidationError("%s must be a string." % field)
    value = "".join(ch for ch in value if ch in "\n\t" or unicodedata.category(ch)[0] != "C").strip()
    if not value:
        raise ValidationError("%s must not be empty." % field)
    if len(value) > limit:
        raise ValidationError("%s is longer than %d characters." % (field, limit))
    return value


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# Reviewer comment file

def parse_comments_json(path: Path) -> dict:
    """Parse a reviewer comment file.

    Accepted shapes:
      [ {id, text, reviewer?, checks?}, ... ]
      { "synthetic": true, "comments": [ ... ] }
    A check is {"label": str, "any": [keywords], "numeric": bool}. Checks are an
    optional author-confirmed acceptance checklist for that comment.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValidationError("Could not read reviewer comments JSON (%s): %s" % (path, exc))
    synthetic = False
    if isinstance(raw, dict):
        synthetic = bool(raw.get("synthetic", False))
        raw = raw.get("comments")
    if not isinstance(raw, list) or not raw:
        raise ValidationError("Reviewer comments file must contain a non-empty list of comments.")
    if len(raw) > config.MAX_COMMENTS:
        raise ValidationError("Too many comments (max %d)." % config.MAX_COMMENTS)
    seen, out = set(), []
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValidationError("Comment #%d is not an object." % i)
        cid = validate_comment_id(str(item.get("id", "")))
        if cid in seen:
            raise ValidationError("Duplicate comment ID %s." % cid)
        seen.add(cid)
        out.append({
            "id": cid,
            "reviewer": clean_text(str(item.get("reviewer", "Reviewer")), 80, "reviewer"),
            "text": clean_text(item.get("text"), field="comment %s text" % cid),
            "checks": normalize_checks(item.get("checks", []), cid),
        })
    return {"comments": out, "synthetic": synthetic}


def normalize_checks(checks, cid: str = "?") -> list[dict]:
    if checks in (None, ""):
        return []
    if not isinstance(checks, list):
        raise ValidationError("checks for %s must be a list." % cid)
    out = []
    for c in checks:
        if not isinstance(c, dict) or not c.get("label") or not c.get("any"):
            raise ValidationError("Each check for %s needs a 'label' and a non-empty 'any' keyword list." % cid)
        kws = [clean_text(str(k), 80, "keyword") for k in c["any"]] if isinstance(c["any"], list) else None
        if not kws:
            raise ValidationError("'any' for a check in %s must be a list of keywords." % cid)
        out.append({"label": clean_text(str(c["label"]), 120, "check label"), "any": kws,
                    "numeric": bool(c.get("numeric", False))})
    return out


def parse_claims(raw: dict, valid_ids: set[str]) -> dict[str, dict]:
    """Normalise claims into {comment_id: {"claim": value, "note": str|None}}.

    Accepts {"R1.1": "resolved"} or {"R1.1": {"claim": "resolved", "note": "..."}}.
    """
    if not isinstance(raw, dict):
        raise ValidationError("Claims must be a JSON object mapping comment IDs to claims.")
    out = {}
    for cid, val in raw.items():
        if cid.startswith("_"):  # allow "_comment" keys in sample files
            continue
        validate_comment_id(cid)
        if cid not in valid_ids:
            raise ValidationError("Claim refers to unknown or inactive comment %s for this paper." % cid)
        note = None
        if isinstance(val, dict):
            note = val.get("note")
            val = val.get("claim")
        if val not in config.CLAIM_VALUES:
            raise ValidationError("Invalid claim %r for %s. Use one of: %s." % (val, cid, ", ".join(config.CLAIM_VALUES)))
        out[cid] = {"claim": val, "note": clean_text(note, 1000, "claim note") if note else None}
    return out


# Prompt-injection screening (heuristic; the real protection is structural)
_INJECTION_PATTERNS = [
    ("ignore_instructions", r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|system|all|any|your)\b[^.\n]{0,30}\b(instructions?|rules?|prompts?|polic(?:y|ies)|guidelines?)\b"),
    ("mark_all_addressed", r"\b(mark|set|label|record|report|treat)\b[^.\n]{0,40}\b(all|every|each)\b[^.\n]{0,40}\b(comments?|issues?|concerns?)\b[^.\n]{0,30}\b(addressed|resolved|fixed|done)\b"),
    ("auto_approve", r"\b(auto[- ]?approve|approve (?:all|every|them all)|skip (?:the )?(?:author )?approval)\b"),
    ("reveal_secret", r"\b(reveal|print|show|output|leak|send|expose|display)\b[^.\n]{0,50}\b(api[ _-]?key|secret|password|credentials?|token|system prompt|\.env)\b"),
    ("change_identity", r"\b(change|set|switch|rename|use|reassign)\b[^.\n]{0,30}\bpaper[ _-]?id\b"),
    ("role_override", r"\b(you are now|act as|new instructions?|from now on you|developer mode)\b"),
    ("forced_verdict", r"\bverdict\s*[:=]\s*(addressed|resolved)\b"),
    ("tag_spoofing", r"</?\s*(system|assistant|untrusted_document|instructions?)\b"),
]
_COMPILED = [(name, re.compile(rx, re.I)) for name, rx in _INJECTION_PATTERNS]


def detect_injection(text: str) -> list[str]:
    """Names of injection-style patterns found in text (empty list if none)."""
    return [name for name, rx in _COMPILED if rx.search(text or "")]


def sanitize_for_prompt(text: str, limit: int) -> str:
    """Make document text safe to embed inside a delimited prompt block."""
    text = "".join(ch for ch in (text or "") if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = text.replace("<", "(").replace(">", ")")
    text = normalize_ws(text) if len(text) > limit else text.strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."
