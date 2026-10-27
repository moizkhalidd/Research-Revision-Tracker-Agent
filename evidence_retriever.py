"""Evidence retrieval over stored passages (plain SQLite + lexical scoring).

No vector database: the corpus is one paper, so an IDF-weighted term overlap is
enough, easy to explain and fully inspectable. Every query is scoped to one
paper_id and one version of that paper.

Tools implemented here:
  search_changed        passages added/modified in this revision, plus neighbours as context
  search_full_document  every passage of the current version
  verify_prior_evidence check whether quotes accepted in an earlier cycle still exist
"""
from __future__ import annotations

import math
import re
from collections import Counter

import config
from models import Hit, Passage
from validators import normalize_ws

_STOP = set("""a about above after again all also an and any are as at be been being below between both but by can
could did do does doing down during each few for from further had has have having he her here hers him his how i if
in into is it its itself just me more most my no nor not now of off on once only or other our out over own same she
should so some such than that the their theirs them then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your paper authors author please
reviewer reviewers provide report show state discuss describe include add given""".split())


def stem(tok: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z][a-z0-9\-]+|\d+(?:\.\d+)?", (text or "").lower())
    return [stem(t) for t in toks if t not in _STOP and len(t) > 2]


def _row_to_passage(r) -> Passage:
    return Passage(r["passage_id"], r["version_id"], r["ord"], r["heading"], r["page"], r["text"],
                   bool(r["injection_flag"]))


def load_passages(conn, paper_id: str, version_id: str) -> list[Passage]:
    rows = conn.execute(
        "SELECT * FROM passages WHERE paper_id = ? AND version_id = ? ORDER BY ord", (paper_id, version_id)
    ).fetchall()
    return [_row_to_passage(r) for r in rows]


def get_passage(conn, paper_id: str, passage_id: str) -> Passage | None:
    r = conn.execute("SELECT * FROM passages WHERE paper_id = ? AND passage_id = ?", (paper_id, passage_id)).fetchone()
    return _row_to_passage(r) if r else None


def changed_new_passages(conn, paper_id: str, diff_id: str) -> list[Passage]:
    rows = conn.execute(
        "SELECT p.* FROM changes c JOIN passages p ON p.passage_id = c.new_passage_id AND p.paper_id = c.paper_id"
        " WHERE c.paper_id = ? AND c.diff_id = ? AND c.kind IN ('added','modified') ORDER BY p.ord",
        (paper_id, diff_id)).fetchall()
    return [_row_to_passage(r) for r in rows]


def removed_old_passages(conn, paper_id: str, diff_id: str) -> list[Passage]:
    rows = conn.execute(
        "SELECT p.* FROM changes c JOIN passages p ON p.passage_id = c.old_passage_id AND p.paper_id = c.paper_id"
        " WHERE c.paper_id = ? AND c.diff_id = ? AND c.kind = 'removed' ORDER BY p.ord",
        (paper_id, diff_id)).fetchall()
    return [_row_to_passage(r) for r in rows]


def build_query(comment_text: str, checks: list[dict], interpretation: str | None = None) -> list[str]:
    terms = tokenize(comment_text) + tokenize(interpretation or "")
    for c in checks or []:
        terms += tokenize(c.get("label", ""))
        for kw in c.get("any", []):
            terms += tokenize(kw)
    return terms


def _idf(corpus: list[Passage]) -> dict[str, float]:
    df: Counter = Counter()
    for p in corpus:
        df.update(set(tokenize(p.text)))
    n = max(len(corpus), 1)
    return {t: math.log(1 + n / (1 + c)) for t, c in df.items()}


def _score(query_terms: list[str], passage: Passage, idf: dict[str, float]) -> tuple[float, list[str]]:
    ptoks = set(tokenize(passage.text))
    matched = sorted({t for t in query_terms if t in ptoks})
    score = sum(idf.get(t, 1.0) for t in matched)
    return round(score, 3), matched


def search_changed(conn, paper_id: str, version_id: str, diff_id: str, query_terms: list[str],
                   top_k: int = config.TOP_K_EVIDENCE) -> list[Hit]:
    """Rank changed passages; add the best hit's neighbours as lower-priority context."""
    corpus = load_passages(conn, paper_id, version_id)
    idf = _idf(corpus)
    changed = changed_new_passages(conn, paper_id, diff_id)
    hits = []
    for p in changed:
        if p.injection_flag:
            continue
        score, matched = _score(query_terms, p, idf)
        if score > 0:
            hits.append(Hit(p, score, "changed", matched))
    hits.sort(key=lambda h: -h.score)
    hits = hits[:top_k]
    seen = {h.passage.passage_id for h in hits}
    for h in list(hits[:2]):
        for nb in corpus:
            if nb.version_id == h.passage.version_id and nb.heading == h.passage.heading \
                    and abs(nb.ord - h.passage.ord) == 1 and nb.passage_id not in seen and not nb.injection_flag:
                score, matched = _score(query_terms, nb, idf)
                hits.append(Hit(nb, score, "context", matched))
                seen.add(nb.passage_id)
    return hits


def search_removed(conn, paper_id: str, diff_id: str, query_terms: list[str], top_k: int = 2) -> list[Hit]:
    """Removed passages that mention the comment's topic (informational, never evidence of resolution)."""
    removed = removed_old_passages(conn, paper_id, diff_id)
    if not removed:
        return []
    idf = _idf(removed)
    hits = []
    for p in removed:
        score, matched = _score(query_terms, p, idf)
        if score > 0:
            hits.append(Hit(p, score, "removed", matched))
    hits.sort(key=lambda h: -h.score)
    return hits[:top_k]


def search_full_document(conn, paper_id: str, version_id: str, query_terms: list[str],
                         top_k: int = config.TOP_K_EVIDENCE) -> list[Hit]:
    corpus = load_passages(conn, paper_id, version_id)
    idf = _idf(corpus)
    hits = []
    for p in corpus:
        if p.injection_flag:
            continue
        score, matched = _score(query_terms, p, idf)
        if score > 0:
            hits.append(Hit(p, score, "full_document", matched))
    hits.sort(key=lambda h: -h.score)
    return hits[:top_k]


def verify_prior_evidence(conn, paper_id: str, version_id: str, prior_items: list[dict]) -> list[dict]:
    """For each earlier evidence quote, report whether it still exists in the current version.

    prior_items: [{"quote", "label", ...}]. Returns the same dicts plus
    "present" (bool) and, when present, "passage_id" of the current passage holding the quote.
    """
    corpus = load_passages(conn, paper_id, version_id)
    norm = [(p, normalize_ws(p.text).lower()) for p in corpus if not p.injection_flag]
    out = []
    for item in prior_items:
        q = normalize_ws(item.get("quote", "")).lower()
        found = next((p for p, t in norm if q and q in t), None)
        entry = dict(item)
        entry["present"] = found is not None
        entry["passage_id"] = found.passage_id if found else None
        out.append(entry)
    return out


def sentences_of(text: str) -> list[str]:
    """Sentence-like units of a passage as exact substrings (table rows count as units)."""
    parts = re.split(r"(?<=[.!?])\s+|\n", text)
    return [p.strip() for p in parts if p and p.strip()]
