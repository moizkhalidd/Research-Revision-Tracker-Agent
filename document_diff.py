"""Version comparison with difflib.

Passages of the old and new version are aligned with SequenceMatcher. Replaced
blocks are paired passage by passage (best similarity first) into 'modified'
changes; unpaired passages become 'added' or 'removed'.

A changed passage is only a lead: the diff proves that text changed, not that
a reviewer comment was addressed. Judging that is the job of the evidence step.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass


@dataclass
class ChangeDraft:
    kind: str                 # added | removed | modified
    old_ord: int | None
    new_ord: int | None
    similarity: float | None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def diff_passages(old_texts: list[str], new_texts: list[str]) -> list[ChangeDraft]:
    a = [_norm(t) for t in old_texts]
    b = [_norm(t) for t in new_texts]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    changes: list[ChangeDraft] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        olds, news = list(range(i1, i2)), list(range(j1, j2))
        pairs = []
        for j in news:
            best, best_r = None, 0.0
            for i in olds:
                r = difflib.SequenceMatcher(None, a[i], b[j]).ratio()
                if r > best_r:
                    best, best_r = i, r
            if best is not None and best_r >= 0.5:
                pairs.append((best, j, best_r))
                olds.remove(best)
        paired_new = {j for _, j, _ in pairs}
        for i, j, r in sorted(pairs, key=lambda x: x[1]):
            changes.append(ChangeDraft("modified", i, j, round(r, 3)))
        for j in news:
            if j not in paired_new:
                changes.append(ChangeDraft("added", None, j, None))
        for i in olds:
            changes.append(ChangeDraft("removed", i, None, None))
    return changes


def summarize(changes: list[ChangeDraft], old_len: int, new_len: int) -> dict:
    counts = {"added": 0, "removed": 0, "modified": 0}
    for c in changes:
        counts[c.kind] += 1
    return {"old_passages": old_len, "new_passages": new_len, **counts}
