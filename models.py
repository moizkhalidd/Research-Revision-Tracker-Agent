"""Plain data classes shared between modules."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PassageDraft:
    """A passage produced by document_reader before it is stored."""
    ord: int
    heading: str
    level: int
    page: int | None
    text: str


@dataclass
class ParsedDocument:
    file_format: str                      # 'pdf' | 'latex' | 'text'
    full_text: str
    passages: list[PassageDraft]
    page_count: int | None
    notes: list[str]
    metadata: dict
    title: str | None = None


@dataclass
class Passage:
    """A stored passage (row of the passages table)."""
    passage_id: str
    version_id: str
    ord: int
    heading: str
    page: int | None
    text: str
    injection_flag: bool = False

    def ref(self) -> str:
        page = "p.%d" % self.page if self.page else "page n/a"
        return "%s (%s)" % (self.heading or "no heading", page)


@dataclass
class Hit:
    passage: Passage
    score: float
    origin: str                           # changed | context | full_document | prior_still_present | removed
    matched: list[str] = field(default_factory=list)
    label: str | None = None              # set for prior_still_present hits


@dataclass
class EvidenceItem:
    passage_id: str
    version_id: str
    heading: str
    page: int | None
    quote: str
    origin: str
    label: str | None = None

    def ref(self) -> str:
        page = "p.%d" % self.page if self.page else "page n/a"
        return "%s (%s)" % (self.heading or "no heading", page)


@dataclass
class VerdictResult:
    comment_id: str
    comment_rev_id: str
    verdict: str
    explanation: str
    missing_evidence: str
    contradiction: str
    claim_assessment: str
    regression: bool
    previous_verdict: str | None
    previous_source: str | None
    action_code: str
    recommended_action: str
    follow_up_question: str
    needs_manual: bool
    injection_flag: int
    evidence: list[EvidenceItem]
    guards: list[str]
    engine: str
    memory_mode: str
    plan: dict
    memory_influenced: bool = False
