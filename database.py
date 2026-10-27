"""SQLite connection, transaction helper and the complete schema.

Design notes (these are the points to explain in the viva)
  * SQLite is the single source of truth. Excel and draft files are views generated from it.
  * Connections run in autocommit mode (isolation_level=None) and every multi-row change is wrapped
    in transaction(conn), which issues BEGIN IMMEDIATE / COMMIT / ROLLBACK explicitly. A failure
    inside the block therefore rolls back every related row together.
  * Isolation: every business table carries paper_id, and child tables use COMPOSITE foreign keys
    such as (paper_id, comment_id) -> comments(paper_id, comment_id). A row of paper B can never
    point at a comment, passage, verdict or cycle that belongs to paper A.
  * History is preserved: corrected or replaced facts are marked is_current = 0 with a
    superseded_by pointer. Triggers abort any DELETE on the history tables.
  * Partial unique indexes guarantee at most one CURRENT row per logical item.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import config


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jdump(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def jload(text, default=None):
    if text in (None, ""):
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


SCHEMA = """
PRAGMA foreign_keys = ON;

-- Small key/value store for application state (for example the currently selected paper).
CREATE TABLE IF NOT EXISTS app_state (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT NOT NULL
);

-- ===================== Papers and the approved baseline =====================
CREATE TABLE IF NOT EXISTS papers (
    paper_id             TEXT PRIMARY KEY CHECK (length(paper_id) BETWEEN 3 AND 40),
    title                TEXT NOT NULL,
    status               TEXT NOT NULL DEFAULT 'active',
    baseline_hash        TEXT NOT NULL UNIQUE,          -- same baseline file cannot be onboarded twice
    baseline_approved_at TEXT NOT NULL,
    baseline_approved_by TEXT NOT NULL,
    is_synthetic         INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT NOT NULL
);

-- ===================== Source knowledge: versions, sections, passages =====================
CREATE TABLE IF NOT EXISTS documents (
    version_id       TEXT PRIMARY KEY,
    paper_id         TEXT NOT NULL REFERENCES papers(paper_id),
    version_no       INTEGER NOT NULL,
    kind             TEXT NOT NULL CHECK (kind IN ('baseline','revision')),
    source_name      TEXT NOT NULL,
    source_path      TEXT,                               -- where the author imported it from
    stored_path      TEXT NOT NULL,                      -- recoverable copy under data/sources/
    file_format      TEXT NOT NULL,
    content_hash     TEXT NOT NULL,                      -- sha256 of the file bytes
    text_hash        TEXT NOT NULL,                      -- sha256 of the normalised extracted text
    page_count       INTEGER,
    extraction_notes TEXT,
    metadata_json    TEXT,
    full_text        TEXT NOT NULL,
    ingested_at      TEXT NOT NULL,
    UNIQUE (paper_id, version_no),
    UNIQUE (paper_id, version_id),
    UNIQUE (paper_id, content_hash)
);

CREATE TABLE IF NOT EXISTS sections (
    paper_id   TEXT NOT NULL,
    version_id TEXT NOT NULL,
    ord        INTEGER NOT NULL,
    heading    TEXT NOT NULL,
    level      INTEGER NOT NULL,
    page_start INTEGER,
    page_end   INTEGER,
    PRIMARY KEY (paper_id, version_id, ord),
    FOREIGN KEY (paper_id, version_id) REFERENCES documents(paper_id, version_id)
);

CREATE TABLE IF NOT EXISTS passages (
    passage_id        TEXT PRIMARY KEY,
    paper_id          TEXT NOT NULL,
    version_id        TEXT NOT NULL,
    ord               INTEGER NOT NULL,
    section_ord       INTEGER,
    heading           TEXT NOT NULL,
    page              INTEGER,
    text              TEXT NOT NULL,
    injection_flag    INTEGER NOT NULL DEFAULT 0,
    injection_patterns TEXT,
    UNIQUE (paper_id, passage_id),
    UNIQUE (paper_id, version_id, ord),
    FOREIGN KEY (paper_id, version_id) REFERENCES documents(paper_id, version_id)
);
CREATE INDEX IF NOT EXISTS idx_passages_version ON passages(paper_id, version_id, ord);

CREATE TABLE IF NOT EXISTS diffs (
    diff_id         TEXT PRIMARY KEY,
    paper_id        TEXT NOT NULL,
    from_version_id TEXT NOT NULL,
    to_version_id   TEXT NOT NULL,
    summary_json    TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE (paper_id, diff_id),
    FOREIGN KEY (paper_id, from_version_id) REFERENCES documents(paper_id, version_id),
    FOREIGN KEY (paper_id, to_version_id)   REFERENCES documents(paper_id, version_id)
);

CREATE TABLE IF NOT EXISTS changes (
    paper_id       TEXT NOT NULL,
    diff_id        TEXT NOT NULL,
    change_no      INTEGER NOT NULL,
    kind           TEXT NOT NULL CHECK (kind IN ('added','removed','modified')),
    old_passage_id TEXT,
    new_passage_id TEXT,
    similarity     REAL,
    PRIMARY KEY (paper_id, diff_id, change_no),
    FOREIGN KEY (paper_id, diff_id)        REFERENCES diffs(paper_id, diff_id),
    FOREIGN KEY (paper_id, old_passage_id) REFERENCES passages(paper_id, passage_id),
    FOREIGN KEY (paper_id, new_passage_id) REFERENCES passages(paper_id, passage_id)
);

-- ===================== Review cycles (working state) =====================
CREATE TABLE IF NOT EXISTS review_cycles (
    cycle_id        TEXT PRIMARY KEY,
    paper_id        TEXT NOT NULL,
    cycle_no        INTEGER NOT NULL,
    version_id      TEXT NOT NULL,
    prev_version_id TEXT NOT NULL,
    diff_id         TEXT NOT NULL,
    status          TEXT NOT NULL,                      -- claims_recorded | evaluated
    created_at      TEXT NOT NULL,
    UNIQUE (paper_id, cycle_no),
    UNIQUE (paper_id, cycle_id),
    FOREIGN KEY (paper_id, version_id)      REFERENCES documents(paper_id, version_id),
    FOREIGN KEY (paper_id, prev_version_id) REFERENCES documents(paper_id, version_id),
    FOREIGN KEY (paper_id, diff_id)         REFERENCES diffs(paper_id, diff_id)
);

-- ===================== Reviewer comments and their revisions =====================
CREATE TABLE IF NOT EXISTS comments (
    paper_id   TEXT NOT NULL REFERENCES papers(paper_id),
    comment_id TEXT NOT NULL,
    reviewer   TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (paper_id, comment_id)
);

CREATE TABLE IF NOT EXISTS comment_revisions (
    rev_id           TEXT PRIMARY KEY,
    paper_id         TEXT NOT NULL,
    comment_id       TEXT NOT NULL,
    rev_no           INTEGER NOT NULL,
    text             TEXT NOT NULL,
    interpretation   TEXT,
    checks_json      TEXT,
    status           TEXT NOT NULL CHECK (status IN ('active','withdrawn')),
    is_current       INTEGER NOT NULL DEFAULT 1,
    superseded_by    TEXT,
    supersede_reason TEXT,
    reason           TEXT,
    created_by       TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    UNIQUE (paper_id, comment_id, rev_no),
    FOREIGN KEY (paper_id, comment_id) REFERENCES comments(paper_id, comment_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_comment_current
    ON comment_revisions(paper_id, comment_id) WHERE is_current = 1;

-- ===================== Author claims =====================
CREATE TABLE IF NOT EXISTS author_claims (
    claim_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id      TEXT NOT NULL,
    cycle_id      TEXT NOT NULL,
    comment_id    TEXT NOT NULL,
    claim         TEXT NOT NULL CHECK (claim IN ('resolved','partial','outstanding','none')),
    note          TEXT,
    is_current    INTEGER NOT NULL DEFAULT 1,
    superseded_by INTEGER,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (paper_id, cycle_id)   REFERENCES review_cycles(paper_id, cycle_id),
    FOREIGN KEY (paper_id, comment_id) REFERENCES comments(paper_id, comment_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_claim_current
    ON author_claims(paper_id, cycle_id, comment_id) WHERE is_current = 1;

-- ===================== Agent suggestions (never approvals) =====================
CREATE TABLE IF NOT EXISTS verdicts (
    verdict_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id           TEXT NOT NULL,
    cycle_id           TEXT NOT NULL,
    comment_id         TEXT NOT NULL,
    comment_rev_id     TEXT NOT NULL,
    engine             TEXT NOT NULL,
    memory_mode        TEXT NOT NULL,
    plan_json          TEXT,
    verdict            TEXT NOT NULL CHECK (verdict IN ('Addressed','Partial','Unverified','Ask Author')),
    explanation        TEXT,
    missing_evidence   TEXT,
    contradiction      TEXT,
    claim_assessment   TEXT,
    regression         INTEGER NOT NULL DEFAULT 0,
    previous_verdict   TEXT,
    previous_source    TEXT,
    action_code        TEXT NOT NULL,
    recommended_action TEXT,
    follow_up_question TEXT,
    needs_manual       INTEGER NOT NULL DEFAULT 0,
    injection_flag     INTEGER NOT NULL DEFAULT 0,
    guards_json        TEXT,
    is_current         INTEGER NOT NULL DEFAULT 1,
    superseded_by      INTEGER,
    supersede_reason   TEXT,
    fingerprint        TEXT,
    created_at         TEXT NOT NULL,
    UNIQUE (paper_id, verdict_id),
    FOREIGN KEY (paper_id, cycle_id)   REFERENCES review_cycles(paper_id, cycle_id),
    FOREIGN KEY (paper_id, comment_id) REFERENCES comments(paper_id, comment_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_verdict_current
    ON verdicts(paper_id, cycle_id, comment_id) WHERE is_current = 1;

CREATE TABLE IF NOT EXISTS verdict_evidence (
    evidence_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    verdict_id    INTEGER NOT NULL,
    paper_id      TEXT NOT NULL,
    version_id    TEXT NOT NULL,
    passage_id    TEXT NOT NULL,
    heading       TEXT,
    page          INTEGER,
    quote         TEXT NOT NULL,
    origin        TEXT,
    element_label TEXT,
    FOREIGN KEY (paper_id, verdict_id) REFERENCES verdicts(paper_id, verdict_id),
    FOREIGN KEY (paper_id, passage_id) REFERENCES passages(paper_id, passage_id)
);

-- ===================== Author approvals and corrections =====================
CREATE TABLE IF NOT EXISTS author_decisions (
    decision_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id         TEXT NOT NULL,
    cycle_id         TEXT NOT NULL,
    comment_id       TEXT NOT NULL,
    verdict_id       INTEGER,
    decision_type    TEXT NOT NULL CHECK (decision_type IN ('approve','correct')),
    final_verdict    TEXT NOT NULL CHECK (final_verdict IN ('Addressed','Partial','Unverified','Ask Author')),
    note             TEXT,
    is_current       INTEGER NOT NULL DEFAULT 1,
    superseded_by    INTEGER,
    supersede_reason TEXT,
    created_at       TEXT NOT NULL,
    FOREIGN KEY (paper_id, cycle_id)   REFERENCES review_cycles(paper_id, cycle_id),
    FOREIGN KEY (paper_id, comment_id) REFERENCES comments(paper_id, comment_id),
    FOREIGN KEY (paper_id, verdict_id) REFERENCES verdicts(paper_id, verdict_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_decision_current
    ON author_decisions(paper_id, cycle_id, comment_id) WHERE is_current = 1;

CREATE TABLE IF NOT EXISTS clarifications (
    clar_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id   TEXT NOT NULL,
    cycle_id   TEXT NOT NULL,
    comment_id TEXT NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN ('clarification','evidence','note')),
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (paper_id, cycle_id)   REFERENCES review_cycles(paper_id, cycle_id),
    FOREIGN KEY (paper_id, comment_id) REFERENCES comments(paper_id, comment_id)
);

-- ===================== Next actions and pending questions (working state) =====================
CREATE TABLE IF NOT EXISTS next_actions (
    action_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id          TEXT NOT NULL,
    cycle_id          TEXT NOT NULL,
    comment_id        TEXT NOT NULL,
    action_code       TEXT NOT NULL,
    description       TEXT,
    rationale         TEXT,
    decided_by        TEXT NOT NULL,                    -- engine name or 'author'
    memory_influenced INTEGER NOT NULL DEFAULT 0,
    status            TEXT NOT NULL CHECK (status IN ('proposed','confirmed','completed','superseded')),
    superseded_reason TEXT,
    confirmed_at      TEXT,
    created_at        TEXT NOT NULL,
    FOREIGN KEY (paper_id, cycle_id)   REFERENCES review_cycles(paper_id, cycle_id),
    FOREIGN KEY (paper_id, comment_id) REFERENCES comments(paper_id, comment_id)
);

CREATE TABLE IF NOT EXISTS pending_questions (
    question_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id      TEXT NOT NULL,
    cycle_id      TEXT NOT NULL,
    comment_id    TEXT NOT NULL,
    question      TEXT NOT NULL,
    reason        TEXT,
    status        TEXT NOT NULL CHECK (status IN ('open','answered','closed','superseded')),
    closed_at     TEXT,
    closed_reason TEXT,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (paper_id, cycle_id)   REFERENCES review_cycles(paper_id, cycle_id),
    FOREIGN KEY (paper_id, comment_id) REFERENCES comments(paper_id, comment_id)
);

-- Why the agent chose what it chose (decision log used for the "LLM decision changes the plan" evidence).
CREATE TABLE IF NOT EXISTS agent_decisions (
    decision_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id           TEXT NOT NULL,
    cycle_id           TEXT NOT NULL,
    comment_id         TEXT NOT NULL,
    stage              TEXT NOT NULL,                   -- plan | next_action
    engine             TEXT NOT NULL,
    memory_mode        TEXT NOT NULL,
    chosen_json        TEXT,
    default_json       TEXT,
    changed_vs_default INTEGER NOT NULL DEFAULT 0,
    rationale          TEXT,
    evidence_seen_json TEXT,
    prior_state_json   TEXT,
    created_at         TEXT NOT NULL,
    FOREIGN KEY (paper_id, cycle_id)   REFERENCES review_cycles(paper_id, cycle_id),
    FOREIGN KEY (paper_id, comment_id) REFERENCES comments(paper_id, comment_id)
);

-- ===================== Follow-up drafts (always UNSENT) =====================
CREATE TABLE IF NOT EXISTS drafts (
    draft_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id     TEXT NOT NULL,
    cycle_id     TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    path         TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'UNSENT' CHECK (status = 'UNSENT'),
    comment_ids  TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    UNIQUE (paper_id, cycle_id, content_hash),
    FOREIGN KEY (paper_id, cycle_id) REFERENCES review_cycles(paper_id, cycle_id)
);

-- ===================== Reliability: operations, checkpoints, logs =====================
CREATE TABLE IF NOT EXISTS operations (
    op_key      TEXT PRIMARY KEY,                       -- idempotency key
    paper_id    TEXT,
    op_type     TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('started','succeeded','failed')),
    attempts    INTEGER NOT NULL DEFAULT 1,
    result_json TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoints (
    paper_id     TEXT NOT NULL,
    cycle_id     TEXT NOT NULL,
    name         TEXT NOT NULL,
    fingerprint  TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (paper_id, cycle_id, name)
);

-- No foreign keys on purpose: the log must be able to record events about anything, including rejected input.
CREATE TABLE IF NOT EXISTS action_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    paper_id        TEXT,
    cycle_id        TEXT,
    kind            TEXT NOT NULL,                      -- tool_call | memory_read | memory_write | decision | security | ...
    tool            TEXT,
    input_ref       TEXT,
    outcome         TEXT,
    detail          TEXT,
    error           TEXT,
    idempotency_key TEXT,
    attempt         INTEGER,
    duration_ms     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_log_paper ON action_log(paper_id, id);

CREATE TABLE IF NOT EXISTS llm_calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT NOT NULL,
    paper_id          TEXT,
    cycle_id          TEXT,
    purpose           TEXT,
    engine            TEXT,
    model             TEXT,
    memory_mode       TEXT,
    latency_ms        INTEGER,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    ok                INTEGER NOT NULL,
    error             TEXT
);

-- ===================== History tables are append-only =====================
CREATE TRIGGER IF NOT EXISTS no_delete_verdicts BEFORE DELETE ON verdicts
BEGIN SELECT RAISE(ABORT, 'history is append-only: verdicts cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS no_delete_decisions BEFORE DELETE ON author_decisions
BEGIN SELECT RAISE(ABORT, 'history is append-only: author decisions cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS no_delete_claims BEFORE DELETE ON author_claims
BEGIN SELECT RAISE(ABORT, 'history is append-only: author claims cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS no_delete_comment_revisions BEFORE DELETE ON comment_revisions
BEGIN SELECT RAISE(ABORT, 'history is append-only: comment revisions cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS no_delete_documents BEFORE DELETE ON documents
BEGIN SELECT RAISE(ABORT, 'history is append-only: document versions cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS no_delete_action_log BEFORE DELETE ON action_log
BEGIN SELECT RAISE(ABORT, 'the action log is append-only'); END;
"""


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    row = conn.execute("SELECT value FROM app_state WHERE key = 'schema_version'").fetchone()
    if not row:
        conn.execute("INSERT INTO app_state(key, value, updated_at) VALUES ('schema_version', ?, ?)",
                     (str(config.SCHEMA_VERSION), utcnow()))


def connect(init: bool = True) -> sqlite3.Connection:
    """Open the application database (creating directories and tables on first use)."""
    paths = config.ensure_dirs()
    conn = sqlite3.connect(str(paths.db), isolation_level=None, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if init:
        init_schema(conn)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    """BEGIN IMMEDIATE ... COMMIT, or ROLLBACK if the block raises."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def get_state(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row and row["value"] is not None else default


def set_state(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute(
        "INSERT INTO app_state(key, value, updated_at) VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET"
        " value = excluded.value, updated_at = excluded.updated_at",
        (key, None if value is None else str(value), utcnow()))
