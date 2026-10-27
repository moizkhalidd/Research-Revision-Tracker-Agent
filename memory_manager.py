"""Persistent memory: storing sources and retrieving task memory from SQLite.

Memory lifecycle implemented here
  Source knowledge   documents, sections, passages, diffs (store_document, ingest_revision)
  Long-term memory   baseline comments, claims, verdicts, decisions, corrections (read via
                     retrieve_comment_memory and current_status)
  Working state      review_cycles.status, pending_questions, next_actions, checkpoints
  Correction         rows are never deleted: is_current = 0 plus superseded_by
  Isolation          every query filters on paper_id; composite foreign keys back this up
  Inspection         paper_overview, current_status, history_for_comment, list_superseded
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import action_logger as alog
import config
import retry_manager
from database import jdump, jload, utcnow
from document_diff import diff_passages, summarize
from models import ParsedDocument
from validators import detect_injection


class NotFound(LookupError):
    """A paper, version, cycle or comment does not exist (for this paper)."""


class DuplicateError(ValueError):
    """The same paper or document was already stored."""


# Papers, versions, cycles, comments (lookups)

def require_paper(conn, paper_id: str):
    row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    if not row:
        raise NotFound("Paper %r is not onboarded. Run 'papers' to list known papers." % paper_id)
    return row


def list_papers(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT p.*, (SELECT COUNT(*) FROM documents d WHERE d.paper_id = p.paper_id) AS versions,"
        " (SELECT COUNT(*) FROM review_cycles c WHERE c.paper_id = p.paper_id) AS cycles"
        " FROM papers p ORDER BY p.created_at, p.paper_id").fetchall()
    return [dict(r) for r in rows]


def latest_version(conn, paper_id: str):
    return conn.execute("SELECT * FROM documents WHERE paper_id = ? ORDER BY version_no DESC LIMIT 1",
                        (paper_id,)).fetchone()


def get_cycle(conn, paper_id: str, cycle_no: int | None = None):
    if cycle_no is None:
        row = conn.execute("SELECT * FROM review_cycles WHERE paper_id = ? ORDER BY cycle_no DESC LIMIT 1",
                           (paper_id,)).fetchone()
    else:
        row = conn.execute("SELECT * FROM review_cycles WHERE paper_id = ? AND cycle_no = ?",
                           (paper_id, int(cycle_no))).fetchone()
    if not row:
        raise NotFound("No review cycle%s for paper %s. Ingest a revision first." %
                       ("" if cycle_no is None else " %s" % cycle_no, paper_id))
    return row


def active_comments(conn, paper_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT c.comment_id, c.reviewer, r.rev_id, r.rev_no, r.text, r.interpretation, r.checks_json, r.status"
        " FROM comments c JOIN comment_revisions r ON r.paper_id = c.paper_id AND r.comment_id = c.comment_id"
        " AND r.is_current = 1 WHERE c.paper_id = ? ORDER BY c.rowid", (paper_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["checks"] = jload(d.pop("checks_json"), [])
        out.append(d)
    return out


def get_comment(conn, paper_id: str, comment_id: str) -> dict:
    for c in active_comments(conn, paper_id):
        if c["comment_id"] == comment_id:
            return c
    raise NotFound("Comment %r does not exist for paper %s." % (comment_id, paper_id))


def get_active_comment(conn, paper_id: str, comment_id: str) -> dict:
    c = get_comment(conn, paper_id, comment_id)
    if c["status"] != "active":
        raise NotFound("Comment %s is withdrawn for paper %s." % (comment_id, paper_id))
    return c


def active_only(conn, paper_id: str) -> list[dict]:
    return [c for c in active_comments(conn, paper_id) if c["status"] == "active"]


# Source storage

def copy_source(paper_id: str, version_no: int, src: Path, content_hash: str) -> str:
    """Copy the imported file under data/sources/<paper_id>/ so the baseline stays recoverable.

    Returns the path relative to the application home (portable across machines).
    """
    paths = config.ensure_dirs()
    dest_dir = paths.sources / paper_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ("v%d_%s%s" % (version_no, content_hash[:12], Path(src).suffix.lower()))
    shutil.copy2(str(src), str(dest))
    return dest.relative_to(paths.home).as_posix()


def remove_source_copy(rel_path: str) -> None:
    try:
        (config.get_paths().home / rel_path).unlink()
    except OSError:
        pass


def store_document(conn, paper_id: str, kind: str, version_no: int, parsed: ParsedDocument, *, source_name: str,
                   source_path: str, stored_path: str, content_hash: str, text_hash: str) -> str:
    version_id = "%s-v%d" % (paper_id, version_no)
    now = utcnow()
    conn.execute(
        "INSERT INTO documents(version_id, paper_id, version_no, kind, source_name, source_path, stored_path,"
        " file_format, content_hash, text_hash, page_count, extraction_notes, metadata_json, full_text, ingested_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (version_id, paper_id, version_no, kind, source_name, source_path, stored_path, parsed.file_format,
         content_hash, text_hash, parsed.page_count, " | ".join(parsed.notes), jdump(parsed.metadata),
         parsed.full_text, now))
    sections: dict[int, dict] = {}
    order: list[tuple] = []
    for p in parsed.passages:
        key = (p.heading, p.level)
        if not order or order[-1][0] != key:
            order.append((key, []))
        order[-1][1].append(p)
    for sec_ord, ((heading, level), plist) in enumerate(order):
        pages = [p.page for p in plist if p.page]
        conn.execute("INSERT INTO sections(paper_id, version_id, ord, heading, level, page_start, page_end)"
                     " VALUES (?,?,?,?,?,?,?)",
                     (paper_id, version_id, sec_ord, heading, level, min(pages) if pages else None,
                      max(pages) if pages else None))
        for p in plist:
            patterns = detect_injection(p.text)
            pid = "%s#%d" % (version_id, p.ord)
            conn.execute(
                "INSERT INTO passages(passage_id, paper_id, version_id, ord, section_ord, heading, page, text,"
                " injection_flag, injection_patterns) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (pid, paper_id, version_id, p.ord, sec_ord, heading, p.page, p.text,
                 1 if patterns else 0, ",".join(patterns) if patterns else None))
            if patterns:
                alog.log_security(conn, paper_id, None, "passage:" + pid,
                                  "Untrusted instruction-like text in document (%s). Stored as data, excluded from "
                                  "evidence, no policy change." % ",".join(patterns))
    alog.log_memory(conn, "write", paper_id, None, "document:" + version_id,
                    "%d passages, hash %s" % (len(parsed.passages), content_hash[:12]))
    return version_id


def _check_duplicate_content(conn, paper_id: str, content_hash: str, text_hash: str) -> None:
    row = conn.execute("SELECT version_no FROM documents WHERE paper_id = ? AND (content_hash = ? OR text_hash = ?)"
                       " ORDER BY version_no LIMIT 1", (paper_id, content_hash, text_hash)).fetchone()
    if row:
        raise DuplicateError("This document is identical to version v%d of paper %s (same file hash or same "
                             "extracted text). Nothing was ingested." % (row["version_no"], paper_id))


def onboard_paper(conn, paper_id: str, title: str, parsed: ParsedDocument, *, source_name: str, source_path: str,
                  src_file: Path, content_hash: str, text_hash: str, comments: list[dict], synthetic: bool,
                  approved_by: str = "author") -> dict:
    """Create the paper, its approved baseline (v1) and baseline reviewer comments in the open transaction."""
    if conn.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)).fetchone():
        raise DuplicateError("Paper ID %s already exists. Choose another ID or use 'ingest' for a new revision." % paper_id)
    other = conn.execute("SELECT paper_id FROM papers WHERE baseline_hash = ?", (content_hash,)).fetchone()
    if other:
        raise DuplicateError("This exact baseline file is already onboarded as paper %s. Nothing was created." %
                             other["paper_id"])
    now = utcnow()
    conn.execute(
        "INSERT INTO papers(paper_id, title, status, baseline_hash, baseline_approved_at, baseline_approved_by,"
        " is_synthetic, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (paper_id, title, "active", content_hash, now, approved_by, 1 if synthetic else 0, now))
    stored = copy_source(paper_id, 1, src_file, content_hash)
    try:
        version_id = store_document(conn, paper_id, "baseline", 1, parsed, source_name=source_name,
                                    source_path=source_path, stored_path=stored, content_hash=content_hash,
                                    text_hash=text_hash)
        for c in comments:
            conn.execute("INSERT INTO comments(paper_id, comment_id, reviewer, created_at) VALUES (?,?,?,?)",
                         (paper_id, c["id"], c["reviewer"], now))
            conn.execute(
                "INSERT INTO comment_revisions(rev_id, paper_id, comment_id, rev_no, text, interpretation,"
                " checks_json, status, is_current, created_by, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("%s:%s:r1" % (paper_id, c["id"]), paper_id, c["id"], 1, c["text"], None, jdump(c["checks"]),
                 "active", 1, "baseline", now))
        alog.log_memory(conn, "write", paper_id, None, "baseline_comments", "%d comments approved as baseline" % len(comments))
    except BaseException:
        remove_source_copy(stored)
        raise
    return {"paper_id": paper_id, "version_id": version_id, "comments": len(comments), "stored_path": stored}


def ingest_revision(conn, paper_id: str, parsed: ParsedDocument, *, source_name: str, source_path: str,
                    src_file: Path, content_hash: str, text_hash: str, claims: dict[str, dict]) -> dict:
    """Store a new version, its diff against the previous version, a review cycle and the author's claims."""
    require_paper(conn, paper_id)
    _check_duplicate_content(conn, paper_id, content_hash, text_hash)
    prev = latest_version(conn, paper_id)
    version_no = prev["version_no"] + 1
    stored = copy_source(paper_id, version_no, src_file, content_hash)
    try:
        version_id = store_document(conn, paper_id, "revision", version_no, parsed, source_name=source_name,
                                    source_path=source_path, stored_path=stored, content_hash=content_hash,
                                    text_hash=text_hash)
        old = conn.execute("SELECT passage_id, text FROM passages WHERE paper_id=? AND version_id=? ORDER BY ord",
                           (paper_id, prev["version_id"])).fetchall()
        new = conn.execute("SELECT passage_id, text FROM passages WHERE paper_id=? AND version_id=? ORDER BY ord",
                           (paper_id, version_id)).fetchall()
        changes = diff_passages([r["text"] for r in old], [r["text"] for r in new])
        diff_id = "%s-d%d-%d" % (paper_id, prev["version_no"], version_no)
        now = utcnow()
        conn.execute("INSERT INTO diffs(diff_id, paper_id, from_version_id, to_version_id, summary_json, created_at)"
                     " VALUES (?,?,?,?,?,?)",
                     (diff_id, paper_id, prev["version_id"], version_id,
                      jdump(summarize(changes, len(old), len(new))), now))
        for n, ch in enumerate(changes, 1):
            conn.execute(
                "INSERT INTO changes(paper_id, diff_id, change_no, kind, old_passage_id, new_passage_id, similarity)"
                " VALUES (?,?,?,?,?,?,?)",
                (paper_id, diff_id, n, ch.kind, old[ch.old_ord]["passage_id"] if ch.old_ord is not None else None,
                 new[ch.new_ord]["passage_id"] if ch.new_ord is not None else None, ch.similarity))
        cycle_no = version_no - 1
        cycle_id = "%s-C%d" % (paper_id, cycle_no)
        conn.execute("INSERT INTO review_cycles(cycle_id, paper_id, cycle_no, version_id, prev_version_id, diff_id,"
                     " status, created_at) VALUES (?,?,?,?,?,?,?,?)",
                     (cycle_id, paper_id, cycle_no, version_id, prev["version_id"], diff_id, "claims_recorded", now))
        # Simulated tool failure (Scenario G): raised after rows exist, so the rollback is visible.
        retry_manager.maybe_fail("ingest_after_insert")
        record_claims(conn, paper_id, cycle_id, claims)
        alog.log_memory(conn, "write", paper_id, cycle_id, "diff:" + diff_id, "%d changed passages" % len(changes))
    except BaseException:
        remove_source_copy(stored)
        raise
    return {"paper_id": paper_id, "version_id": version_id, "version_no": version_no, "cycle_id": cycle_id,
            "cycle_no": cycle_no, "diff_id": diff_id, "changes": len(changes), "stored_path": stored,
            "diff_summary": summarize(changes, len(old), len(new))}


def record_claims(conn, paper_id: str, cycle_id: str, claims: dict[str, dict]) -> int:
    """Store the author's resolution claims for this cycle. Changed claims supersede the old row."""
    written = 0
    now = utcnow()
    for c in active_only(conn, paper_id):
        cid = c["comment_id"]
        want = claims.get(cid, {"claim": "none", "note": None}) if claims is not None else {"claim": "none", "note": None}
        cur = conn.execute("SELECT * FROM author_claims WHERE paper_id=? AND cycle_id=? AND comment_id=? AND is_current=1",
                           (paper_id, cycle_id, cid)).fetchone()
        if cur and cur["claim"] == want["claim"] and (cur["note"] or None) == (want.get("note") or None):
            continue
        if cur:
            conn.execute("UPDATE author_claims SET is_current=0 WHERE claim_id=?", (cur["claim_id"],))
        new_id = conn.execute(
            "INSERT INTO author_claims(paper_id, cycle_id, comment_id, claim, note, is_current, created_at)"
            " VALUES (?,?,?,?,?,1,?)", (paper_id, cycle_id, cid, want["claim"], want.get("note"), now)).lastrowid
        if cur:
            conn.execute("UPDATE author_claims SET superseded_by=? WHERE claim_id=?", (new_id, cur["claim_id"]))
        written += 1
    alog.log_memory(conn, "write", paper_id, cycle_id, "author_claims", "%d claim rows written" % written)
    return written


# Memory retrieval

def _prev_decision_for(conn, paper_id, cycle_id, comment_id, verdict_id):
    d = conn.execute("SELECT * FROM author_decisions WHERE paper_id=? AND cycle_id=? AND comment_id=? AND is_current=1",
                     (paper_id, cycle_id, comment_id)).fetchone()
    if d and (d["decision_type"] == "correct" or d["verdict_id"] == verdict_id):
        return d
    return None


def retrieve_comment_memory(conn, paper_id: str, cycle_id: str, comment_id: str, enabled: bool = True) -> dict:
    """Everything the agent may remember about one comment, as of the given cycle.

    With enabled=False (diagnostic mode only) prior task records are NOT retrieved: no previous verdicts,
    evidence, decisions, clarifications, open questions or comment corrections. The current claim and the
    documents are still provided because they are the input of the step, and saved data is never touched.
    """
    cyc = conn.execute("SELECT * FROM review_cycles WHERE paper_id=? AND cycle_id=?", (paper_id, cycle_id)).fetchone()
    if not cyc:
        raise NotFound("Cycle %r does not belong to paper %s." % (cycle_id, paper_id))
    comment = get_comment(conn, paper_id, comment_id)
    base = conn.execute("SELECT * FROM comment_revisions WHERE paper_id=? AND comment_id=? AND rev_no=1",
                        (paper_id, comment_id)).fetchone()
    claim = conn.execute("SELECT claim, note FROM author_claims WHERE paper_id=? AND cycle_id=? AND comment_id=? AND is_current=1",
                         (paper_id, cycle_id, comment_id)).fetchone()
    mem = {
        "memory_enabled": bool(enabled),
        "comment": {"comment_id": comment_id, "rev_id": comment["rev_id"], "rev_no": comment["rev_no"],
                    "text": comment["text"], "interpretation": comment["interpretation"],
                    "checks": comment["checks"], "corrected": comment["rev_no"] > 1},
        "claim": {"value": claim["claim"] if claim else "none", "note": claim["note"] if claim else None},
        "previous": None, "claim_history": [], "clarifications": [], "open_questions": [], "corrections": [],
    }
    if not enabled:
        mem["comment"].update(rev_id=base["rev_id"], rev_no=1, text=base["text"], interpretation=None,
                              checks=jload(base["checks_json"], []), corrected=False)
        alog.log_memory(conn, "read", paper_id, cycle_id, "comment_memory:%s" % comment_id,
                        "DIAGNOSTIC: prior task records NOT retrieved (memory disabled)")
        return mem

    prev = conn.execute(
        "SELECT v.*, c.cycle_no FROM verdicts v JOIN review_cycles c ON c.paper_id=v.paper_id AND c.cycle_id=v.cycle_id"
        " WHERE v.paper_id=? AND v.comment_id=? AND v.comment_rev_id=? AND v.is_current=1 AND c.cycle_no < ?"
        " ORDER BY c.cycle_no DESC LIMIT 1", (paper_id, comment_id, comment["rev_id"], cyc["cycle_no"])).fetchone()
    if prev:
        dec = _prev_decision_for(conn, paper_id, prev["cycle_id"], comment_id, prev["verdict_id"])
        ev = conn.execute("SELECT quote, passage_id, version_id, heading, page, element_label AS label, origin"
                          " FROM verdict_evidence WHERE verdict_id=? AND paper_id=? ORDER BY evidence_id",
                          (prev["verdict_id"], paper_id)).fetchall()
        mem["previous"] = {
            "verdict_id": prev["verdict_id"], "cycle_no": prev["cycle_no"], "cycle_id": prev["cycle_id"],
            "verdict": dec["final_verdict"] if dec else prev["verdict"],
            "source": "author" if dec else "agent",
            "agent_verdict": prev["verdict"], "explanation": prev["explanation"],
            "missing_evidence": prev["missing_evidence"], "follow_up_question": prev["follow_up_question"],
            "action_code": prev["action_code"], "evidence": [dict(r) for r in ev],
        }
    rows = conn.execute(
        "SELECT a.claim, a.note, c.cycle_no FROM author_claims a JOIN review_cycles c ON c.paper_id=a.paper_id AND c.cycle_id=a.cycle_id"
        " WHERE a.paper_id=? AND a.comment_id=? AND a.is_current=1 AND c.cycle_no < ? ORDER BY c.cycle_no",
        (paper_id, comment_id, cyc["cycle_no"])).fetchall()
    mem["claim_history"] = [dict(r) for r in rows]
    rows = conn.execute(
        "SELECT k.kind, k.text, c.cycle_no FROM clarifications k JOIN review_cycles c ON c.paper_id=k.paper_id AND c.cycle_id=k.cycle_id"
        " WHERE k.paper_id=? AND k.comment_id=? AND c.cycle_no <= ? ORDER BY k.clar_id",
        (paper_id, comment_id, cyc["cycle_no"])).fetchall()
    mem["clarifications"] = [dict(r) for r in rows]
    rows = conn.execute("SELECT question, reason, cycle_id FROM pending_questions WHERE paper_id=? AND comment_id=? AND status='open'",
                        (paper_id, comment_id)).fetchall()
    mem["open_questions"] = [dict(r) for r in rows]
    rows = conn.execute("SELECT rev_no, text, interpretation, reason FROM comment_revisions WHERE paper_id=? AND comment_id=? AND rev_no > 1"
                        " ORDER BY rev_no", (paper_id, comment_id)).fetchall()
    mem["corrections"] = [dict(r) for r in rows]
    alog.log_memory(conn, "read", paper_id, cycle_id, "comment_memory:%s" % comment_id,
                    "previous=%s claims=%d clarifications=%d open_questions=%d corrections=%d prior_evidence=%d" % (
                        mem["previous"]["verdict"] if mem["previous"] else "none", len(mem["claim_history"]),
                        len(mem["clarifications"]), len(mem["open_questions"]), len(mem["corrections"]),
                        len(mem["previous"]["evidence"]) if mem["previous"] else 0))
    return mem


def evaluation_fingerprint(conn, paper_id: str, cycle_id: str, comment_ids: list[str], engine_name: str,
                           memory_enabled: bool) -> str:
    """Hash of every input that can change an evaluation. Same fingerprint means a retry, not new work."""
    parts: list = [engine_name, bool(memory_enabled), cycle_id]
    cyc = conn.execute("SELECT version_id FROM review_cycles WHERE paper_id=? AND cycle_id=?", (paper_id, cycle_id)).fetchone()
    parts.append(cyc["version_id"] if cyc else None)
    for cid in sorted(comment_ids):
        rev = conn.execute("SELECT rev_id FROM comment_revisions WHERE paper_id=? AND comment_id=? AND is_current=1",
                           (paper_id, cid)).fetchone()
        claim = conn.execute("SELECT claim_id FROM author_claims WHERE paper_id=? AND cycle_id=? AND comment_id=? AND is_current=1",
                             (paper_id, cycle_id, cid)).fetchone()
        decs = [r[0] for r in conn.execute("SELECT decision_id FROM author_decisions WHERE paper_id=? AND comment_id=? AND is_current=1 ORDER BY decision_id", (paper_id, cid))]
        clar = conn.execute("SELECT COALESCE(MAX(clar_id),0) FROM clarifications WHERE paper_id=? AND comment_id=?", (paper_id, cid)).fetchone()[0]
        vids = [r[0] for r in conn.execute(
            "SELECT v.verdict_id FROM verdicts v JOIN review_cycles c ON c.paper_id=v.paper_id AND c.cycle_id=v.cycle_id"
            " WHERE v.paper_id=? AND v.comment_id=? AND v.is_current=1 AND c.cycle_no < (SELECT cycle_no FROM review_cycles WHERE paper_id=? AND cycle_id=?)"
            " ORDER BY v.verdict_id", (paper_id, cid, paper_id, cycle_id))]
        parts.append([cid, rev["rev_id"] if rev else None, claim["claim_id"] if claim else None, decs, clar, vids])
    return hashlib.sha256(jdump(parts).encode("utf-8")).hexdigest()[:32]


# State views (for status, export, draft, review)

def _evidence_rows(conn, paper_id, verdict_id):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM verdict_evidence WHERE paper_id=? AND verdict_id=? ORDER BY evidence_id", (paper_id, verdict_id))]


def current_status(conn, paper_id: str) -> list[dict]:
    """One dict per active comment: latest suggestion, approval state, effective verdict and next action."""
    out = []
    for c in active_only(conn, paper_id):
        cid = c["comment_id"]
        v = conn.execute(
            "SELECT v.*, cy.cycle_no FROM verdicts v JOIN review_cycles cy ON cy.paper_id=v.paper_id AND cy.cycle_id=v.cycle_id"
            " WHERE v.paper_id=? AND v.comment_id=? AND v.comment_rev_id=? AND v.is_current=1 ORDER BY cy.cycle_no DESC LIMIT 1",
            (paper_id, cid, c["rev_id"])).fetchone()
        entry = {"comment_id": cid, "reviewer": c["reviewer"], "text": c["text"], "rev_no": c["rev_no"],
                 "interpretation": c["interpretation"], "verdict": None, "decision": None, "approval_state": "none",
                 "effective_verdict": None, "effective_source": None, "unresolved": True, "action": None,
                 "questions": [], "evidence": [], "claim": None, "cycle_no": None, "cycle_id": None}
        if v:
            entry.update(verdict=dict(v), cycle_no=v["cycle_no"], cycle_id=v["cycle_id"],
                         evidence=_evidence_rows(conn, paper_id, v["verdict_id"]))
            d = conn.execute("SELECT * FROM author_decisions WHERE paper_id=? AND cycle_id=? AND comment_id=? AND is_current=1",
                             (paper_id, v["cycle_id"], cid)).fetchone()
            claim = conn.execute("SELECT claim, note FROM author_claims WHERE paper_id=? AND cycle_id=? AND comment_id=? AND is_current=1",
                                 (paper_id, v["cycle_id"], cid)).fetchone()
            entry["claim"] = dict(claim) if claim else None
            if d and (d["decision_type"] == "correct" or d["verdict_id"] == v["verdict_id"]):
                entry.update(decision=dict(d), approval_state="corrected" if d["decision_type"] == "correct" else "approved",
                             effective_verdict=d["final_verdict"], effective_source="author")
            else:
                entry.update(approval_state="stale" if d else "pending", effective_verdict=v["verdict"],
                             effective_source="agent (suggested, not approved)")
            entry["unresolved"] = entry["effective_verdict"] != "Addressed"
            act = conn.execute("SELECT * FROM next_actions WHERE paper_id=? AND cycle_id=? AND comment_id=? AND status IN ('proposed','confirmed')"
                               " ORDER BY action_id DESC LIMIT 1", (paper_id, v["cycle_id"], cid)).fetchone()
            entry["action"] = dict(act) if act else None
        else:
            act = conn.execute("SELECT * FROM next_actions WHERE paper_id=? AND comment_id=? AND status IN ('proposed','confirmed')"
                               " ORDER BY action_id DESC LIMIT 1", (paper_id, cid)).fetchone()
            entry["action"] = dict(act) if act else None
        entry["questions"] = [dict(r) for r in conn.execute(
            "SELECT * FROM pending_questions WHERE paper_id=? AND comment_id=? AND status='open' ORDER BY question_id", (paper_id, cid))]
        out.append(entry)
    alog.log_memory(conn, "read", paper_id, None, "current_status", "%d active comments" % len(out))
    return out


def paper_overview(conn, paper_id: str) -> dict:
    p = require_paper(conn, paper_id)
    versions = [dict(r) for r in conn.execute(
        "SELECT version_id, version_no, kind, file_format, source_name, stored_path, content_hash, page_count, ingested_at, extraction_notes"
        " FROM documents WHERE paper_id=? ORDER BY version_no", (paper_id,))]
    cycles = [dict(r) for r in conn.execute("SELECT * FROM review_cycles WHERE paper_id=? ORDER BY cycle_no", (paper_id,))]
    return {"paper": dict(p), "versions": versions, "cycles": cycles}


def history_for_comment(conn, paper_id: str, comment_id: str) -> dict:
    get_comment(conn, paper_id, comment_id)  # isolation: raises NotFound for another paper's comment
    q = lambda sql: [dict(r) for r in conn.execute(sql, (paper_id, comment_id))]  # noqa: E731
    alog.log_memory(conn, "read", paper_id, None, "history:%s" % comment_id, "full comment history")
    return {
        "revisions": q("SELECT rev_no, text, interpretation, status, is_current, superseded_by, reason, supersede_reason, created_by, created_at FROM comment_revisions WHERE paper_id=? AND comment_id=? ORDER BY rev_no"),
        "claims": q("SELECT a.cycle_id, a.claim, a.note, a.is_current, a.created_at FROM author_claims a WHERE a.paper_id=? AND a.comment_id=? ORDER BY a.claim_id"),
        "verdicts": q("SELECT verdict_id, cycle_id, engine, memory_mode, verdict, regression, action_code, is_current, supersede_reason, created_at FROM verdicts WHERE paper_id=? AND comment_id=? ORDER BY verdict_id"),
        "decisions": q("SELECT decision_id, cycle_id, decision_type, final_verdict, note, is_current, supersede_reason, created_at FROM author_decisions WHERE paper_id=? AND comment_id=? ORDER BY decision_id"),
        "clarifications": q("SELECT cycle_id, kind, text, created_at FROM clarifications WHERE paper_id=? AND comment_id=? ORDER BY clar_id"),
        "questions": q("SELECT cycle_id, question, status, closed_reason, created_at FROM pending_questions WHERE paper_id=? AND comment_id=? ORDER BY question_id"),
        "actions": q("SELECT action_id, cycle_id, action_code, decided_by, status, superseded_reason, created_at FROM next_actions WHERE paper_id=? AND comment_id=? ORDER BY action_id"),
    }


def list_superseded(conn, paper_id: str) -> list[dict]:
    require_paper(conn, paper_id)
    out = []

    def add(table, rows, fmt):
        for r in rows:
            out.append({"table": table, **fmt(dict(r))})

    add("comment_revisions", conn.execute("SELECT * FROM comment_revisions WHERE paper_id=? AND is_current=0", (paper_id,)),
        lambda r: {"id": r["rev_id"], "comment_id": r["comment_id"], "summary": "rev %d: %s" % (r["rev_no"], r["text"][:80]),
                   "superseded_by": r["superseded_by"], "reason": r["reason"], "created_at": r["created_at"]})
    add("author_claims", conn.execute("SELECT * FROM author_claims WHERE paper_id=? AND is_current=0", (paper_id,)),
        lambda r: {"id": r["claim_id"], "comment_id": r["comment_id"], "summary": "%s claim '%s'" % (r["cycle_id"], r["claim"]),
                   "superseded_by": r["superseded_by"], "reason": "claim updated", "created_at": r["created_at"]})
    add("verdicts", conn.execute("SELECT * FROM verdicts WHERE paper_id=? AND is_current=0", (paper_id,)),
        lambda r: {"id": r["verdict_id"], "comment_id": r["comment_id"], "summary": "%s suggested '%s'" % (r["cycle_id"], r["verdict"]),
                   "superseded_by": r["superseded_by"], "reason": r["supersede_reason"], "created_at": r["created_at"]})
    add("author_decisions", conn.execute("SELECT * FROM author_decisions WHERE paper_id=? AND is_current=0", (paper_id,)),
        lambda r: {"id": r["decision_id"], "comment_id": r["comment_id"], "summary": "%s %s '%s'" % (r["cycle_id"], r["decision_type"], r["final_verdict"]),
                   "superseded_by": r["superseded_by"], "reason": r["supersede_reason"], "created_at": r["created_at"]})
    add("next_actions", conn.execute("SELECT * FROM next_actions WHERE paper_id=? AND status='superseded'", (paper_id,)),
        lambda r: {"id": r["action_id"], "comment_id": r["comment_id"], "summary": "%s %s" % (r["cycle_id"], r["action_code"]),
                   "superseded_by": None, "reason": r["superseded_reason"], "created_at": r["created_at"]})
    add("pending_questions", conn.execute("SELECT * FROM pending_questions WHERE paper_id=? AND status IN ('closed','superseded')", (paper_id,)),
        lambda r: {"id": r["question_id"], "comment_id": r["comment_id"], "summary": r["question"][:80],
                   "superseded_by": None, "reason": r["closed_reason"], "created_at": r["created_at"]})
    alog.log_memory(conn, "read", paper_id, None, "superseded_records", "%d records" % len(out))
    return out


def recommend_next_step(conn, paper_id: str, status: list[dict] | None = None) -> str:
    """Rule-based, cycle-level recommendation shown by 'status'."""
    status = status if status is not None else current_status(conn, paper_id)
    if not status:
        return "All reviewer comments are withdrawn. Nothing to track."
    cyc = conn.execute("SELECT * FROM review_cycles WHERE paper_id=? ORDER BY cycle_no DESC LIMIT 1", (paper_id,)).fetchone()
    if not cyc:
        return "Ingest the first revision: python main.py ingest --file <revised file> --claims <claims.json>"
    unevaluated = [s for s in status if s["verdict"] is None]
    if unevaluated:
        return "Evaluate cycle %d: python main.py evaluate --engine groq   (or --engine offline)" % cyc["cycle_no"]
    pending = [s for s in status if s["approval_state"] in ("pending", "stale")]
    if pending:
        return "Approve or correct the %d suggested verdict(s): python main.py approve --comment <ID>  (or correct-verdict)" % len(pending)
    unresolved = [s for s in status if s["unresolved"]]
    if unresolved:
        return ("%d comment(s) remain unresolved. Generate the UNSENT follow-up (python main.py draft), send it yourself "
                "outside this tool, then ingest the next revision." % len(unresolved))
    return "All active comments are Addressed and approved by the author."
