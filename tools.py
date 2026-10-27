"""The agent's tools. Each one is a distinct capability, logged with paper ID, cycle ID, input reference,
outcome, duration and idempotency information (see action_logger.tool_call and retry_manager.run_operation).

  1. ingest_and_diff           read a PDF/LaTeX file, store a version, diff it against the previous one  (persistent write)
  2. retrieve_memory           read approved baseline, verdicts, claims, corrections, open questions from SQLite
  3. retrieve_evidence         search changed passages / the full document / earlier evidence for a comment
  4. record_verdicts           store suggested verdicts, evidence, questions and next actions               (persistent write)
  5. apply_author_correction   author approvals and corrections that supersede earlier records (author_review.py)
  6. export_excel              regenerate the Excel review matrix from SQLite (excel_exporter.py)
  7. draft_followup            create an UNSENT follow-up draft from unresolved comments (followup_drafter.py)
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import action_logger as alog
import config
import evidence_retriever as er
import memory_manager as mm
import retry_manager as rm
from database import jdump, set_state
from document_diff import diff_passages, summarize
from document_reader import DocumentReadError, read_document, sha256_bytes, sha256_text
from validators import (ValidationError, parse_claims, parse_comments_json, validate_input_file,
                        validate_paper_id)

TOOLS = {
    "ingest_and_diff": "Extract a document, record version metadata and hash, diff against the previous version",
    "retrieve_memory": "Retrieve baseline, claims, previous verdicts, corrections and open questions from SQLite",
    "retrieve_evidence": "Search changed passages, the full document, or verify earlier evidence still exists",
    "record_verdicts": "Persist suggested verdicts, evidence references, questions and next actions (transactional)",
    "apply_author_correction": "Apply author approvals/corrections that supersede earlier records",
    "export_excel": "Regenerate the Excel revision matrix from the database (never writes decisions back)",
    "draft_followup": "Create an UNSENT follow-up draft from unresolved comments and stored history",
}


class Declined(RuntimeError):
    """The author declined the confirmation prompt; nothing was written."""


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return s[:24] or "paper"


def _read(path_str: str):
    path = validate_input_file(path_str)
    data = path.read_bytes()
    try:
        parsed = read_document(path)
    except DocumentReadError as exc:
        raise ValidationError(str(exc)) from exc
    return path, parsed, sha256_bytes(data), sha256_text(parsed.full_text)


# Tool 1: onboarding and revision ingestion

def onboard(conn, *, file: str, comments_file: str, paper_id: str | None = None, title: str | None = None,
            confirm=None) -> dict:
    """First-time onboarding. `confirm(preview_dict) -> bool` is the author's approval of the baseline."""
    path, parsed, chash, thash = _read(file)
    cdata = parse_comments_json(Path(comments_file))
    title = (title or parsed.title or path.stem).strip()[:200]
    paper_id = validate_paper_id(paper_id) if paper_id else "P-%s-%s" % (_slug(title), chash[:6])
    if conn.execute("SELECT 1 FROM papers WHERE paper_id=?", (paper_id,)).fetchone():
        raise mm.DuplicateError("Paper ID %s already exists. Nothing was changed." % paper_id)
    other = conn.execute("SELECT paper_id FROM papers WHERE baseline_hash=?", (chash,)).fetchone()
    if other:
        alog.log_event(conn, kind="tool_call", tool="ingest_and_diff", paper_id=other["paper_id"], outcome="rejected",
                       detail="duplicate onboarding attempt (same baseline hash)", input_ref=path.name)
        raise mm.DuplicateError("This exact file is already onboarded as paper %s. Nothing was created." % other["paper_id"])
    headings = []
    for p in parsed.passages:
        if p.heading not in headings:
            headings.append(p.heading)
    preview = {"paper_id": paper_id, "title": title, "file": path.name, "format": parsed.file_format,
               "pages": parsed.page_count, "passages": len(parsed.passages), "headings": headings,
               "notes": parsed.notes, "comments": cdata["comments"], "synthetic": cdata["synthetic"],
               "sha256": chash}
    if confirm is not None and not confirm(preview):
        raise Declined("Onboarding was not approved; nothing was saved.")
    key = "onboard:%s:%s" % (paper_id, chash)

    def work():
        info = mm.onboard_paper(conn, paper_id, title, parsed, source_name=path.name, source_path=str(path),
                                src_file=path, content_hash=chash, text_hash=thash, comments=cdata["comments"],
                                synthetic=cdata["synthetic"])
        set_state(conn, "selected_paper", paper_id)
        return info

    with alog.tool_call(conn, "ingest_and_diff", paper_id, None, "onboard:" + path.name, key):
        result, replayed = rm.run_operation(conn, key, "onboard", paper_id, work, tool="onboard")
    result["replayed"] = replayed
    return result


def ingest_revision(conn, *, paper_id: str, file: str, claims_raw: dict | None, confirm=None) -> dict:
    mm.require_paper(conn, paper_id)
    path, parsed, chash, thash = _read(file)
    dup = conn.execute("SELECT version_no FROM documents WHERE paper_id=? AND (content_hash=? OR text_hash=?)",
                       (paper_id, chash, thash)).fetchone()
    if dup:
        alog.log_event(conn, kind="tool_call", tool="ingest_and_diff", paper_id=paper_id, outcome="rejected",
                       detail="duplicate ingestion rejected: identical to v%d" % dup["version_no"], input_ref=path.name)
        raise mm.DuplicateError("This document is identical to version v%d of paper %s (same file hash or same "
                                "extracted text). Nothing was ingested." % (dup["version_no"], paper_id))
    valid_ids = {c["comment_id"] for c in mm.active_only(conn, paper_id)}
    claims = parse_claims(claims_raw or {}, valid_ids)
    prev = mm.latest_version(conn, paper_id)
    old = er.load_passages(conn, paper_id, prev["version_id"])
    changes = diff_passages([p.text for p in old], [p.text for p in parsed.passages])
    preview = {"paper_id": paper_id, "file": path.name, "format": parsed.file_format, "new_version_no": prev["version_no"] + 1,
               "diff": summarize(changes, len(old), len(parsed.passages)), "claims": claims, "notes": parsed.notes,
               "active_comments": sorted(valid_ids), "sha256": chash}
    if confirm is not None and not confirm(preview):
        raise Declined("Ingestion was not approved; nothing was saved.")
    key = "ingest:%s:%s" % (paper_id, chash)

    def work():
        return mm.ingest_revision(conn, paper_id, parsed, source_name=path.name, source_path=str(path), src_file=path,
                                  content_hash=chash, text_hash=thash, claims=claims)

    with alog.tool_call(conn, "ingest_and_diff", paper_id, None, "ingest:" + path.name, key):
        result, replayed = rm.run_operation(conn, key, "ingest", paper_id, work, cycle_id="%s-C%d" % (paper_id, prev["version_no"]),
                                            tool="ingest_revision")
    result["replayed"] = replayed
    set_state(conn, "selected_paper", paper_id)
    return result


def set_claim(conn, *, paper_id: str, cycle_no: int | None, comment_id: str, value: str, note: str | None) -> dict:
    cyc = mm.get_cycle(conn, paper_id, cycle_no)
    mm.get_active_comment(conn, paper_id, comment_id)
    claims = parse_claims({comment_id: {"claim": value, "note": note}}, {comment_id})
    cur = conn.execute("SELECT claim_id FROM author_claims WHERE paper_id=? AND cycle_id=? AND comment_id=? AND is_current=1",
                       (paper_id, cyc["cycle_id"], comment_id)).fetchone()
    key = "claim:%s:%s:%s:%s:%s" % (cyc["cycle_id"], comment_id, cur["claim_id"] if cur else 0, value,
                                    hashlib.sha256((note or "").encode()).hexdigest()[:8])

    def work():
        only = claims[comment_id]
        cur_all = {c["comment_id"]: None for c in mm.active_only(conn, paper_id)}
        # record_claims rewrites only rows that differ; keep other comments' current claims untouched
        rows = conn.execute("SELECT comment_id, claim, note FROM author_claims WHERE paper_id=? AND cycle_id=? AND is_current=1",
                            (paper_id, cyc["cycle_id"])).fetchall()
        merged = {r["comment_id"]: {"claim": r["claim"], "note": r["note"]} for r in rows}
        merged[comment_id] = only
        n = mm.record_claims(conn, paper_id, cyc["cycle_id"], merged)
        del cur_all
        return {"cycle_id": cyc["cycle_id"], "comment_id": comment_id, "claim": value, "rows_written": n}

    with alog.tool_call(conn, "ingest_and_diff", paper_id, cyc["cycle_id"], "claim:%s=%s" % (comment_id, value), key):
        result, replayed = rm.run_operation(conn, key, "set_claim", paper_id, work, cycle_id=cyc["cycle_id"])
    result["replayed"] = replayed
    return result


# Tool 2 and 3: memory and evidence

def retrieve_memory(conn, paper_id, cycle_id, comment_id, enabled=True) -> dict:
    with alog.tool_call(conn, "retrieve_memory", paper_id, cycle_id, "comment:%s memory=%s" % (comment_id, "on" if enabled else "OFF")):
        return mm.retrieve_comment_memory(conn, paper_id, cycle_id, comment_id, enabled)


def search_changed(conn, paper_id, cycle, query_terms, comment_id):
    with alog.tool_call(conn, "retrieve_evidence:search_changed", paper_id, cycle["cycle_id"], "comment:%s terms=%d" % (comment_id, len(query_terms))):
        hits = er.search_changed(conn, paper_id, cycle["version_id"], cycle["diff_id"], query_terms)
        removed = er.search_removed(conn, paper_id, cycle["diff_id"], query_terms)
        flagged = sum(1 for p in er.changed_new_passages(conn, paper_id, cycle["diff_id"]) if p.injection_flag)
        return hits, removed, flagged


def search_full(conn, paper_id, cycle, query_terms, comment_id):
    with alog.tool_call(conn, "retrieve_evidence:search_full_document", paper_id, cycle["cycle_id"], "comment:%s" % comment_id):
        return er.search_full_document(conn, paper_id, cycle["version_id"], query_terms)


def verify_prior(conn, paper_id, cycle, prior_items, comment_id):
    with alog.tool_call(conn, "retrieve_evidence:verify_prior_evidence", paper_id, cycle["cycle_id"],
                        "comment:%s items=%d" % (comment_id, len(prior_items))):
        return er.verify_prior_evidence(conn, paper_id, cycle["version_id"], prior_items)


# Tool 4: record verdicts (called inside run_operation's transaction by agent.py)

def record_verdicts(conn, paper_id: str, cycle, results: list, decisions: list, fingerprint: str,
                    memory_mode: str) -> dict:
    now = None
    from database import utcnow
    now = utcnow()
    cid = cycle["cycle_id"]
    ids = []
    for n, r in enumerate(results):
        old = conn.execute("SELECT verdict_id FROM verdicts WHERE paper_id=? AND cycle_id=? AND comment_id=? AND is_current=1",
                           (paper_id, cid, r.comment_id)).fetchall()
        conn.execute("UPDATE verdicts SET is_current=0, supersede_reason=? WHERE paper_id=? AND cycle_id=? AND comment_id=? AND is_current=1",
                     ("re-evaluated (engine=%s)" % r.engine, paper_id, cid, r.comment_id))
        vid = conn.execute(
            "INSERT INTO verdicts(paper_id, cycle_id, comment_id, comment_rev_id, engine, memory_mode, plan_json, verdict,"
            " explanation, missing_evidence, contradiction, claim_assessment, regression, previous_verdict, previous_source,"
            " action_code, recommended_action, follow_up_question, needs_manual, injection_flag, guards_json, is_current,"
            " fingerprint, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?)",
            (paper_id, cid, r.comment_id, r.comment_rev_id, r.engine, memory_mode, jdump(r.plan), r.verdict, r.explanation,
             r.missing_evidence, r.contradiction, r.claim_assessment, int(r.regression), r.previous_verdict,
             r.previous_source, r.action_code, r.recommended_action, r.follow_up_question, int(r.needs_manual),
             r.injection_flag, jdump(r.guards), fingerprint, now)).lastrowid
        for o in old:
            conn.execute("UPDATE verdicts SET superseded_by=? WHERE verdict_id=?", (vid, o["verdict_id"]))
        for ev in r.evidence:
            conn.execute("INSERT INTO verdict_evidence(verdict_id, paper_id, version_id, passage_id, heading, page, quote, origin,"
                         " element_label) VALUES (?,?,?,?,?,?,?,?,?)",
                         (vid, paper_id, ev.version_id, ev.passage_id, ev.heading, ev.page, ev.quote, ev.origin, ev.label))
        conn.execute("UPDATE next_actions SET status='superseded', superseded_reason='new evaluation' WHERE paper_id=? AND cycle_id=?"
                     " AND comment_id=? AND status IN ('proposed','confirmed')", (paper_id, cid, r.comment_id))
        conn.execute("INSERT INTO next_actions(paper_id, cycle_id, comment_id, action_code, description, rationale, decided_by,"
                     " memory_influenced, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (paper_id, cid, r.comment_id, r.action_code, config.ACTION_DESCRIPTIONS[r.action_code],
                      r.recommended_action, r.engine, int(r.memory_influenced), "proposed", now))
        conn.execute("UPDATE pending_questions SET status='superseded', closed_at=?, closed_reason='new evaluation' WHERE paper_id=?"
                     " AND cycle_id=? AND comment_id=? AND status='open'", (now, paper_id, cid, r.comment_id))
        if r.verdict != "Addressed" and r.follow_up_question:
            conn.execute("INSERT INTO pending_questions(paper_id, cycle_id, comment_id, question, reason, status, created_at)"
                         " VALUES (?,?,?,?,?,'open',?)",
                         (paper_id, cid, r.comment_id, r.follow_up_question, r.action_code, now))
        ids.append(vid)
        if n == 0:
            # Scenario G: a failure after the first verdict was written proves the whole batch rolls back.
            rm.maybe_fail("evaluate_before_commit")
    for d in decisions:
        conn.execute("INSERT INTO agent_decisions(paper_id, cycle_id, comment_id, stage, engine, memory_mode, chosen_json,"
                     " default_json, changed_vs_default, rationale, evidence_seen_json, prior_state_json, created_at)"
                     " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (paper_id, cid, d["comment_id"], d["stage"], d["engine"], memory_mode, jdump(d["chosen"]),
                      jdump(d["default"]), int(d["changed"]), d["rationale"], jdump(d["evidence_seen"]),
                      jdump(d["prior_state"]), now))
        alog.log_decision(conn, paper_id, cid, "%s:%s" % (d["stage"], d["comment_id"]),
                          "chosen=%s default=%s changed=%s | %s" % (jdump(d["chosen"]), jdump(d["default"]), d["changed"], d["rationale"][:300]))
    conn.execute("UPDATE review_cycles SET status='evaluated' WHERE paper_id=? AND cycle_id=?", (paper_id, cid))
    alog.log_memory(conn, "write", paper_id, cid, "verdicts", "%d suggested verdicts saved (agent suggestions, not approvals)" % len(ids))
    return {"verdict_ids": ids, "count": len(ids)}


# Tools 5-7: thin logged wrappers so the registry is complete

def export_excel(conn, paper_id: str) -> dict:
    import excel_exporter
    with alog.tool_call(conn, "export_excel", paper_id, None, "matrix"):
        return excel_exporter.export_workbook(conn, paper_id)


def draft_followup(conn, paper_id: str, cycle_no: int | None = None) -> dict:
    import followup_drafter
    cyc = mm.get_cycle(conn, paper_id, cycle_no)
    with alog.tool_call(conn, "draft_followup", paper_id, cyc["cycle_id"], "cycle:%d" % cyc["cycle_no"]):
        return followup_drafter.create_draft(conn, paper_id, cyc)
