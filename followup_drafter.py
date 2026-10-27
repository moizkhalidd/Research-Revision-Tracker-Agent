"""UNSENT follow-up draft (tool 7: draft_followup).

The draft is built from stored history (suggested and approved verdicts, missing evidence, earlier questions,
author clarifications and regressions) for every comment that is still unresolved. It is saved as a local text file
labelled UNSENT and recorded in the drafts table and the action log.

This application has NO email integration. Nothing is ever sent; the author copies the text if they want to use it.

Idempotency: the draft's identity is a hash of its body (without the generation timestamp). Creating the same draft
again returns the stored one and writes nothing. If the process dies after the file was written but before the
database row was committed (simulated failure stage 'draft_after_file_write'), the file is removed and a retry
produces exactly one file and one row.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import action_logger as alog
import config
import memory_manager as mm
import retry_manager as rm
from database import utcnow

UNSENT_BANNER = "UNSENT DRAFT. Nothing has been sent; this tool has no email integration."


def _history_lines(conn, paper_id: str, comment_id: str, upto_cycle_no: int) -> list[str]:
    h = mm.history_for_comment(conn, paper_id, comment_id)
    cyc_no = {r["cycle_id"]: r["cycle_no"] for r in conn.execute(
        "SELECT cycle_id, cycle_no FROM review_cycles WHERE paper_id=?", (paper_id,))}
    lines = []
    seen = set()
    for v in h["verdicts"]:
        if v["is_current"] and cyc_no.get(v["cycle_id"], 0) <= upto_cycle_no:
            key = (v["cycle_id"], v["verdict"])
            if key not in seen:
                seen.add(key)
                lines.append("cycle %d: suggested %s%s" % (cyc_no[v["cycle_id"]], v["verdict"], " (regression)" if v["regression"] else ""))
    for q in h["questions"]:
        if q["status"] in ("open", "answered", "superseded", "closed") and q["cycle_id"] in cyc_no and cyc_no[q["cycle_id"]] < upto_cycle_no:
            lines.append('cycle %d asked: "%s" (%s)' % (cyc_no[q["cycle_id"]], q["question"][:200], q["status"]))
    for k in h["clarifications"]:
        lines.append("author %s: %s" % (k["kind"], k["text"][:200]))
    return lines


def build_body(conn, paper_id: str, cyc) -> tuple[str, list[str]]:
    paper = mm.require_paper(conn, paper_id)
    status = mm.current_status(conn, paper_id)
    unresolved = [s for s in status if s["verdict"] and s["unresolved"]]
    pending_ok = [s for s in status if s["verdict"] and not s["unresolved"] and s["approval_state"] in ("pending", "stale")]
    not_evaluated = [s for s in status if not s["verdict"]]
    ids = [s["comment_id"] for s in unresolved]
    lines = [
        "Subject: Revision follow-up for %s (%s), outstanding reviewer comments after cycle %d" % (paper["title"], paper_id, cyc["cycle_no"]),
        "To: (no recipient set; fill in yourself)",
        "",
        "Hello,",
        "",
        "I re-checked the latest revision (%s) against the approved reviewer comments. %d comment(s) still need attention:" % (
            cyc["version_id"], len(unresolved)),
        "",
    ]
    for s in unresolved:
        v = s["verdict"]
        lines.append("%s - %s" % (s["comment_id"], s["text"]))
        src = "author-set" if s["effective_source"] == "author" else "agent-suggested, %s" % ("not yet approved" if s["approval_state"] == "pending" else s["approval_state"])
        lines.append("  Status: %s (%s)" % (s["effective_verdict"], src))
        if s["claim"] and s["claim"]["claim"] not in ("none",):
            lines.append("  Claim in cycle %d: %s%s" % (s["cycle_no"], s["claim"]["claim"], " - " + s["claim"]["note"] if s["claim"]["note"] else ""))
        if v["claim_assessment"] in ("unsupported", "overstated"):
            lines.append("  Claim check: %s by the evidence found." % v["claim_assessment"])
        if v["missing_evidence"]:
            lines.append("  Missing: %s" % v["missing_evidence"])
        if v["contradiction"]:
            lines.append("  Conflict: %s" % v["contradiction"])
        if s["evidence"]:
            ev = s["evidence"][0]
            lines.append('  Already found: "%s" [%s]' % (ev["quote"][:180], "%s, %s" % (ev["heading"], "p.%d" % ev["page"] if ev["page"] else "page n/a")))
        q = (s["questions"][0]["question"] if s["questions"] else v["follow_up_question"]) or ""
        if q:
            lines.append("  Question: %s" % q)
        lines.append("  Next step: %s" % (v["recommended_action"] or config.ACTION_DESCRIPTIONS.get(v["action_code"], "")))
        hist = _history_lines(conn, paper_id, s["comment_id"], s["cycle_no"] or cyc["cycle_no"])
        if hist:
            lines.append("  History: " + " | ".join(hist))
        lines.append("")
    if not_evaluated:
        lines.append("Not yet evaluated against this revision: %s." % ", ".join(s["comment_id"] for s in not_evaluated))
        lines.append("")
    if pending_ok:
        lines.append("Suggested as Addressed and waiting for your approval (not counted above): %s." % ", ".join(s["comment_id"] for s in pending_ok))
        lines.append("")
    lines += ["Please reply with the specific sections, tables or results for the items above, or tell me how to handle them.", "",
              "(Generated from stored review history. Verdicts marked agent-suggested are suggestions, not decisions.)"]
    return "\n".join(lines), ids


def create_draft(conn, paper_id: str, cyc) -> dict:
    body, ids = build_body(conn, paper_id, cyc)
    if not ids:
        alog.log_event(conn, kind="tool_call", tool="draft_followup", paper_id=paper_id, cycle_id=cyc["cycle_id"], outcome="skipped",
                       detail="no unresolved comments with a verdict; no draft created")
        return {"created": False, "reason": "No unresolved comments with a verdict, so there is nothing to follow up.", "replayed": False}
    chash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    key = "draft:%s:%s:%s" % (paper_id, cyc["cycle_id"], chash[:16])
    paths = config.ensure_dirs()
    fname = "UNSENT_%s_%s.txt" % (cyc["cycle_id"], chash[:10])
    out_dir = paths.drafts / paper_id
    out_path = out_dir / fname
    rel = out_path.relative_to(paths.home).as_posix()

    def work():
        out_dir.mkdir(parents=True, exist_ok=True)
        existed = out_path.exists()
        text = "%s\nGenerated (UTC): %s | Paper: %s | Cycle: %s\n\n%s\n" % (UNSENT_BANNER, utcnow(), paper_id, cyc["cycle_id"], body)
        tmp = out_path.with_name(out_path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(str(tmp), str(out_path))
        try:
            rm.maybe_fail("draft_after_file_write")
            conn.execute("INSERT INTO drafts(paper_id, cycle_id, content_hash, path, status, comment_ids, created_at) VALUES (?,?,?,?, 'UNSENT',?,?)",
                         (paper_id, cyc["cycle_id"], chash, rel, ",".join(ids), utcnow()))
        except BaseException:
            if not existed:
                try:
                    Path(out_path).unlink()
                except OSError:
                    pass
            raise
        alog.log_event(conn, kind="memory_write", tool="draft_followup", paper_id=paper_id, cycle_id=cyc["cycle_id"], input_ref="draft:" + fname,
                       detail="UNSENT draft created for %s (not sent; no email integration)" % ", ".join(ids))
        return {"created": True, "path": rel, "comment_ids": ids, "content_hash": chash[:16]}

    result, replayed = rm.run_operation(conn, key, "draft", paper_id, work, cycle_id=cyc["cycle_id"], tool="draft_followup")
    result["replayed"] = replayed
    return result


def read_draft(path_rel: str) -> str:
    return (config.get_paths().home / path_rel).read_text(encoding="utf-8")
