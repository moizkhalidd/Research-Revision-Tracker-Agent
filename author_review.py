"""Author approval and correction (tool 5: apply_author_correction).

Only the human author can create approved state, and only through the functions in this module. The LLM engines
never call them. Every function

  * validates its input and that the comment belongs to THIS paper (isolation),
  * asks for explicit confirmation through the `confirm` callback (the CLI prompts unless --yes is given),
  * runs inside retry_manager.run_operation, so it is atomic and idempotent,
  * never deletes or overwrites history: old rows get is_current = 0, superseded_by and a reason.

Agent suggestion versus author decision
  verdicts          what the agent SUGGESTED (never an approval)
  author_decisions  what the author APPROVED or CORRECTED. A 'correct' decision overrides the suggestion.
"""
from __future__ import annotations

import hashlib

import action_logger as alog
import config
import memory_manager as mm
import retry_manager as rm
from database import utcnow
from tools import Declined
from validators import ValidationError, clean_text, normalize_checks

TOOL = "apply_author_correction"


def _h(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12]


def _confirmed(confirm, preview: dict) -> None:
    if confirm is not None and not confirm(preview):
        raise Declined("Not confirmed; nothing was changed.")


def _entry(conn, paper_id: str, comment_id: str) -> dict:
    mm.get_active_comment(conn, paper_id, comment_id)  # NotFound for unknown, withdrawn or another paper's comment
    for s in mm.current_status(conn, paper_id):
        if s["comment_id"] == comment_id:
            return s
    raise mm.NotFound("Comment %r does not exist for paper %s." % (comment_id, paper_id))


def _supersede_decision(conn, paper_id, cycle_id, comment_id, reason) -> int | None:
    cur = conn.execute("SELECT decision_id FROM author_decisions WHERE paper_id=? AND cycle_id=? AND comment_id=? AND is_current=1",
                       (paper_id, cycle_id, comment_id)).fetchone()
    if cur:
        conn.execute("UPDATE author_decisions SET is_current=0, supersede_reason=? WHERE decision_id=?", (reason, cur["decision_id"]))
        return cur["decision_id"]
    return None


def _insert_decision(conn, paper_id, cycle_id, comment_id, verdict_id, dtype, final, note, old_id, reason) -> int:
    new_id = conn.execute(
        "INSERT INTO author_decisions(paper_id, cycle_id, comment_id, verdict_id, decision_type, final_verdict, note,"
        " is_current, created_at) VALUES (?,?,?,?,?,?,?,1,?)",
        (paper_id, cycle_id, comment_id, verdict_id, dtype, final, note, utcnow())).lastrowid
    if old_id:
        conn.execute("UPDATE author_decisions SET superseded_by=? WHERE decision_id=?", (new_id, old_id))
    return new_id


def _close_open_questions(conn, paper_id, comment_id, status, reason) -> int:
    cur = conn.execute("UPDATE pending_questions SET status=?, closed_at=?, closed_reason=? WHERE paper_id=? AND comment_id=?"
                       " AND status='open'", (status, utcnow(), reason, paper_id, comment_id))
    return cur.rowcount


def _run(conn, paper_id, cycle_id, key, op_type, ref, work):
    with alog.tool_call(conn, TOOL, paper_id, cycle_id, ref, key):
        result, replayed = rm.run_operation(conn, key, op_type, paper_id, work, cycle_id=cycle_id, tool=TOOL + ":" + op_type)
    result["replayed"] = replayed
    return result


# Approve / correct a suggested verdict

def approve_verdict(conn, paper_id: str, comment_id: str, note: str | None = None, confirm=None) -> dict:
    entry = _entry(conn, paper_id, comment_id)
    v = entry["verdict"]
    if not v:
        raise ValidationError("Comment %s has no suggested verdict yet. Run 'evaluate' first." % comment_id)
    note = clean_text(note, 1000, "note") if note else None
    cur = entry["decision"]
    cid = v["cycle_id"]
    if cur and cur["decision_type"] == "approve" and cur["verdict_id"] == v["verdict_id"]:
        alog.log_event(conn, kind="tool_call", tool=TOOL + ":approve", paper_id=paper_id, cycle_id=cid, outcome="replayed",
                       input_ref="approve:%s" % comment_id, detail="Already approved; no new records written")
        return {"comment_id": comment_id, "cycle_id": cid, "final_verdict": cur["final_verdict"],
                "decision_id": cur["decision_id"], "replayed": True}
    preview = {"action": "approve", "paper_id": paper_id, "comment_id": comment_id, "cycle_id": cid,
               "suggested_verdict": v["verdict"], "engine": v["engine"], "note": note}
    _confirmed(confirm, preview)
    key = "approve:%s:%s:%s:%s:%s" % (cid, comment_id, v["verdict_id"], cur["decision_id"] if cur else 0, _h(note))

    def work():
        old = _supersede_decision(conn, paper_id, cid, comment_id, "re-approved" if cur else "replaced by approval")
        did = _insert_decision(conn, paper_id, cid, comment_id, v["verdict_id"], "approve", v["verdict"], note, old, None)
        now = utcnow()
        if v["verdict"] == "Addressed":
            conn.execute("UPDATE next_actions SET status='completed', confirmed_at=? WHERE paper_id=? AND cycle_id=? AND comment_id=?"
                         " AND status IN ('proposed','confirmed')", (now, paper_id, cid, comment_id))
            _close_open_questions(conn, paper_id, comment_id, "closed", "author approved Addressed")
        else:
            conn.execute("UPDATE next_actions SET status='confirmed', confirmed_at=? WHERE paper_id=? AND cycle_id=? AND comment_id=?"
                         " AND status='proposed'", (now, paper_id, cid, comment_id))
        alog.log_memory(conn, "write", paper_id, cid, "author_decision:%d" % did,
                        "author APPROVED suggested verdict '%s' for %s" % (v["verdict"], comment_id))
        return {"comment_id": comment_id, "cycle_id": cid, "final_verdict": v["verdict"], "decision_id": did}

    return _run(conn, paper_id, cid, key, "approve", "approve:%s" % comment_id, work)


def correct_verdict(conn, paper_id: str, comment_id: str, final_verdict: str, note: str, confirm=None) -> dict:
    if final_verdict not in config.VERDICTS:
        raise ValidationError("Invalid verdict %r. Use one of: %s." % (final_verdict, ", ".join(config.VERDICTS)))
    note = clean_text(note, 1000, "note (the reason for the correction)")
    entry = _entry(conn, paper_id, comment_id)
    v = entry["verdict"]
    if not v:
        raise ValidationError("Comment %s has no suggested verdict yet. Run 'evaluate' first." % comment_id)
    if final_verdict == v["verdict"]:
        raise ValidationError("The suggested verdict is already '%s'. Use 'approve' to accept it." % final_verdict)
    cid = v["cycle_id"]
    cur = entry["decision"]
    if cur and cur["decision_type"] == "correct" and cur["final_verdict"] == final_verdict and cur["verdict_id"] == v["verdict_id"]:
        alog.log_event(conn, kind="tool_call", tool=TOOL + ":correct_verdict", paper_id=paper_id, cycle_id=cid, outcome="replayed",
                       input_ref="correct:%s" % comment_id, detail="Same correction already stored; no new records written")
        return {"comment_id": comment_id, "cycle_id": cid, "final_verdict": final_verdict, "decision_id": cur["decision_id"],
                "replayed": True}
    new_action = config.ACTION_FOR_AUTHOR_VERDICT[final_verdict]
    preview = {"action": "correct verdict", "paper_id": paper_id, "comment_id": comment_id, "cycle_id": cid,
               "agent_suggested": v["verdict"], "author_final": final_verdict, "reason": note,
               "next_action_after": new_action}
    _confirmed(confirm, preview)
    key = "correct_verdict:%s:%s:%s:%s:%s" % (cid, comment_id, v["verdict_id"], cur["decision_id"] if cur else 0, _h(final_verdict, note))

    def work():
        old = _supersede_decision(conn, paper_id, cid, comment_id, "superseded by author correction")
        did = _insert_decision(conn, paper_id, cid, comment_id, v["verdict_id"], "correct", final_verdict, note, old, None)
        now = utcnow()
        conn.execute("UPDATE next_actions SET status='superseded', superseded_reason='author corrected the verdict to ' || ?"
                     " WHERE paper_id=? AND cycle_id=? AND comment_id=? AND status IN ('proposed','confirmed')",
                     (final_verdict, paper_id, cid, comment_id))
        conn.execute(
            "INSERT INTO next_actions(paper_id, cycle_id, comment_id, action_code, description, rationale, decided_by,"
            " memory_influenced, status, confirmed_at, created_at) VALUES (?,?,?,?,?,?,?,1,?,?,?)",
            (paper_id, cid, comment_id, new_action, config.ACTION_DESCRIPTIONS[new_action],
             "Author correction: " + note, "author", "completed" if new_action == "CLOSE_COMMENT" else "confirmed", now, now))
        if final_verdict == "Addressed":
            _close_open_questions(conn, paper_id, comment_id, "closed", "author set the verdict to Addressed")
        elif not conn.execute("SELECT 1 FROM pending_questions WHERE paper_id=? AND comment_id=? AND status='open'",
                              (paper_id, comment_id)).fetchone():
            conn.execute("INSERT INTO pending_questions(paper_id, cycle_id, comment_id, question, reason, status, created_at)"
                         " VALUES (?,?,?,?,?,'open',?)",
                         (paper_id, cid, comment_id,
                          "You set %s to '%s' (%s). What change or evidence is still needed?" % (comment_id, final_verdict, note[:120]),
                          "AUTHOR_CORRECTION", now))
        alog.log_memory(conn, "write", paper_id, cid, "author_decision:%d" % did,
                        "author CORRECTED %s: agent '%s' -> author '%s'" % (comment_id, v["verdict"], final_verdict))
        return {"comment_id": comment_id, "cycle_id": cid, "agent_verdict": v["verdict"], "final_verdict": final_verdict,
                "decision_id": did, "next_action": new_action}

    return _run(conn, paper_id, cid, key, "correct_verdict", "correct:%s" % comment_id, work)


def confirm_action(conn, paper_id: str, comment_id: str, confirm=None) -> dict:
    """The author confirms that the proposed next action for a comment should stand."""
    entry = _entry(conn, paper_id, comment_id)
    act = entry["action"]
    if not act or act["status"] != "proposed":
        raise ValidationError("Comment %s has no proposed next action waiting for confirmation." % comment_id)
    _confirmed(confirm, {"action": "confirm next action", "paper_id": paper_id, "comment_id": comment_id,
                         "next_action": act["action_code"], "description": act["description"]})
    key = "confirm_action:%s:%s" % (comment_id, act["action_id"])

    def work():
        conn.execute("UPDATE next_actions SET status='confirmed', confirmed_at=? WHERE action_id=? AND paper_id=?",
                     (utcnow(), act["action_id"], paper_id))
        alog.log_memory(conn, "write", paper_id, act["cycle_id"], "next_action:%d" % act["action_id"], "author confirmed next action")
        return {"comment_id": comment_id, "action_id": act["action_id"], "action_code": act["action_code"]}

    return _run(conn, paper_id, act["cycle_id"], key, "confirm_action", "confirm:%s" % comment_id, work)


# Clarifications and additional evidence from the author

def add_clarification(conn, paper_id: str, comment_id: str, text: str, kind: str = "clarification",
                      cycle_no: int | None = None, confirm=None) -> dict:
    if kind not in ("clarification", "evidence", "note"):
        raise ValidationError("kind must be clarification, evidence or note.")
    text = clean_text(text, 2000, "clarification text")
    mm.get_active_comment(conn, paper_id, comment_id)
    cyc = mm.get_cycle(conn, paper_id, cycle_no)
    _confirmed(confirm, {"action": "record " + kind, "paper_id": paper_id, "comment_id": comment_id,
                         "cycle_id": cyc["cycle_id"], "text": text,
                         "note": "Stored as an unverified author statement; it does not change any verdict by itself."})
    key = "clarify:%s:%s:%s" % (cyc["cycle_id"], comment_id, _h(kind, text))

    def work():
        clar_id = conn.execute("INSERT INTO clarifications(paper_id, cycle_id, comment_id, kind, text, created_at) VALUES (?,?,?,?,?,?)",
                               (paper_id, cyc["cycle_id"], comment_id, kind, text, utcnow())).lastrowid
        answered = _close_open_questions(conn, paper_id, comment_id, "answered", "author replied (%s)" % kind)
        alog.log_memory(conn, "write", paper_id, cyc["cycle_id"], "clarification:%d" % clar_id,
                        "%s for %s stored; %d open question(s) marked answered" % (kind, comment_id, answered))
        return {"comment_id": comment_id, "cycle_id": cyc["cycle_id"], "clar_id": clar_id, "questions_answered": answered}

    return _run(conn, paper_id, cyc["cycle_id"], key, "clarify", "clarify:%s" % comment_id, work)


# Correct or withdraw a reviewer comment (supersedes the recorded interpretation)

def _retire_dependents(conn, paper_id: str, comment_id: str, old_rev_id: str, reason: str) -> dict:
    v = conn.execute("UPDATE verdicts SET is_current=0, supersede_reason=? WHERE paper_id=? AND comment_id=? AND comment_rev_id=?"
                     " AND is_current=1", (reason, paper_id, comment_id, old_rev_id)).rowcount
    d = conn.execute("UPDATE author_decisions SET is_current=0, supersede_reason=? WHERE paper_id=? AND comment_id=? AND is_current=1"
                     " AND verdict_id IN (SELECT verdict_id FROM verdicts WHERE paper_id=? AND comment_rev_id=?)",
                     (reason, paper_id, comment_id, paper_id, old_rev_id)).rowcount
    a = conn.execute("UPDATE next_actions SET status='superseded', superseded_reason=? WHERE paper_id=? AND comment_id=?"
                     " AND status IN ('proposed','confirmed')", (reason, paper_id, comment_id)).rowcount
    q = _close_open_questions(conn, paper_id, comment_id, "superseded", reason)
    return {"verdicts_superseded": v, "decisions_superseded": d, "actions_superseded": a, "questions_superseded": q}


def _new_revision(conn, paper_id, cur, *, text, interpretation, checks, status, reason) -> str:
    from database import jdump
    new_rev = "%s:%s:r%d" % (paper_id, cur["comment_id"], cur["rev_no"] + 1)
    conn.execute("UPDATE comment_revisions SET is_current=0, superseded_by=?, supersede_reason=? WHERE rev_id=?",
                 (new_rev, reason, cur["rev_id"]))
    conn.execute(
        "INSERT INTO comment_revisions(rev_id, paper_id, comment_id, rev_no, text, interpretation, checks_json, status,"
        " is_current, reason, created_by, created_at) VALUES (?,?,?,?,?,?,?,?,1,?,?,?)",
        (new_rev, paper_id, cur["comment_id"], cur["rev_no"] + 1, text, interpretation, jdump(checks), status, reason, "author", utcnow()))
    return new_rev


def correct_comment(conn, paper_id: str, comment_id: str, *, text: str | None = None, interpretation: str | None = None,
                    checks=None, reason: str, confirm=None) -> dict:
    reason = clean_text(reason, 1000, "reason")
    cur = mm.get_active_comment(conn, paper_id, comment_id)
    new_text = clean_text(text, field="comment text") if text else cur["text"]
    new_interp = clean_text(interpretation, 1500, "interpretation") if interpretation else cur["interpretation"]
    new_checks = normalize_checks(checks, comment_id) if checks is not None else cur["checks"]
    if new_text == cur["text"] and new_interp == cur["interpretation"] and new_checks == cur["checks"]:
        raise ValidationError("Nothing to correct: text, interpretation and checklist are all unchanged.")
    preview = {"action": "correct comment", "paper_id": paper_id, "comment_id": comment_id,
               "old": {"text": cur["text"], "interpretation": cur["interpretation"], "checks": [c["label"] for c in cur["checks"]]},
               "new": {"text": new_text, "interpretation": new_interp, "checks": [c["label"] for c in new_checks]},
               "reason": reason,
               "effect": "The old revision is kept as superseded; earlier verdicts for it stop counting; open questions and "
                         "proposed actions for this comment are superseded; it must be re-evaluated."}
    _confirmed(confirm, preview)
    from database import jdump
    key = "correct_comment:%s:%s:%s:%s" % (paper_id, comment_id, cur["rev_id"], _h(new_text, new_interp, jdump(new_checks), reason))

    def work():
        why = "comment corrected (%s)" % reason[:120]
        new_rev = _new_revision(conn, paper_id, cur, text=new_text, interpretation=new_interp, checks=new_checks, status="active", reason=reason)
        counts = _retire_dependents(conn, paper_id, comment_id, cur["rev_id"], why)
        alog.log_memory(conn, "write", paper_id, None, "comment_revision:" + new_rev,
                        "author CORRECTED %s (rev %d -> %d); %s" % (comment_id, cur["rev_no"], cur["rev_no"] + 1, jdump(counts)))
        return {"comment_id": comment_id, "old_rev": cur["rev_id"], "new_rev": new_rev, **counts}

    return _run(conn, paper_id, None, key, "correct_comment", "correct_comment:%s" % comment_id, work)


def withdraw_comment(conn, paper_id: str, comment_id: str, reason: str, confirm=None) -> dict:
    reason = clean_text(reason, 1000, "reason")
    cur = mm.get_active_comment(conn, paper_id, comment_id)
    _confirmed(confirm, {"action": "withdraw comment", "paper_id": paper_id, "comment_id": comment_id, "text": cur["text"],
                         "reason": reason,
                         "effect": "The comment is no longer evaluated or exported as active; all earlier records stay as history."})
    key = "withdraw_comment:%s:%s:%s" % (paper_id, comment_id, cur["rev_id"])

    def work():
        from database import jdump
        why = "comment withdrawn (%s)" % reason[:120]
        new_rev = _new_revision(conn, paper_id, cur, text=cur["text"], interpretation=cur["interpretation"], checks=cur["checks"],
                                status="withdrawn", reason=reason)
        counts = _retire_dependents(conn, paper_id, comment_id, cur["rev_id"], why)
        alog.log_memory(conn, "write", paper_id, None, "comment_revision:" + new_rev,
                        "author WITHDREW %s; %s" % (comment_id, jdump(counts)))
        return {"comment_id": comment_id, "old_rev": cur["rev_id"], "new_rev": new_rev, **counts}

    return _run(conn, paper_id, None, key, "withdraw_comment", "withdraw:%s" % comment_id, work)
