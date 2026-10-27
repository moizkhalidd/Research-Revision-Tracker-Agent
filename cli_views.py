"""Turns saved data into plain text for the terminal. ASCII only, so the Windows console never fails on encoding."""
from __future__ import annotations

import json
import textwrap

import config

WIDTH = 100


def wrap(text: str, indent: str = "    ", first: str | None = None) -> str:
    return textwrap.fill(str(text), width=WIDTH, initial_indent=first if first is not None else indent,
                         subsequent_indent=indent, break_long_words=False, replace_whitespace=True)


def ref(heading, page) -> str:
    return "%s (%s)" % (heading or "no heading", "p.%d" % page if page else "page n/a")


def engine_banner(engine_name: str) -> str:
    if engine_name.startswith("offline"):
        return ("ENGINE: offline-deterministic keyword rubric. This is NOT an LLM evaluation; use --engine groq for "
                "genuine model judgment.")
    return "ENGINE: %s (LLM judgment; every verdict is only a suggestion until the author approves it)" % engine_name


def render_paper_header(overview: dict) -> str:
    p = overview["paper"]
    vs = overview["versions"]
    lines = ["Paper %s: %s%s" % (p["paper_id"], p["title"], "   [SYNTHETIC SAMPLE]" if p["is_synthetic"] else ""),
             "Baseline approved %s by %s | hash %s" % (p["baseline_approved_at"], p["baseline_approved_by"], p["baseline_hash"][:12]),
             "Versions: %s | Review cycles: %d" % (", ".join("v%d" % v["version_no"] for v in vs), len(overview["cycles"]))]
    return "\n".join(lines)


def render_status(overview: dict, status: list[dict], recommendation: str) -> str:
    out = [render_paper_header(overview), ""]
    if not status:
        out.append("(no active reviewer comments)")
    hdr = "%-7s %-12s %-26s %-6s %s" % ("Comment", "Verdict", "Source / approval", "Open", "Next action")
    out.append(hdr)
    for s in status:
        src = "none yet" if not s["verdict"] else ("AUTHOR (%s)" % s["approval_state"] if s["effective_source"] == "author"
                                                  else "agent, %s" % ("not approved" if s["approval_state"] == "pending" else s["approval_state"]))
        act = s["action"]
        act_txt = "%s [%s]" % (act["action_code"], act["status"]) if act else "-"
        out.append("%-7s %-12s %-26s %-6s %s" % (s["comment_id"], s["effective_verdict"] or "(not evaluated)", src,
                                                 "yes" if s["unresolved"] else "no", act_txt))
    questions = [(s["comment_id"], q) for s in status for q in s["questions"]]
    if questions:
        out += ["", "Open questions for the author:"]
        for cid, q in questions:
            out.append(wrap(q["question"], "      ", "  %-6s " % cid))
    out += ["", "Recommended next step:", wrap(recommendation, "  ")]
    out.append("")
    out.append("Legend: 'agent, not approved' is a SUGGESTION. AUTHOR means you approved or corrected it.")
    return "\n".join(out)


def render_evaluation(out: dict) -> str:
    lines = [engine_banner(out["engine"]), "Cycle %s | memory %s" % (out["cycle_id"], out["memory_mode"])]
    if out.get("replayed"):
        lines.append("Replayed: identical inputs were already evaluated; nothing new was written (idempotent).")
        return "\n".join(lines)
    for r in out["results"]:
        lines.append("")
        lines.append("%-6s %-11s action=%-20s claim=%s%s" % (r.comment_id, r.verdict, r.action_code, r.claim_assessment,
                                                           "  REGRESSION" if r.regression else ""))
        tools = r.plan.get("followup_tools") or []
        lines.append(wrap("Plan: %s | %s" % (", ".join(tools) if tools else "no extra tools", r.plan.get("reason", "")), "      ", "  "))
        lines.append(wrap(r.explanation, "      ", "  "))
        for e in r.evidence[:3]:
            lines.append(wrap('"%s"  [%s, %s]' % (e.quote[:240], ref(e.heading, e.page), e.origin), "        ", "    evidence: "))
        if r.missing_evidence:
            lines.append(wrap("missing: " + r.missing_evidence, "        ", "    "))
        if r.contradiction:
            lines.append(wrap("conflict: " + r.contradiction, "        ", "    "))
        if r.follow_up_question and r.verdict != "Addressed":
            lines.append(wrap("question: " + r.follow_up_question, "        ", "    "))
        for g in r.guards:
            lines.append(wrap("guard: " + g, "        ", "    "))
    changed = [d for d in out.get("decisions", []) if d["changed"]]
    lines += ["", "Decisions that differ from the default path: %d of %d logged" % (len(changed), len(out.get("decisions", [])))]
    for d in changed:
        lines.append(wrap("%s/%s: chose %s instead of %s because %s" % (d["comment_id"], d["stage"], d["chosen"], d["default"], d["rationale"][:220]),
                          "      ", "  - "))
    return "\n".join(lines)


def render_history(h: dict, comment_id: str) -> str:
    out = ["History of comment %s (nothing is deleted; superseded records are marked)" % comment_id, ""]
    out.append("Comment revisions:")
    for r in h["revisions"]:
        out.append(wrap("rev %d [%s, %s] %s%s%s" % (r["rev_no"], r["status"], "CURRENT" if r["is_current"] else "superseded", r["text"][:140],
                                                    " | interpretation: " + r["interpretation"] if r["interpretation"] else "",
                                                    " | reason: " + (r["reason"] or r["supersede_reason"] or "") if (r["reason"] or r["supersede_reason"]) else ""),
                        "      ", "  "))
    out.append("Author claims:")
    for c in h["claims"] or []:
        out.append("  %s: %s%s" % (c["cycle_id"], c["claim"], "" if c["is_current"] else " (superseded)"))
    out.append("Agent suggestions:")
    for v in h["verdicts"]:
        out.append("  #%d %s: %s via %s memory=%s action=%s%s%s" % (v["verdict_id"], v["cycle_id"], v["verdict"], v["engine"], v["memory_mode"], v["action_code"],
                                                                 " REGRESSION" if v["regression"] else "", "" if v["is_current"] else " (superseded: %s)" % (v["supersede_reason"] or "")))
    out.append("Author decisions:")
    for d in h["decisions"]:
        out.append("  #%d %s: %s -> %s%s%s" % (d["decision_id"], d["cycle_id"], d["decision_type"], d["final_verdict"],
                                                " | " + d["note"] if d["note"] else "", "" if d["is_current"] else " (superseded: %s)" % (d["supersede_reason"] or "")))
    out.append("Author clarifications:")
    for k in h["clarifications"]:
        out.append(wrap("%s %s: %s" % (k["cycle_id"], k["kind"], k["text"]), "      ", "  "))
    out.append("Questions asked:")
    for q in h["questions"]:
        out.append(wrap("%s [%s]: %s" % (q["cycle_id"], q["status"], q["question"]), "      ", "  "))
    out.append("Next actions:")
    for a in h["actions"]:
        out.append("  #%d %s: %s by %s [%s]%s" % (a["action_id"], a["cycle_id"], a["action_code"], a["decided_by"], a["status"],
                                                 " (%s)" % a["superseded_reason"] if a["superseded_reason"] else ""))
    return "\n".join(out)


def render_superseded(rows: list[dict]) -> str:
    out = ["Superseded records (kept as history, no longer current): %d" % len(rows), ""]
    for r in rows:
        out.append(wrap("%s #%s [%s] %s | superseded by %s | %s" % (r["table"], r["id"], r["comment_id"], r["summary"], r["superseded_by"] or "-",
                                                               r["reason"] or ""), "      ", "  "))
    return "\n".join(out)


def render_versions(overview: dict) -> str:
    out = [render_paper_header(overview), ""]
    for v in overview["versions"]:
        out.append("v%d %-8s %s (%s, %s pages) hash=%s ingested %s" % (v["version_no"], v["kind"], v["source_name"], v["file_format"],
                                                                      v["page_count"] or "n/a", v["content_hash"][:12], v["ingested_at"]))
        out.append("     stored copy: %s" % v["stored_path"])
        if v["extraction_notes"]:
            out.append(wrap("limitation: " + v["extraction_notes"], "           ", "     "))
    for c in overview["cycles"]:
        out.append("cycle %d (%s): %s -> %s, status %s" % (c["cycle_no"], c["cycle_id"], c["prev_version_id"], c["version_id"], c["status"]))
    return "\n".join(out)


def render_log(rows) -> str:
    out = ["%-5s %-20s %-8s %-14s %-34s %-9s %s" % ("id", "time (UTC)", "cycle", "kind", "tool", "outcome", "detail / error")]
    for r in rows:
        txt = (r["error"] or r["detail"] or r["input_ref"] or "")
        out.append("%-5d %-20s %-8s %-14s %-34s %-9s %s" % (r["id"], r["ts"], (r["cycle_id"] or "-").rsplit("-", 1)[-1], r["kind"], (r["tool"] or "")[:34],
                                                          r["outcome"] or "", txt[:160]))
    return "\n".join(out)


def render_preview(preview: dict) -> str:
    out = [""]
    for k, v in preview.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=True)
        out.append(wrap("%s: %s" % (k, v), "      ", "  "))
    return "\n".join(out)


VERDICT_CHOICES = config.VERDICTS
