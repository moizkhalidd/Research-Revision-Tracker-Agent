"""Command line interface. Run `python main.py --help` to see the commands.

Each command is its own short process that opens data/tracker.db, does its work and exits, so closing the
terminal and coming back later loses nothing.

Exit codes: 0 fine, 1 bad input or state, 2 LLM unavailable, 3 tool failure (simulated or unexpected).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import action_logger as alog
import agent
import author_review as ar
import cli_views as views
import config
import memory_manager as mm
import retry_manager as rm
import tools
from database import connect, get_state, set_state
from llm_client import EngineError, EngineUnavailable, GroqLLM
from validators import ValidationError, validate_comment_id, validate_paper_id

EXIT_OK, EXIT_INPUT, EXIT_ENGINE, EXIT_FAILURE = 0, 1, 2, 3


def out(text: str = "") -> None:
    print(text)


def resolve_paper(conn, args) -> str:
    pid = getattr(args, "paper", None)
    if pid:
        validate_paper_id(pid)
        mm.require_paper(conn, pid)
        return pid
    sel = get_state(conn, "selected_paper")
    if sel and conn.execute("SELECT 1 FROM papers WHERE paper_id=?", (sel,)).fetchone():
        return sel
    papers = mm.list_papers(conn)
    if len(papers) == 1:
        return papers[0]["paper_id"]
    raise ValidationError("No paper selected. Run 'python main.py papers', then 'python main.py select --paper <ID>' or pass --paper <ID>.")


def make_confirm(args):
    """Author confirmation. --yes confirms without prompting (the preview is still printed)."""
    def confirm(preview: dict) -> bool:
        out(views.render_preview(preview))
        if getattr(args, "yes", False):
            out("Confirmed with --yes.")
            return True
        if not sys.stdin.isatty():
            out("No interactive terminal and no --yes: not confirmed, nothing changed.")
            return False
        return input("Confirm? [y/N] ").strip().lower() in ("y", "yes")
    return confirm


def ask_claims(conn, paper_id: str) -> dict:
    """Interactively ask the author which comments they believe are resolved."""
    out("For each reviewer comment, state what you believe about this revision.")
    out("  r = resolved   p = partially resolved   o = outstanding   Enter = no claim")
    claims = {}
    names = {"r": "resolved", "p": "partial", "o": "outstanding"}
    for c in mm.active_only(conn, paper_id):
        out("\n%s: %s" % (c["comment_id"], c["text"]))
        a = input("  claim [r/p/o/Enter]: ").strip().lower()
        if a in names:
            note = input("  short note (optional): ").strip()
            claims[c["comment_id"]] = {"claim": names[a], "note": note or None}
    return claims


def read_json_file(path: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValidationError("Could not read JSON file %s: %s" % (path, exc))


def cmd_papers(conn, args):
    papers = mm.list_papers(conn)
    sel = get_state(conn, "selected_paper")
    if not papers:
        out("No papers onboarded yet. Start with: python main.py onboard --file <paper.tex|pdf> --comments <comments.json>")
        return EXIT_OK
    out("%-24s %-9s %-8s %-7s %s" % ("Paper ID", "Versions", "Cycles", "Synth.", "Title"))
    for p in papers:
        out("%-24s %-9d %-8d %-7s %s%s" % (p["paper_id"], p["versions"], p["cycles"], "yes" if p["is_synthetic"] else "no", p["title"],
                                         "   <- selected" if p["paper_id"] == sel else ""))
    return EXIT_OK


def cmd_select(conn, args):
    pid = validate_paper_id(args.paper)
    mm.require_paper(conn, pid)
    set_state(conn, "selected_paper", pid)
    out("Selected paper %s. Other papers stay isolated; their records are never read for this paper." % pid)
    return EXIT_OK


def cmd_onboard(conn, args):
    res = tools.onboard(conn, file=args.file, comments_file=args.comments, paper_id=args.paper_id, title=args.title,
                        confirm=make_confirm_onboard(args))
    out("Paper %s onboarded: %d reviewer comments approved as baseline." % (res["paper_id"], res["comments"]))
    out("Baseline stored as %s (recoverable without re-upload)." % res["stored_path"])
    out("Next: python main.py ingest --paper %s --file <first revision> --claims <claims.json>" % res["paper_id"])
    return EXIT_OK


def make_confirm_onboard(args):
    base = make_confirm(args)

    def confirm(preview: dict) -> bool:
        out("\nBaseline preview (nothing is saved until you confirm)")
        out("Paper ID: %s" % preview["paper_id"])
        out("Title: %s" % preview["title"])
        out("File: %s (%s, pages: %s), %d passages" % (preview["file"], preview["format"], preview["pages"] or "n/a", preview["passages"]))
        if preview["synthetic"]:
            out("This is a synthetic sample paper.")
        for n in preview["notes"]:
            out(views.wrap("Note about reading this file: " + n, "    ", "  "))
        out("Reviewer comments that will be saved as the baseline:")
        for c in preview["comments"]:
            out(views.wrap("%s (%s): %s" % (c["id"], c["reviewer"], c["text"]), "      ", "  "))
        return base({"action": "approve baseline and reviewer comments"})
    return confirm


def cmd_ingest(conn, args):
    pid = resolve_paper(conn, args)
    claims_raw = read_json_file(args.claims) if args.claims else None
    if claims_raw is None and sys.stdin.isatty():
        claims_raw = ask_claims(conn, pid)
    elif claims_raw is None:
        out("No --claims file and no interactive terminal: all claims recorded as 'none'.")

    def confirm(preview: dict) -> bool:
        out("\nRevision preview")
        out("File %s (%s) will become version v%d. Passage changes: %s" % (preview["file"], preview["format"], preview["new_version_no"], preview["diff"]))
        out("Claims: " + (", ".join("%s=%s" % (k, v["claim"]) for k, v in preview["claims"].items()) or "none"))
        return make_confirm(args)({"action": "ingest this revision and record the claims"})

    res = tools.ingest_revision(conn, paper_id=pid, file=args.file, claims_raw=claims_raw, confirm=confirm)
    out("%s ingested as v%d, review cycle %s%s." % (res["paper_id"], res["version_no"], res["cycle_id"], " (replayed, nothing new written)" if res["replayed"] else ""))
    out("Diff: %s" % res["diff_summary"])
    out("Next: python main.py evaluate --paper %s --engine groq    (or --engine offline for the deterministic test engine)" % pid)
    return EXIT_OK


def cmd_claim(conn, args):
    pid = resolve_paper(conn, args)
    res = tools.set_claim(conn, paper_id=pid, cycle_no=args.cycle, comment_id=validate_comment_id(args.comment), value=args.value, note=args.note)
    out("Claim for %s in %s set to '%s'%s." % (res["comment_id"], res["cycle_id"], res["claim"], " (replayed)" if res["replayed"] else ""))
    return EXIT_OK


def cmd_evaluate(conn, args):
    pid = resolve_paper(conn, args)
    only = [validate_comment_id(c) for c in args.comment] if args.comment else None
    if args.no_memory and not args.dry_run:
        raise ValidationError("--no-memory is a diagnostic option and requires --dry-run (nothing is saved).")
    res = agent.evaluate_cycle(conn, pid, args.cycle, args.engine, only=only, memory_enabled=not args.no_memory, persist=not args.dry_run)
    out(views.render_evaluation(res))
    if args.dry_run:
        out("\nDry run, nothing was saved.")
    elif not res.get("replayed"):
        out("\nSaved %d suggested verdicts. They are only suggestions until you approve them (python main.py approve --comment <ID>)." % len(res["results"]))
    return EXIT_OK


def cmd_status(conn, args):
    pid = resolve_paper(conn, args)
    status = mm.current_status(conn, pid)
    rec = mm.recommend_next_step(conn, pid, status)
    ov = mm.paper_overview(conn, pid)
    if args.json:
        out(json.dumps({
            "paper_id": pid, "title": ov["paper"]["title"], "versions": [v["version_no"] for v in ov["versions"]],
            "baseline_hash": ov["paper"]["baseline_hash"], "cycles": [c["cycle_no"] for c in ov["cycles"]],
            "comments": [{"comment_id": s["comment_id"], "effective_verdict": s["effective_verdict"], "approval_state": s["approval_state"],
                          "unresolved": s["unresolved"], "rev_no": s["rev_no"], "action_code": s["action"]["action_code"] if s["action"] else None,
                          "action_status": s["action"]["status"] if s["action"] else None,
                          "open_questions": [q["question"] for q in s["questions"]]} for s in status],
            "recommended_next_step": rec}, indent=2))
    else:
        out(views.render_status(ov, status, rec))
    return EXIT_OK


def cmd_approve(conn, args):
    pid = resolve_paper(conn, args)
    r = ar.approve_verdict(conn, pid, validate_comment_id(args.comment), args.note, confirm=make_confirm(args))
    out("%s approved as '%s'%s." % (r["comment_id"], r["final_verdict"], " (already approved, replayed)" if r["replayed"] else ""))
    return EXIT_OK


def cmd_correct_verdict(conn, args):
    pid = resolve_paper(conn, args)
    r = ar.correct_verdict(conn, pid, validate_comment_id(args.comment), args.verdict, args.note, confirm=make_confirm(args))
    out("%s corrected to '%s'; next action is now %s." % (r["comment_id"], r["final_verdict"], r.get("next_action")))
    return EXIT_OK


def cmd_clarify(conn, args):
    pid = resolve_paper(conn, args)
    r = ar.add_clarification(conn, pid, validate_comment_id(args.comment), args.text, args.kind, args.cycle, confirm=make_confirm(args))
    out("Stored %s for %s (%d open question(s) marked answered). Re-evaluate to let it inform the next suggestion." % (args.kind, r["comment_id"], r["questions_answered"]))
    return EXIT_OK


def cmd_correct_comment(conn, args):
    pid = resolve_paper(conn, args)
    checks = read_json_file(args.checks_file) if args.checks_file else None
    r = ar.correct_comment(conn, pid, validate_comment_id(args.comment), text=args.text, interpretation=args.interpretation,
                           checks=checks, reason=args.reason, confirm=make_confirm(args))
    out("%s corrected: %s superseded by %s. Superseded: %d verdict(s), %d action(s), %d question(s)." % (
        r["comment_id"], r["old_rev"], r["new_rev"], r["verdicts_superseded"], r["actions_superseded"], r["questions_superseded"]))
    out("Status now shows it as not evaluated. Re-check it with: python main.py evaluate --comment %s --engine groq   (or wait for the next revision)." % r["comment_id"])
    if args.reevaluate:
        res = agent.evaluate_cycle(conn, pid, None, args.engine, only=[r["comment_id"]])
        out(views.render_evaluation(res))
    return EXIT_OK


def cmd_withdraw(conn, args):
    pid = resolve_paper(conn, args)
    r = ar.withdraw_comment(conn, pid, validate_comment_id(args.comment), args.reason, confirm=make_confirm(args))
    out("%s withdrawn (%s superseded by %s). It will no longer be evaluated; its history is kept." % (r["comment_id"], r["old_rev"], r["new_rev"]))
    return EXIT_OK


def cmd_confirm_action(conn, args):
    pid = resolve_paper(conn, args)
    r = ar.confirm_action(conn, pid, validate_comment_id(args.comment), confirm=make_confirm(args))
    out("Next action %s for %s confirmed." % (r["action_code"], r["comment_id"]))
    return EXIT_OK


def cmd_history(conn, args):
    pid = resolve_paper(conn, args)
    validate_comment_id(args.comment)
    out(views.render_history(mm.history_for_comment(conn, pid, args.comment), args.comment))
    return EXIT_OK


def cmd_superseded(conn, args):
    out(views.render_superseded(mm.list_superseded(conn, resolve_paper(conn, args))))
    return EXIT_OK


def cmd_versions(conn, args):
    out(views.render_versions(mm.paper_overview(conn, resolve_paper(conn, args))))
    return EXIT_OK


def cmd_diff(conn, args):
    pid = resolve_paper(conn, args)
    cyc = mm.get_cycle(conn, pid, args.cycle)
    rows = conn.execute(
        "SELECT c.change_no, c.kind, c.similarity, o.heading AS oh, n.heading AS nh, o.text AS ot, n.text AS nt FROM changes c"
        " LEFT JOIN passages o ON o.paper_id=c.paper_id AND o.passage_id=c.old_passage_id"
        " LEFT JOIN passages n ON n.paper_id=c.paper_id AND n.passage_id=c.new_passage_id"
        " WHERE c.paper_id=? AND c.diff_id=? ORDER BY c.change_no", (pid, cyc["diff_id"])).fetchall()
    alog.log_memory(conn, "read", pid, cyc["cycle_id"], "diff:" + cyc["diff_id"], "%d changes" % len(rows))
    out("Cycle %d: %s -> %s, %d changed passages" % (cyc["cycle_no"], cyc["prev_version_id"], cyc["version_id"], len(rows)))
    for r in rows:
        out("")
        out("%s in %s%s" % (r["kind"].upper(), r["nh"] or r["oh"], " (similarity %.2f)" % r["similarity"] if r["similarity"] else ""))
        if r["ot"]:
            out(views.wrap(r["ot"], "      ", "  - "))
        if r["nt"]:
            out(views.wrap(r["nt"], "      ", "  + "))
    return EXIT_OK


def cmd_log(conn, args):
    pid = None if args.all else resolve_paper(conn, args)
    sql, params = "SELECT * FROM action_log WHERE 1=1", []
    if pid:
        sql += " AND paper_id = ?"
        params.append(pid)
    for col, val in (("kind", args.kind), ("tool", args.tool), ("cycle_id", args.cycle_id)):
        if val:
            sql += " AND %s LIKE ?" % col
            params.append(val + "%")
    if args.errors:
        sql += " AND (outcome IN ('error','flagged','rejected') OR error IS NOT NULL)"
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(args.limit)
    rows = list(reversed(conn.execute(sql, params).fetchall()))
    out(views.render_log(rows))
    return EXIT_OK


def cmd_export(conn, args):
    pid = resolve_paper(conn, args)
    r = tools.export_excel(conn, pid) if not args.out else _export_to(conn, pid, args.out)
    out("Workbook written: %s" % r["path"])
    out("%d rows written. The workbook is generated from the database; the database was not changed." % r["matrix_rows"])
    return EXIT_OK


def _export_to(conn, pid, out_path):
    import excel_exporter
    with alog.tool_call(conn, "export_excel", pid, None, "matrix:" + str(out_path)):
        return excel_exporter.export_workbook(conn, pid, out_path)


def cmd_draft(conn, args):
    pid = resolve_paper(conn, args)
    r = tools.draft_followup(conn, pid, args.cycle)
    if not r.get("created"):
        out(r.get("reason", "No draft created."))
        return EXIT_OK
    import followup_drafter
    out("UNSENT draft %s: %s" % ("already existed (replayed, nothing new written)" if r["replayed"] else "created", r["path"]))
    out("")
    out(followup_drafter.read_draft(r["path"]))
    return EXIT_OK


def cmd_simulate_failure(conn, args):
    if args.clear:
        rm.disarm_failure()
        out("Simulated failure disarmed.")
    elif args.status or not args.stage:
        out("Armed: %s" % (rm.failure_status() or "nothing"))
        out("Stages: " + "; ".join("%s (%s)" % (k, v) for k, v in rm.FAILURE_STAGES.items()))
    else:
        flag = rm.arm_failure(args.stage, args.times)
        out("Armed: next %d call(s) at stage '%s' will fail, roll back and be logged. Re-run the same command to retry." % (flag["remaining"], args.stage))
    return EXIT_OK


def cmd_llm_check(conn, args):
    s = config.get_llm_settings()
    out("Model: %s | API key: %s" % (s.model, "set" if s.api_key else "NOT SET (copy .env.example to .env and add your key)"))
    llm = GroqLLM(conn, s)
    t = time.perf_counter()
    r = llm.ping()
    out("Live call OK in %d ms, reply: %s" % ((time.perf_counter() - t) * 1000, r))
    return EXIT_OK


def cmd_eval(conn, args):
    import evaluation
    if args.eval_cmd == "run":
        return evaluation.cli_run(args.engine)
    if args.eval_cmd == "compare":
        return evaluation.cli_compare(args.engine)
    if args.eval_cmd == "metrics":
        return evaluation.cli_metrics()
    raise ValidationError("Unknown eval command.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python main.py", description=config.APP_NAME + " (SQLite-backed, evidence-based, author-approved).")
    sub = ap.add_subparsers(dest="command", metavar="command")

    def add(name, fn, help_, paper=True):
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(fn=fn)
        if paper:
            p.add_argument("--paper", help="paper ID (default: the selected paper)")
        return p

    add("papers", cmd_papers, "list onboarded papers", paper=False)
    p = add("select", cmd_select, "select the paper later commands use", paper=False)
    p.add_argument("--paper", required=True)

    p = add("onboard", cmd_onboard, "onboard a new paper and its reviewer comments (first time only)", paper=False)
    p.add_argument("--file", required=True, help="original paper (.pdf, .tex or .txt)")
    p.add_argument("--comments", required=True, help="reviewer comments JSON file")
    p.add_argument("--paper-id", help="stable paper ID (default: generated from title and hash)")
    p.add_argument("--title")
    p.add_argument("--yes", action="store_true", help="approve the baseline without the y/N prompt")

    p = add("ingest", cmd_ingest, "ingest a new revision and record the author's resolution claims")
    p.add_argument("--file", required=True)
    p.add_argument("--claims", help="claims JSON {comment_id: resolved|partial|outstanding}; omit to be asked")
    p.add_argument("--yes", action="store_true")

    p = add("claim", cmd_claim, "change the author's claim for one comment in a cycle")
    p.add_argument("--comment", required=True)
    p.add_argument("--value", required=True, choices=config.CLAIM_VALUES)
    p.add_argument("--note")
    p.add_argument("--cycle", type=int)

    p = add("evaluate", cmd_evaluate, "retrieve memory and evidence, then suggest a verdict for each comment")
    p.add_argument("--cycle", type=int, help="cycle number (default: latest)")
    p.add_argument("--engine", choices=("groq", "offline"), default="groq")
    p.add_argument("--comment", action="append", help="evaluate only this comment (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="evaluate but save nothing")
    p.add_argument("--no-memory", action="store_true", help="DIAGNOSTIC: do not retrieve prior task records (requires --dry-run)")

    p = add("status", cmd_status, "show saved state, unresolved comments and the recommended next step")
    p.add_argument("--json", action="store_true")

    p = add("approve", cmd_approve, "approve the agent's suggested verdict for a comment")
    p.add_argument("--comment", required=True)
    p.add_argument("--note")
    p.add_argument("--yes", action="store_true")
    p = add("correct-verdict", cmd_correct_verdict, "override the suggested verdict (the suggestion is kept as history)")
    p.add_argument("--comment", required=True)
    p.add_argument("--verdict", required=True, choices=config.VERDICTS)
    p.add_argument("--note", required=True, help="why the suggestion is wrong")
    p.add_argument("--yes", action="store_true")
    p = add("clarify", cmd_clarify, "add a clarification or evidence pointer (an unverified author statement)")
    p.add_argument("--comment", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--kind", choices=("clarification", "evidence", "note"), default="clarification")
    p.add_argument("--cycle", type=int)
    p.add_argument("--yes", action="store_true")
    p = add("correct-comment", cmd_correct_comment, "correct a comment's text, interpretation or checklist (old revision is superseded)")
    p.add_argument("--comment", required=True)
    p.add_argument("--text")
    p.add_argument("--interpretation")
    p.add_argument("--checks-file", help="JSON list of checks replacing the checklist")
    p.add_argument("--reason", required=True)
    p.add_argument("--reevaluate", action="store_true", help="immediately re-evaluate this comment against the latest cycle")
    p.add_argument("--engine", choices=("groq", "offline"), default="groq")
    p.add_argument("--yes", action="store_true")
    p = add("withdraw", cmd_withdraw, "withdraw a reviewer comment (history is kept)")
    p.add_argument("--comment", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--yes", action="store_true")
    p = add("confirm-action", cmd_confirm_action, "confirm the proposed next action for a comment")
    p.add_argument("--comment", required=True)
    p.add_argument("--yes", action="store_true")

    p = add("history", cmd_history, "full history of one comment, including superseded records")
    p.add_argument("--comment", required=True)
    add("superseded", cmd_superseded, "list every superseded record of the paper")
    add("versions", cmd_versions, "list document versions, hashes and stored copies")
    p = add("diff", cmd_diff, "show passage-level changes of a cycle")
    p.add_argument("--cycle", type=int)
    p = add("log", cmd_log, "show the action log (tool calls, memory reads/writes, decisions, errors)")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--kind")
    p.add_argument("--tool")
    p.add_argument("--cycle-id", dest="cycle_id")
    p.add_argument("--errors", action="store_true", help="only errors, rejections and security events")
    p.add_argument("--all", action="store_true", help="include events of every paper and global events")

    p = add("export", cmd_export, "generate the Excel revision matrix from SQLite")
    p.add_argument("--out")
    p = add("draft", cmd_draft, "create an UNSENT follow-up draft for unresolved comments")
    p.add_argument("--cycle", type=int)

    p = add("simulate-failure", cmd_simulate_failure, "arm a one-shot simulated tool failure (Scenario G)", paper=False)
    p.add_argument("--stage", choices=sorted(rm.FAILURE_STAGES))
    p.add_argument("--times", type=int, default=1)
    p.add_argument("--clear", action="store_true")
    p.add_argument("--status", action="store_true")

    add("llm-check", cmd_llm_check, "test the Groq key and model with one tiny live call", paper=False)

    p = add("eval", cmd_eval, "scenario evaluation, memory comparison and metrics", paper=False)
    esub = p.add_subparsers(dest="eval_cmd", required=True)
    for name, h in (("run", "run the 14 evaluation scenarios in an isolated sandbox"),
                    ("compare", "memory-enabled versus memory-disabled diagnostic (3 cases)"),
                    ("metrics", "pass rate, duplicate side effects, LLM calls, latency")):
        e = esub.add_parser(name, help=h)
        if name in ("run", "compare"):
            e.add_argument("--engine", choices=("offline", "groq"), default="offline")

    return ap


def dispatch(argv: list[str]) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not args.command:
        ap.print_help()
        return EXIT_OK
    conn = connect()
    try:
        return args.fn(conn, args)
    except tools.Declined as exc:
        out("Cancelled: %s" % exc)
        return EXIT_INPUT
    except (ValidationError, mm.NotFound, mm.DuplicateError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return EXIT_INPUT
    except EngineUnavailable as exc:
        print("ENGINE UNAVAILABLE: %s" % exc, file=sys.stderr)
        print("Saved data is untouched. Re-run the same command after fixing the problem.", file=sys.stderr)
        return EXIT_ENGINE
    except EngineError as exc:
        print("ENGINE ERROR: %s" % exc, file=sys.stderr)
        return EXIT_ENGINE
    except rm.SimulatedToolFailure as exc:
        print("TOOL FAILURE (simulated): %s" % exc, file=sys.stderr)
        print("The transaction was rolled back and the failure was logged. Re-run the SAME command to retry safely.", file=sys.stderr)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        print("\nInterrupted. Nothing half-written is kept; re-run the command.", file=sys.stderr)
        return 130
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        return dispatch(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:  # noqa: BLE001 - last-resort handler: show a clear message instead of a stack trace
        print("UNEXPECTED ERROR (%s): %s" % (type(exc).__name__, exc), file=sys.stderr)
        print("Your saved data is safe: every write is transactional. Check logs/app.log and re-run the command.", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
