"""The decision loop.

For one review cycle the agent does, per comment:

  1. RETRIEVE MEMORY       approved text/corrections, claim, previous verdict + evidence, clarifications, open questions
  2. FIRST-PASS EVIDENCE   search the passages that changed in this revision
  3. PLAN (decision #1)    the engine sees the claim, the memory and the first-pass evidence and chooses follow-up
                           tools: search_full_document and/or verify_prior_evidence (or none).
                           THIS choice changes which tools run and is logged with its rationale.
  4. RUN CHOSEN TOOLS
  5. JUDGE                 the engine suggests a verdict with verbatim evidence quotes
  6. GUARD + NEXT ACTION   (decision #2) Python validates the verdict, computes regression and claim support from
                           stored memory, and fixes the next action. The engine's proposal is only accepted if the
                           rules allow it for that verdict.
  7. SAVE                  one transaction: verdicts, evidence, questions, next actions, decision log.

Suggested verdicts are never approvals. Approval lives in author_decisions (author_review.py).
Heavy work (LLM calls) happens outside the write transaction; results are checkpointed so a failed save can be
retried without repeating model calls or creating duplicates.
"""
from __future__ import annotations

import dataclasses

import action_logger as alog
import config
import evidence_retriever as er
import memory_manager as mm
import retry_manager as rm
import tools
from database import jdump, jload, utcnow
from llm_client import EngineError, GroqLLM
from models import EvidenceItem, VerdictResult
from verdict_engine import GroqEngine, OfflineEngine, finalize_verdict


def make_engine(conn, kind: str, paper_id=None, cycle_id=None):
    if kind == "offline":
        return OfflineEngine()
    if kind == "groq":
        return GroqEngine(GroqLLM(conn), paper_id, cycle_id)
    raise ValueError("Unknown engine %r (use 'groq' or 'offline')." % kind)


def _passage_dict(item: dict, passage, origin: str, label=None, quote=None) -> dict:
    existing = item["_by_pid"].get(passage.passage_id)
    if existing is None:
        existing = {"alias": "E%d" % (len(item["passages"]) + 1), "passage_id": passage.passage_id,
                    "version_id": passage.version_id, "heading": passage.heading, "page": passage.page,
                    "text": passage.text, "origins": [], "injection_flag": passage.injection_flag,
                    "prior_labels": [], "prior_quotes": {}}
        item["passages"].append(existing)
        item["_by_pid"][passage.passage_id] = existing
    if origin not in existing["origins"]:
        existing["origins"].append(origin)
    if label:
        existing["prior_labels"].append(label)
        existing["prior_quotes"][label] = quote
    return existing


def _build_item(conn, paper_id: str, cycle, comment: dict, memory_enabled: bool) -> dict:
    cid = comment["comment_id"]
    mem = tools.retrieve_memory(conn, paper_id, cycle["cycle_id"], cid, memory_enabled)
    c = mem["comment"]
    terms = er.build_query(c["text"], c["checks"], c["interpretation"])
    hits, removed, flagged = tools.search_changed(conn, paper_id, cycle, terms, cid)
    item = {"comment": c, "claim": mem["claim"], "previous": mem["previous"], "claim_history": mem["claim_history"],
            "clarifications": mem["clarifications"], "open_questions": mem["open_questions"],
            "memory_enabled": memory_enabled, "passages": [], "_by_pid": {}, "first_pass_passages": [],
            "prior_check": [], "injection_excluded": flagged, "terms": terms, "plan": {"followup_tools": []}}
    for h in hits:
        d = _passage_dict(item, h.passage, h.origin)
        if h.origin == "changed":
            item["first_pass_passages"].append(d)
    for h in removed:
        _passage_dict(item, h.passage, "removed")
    if flagged:
        alog.log_security(conn, paper_id, cycle["cycle_id"], "comment:" + cid,
                          "%d instruction-like passage(s) in the changed text were excluded from evidence" % flagged)
    return item


def _validate_plan(conn, paper_id, cycle_id, item: dict, raw_plan: dict | None) -> dict:
    tools_req = (raw_plan or {}).get("followup_tools", [])
    reason = (raw_plan or {}).get("reason", "") or "(no reason given)"
    chosen = []
    for t in tools_req if isinstance(tools_req, list) else []:
        if t not in config.FOLLOWUP_TOOLS:
            alog.log_event(conn, kind="decision", tool="agent", paper_id=paper_id, cycle_id=cycle_id, outcome="rejected",
                           input_ref="plan:%s" % item["comment"]["comment_id"], detail="unknown tool %r ignored" % (t,))
            continue
        if t == "verify_prior_evidence" and not (item["previous"] and item["previous"]["evidence"]):
            alog.log_event(conn, kind="decision", tool="agent", paper_id=paper_id, cycle_id=cycle_id, outcome="rejected",
                           input_ref="plan:%s" % item["comment"]["comment_id"], detail="verify_prior_evidence unavailable: no prior evidence in memory")
            continue
        if t not in chosen:
            chosen.append(t)
    return {"followup_tools": chosen, "reason": str(reason)[:400]}


def _run_followups(conn, paper_id: str, cycle, item: dict) -> None:
    cid = item["comment"]["comment_id"]
    for t in item["plan"]["followup_tools"]:
        if t == "search_full_document":
            for h in tools.search_full(conn, paper_id, cycle, item["terms"], cid):
                _passage_dict(item, h.passage, "full_document")
        elif t == "verify_prior_evidence":
            prior = item["previous"]["evidence"]
            checked = tools.verify_prior(conn, paper_id, cycle, prior, cid)
            item["prior_check"] = checked
            for pe in checked:
                if pe["present"]:
                    p = er.get_passage(conn, paper_id, pe["passage_id"])
                    if p:
                        _passage_dict(item, p, "prior_still_present", label=pe.get("label"), quote=pe["quote"])


def _result_to_json(r: VerdictResult) -> dict:
    return dataclasses.asdict(r)


def _result_from_json(d: dict) -> VerdictResult:
    d = dict(d)
    d["evidence"] = [EvidenceItem(**e) for e in d["evidence"]]
    return VerdictResult(**d)


def _decision_records(item: dict, plan: dict, result: VerdictResult, engine_name: str, hit_count: int) -> list[dict]:
    cid = item["comment"]["comment_id"]
    prev = item["previous"]
    prior_state = {"previous_verdict": prev["verdict"] if prev else None, "previous_source": prev["source"] if prev else None,
                   "prior_evidence_items": len(prev["evidence"]) if prev else 0, "claim": item["claim"]["value"],
                   "corrected_comment": bool(item["comment"].get("corrected")), "memory_enabled": item["memory_enabled"]}
    seen = {"first_pass_changed_hits": len(item["first_pass_passages"]), "passages_given_to_engine": len(item["passages"]),
            "prior_still_present": [bool(p["present"]) for p in item["prior_check"]], "hit_count": hit_count}
    return [
        {"comment_id": cid, "stage": "plan", "engine": engine_name,
         "chosen": {"followup_tools": plan["followup_tools"]}, "default": {"followup_tools": []},
         "changed": bool(plan["followup_tools"]), "rationale": plan["reason"],
         "evidence_seen": seen, "prior_state": prior_state},
        {"comment_id": cid, "stage": "next_action", "engine": engine_name,
         "chosen": {"verdict": result.verdict, "action_code": result.action_code, "regression": result.regression},
         "default": {"action_code": config.DEFAULT_ACTION_BY_VERDICT[result.verdict]},
         "changed": result.action_code != config.DEFAULT_ACTION_BY_VERDICT[result.verdict],
         "rationale": "%s | guards: %s" % (result.recommended_action, "; ".join(result.guards) or "none"),
         "evidence_seen": {"evidence_items": len(result.evidence), "claim_assessment": result.claim_assessment},
         "prior_state": prior_state},
    ]


def evaluate_cycle(conn, paper_id: str, cycle_no: int | None = None, engine_kind: str = "groq",
                   only: list[str] | None = None, memory_enabled: bool = True, persist: bool = True) -> dict:
    """Evaluate the active comments of one cycle. With persist=False nothing is written (diagnostic runs)."""
    mm.require_paper(conn, paper_id)
    cycle = mm.get_cycle(conn, paper_id, cycle_no)
    cid = cycle["cycle_id"]
    comments = mm.active_only(conn, paper_id)
    if only:
        unknown = [x for x in only if x not in {c["comment_id"] for c in comments}]
        if unknown:
            raise mm.NotFound("Unknown or withdrawn comment(s) for paper %s: %s" % (paper_id, ", ".join(unknown)))
        comments = [c for c in comments if c["comment_id"] in only]
    if not comments:
        raise mm.NotFound("No active comments to evaluate.")
    engine = make_engine(conn, engine_kind, paper_id, cid)
    mode = "enabled" if memory_enabled else "disabled-diagnostic"
    fp = mm.evaluation_fingerprint(conn, paper_id, cid, [c["comment_id"] for c in comments], engine.name, memory_enabled)
    op_key = "evaluate:%s:%s" % (cid, fp)

    if persist:
        done = conn.execute("SELECT * FROM operations WHERE op_key=? AND status='succeeded'", (op_key,)).fetchone()
        if done:
            alog.log_event(conn, kind="tool_call", tool="record_verdicts", paper_id=paper_id, cycle_id=cid, input_ref=op_key,
                           outcome="replayed", detail="Same inputs already evaluated; no new records written", idempotency_key=op_key)
            return {"replayed": True, "engine": engine.name, "cycle_id": cid, "results": [], "saved": jload(done["result_json"], {}),
                    "fingerprint": fp, "memory_mode": mode}

    # Phase 1: memory + first pass (no model calls)
    items = [_build_item(conn, paper_id, cycle, c, memory_enabled) for c in comments]

    # Phase 2: plan (one model call for the whole cycle) and per-comment judgment, with checkpoints
    ckpt_name = "evaluate"
    ckpt = rm.load_checkpoint(conn, paper_id, cid, ckpt_name, fp) if persist else None
    ckpt = ckpt or {"plans": None, "results": {}}
    with alog.tool_call(conn, "plan_followup_tools", paper_id, cid, "engine=%s comments=%d" % (engine.name, len(items))):
        if ckpt["plans"] is None:
            ckpt["plans"] = engine.plan(items, mode)
            if persist:
                rm.save_checkpoint(conn, paper_id, cid, ckpt_name, fp, ckpt)
    results: list[VerdictResult] = []
    decisions: list[dict] = []
    for item in items:
        k = item["comment"]["comment_id"]
        item["plan"] = _validate_plan(conn, paper_id, cid, item, ckpt["plans"].get(k))
        _run_followups(conn, paper_id, cycle, item)
        if k in ckpt["results"]:
            result = _result_from_json(ckpt["results"][k]["result"])
        else:
            with alog.tool_call(conn, "judge_comment", paper_id, cid, "comment:%s engine=%s" % (k, engine.name)):
                raw = engine.evaluate(item, mode)
                result = finalize_verdict(item, raw, engine.name, mode)
            ckpt["results"][k] = {"result": _result_to_json(result)}
            if persist:
                rm.save_checkpoint(conn, paper_id, cid, ckpt_name, fp, ckpt)
        results.append(result)
        decisions.extend(_decision_records(item, item["plan"], result, engine.name, len(item["passages"])))
        if result.guards:
            alog.log_event(conn, kind="decision", tool="guard", paper_id=paper_id, cycle_id=cid, input_ref="comment:" + k,
                           outcome="guarded", detail="; ".join(result.guards)[:900])
    out = {"replayed": False, "engine": engine.name, "cycle_id": cid, "results": results, "decisions": decisions,
           "fingerprint": fp, "memory_mode": mode, "items": items}
    if not persist:
        return out

    # Phase 3: one atomic, idempotent save
    def save():
        return tools.record_verdicts(conn, paper_id, cycle, results, decisions, fp, mode)

    with alog.tool_call(conn, "record_verdicts", paper_id, cid, "comments=%d" % len(results), op_key):
        saved, replayed = rm.run_operation(conn, op_key, "evaluate", paper_id, save, cycle_id=cid, tool="record_verdicts")
    rm.clear_checkpoint(conn, paper_id, cid, ckpt_name)
    out["saved"], out["replayed"] = saved, replayed
    return out
