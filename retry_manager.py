"""Idempotent operations, simulated failures and resumable checkpoints.

run_operation() is the single entry point for every state-changing operation:

  1. If the idempotency key already succeeded, the stored result is returned and
     nothing is written again (replay).
  2. Otherwise the work runs inside one SQLite transaction together with the
     'succeeded' marker. Any exception rolls the whole transaction back, so a
     failed attempt leaves no partial records.
  3. After a rollback the failure is recorded (operations.status = 'failed' and
     an action_log error row). A later retry with the same key starts clean.

Simulated failures (Scenario G) are controlled by a small JSON flag file kept
OUTSIDE the database so that arming/consuming the flag is not rolled back with
the failed transaction.
"""
from __future__ import annotations

import json

import action_logger as alog
import config
from database import jdump, jload, transaction, utcnow


class SimulatedToolFailure(RuntimeError):
    """Raised on purpose by maybe_fail() when a failure has been armed."""


# Stages at which a failure can be simulated.
FAILURE_STAGES = {
    "ingest_after_insert": "inside the ingest transaction, after document rows were written",
    "evaluate_before_commit": "inside the verdict-saving transaction, after some verdicts were written",
    "draft_after_file_write": "after the draft file is written, before its database row",
}


def arm_failure(stage: str, times: int = 1) -> dict:
    if stage not in FAILURE_STAGES:
        raise ValueError("Unknown stage %r. Choose one of: %s" % (stage, ", ".join(FAILURE_STAGES)))
    flag = {"stage": stage, "remaining": int(times), "armed_at": utcnow()}
    config.ensure_dirs()
    config.get_paths().failure_flag.write_text(json.dumps(flag), encoding="utf-8")
    return flag


def disarm_failure() -> None:
    p = config.get_paths().failure_flag
    if p.exists():
        p.unlink()


def failure_status() -> dict | None:
    p = config.get_paths().failure_flag
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def maybe_fail(stage: str) -> None:
    """Raise SimulatedToolFailure if this stage is armed; consumes one shot."""
    flag = failure_status()
    if not flag or flag.get("stage") != stage or int(flag.get("remaining", 0)) <= 0:
        return
    flag["remaining"] = int(flag["remaining"]) - 1
    if flag["remaining"] <= 0:
        disarm_failure()
    else:
        config.get_paths().failure_flag.write_text(json.dumps(flag), encoding="utf-8")
    raise SimulatedToolFailure("SIMULATED failure at stage '%s' (%s)" % (stage, FAILURE_STAGES[stage]))


def run_operation(conn, op_key: str, op_type: str, paper_id, fn, cycle_id=None, tool: str | None = None):
    """Run fn() atomically and idempotently. Returns (result_dict, replayed_bool).

    fn must return a JSON-serialisable dict and must do all of its database
    writes through `conn` so they are covered by the transaction.
    """
    tool = tool or op_type
    row = conn.execute("SELECT * FROM operations WHERE op_key = ?", (op_key,)).fetchone()
    if row and row["status"] == "succeeded":
        alog.log_event(conn, kind="tool_call", tool=tool, paper_id=paper_id, cycle_id=cycle_id,
                       input_ref=op_key, outcome="replayed",
                       detail="Idempotency key already succeeded; no new records written",
                       idempotency_key=op_key, attempt=row["attempts"])
        return jload(row["result_json"], {}), True

    attempts = (row["attempts"] if row else 0) + 1
    recovering = bool(row)  # previous attempt failed or was interrupted
    now = utcnow()
    conn.execute(
        "INSERT INTO operations(op_key, paper_id, op_type, status, attempts, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?) ON CONFLICT(op_key) DO UPDATE SET status='started',"
        " attempts=excluded.attempts, updated_at=excluded.updated_at, error=NULL",
        (op_key, paper_id, op_type, "started", attempts, now, now),
    )
    alog.clear_deferred()
    start_detail = "retry after earlier failure or interruption" if recovering else "first attempt"
    try:
        with transaction(conn):
            result = fn()
            conn.execute(
                "UPDATE operations SET status='succeeded', result_json=?, updated_at=? WHERE op_key=?",
                (jdump(result), utcnow(), op_key),
            )
            alog.log_event(conn, kind="tool_call", tool=tool, paper_id=paper_id, cycle_id=cycle_id,
                           input_ref=op_key, outcome="ok", detail=start_detail,
                           idempotency_key=op_key, attempt=attempts)
        return result, False
    except BaseException as exc:  # noqa: BLE001 - recorded then re-raised
        err = "%s: %s" % (type(exc).__name__, exc)
        conn.execute(
            "UPDATE operations SET status='failed', error=?, updated_at=? WHERE op_key=?",
            (err[:1000], utcnow(), op_key),
        )
        alog.log_event(conn, kind="tool_call", tool=tool, paper_id=paper_id, cycle_id=cycle_id,
                       input_ref=op_key, outcome="error",
                       detail="Transaction rolled back; no partial records were kept. Safe to retry.",
                       error=err, idempotency_key=op_key, attempt=attempts)
        alog.flush_deferred(conn)
        raise


# Checkpoints (working state that lets a failed operation resume)

def save_checkpoint(conn, paper_id, cycle_id, name, fingerprint, payload) -> None:
    conn.execute(
        "INSERT INTO checkpoints(paper_id, cycle_id, name, fingerprint, payload_json, created_at)"
        " VALUES (?,?,?,?,?,?) ON CONFLICT(paper_id, cycle_id, name) DO UPDATE SET"
        " fingerprint=excluded.fingerprint, payload_json=excluded.payload_json, created_at=excluded.created_at",
        (paper_id, cycle_id, name, fingerprint, jdump(payload), utcnow()),
    )
    alog.log_memory(conn, "write", paper_id, cycle_id, "checkpoint:" + name, "working-state checkpoint saved")


def load_checkpoint(conn, paper_id, cycle_id, name, fingerprint):
    row = conn.execute(
        "SELECT payload_json FROM checkpoints WHERE paper_id=? AND cycle_id=? AND name=? AND fingerprint=?",
        (paper_id, cycle_id, name, fingerprint),
    ).fetchone()
    if row:
        alog.log_memory(conn, "read", paper_id, cycle_id, "checkpoint:" + name, "resumed from checkpoint")
        return jload(row["payload_json"])
    return None


def clear_checkpoint(conn, paper_id, cycle_id, name) -> None:
    conn.execute("DELETE FROM checkpoints WHERE paper_id=? AND cycle_id=? AND name=?",
                 (paper_id, cycle_id, name))
