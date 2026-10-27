"""Action log: tool calls, memory reads/writes, decisions, security events, errors.

Every row goes into the action_log table (authoritative) and, best effort, to
logs/app.log as a plain text line. Rows written inside a transaction commit or
roll back together with the data they describe. Errors raised inside a
transaction are deferred and written by retry_manager after the rollback, so
a failed attempt is always visible in the log.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

import config
from database import utcnow

_file_logger: logging.Logger | None = None
_file_logger_path = None

# Errors that happened inside an open transaction; flushed after ROLLBACK.
_DEFERRED: list[dict] = []


def _get_file_logger() -> logging.Logger | None:
    global _file_logger, _file_logger_path
    try:
        path = config.ensure_dirs().logs / "app.log"
        if _file_logger is None or _file_logger_path != path:
            logger = logging.getLogger("rrt.%s" % path)
            logger.setLevel(logging.INFO)
            logger.propagate = False
            if not logger.handlers:
                handler = logging.FileHandler(path, encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
                logger.addHandler(handler)
            _file_logger, _file_logger_path = logger, path
        return _file_logger
    except OSError:
        return None


def log_event(conn, *, kind, tool=None, paper_id=None, cycle_id=None, input_ref=None,
              outcome="ok", detail=None, error=None, idempotency_key=None,
              attempt=None, duration_ms=None) -> int:
    cur = conn.execute(
        "INSERT INTO action_log(ts, paper_id, cycle_id, kind, tool, input_ref, outcome, detail, error,"
        " idempotency_key, attempt, duration_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (utcnow(), paper_id, cycle_id, kind, tool, _clip(input_ref), outcome, _clip(detail, 2000),
         _clip(error, 2000), idempotency_key, attempt, duration_ms),
    )
    fl = _get_file_logger()
    if fl is not None:
        fl.info("%s tool=%s paper=%s cycle=%s outcome=%s detail=%s error=%s", kind, tool, paper_id,
                cycle_id, outcome, _clip(detail, 300), _clip(error, 300))
    return cur.lastrowid


def _clip(value, limit=500):
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def log_memory(conn, direction: str, paper_id, cycle_id, what: str, detail: str = "") -> None:
    """direction is 'read' or 'write'."""
    log_event(conn, kind="memory_" + direction, tool="memory", paper_id=paper_id, cycle_id=cycle_id,
              input_ref=what, detail=detail)


def log_decision(conn, paper_id, cycle_id, what: str, detail: str) -> None:
    log_event(conn, kind="decision", tool="agent", paper_id=paper_id, cycle_id=cycle_id,
              input_ref=what, detail=detail)


def log_security(conn, paper_id, cycle_id, what: str, detail: str) -> None:
    log_event(conn, kind="security", tool="validators", paper_id=paper_id, cycle_id=cycle_id,
              input_ref=what, outcome="flagged", detail=detail)


@contextmanager
def tool_call(conn, tool: str, paper_id=None, cycle_id=None, input_ref=None, idempotency_key=None):
    """Log one tool invocation with its outcome, duration and any error."""
    start = time.perf_counter()
    try:
        yield
    except BaseException as exc:  # noqa: BLE001 - logged then re-raised
        ms = int((time.perf_counter() - start) * 1000)
        row = dict(kind="tool_call", tool=tool, paper_id=paper_id, cycle_id=cycle_id, input_ref=input_ref,
                   outcome="error", error="%s: %s" % (type(exc).__name__, exc),
                   idempotency_key=idempotency_key, duration_ms=ms)
        if conn.in_transaction:
            _DEFERRED.append(row)
        else:
            log_event(conn, **row)
        raise
    else:
        ms = int((time.perf_counter() - start) * 1000)
        log_event(conn, kind="tool_call", tool=tool, paper_id=paper_id, cycle_id=cycle_id,
                  input_ref=input_ref, outcome="ok", idempotency_key=idempotency_key, duration_ms=ms)


def flush_deferred(conn) -> None:
    """Write errors that occurred inside a transaction that has since rolled back."""
    while _DEFERRED:
        row = _DEFERRED.pop(0)
        log_event(conn, **row)


def clear_deferred() -> None:
    _DEFERRED.clear()
