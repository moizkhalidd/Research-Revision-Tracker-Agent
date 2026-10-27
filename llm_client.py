"""Thin wrapper around the official Groq SDK.

Responsibilities: read the key from the environment, call chat.completions in
JSON mode, retry once if the reply is not valid JSON, and record every call
(latency, token usage, success/failure) in the llm_calls table so cost and
latency can be reported honestly.

If the API cannot be used this module raises EngineUnavailable with a message
that explains how to recover. It never fabricates a reply.
"""
from __future__ import annotations

import json
import re
import sys
import time

import config
from database import utcnow


class EngineError(RuntimeError):
    """The engine produced something unusable (for example an invalid verdict)."""


class EngineUnavailable(EngineError):
    """The LLM cannot be reached or is not configured."""


RECOVERY_HINT = (
    "Recovery options: (1) check GROQ_API_KEY in your .env file (copy .env.example to .env); "
    "(2) check your network; (3) set GROQ_MODEL to a model listed on https://console.groq.com/docs/models "
    "(run 'python main.py llm-check' to test); (4) your saved data is untouched, so simply re-run the same command; "
    "(5) to test everything except LLM judgment use '--engine offline' (deterministic rules, clearly labelled, "
    "NOT an LLM evaluation)."
)


# When Groq says "too many requests" we wait and try again a few times before giving up.
RATE_LIMIT_RETRIES = 4
RATE_LIMIT_MAX_WAIT = 65      # seconds; the per minute limit resets within a minute
RATE_LIMIT_DEFAULT_WAIT = 20  # used when Groq does not say how long to wait


def _rate_limit_wait(exc) -> float:
    """How many seconds Groq asks us to wait, read from the Retry-After header or the error text."""
    try:
        value = exc.response.headers.get("retry-after")
        if value:
            return min(float(value) + 1, RATE_LIMIT_MAX_WAIT)
    except (AttributeError, ValueError):
        pass
    m = re.search(r"try again in (?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", str(exc))
    if m and (m.group(1) or m.group(2)):
        seconds = int(m.group(1) or 0) * 60 + float(m.group(2) or 0)
        return min(seconds + 1, RATE_LIMIT_MAX_WAIT)
    return RATE_LIMIT_DEFAULT_WAIT


def _extract_json(text: str) -> dict:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    obj = json.loads(t[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object")
    return obj


class GroqLLM:
    def __init__(self, conn, settings: config.LLMSettings | None = None):
        self.conn = conn
        self.settings = settings or config.get_llm_settings()
        self._client = None
        self.calls = 0

    @property
    def name(self) -> str:
        return "groq:%s" % self.settings.model

    def _get_client(self):
        if self._client is None:
            if not self.settings.api_key:
                raise EngineUnavailable("GROQ_API_KEY is not set. " + RECOVERY_HINT)
            try:
                from groq import Groq
            except ImportError as exc:
                raise EngineUnavailable("The 'groq' package is not installed. Run: pip install -r requirements.txt") from exc
            self._client = Groq(api_key=self.settings.api_key, timeout=self.settings.timeout,
                                max_retries=self.settings.max_retries)
        return self._client

    def _record(self, purpose, paper_id, cycle_id, memory_mode, latency_ms, usage, ok, error=None):
        self.conn.execute(
            "INSERT INTO llm_calls(ts, paper_id, cycle_id, purpose, engine, model, memory_mode, latency_ms,"
            " prompt_tokens, completion_tokens, ok, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (utcnow(), paper_id, cycle_id, purpose, "groq", self.settings.model, memory_mode, latency_ms,
             getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None),
             1 if ok else 0, (error or None) and str(error)[:500]))

    def _create(self, kwargs):
        import groq
        client = self._get_client()
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                return client.chat.completions.create(**kwargs)
            except groq.RateLimitError as exc:
                if attempt == RATE_LIMIT_RETRIES:
                    raise EngineUnavailable("Groq rate limit still reached after waiting and retrying: %s. Wait a few "
                                            "minutes and re-run the same command; finished work is checkpointed. %s"
                                            % (exc, RECOVERY_HINT)) from exc
                wait = _rate_limit_wait(exc)
                print("Groq rate limit reached, waiting %d s before retrying (%d of %d)."
                      % (wait, attempt + 1, RATE_LIMIT_RETRIES), file=sys.stderr, flush=True)
                time.sleep(wait)
            except groq.AuthenticationError as exc:
                raise EngineUnavailable("Groq rejected the API key (authentication error). " + RECOVERY_HINT) from exc
            except groq.BadRequestError as exc:
                raise EngineUnavailable("Groq rejected the request: %s. %s" % (exc, RECOVERY_HINT)) from exc
            except groq.PermissionDeniedError as exc:
                raise EngineUnavailable("Groq denied access for this key or model: %s. %s" % (exc, RECOVERY_HINT)) from exc
            except groq.NotFoundError as exc:
                raise EngineUnavailable("Model %r was not found (it may have been retired): %s. %s" %
                                        (self.settings.model, exc, RECOVERY_HINT)) from exc
            except (groq.APIConnectionError, groq.APITimeoutError) as exc:
                raise EngineUnavailable("Could not reach Groq (%s). %s" % (type(exc).__name__, RECOVERY_HINT)) from exc
            except groq.InternalServerError as exc:
                raise EngineUnavailable("Groq server error: %s. %s" % (exc, RECOVERY_HINT)) from exc

    def chat_json(self, system: str, user: str, *, purpose: str, paper_id=None, cycle_id=None,
                  memory_mode: str = "enabled") -> dict:
        import groq
        s = self.settings
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kwargs = {"model": s.model, "messages": messages, "temperature": 0,
                  "max_completion_tokens": s.max_completion_tokens,
                  "response_format": {"type": "json_object"}}
        last_error = None
        for attempt in range(2):
            start = time.perf_counter()
            try:
                resp = self._create(kwargs)
            except EngineUnavailable as exc:
                self._record(purpose, paper_id, cycle_id, memory_mode, int((time.perf_counter() - start) * 1000), None, False, exc)
                raise
            ms = int((time.perf_counter() - start) * 1000)
            self.calls += 1
            content = (resp.choices[0].message.content or "") if resp.choices else ""
            try:
                obj = _extract_json(content)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                self._record(purpose, paper_id, cycle_id, memory_mode, ms, getattr(resp, "usage", None), False,
                             "invalid JSON: %s" % exc)
                messages = messages + [{"role": "assistant", "content": content[:2000]},
                                       {"role": "user", "content": "That was not a single valid JSON object. "
                                        "Reply again with ONLY the JSON object, no prose, no code fences."}]
                kwargs["messages"] = messages
                continue
            self._record(purpose, paper_id, cycle_id, memory_mode, ms, getattr(resp, "usage", None), True)
            return obj
        raise EngineError("The model did not return valid JSON twice in a row (%s)." % last_error)

    def ping(self) -> dict:
        """Small live call used by 'llm-check'."""
        obj = self.chat_json('Reply with exactly this JSON object: {"ok": true}', "ping", purpose="ping")
        return obj