# Research Document Revision Tracker

A command-line agent that tracks how an author revises a research paper in response to numbered reviewer comments.
It remembers the baseline paper, comments, every later version, the author's resolution claims, the evidence it found,
the verdicts, and every correction, in SQLite. It checks whether each comment was really addressed instead of trusting
the author's claim, proposes the next action, and writes an UNSENT follow-up draft. It never sends anything.

Python 3.10 or newer. Dependencies are pinned in `requirements.txt`. Developed and tested on Python 3.13.

## What is real and what is not

| Part | Status |
|---|---|
| Everything except LLM judgment (ingestion, diff, memory, corrections, isolation, retries, Excel, drafts, evaluation runner) | Executed offline; `python -m unittest discover -s tests -t .` runs 26 tests and the 14-scenario evaluation passes with the offline engine |
| `--engine offline` | A deterministic keyword rubric, NOT an LLM. It needs the author-confirmed "checks" in the comment file. Every verdict it creates is labelled `offline-deterministic` |
| `--engine groq` | The real LLM path. It has been run live: `eval run --engine groq` passed 14/14 scenarios and `eval compare --engine groq` was run with the real model (reports in `exports/`). Verdicts can vary between runs |
| PDF reading | Tested with a small generated text PDF. Heading detection is heuristic; scanned PDFs are rejected (no OCR) |
| LaTeX reading | Regex based. Page numbers do not exist in LaTeX, so references say "page n/a" instead of inventing one |

Groq model: the default is `openai/gpt-oss-120b`. Run `python main.py llm-check` once with your key.
Set `GROQ_MODEL` in `.env` to change it. The recorded Groq evaluation and memory comparison runs used `openai/gpt-oss-20b`.

## Architecture in one page

```
author --CLI--> main.py --> tools.py ---------------------------> SQLite (authoritative)
                              |  ingest_and_diff   (document_reader, document_diff)
                              |  retrieve_memory   (memory_manager)
                              |  retrieve_evidence (evidence_retriever)
                              |  record_verdicts   (one transaction)
                              |  apply_author_correction (author_review)
                              |  export_excel      (excel_exporter, read-only view)
                              |  draft_followup    (followup_drafter, UNSENT)
agent.py: decision loop            verdict_engine.py: Groq or offline engine + deterministic guards
```

Decision loop for each review cycle (`agent.py`):

1. **Retrieve memory** for each comment: approved text and corrections, claim, previous verdict and evidence, clarifications, open questions.
2. **First-pass evidence**: search the passages that changed in this revision.
3. **Plan (decision 1)**: the engine sees the claim, memory and first-pass evidence and chooses follow-up tools
   (`search_full_document`, `verify_prior_evidence`, or none). This changes which tools run and is logged with its rationale.
4. **Judge**: the engine suggests Addressed / Partial / Unverified / Ask Author with verbatim evidence quotes.
5. **Guards and next action (decision 2)**: Python validates the answer. Quotes must be verbatim in a stored passage, "Addressed" without
   a verifiable quote becomes "Unverified", regression is computed from stored memory, an unsupported "resolved" claim forces
   `REQUEST_EVIDENCE`, and the engine's proposed action is only accepted if allowed for that verdict.
6. **Save** verdicts, evidence, questions, actions and the decision log in one transaction.

Suggested verdicts live in `verdicts`. Author approvals and corrections live in `author_decisions`. They are never mixed.

Memory lifecycle: working state (cycle status, pending questions, checkpoints, selected paper), long-term memory (baseline, claims,
verdicts, decisions, corrections), source knowledge (documents, passages, diffs, hashes). Corrected records are kept with
`is_current = 0` and `superseded_by`. Every query filters on `paper_id` and child tables use composite foreign keys, so one paper cannot
point at another's rows. Excel and drafts are generated views; edits typed into a workbook are never imported.

## Install on Windows (PowerShell)

```powershell
mkdir C:\Projects\research_revision_tracker
cd C:\Projects\research_revision_tracker
# copy the project files here, then:
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
copy .env.example .env
notepad .env                           # paste your key after GROQ_API_KEY=
python main.py llm-check               # needs a key and network; everything else below can run offline
```

## Run the demonstration by hand

Use `--engine offline` to run without a key, or `--engine groq` (default for `evaluate`) for the real LLM.
Each command is a separate process, so closing the window and reopening it is a genuine restart.

```powershell
# 1 Onboard paper A once (confirm the baseline, or add --yes)
python main.py onboard --file sample_data\paper_v1.tex --comments sample_data\reviewer_comments.json --paper-id EDGE-DEMO

# 2 First revision, with the author's claims, then evaluate
python main.py ingest --file sample_data\paper_v2.tex --claims sample_data\claims_v2.json
python main.py evaluate --engine offline
python main.py status                   # suggestions, evidence needs, recommended next step
python main.py approve --comment R2.2

# 3 Restart: close PowerShell, open it again, activate the venv, then:
python main.py papers
python main.py status                   # state is back; nothing is uploaded again

# 4 Second revision. R2.2 was approved earlier, v3 drops the Limitations section, so it is flagged as a regression
python main.py ingest --file sample_data\paper_v3.tex --claims sample_data\claims_v3.json
python main.py evaluate --engine offline
python main.py history --comment R2.2

# 5 Correct or withdraw comments (old records stay as superseded history)
python main.py correct-comment --comment R3.1 --text "Please justify the choice of the 64-reading window." --interpretation "Justifying the 64-reading window is enough" --checks-file sample_data\comment_r31_corrected_checks.json --reason "reviewer clarified" --reevaluate --engine offline
python main.py withdraw --comment R3.2 --reason "withdrawn by the reviewer"
python main.py superseded
python main.py correct-verdict --comment R1.1 --verdict Addressed --note "Both baselines are named in Setup and their F1 scores are given in Results"
python main.py clarify --comment R1.1 --text "The first F1 value (0.84) is One-Class SVM and the second (0.79) is the LSTM autoencoder"

# 6 Excel and the UNSENT draft
python main.py export                   # exports\EDGE-DEMO_revision_matrix.xlsx
python main.py draft                    # drafts\EDGE-DEMO\UNSENT_*.txt (printed as well)
python main.py log --limit 40           # tool calls, memory reads and writes, decisions, errors
python main.py log --errors

# 7 Simulated failure and retry (stages: ingest_after_insert, evaluate_before_commit, draft_after_file_write)
python main.py simulate-failure --stage ingest_after_insert
python main.py ingest --file sample_data\paper_v3.tex --claims sample_data\claims_v3.json   # fails, rolls back, exit code 3
python main.py ingest --file sample_data\paper_v3.tex --claims sample_data\claims_v3.json   # retry succeeds once

# 8 Isolation and prompt injection: run them through the evaluation (below), or by hand:
python main.py onboard --file sample_data\paper_b_v1.tex --comments sample_data\reviewer_comments_b.json --paper-id FED-DEMO
python main.py select --paper FED-DEMO
python main.py status                   # only B's two comments appear
```

## Evaluation (all commands run in throw-away sandboxes; your real data is never touched)

```powershell
python -m unittest discover -s tests -t .           # offline unit tests (26)
python main.py eval run --engine offline            # 14 scenarios, no key needed
python main.py eval run --engine groq               # same 14 scenarios with the real LLM (needs key; passed 14/14 in the recorded run)
python main.py eval compare --engine offline        # memory enabled vs disabled, 3 cases
python main.py eval compare --engine groq           # API-dependent version
python main.py eval metrics                         # pass rate, duplicate side effects, LLM calls, latency
# the full report is saved to exports\evaluation_report.md after each eval run
```

The 14 scenarios cover the 8 compulsory ones (initial use, later use, restart,
correction, isolation, missing or conflicting evidence, tool failure and retry, untrusted instruction) plus partial-to-addressed, regression,
duplicates, invalid input and missing API key, Excel consistency and the UNSENT draft.

Needs a Groq key: `evaluate --engine groq`, `correct-comment --reevaluate --engine groq`, `eval run --engine groq`, `eval compare --engine groq`, `llm-check`.
Offline: everything else.

## Known limits

- The offline engine cannot judge meaning. It matches author-confirmed keywords. Do not present its results as LLM evaluation.
- With the Groq engine, verdicts vary between runs. Guards limit the damage (fabricated quotes are dropped, unsupported "Addressed" is downgraded,
  regressions come from stored memory) but the quality of semantic judgment is the model's.
- Text comparison cannot confirm a new experiment, a figure or a changed method. Such cases should end as Unverified or "manual inspection".
- Prompt-injection resistance is structural (untrusted text is delimited, flagged passages are never evidence, the model cannot write approvals, IDs or permissions).
  It is tested with a stand-in engine that obeys the attack; behaviour of a real model against new attacks is not guaranteed.
- No email integration exists. Drafts are files labelled UNSENT.
