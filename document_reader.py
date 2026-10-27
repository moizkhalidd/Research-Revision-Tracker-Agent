"""Document ingestion: PDF (pdfplumber), LaTeX (lightweight regex processing) and plain text.

The reader turns a file into ordered passages, each with a heading and, for PDF,
a page number. Limitations are reported in ParsedDocument.notes instead of
being hidden:

* PDF: headings are detected heuristically (numbered headings and a few
  well-known names); paragraph boundaries are approximated from line breaks.
  Scanned PDFs without a text layer are rejected (no OCR).
* LaTeX: the source has no page numbers, so page references are None and are
  shown as "page n/a". Macros are stripped, not expanded; \\input is not followed.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import config
from models import ParsedDocument, PassageDraft


class DocumentReadError(ValueError):
    """The file could not be turned into usable text."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


def read_document(path: Path) -> ParsedDocument:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".tex":
        return read_latex(Path(path).read_text(encoding="utf-8", errors="replace"))
    if suffix == ".txt":
        return read_plain_text(Path(path).read_text(encoding="utf-8", errors="replace"))
    raise DocumentReadError("Unsupported file type: %s" % suffix)


# Shared helpers
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NAMED_HEADINGS = {"abstract", "introduction", "related work", "conclusion", "conclusions", "references",
                   "acknowledgments", "acknowledgements", "appendix", "supplementary material"}
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,2})\.?\s+([A-Z][^\n]{1,90})$")


def split_long(text: str, limit: int = config.MAX_PASSAGE_CHARS) -> list[str]:
    """Split text longer than limit at sentence boundaries."""
    if len(text) <= limit:
        return [text]
    chunks, cur = [], ""
    for sent in _SENT_SPLIT.split(text):
        if cur and len(cur) + 1 + len(sent) > limit:
            chunks.append(cur)
            cur = sent
        else:
            cur = (cur + " " + sent).strip()
    if cur:
        chunks.append(cur)
    return chunks


def _heading_of_line(line: str):
    """Return (title, level) if the line looks like a section heading, else None."""
    s = line.strip()
    if not s or len(s) > 100 or s.endswith((".", ",", ";", ":")):
        return None
    if s.lower() in _NAMED_HEADINGS:
        return s.title(), 1
    m = _NUMBERED_HEADING.match(s)
    if m and len(s.split()) <= 12:
        return "%s %s" % (m.group(1), m.group(2)), m.group(1).count(".") + 1
    return None


def _assemble(passages: list[PassageDraft]) -> str:
    out, last = [], None
    for p in passages:
        if p.heading != last:
            out.append("## " + p.heading)
            last = p.heading
        out.append(p.text)
    return "\n\n".join(out)


# PDF

def read_pdf(path: Path) -> ParsedDocument:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise DocumentReadError("pdfplumber is not installed. Run: pip install -r requirements.txt") from exc
    try:
        with pdfplumber.open(str(path)) as pdf:
            pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(pdf.pages)]
            meta = {k: str(v) for k, v in (pdf.metadata or {}).items() if isinstance(v, (str, int, float))}
    except Exception as exc:  # noqa: BLE001 - pdfminer raises many types
        raise DocumentReadError("Could not open PDF (%s): %s" % (Path(path).name, exc)) from exc
    if sum(len(t.strip()) for _, t in pages) < 20:
        raise DocumentReadError(
            "The PDF has no extractable text layer (scanned?). OCR is not supported; use LaTeX or a text PDF.")
    passages: list[PassageDraft] = []
    heading, level = "(front matter)", 1
    buf: list[str] = []
    buf_page = 1

    def flush():
        nonlocal buf
        text = " ".join(x.strip() for x in buf).strip()
        buf = []
        if text:
            for chunk in split_long(text):
                passages.append(PassageDraft(len(passages), heading, level, buf_page, chunk))

    for page_no, text in pages:
        if buf and buf_page != page_no:
            flush()
        for line in text.splitlines():
            if not line.strip():
                flush()
                continue
            h = _heading_of_line(line)
            if h:
                flush()
                heading, level = h
                buf_page = page_no
                continue
            if not buf:
                buf_page = page_no
            buf.append(line)
            if sum(len(x) for x in buf) >= 700 and line.rstrip().endswith("."):
                flush()
    flush()
    notes = ["PDF: section headings are detected heuristically and paragraph boundaries are approximated; "
             "page numbers are reliable, section names may be '(front matter)' when no heading was recognised."]
    return ParsedDocument("pdf", _assemble(passages), passages, len(pages), notes, meta, meta.get("Title"))


# LaTeX
_HEAD_RE = re.compile(r"\\(subsubsection|subsection|section)\*?\{((?:[^{}]|\{[^{}]*\})*)\}")


def read_latex(src: str) -> ParsedDocument:
    s = re.sub(r"(?<!\\)%.*$", "", src, flags=re.M)
    tm = re.search(r"\\title\{((?:[^{}]|\{[^{}]*\})*)\}", s)
    title = re.sub(r"\s+", " ", re.sub(r"\\[a-zA-Z]+", "", tm.group(1))).strip() if tm else None
    if "\\begin{document}" in s:
        s = s.split("\\begin{document}", 1)[1]
    s = s.split("\\end{document}", 1)[0]
    s = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", s, flags=re.S)
    s = re.sub(r"\\(bibliography|bibliographystyle|maketitle|tableofcontents|newpage|clearpage|centering)\b(\{[^{}]*\})?", "", s)

    def head(m):
        lvl = {"section": 1, "subsection": 2, "subsubsection": 3}[m.group(1)]
        return "\n\n@@H%d@@%s\n\n" % (lvl, m.group(2))

    s = _HEAD_RE.sub(head, s)
    s = re.sub(r"\\begin\{abstract\}", "\n\n@@H1@@Abstract\n\n", s)
    s = re.sub(r"\\end\{abstract\}", "\n\n", s)
    s = s.replace("\\&", "\x01AMP\x01").replace("\\%", "\x01PCT\x01").replace("\\_", "\x01US\x01").replace("\\#", "\x01HASH\x01")
    s = re.sub(r"\\begin\{tabular\}\{[^{}]*\}", "\n", s)
    s = re.sub(r"\\(hline|toprule|midrule|bottomrule|cline\{[^{}]*\})", "", s)
    s = re.sub(r"\\\\(\[[^\]]*\])?", "\n", s)
    s = s.replace("&", " | ")
    s = re.sub(r"\\caption\{", "\n\nCaption: {", s)
    s = re.sub(r"\\cite[a-zA-Z]*\{([^{}]*)\}", r"[cite:\1]", s)
    s = re.sub(r"\\(?:eq)?ref\{([^{}]*)\}", r"[ref:\1]", s)
    s = re.sub(r"\\label\{[^{}]*\}", "", s)
    s = re.sub(r"\\item\s*", "\n- ", s)
    s = re.sub(r"\\(?:begin|end)\{[^{}]*\}(\[[^\]]*\])?", "\n\n", s)
    for _ in range(6):
        new = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", s)
        if new == s:
            break
        s = new
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    s = s.replace("{", "").replace("}", "").replace("~", " ")
    s = s.replace("\x01AMP\x01", "&").replace("\x01PCT\x01", "%").replace("\x01US\x01", "_").replace("\x01HASH\x01", "#")

    passages: list[PassageDraft] = []
    heading, level = "(front matter)", 1
    for block in re.split(r"\n\s*\n", s):
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        m = re.match(r"@@H(\d)@@(.*)", lines[0])
        if m:
            heading, level = re.sub(r"\s+", " ", m.group(2)).strip(), int(m.group(1))
            lines = lines[1:]
            if not lines:
                continue
        keep_lines = any(" | " in ln or ln.startswith("- ") for ln in lines)
        text = ("\n" if keep_lines else " ").join(re.sub(r"[ \t]+", " ", ln) for ln in lines).strip()
        if len(text) < 3:
            continue
        for chunk in split_long(text):
            passages.append(PassageDraft(len(passages), heading, level, None, chunk))
    if not passages:
        raise DocumentReadError("No readable text was found in the LaTeX source.")
    notes = ["LaTeX source: page numbers are not available, so page references are reported as 'page n/a'. "
             "Macros are stripped rather than expanded and \\input files are not followed."]
    return ParsedDocument("latex", _assemble(passages), passages, None, notes, {"title": title or ""}, title)


# Plain text

def read_plain_text(text: str) -> ParsedDocument:
    passages: list[PassageDraft] = []
    heading, level = "(front matter)", 1
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        h = _heading_of_line(lines[0])
        if h:
            heading, level = h
            lines = lines[1:]
        if lines:
            for chunk in split_long(" ".join(lines)):
                passages.append(PassageDraft(len(passages), heading, level, None, chunk))
    if not passages:
        raise DocumentReadError("The text file contains no readable text.")
    return ParsedDocument("text", _assemble(passages), passages, None,
                          ["Plain text: page numbers are not available; headings are detected heuristically."], {})
