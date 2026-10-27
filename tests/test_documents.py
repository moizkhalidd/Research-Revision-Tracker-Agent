import unittest
from tests.helpers import IsolatedHome, make_pdf, sample, PROJECT  # noqa: F401
import document_reader as dr
import tools
from validators import ValidationError


class TestReaders(IsolatedHome):
    def test_latex_sections_and_no_pages(self):
        doc = dr.read_document(PROJECT / "sample_data" / "paper_v2.tex")
        headings = {p.heading for p in doc.passages}
        self.assertIn("Limitations", headings)
        self.assertTrue(all(p.page is None for p in doc.passages))
        self.assertIn("page numbers are not available", " ".join(doc.notes))

    def test_pdf_pages_and_headings(self):
        pdf = make_pdf(self.home / "p.pdf", [
            ["Abstract", "We study reminder scheduling in clinics.", "1 Introduction", "Missed appointments waste capacity."],
            ["2 Method", "Each clinic trains a local model.", "Updates are protected with epsilon 2.0."]])
        doc = dr.read_document(pdf)
        self.assertEqual(doc.page_count, 2)
        by_page = {p.page for p in doc.passages}
        self.assertEqual(by_page, {1, 2})
        self.assertIn("2 Method", {p.heading for p in doc.passages})

    def test_pdf_onboarding_and_ingest_keep_page_references(self):
        import json
        c = self.home / "c.json"
        c.write_text(json.dumps([{"id": "R1", "text": "State the privacy budget epsilon.",
                                  "checks": [{"label": "epsilon", "any": ["epsilon"], "numeric": True}]}]))
        v1 = make_pdf(self.home / "v1.pdf", [["1 Method", "Each clinic trains a local model."]])
        v2 = make_pdf(self.home / "v2.pdf", [["1 Method", "Each clinic trains a local model."], ["2 Privacy", "Updates use epsilon 2.0."]])
        tools.onboard(self.conn, file=str(v1), comments_file=str(c), paper_id="PDF-PAPER", confirm=lambda p: True)
        tools.ingest_revision(self.conn, paper_id="PDF-PAPER", file=str(v2), claims_raw={"R1": "resolved"}, confirm=lambda p: True)
        import agent
        agent.evaluate_cycle(self.conn, "PDF-PAPER", 1, "offline")
        row = self.conn.execute("SELECT page, quote FROM verdict_evidence WHERE paper_id='PDF-PAPER'").fetchone()
        self.assertEqual(row["page"], 2)
        self.assertIn("epsilon 2.0", row["quote"])

    def test_scanned_or_empty_pdf_is_rejected(self):
        pdf = make_pdf(self.home / "empty.pdf", [[""]])
        with self.assertRaises(dr.DocumentReadError):
            dr.read_document(pdf)

    def test_bad_extension_and_missing_file_rejected(self):
        with self.assertRaises(ValidationError):
            tools.onboard(self.conn, file=str(self.home / "x.exe"), comments_file=sample("reviewer_comments.json"), confirm=lambda p: True)


if __name__ == "__main__":
    unittest.main()
