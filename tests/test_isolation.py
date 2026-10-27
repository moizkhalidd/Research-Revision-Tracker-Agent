import unittest
from tests.helpers import IsolatedHome
import evaluation as ev
import memory_manager as mm
import author_review as ar


class TestIsolation(IsolatedHome):
    def setUp(self):
        super().setUp()
        ev.onboard_a(self.conn)
        ev.ingest(self.conn, ev.A_PID, "paper_v2.tex", "claims_v2.json")
        ev.evaluate(self.conn, ev.A_PID, "offline", 1)
        ev.onboard_b(self.conn)
        ev.ingest(self.conn, ev.B_PID, "paper_b_v2.tex", "claims_b_v2.json")
        ev.evaluate(self.conn, ev.B_PID, "offline", 1)

    def test_same_comment_id_resolves_per_paper(self):
        a = mm.get_comment(self.conn, ev.A_PID, "R1.1")["text"]
        b = mm.get_comment(self.conn, ev.B_PID, "R1.1")["text"]
        self.assertNotEqual(a, b)

    def test_cross_paper_access_is_refused(self):
        with self.assertRaises(mm.NotFound):
            mm.get_comment(self.conn, ev.B_PID, "R3.1")
        with self.assertRaises(mm.NotFound):
            mm.retrieve_comment_memory(self.conn, ev.B_PID, ev.A_PID + "-C1", "R1.1")
        with self.assertRaises(Exception):
            ar.approve_verdict(self.conn, ev.B_PID, "R3.1", confirm=lambda p: True)

    def test_no_row_of_one_paper_references_the_other(self):
        for table, col in (("verdict_evidence", "version_id"), ("verdicts", "cycle_id"), ("passages", "version_id")):
            n = self.conn.execute("SELECT COUNT(*) FROM %s WHERE paper_id=? AND %s NOT LIKE ?" % (table, col),
                                  (ev.B_PID, ev.B_PID + "-%")).fetchone()[0]
            self.assertEqual(n, 0, table)

    def test_composite_foreign_key_blocks_cross_paper_insert(self):
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute("INSERT INTO clarifications(paper_id, cycle_id, comment_id, kind, text, created_at)"
                              " VALUES (?,?,?,?,?,?)", (ev.B_PID, ev.A_PID + "-C1", "R1.1", "clarification", "x", "now"))


if __name__ == "__main__":
    unittest.main()
