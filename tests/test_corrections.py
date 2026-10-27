import json
import unittest
from tests.helpers import IsolatedHome
import author_review as ar
import evaluation as ev
import memory_manager as mm

YES = lambda p: True  # noqa: E731


class TestCorrections(IsolatedHome):
    def setUp(self):
        super().setUp()
        ev.onboard_a(self.conn)
        ev.ingest(self.conn, ev.A_PID, "paper_v2.tex", "claims_v2.json")
        ev.evaluate(self.conn, ev.A_PID, "offline", 1)

    def test_verdict_correction_supersedes_and_keeps_history(self):
        ar.approve_verdict(self.conn, ev.A_PID, "R1.1", confirm=YES)
        ar.correct_verdict(self.conn, ev.A_PID, "R1.1", "Addressed", "Table 2 is enough", confirm=YES)
        rows = self.conn.execute("SELECT decision_type, is_current FROM author_decisions WHERE paper_id=? AND comment_id='R1.1' ORDER BY decision_id", (ev.A_PID,)).fetchall()
        self.assertEqual([(r[0], r[1]) for r in rows], [("approve", 0), ("correct", 1)])
        st = {s["comment_id"]: s for s in mm.current_status(self.conn, ev.A_PID)}["R1.1"]
        self.assertEqual((st["effective_verdict"], st["approval_state"]), ("Addressed", "corrected"))
        self.assertEqual(st["verdict"]["verdict"], "Partial")  # the agent suggestion is untouched

    def test_comment_correction_supersedes_and_changes_next_decision(self):
        ev.ingest(self.conn, ev.A_PID, "paper_v3.tex", "claims_v3.json")
        ev.evaluate(self.conn, ev.A_PID, "offline", 2)
        before = ev.cycle_verdicts(self.conn, ev.A_PID, 2)["R3.1"]
        checks = json.load(open(ev.S("comment_r31_corrected_checks.json"), encoding="utf-8"))
        res = ar.correct_comment(self.conn, ev.A_PID, "R3.1", interpretation="justification is enough", checks=checks, reason="rebuttal", confirm=YES)
        old = self.conn.execute("SELECT is_current FROM comment_revisions WHERE rev_id=?", (ev.A_PID + ":R3.1:r1",)).fetchone()[0]
        self.assertEqual(old, 0)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM verdicts WHERE paper_id=? AND comment_rev_id=? AND is_current=1", (ev.A_PID, ev.A_PID + ":R3.1:r1")).fetchone()[0], 0)
        ev.evaluate(self.conn, ev.A_PID, "offline", 2, only=["R3.1"])
        after = ev.cycle_verdicts(self.conn, ev.A_PID, 2)["R3.1"]
        self.assertEqual(after["comment_rev_id"], res["new_rev"])
        self.assertNotEqual(before["action_code"], after["action_code"])

    def test_withdrawn_comment_is_not_evaluated_but_kept(self):
        ar.withdraw_comment(self.conn, ev.A_PID, "R3.2", "withdrawn by reviewer", confirm=YES)
        self.assertNotIn("R3.2", {c["comment_id"] for c in mm.active_only(self.conn, ev.A_PID)})
        with self.assertRaises(mm.NotFound):
            ev.evaluate(self.conn, ev.A_PID, "offline", 1, only=["R3.2"], persist=False)
        self.assertGreaterEqual(self.count("verdicts", ev.A_PID), 6)

    def test_declined_confirmation_changes_nothing(self):
        before = ev.counts(self.conn, ev.A_PID)
        with self.assertRaises(Exception):
            ar.withdraw_comment(self.conn, ev.A_PID, "R3.2", "no", confirm=lambda p: False)
        self.assertEqual(before, ev.counts(self.conn, ev.A_PID))


if __name__ == "__main__":
    unittest.main()
