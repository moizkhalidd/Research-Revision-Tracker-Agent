import unittest
from tests.helpers import IsolatedHome
import evaluation as ev
import memory_manager as mm
import retry_manager as rm
import tools


class TestIdempotency(IsolatedHome):
    def test_ingest_failure_rolls_back_and_retry_is_clean(self):
        ev.onboard_a(self.conn)
        base = ev.counts(self.conn, ev.A_PID)
        rm.arm_failure("ingest_after_insert")
        with self.assertRaises(rm.SimulatedToolFailure):
            ev.ingest(self.conn, ev.A_PID, "paper_v2.tex", "claims_v2.json")
        self.assertEqual(ev.counts(self.conn, ev.A_PID), base)
        self.assertGreaterEqual(ev.log_rows(self.conn, ev.A_PID, outcome="error"), 1)
        ev.ingest(self.conn, ev.A_PID, "paper_v2.tex", "claims_v2.json")
        with self.assertRaises(mm.DuplicateError):
            ev.ingest(self.conn, ev.A_PID, "paper_v2.tex", "claims_v2.json")
        self.assertEqual(self.count("documents", ev.A_PID), 2)
        self.assertEqual(self.count("review_cycles", ev.A_PID), 1)

    def test_evaluate_failure_keeps_checkpoint_and_repeat_is_replay(self):
        ev.onboard_a(self.conn)
        ev.ingest(self.conn, ev.A_PID, "paper_v2.tex", "claims_v2.json")
        rm.arm_failure("evaluate_before_commit")
        with self.assertRaises(rm.SimulatedToolFailure):
            ev.evaluate(self.conn, ev.A_PID, "offline", 1)
        self.assertEqual(self.count("verdicts", ev.A_PID), 0)
        self.assertEqual(self.count("checkpoints", ev.A_PID), 1)
        ev.evaluate(self.conn, ev.A_PID, "offline", 1)
        again = ev.evaluate(self.conn, ev.A_PID, "offline", 1)
        self.assertTrue(again["replayed"])
        self.assertEqual(self.count("verdicts", ev.A_PID), 6)
        self.assertEqual(self.count("next_actions", ev.A_PID), 6)

    def test_draft_failure_leaves_no_orphan_file(self):
        ev.onboard_a(self.conn)
        ev.ingest(self.conn, ev.A_PID, "paper_v2.tex", "claims_v2.json")
        ev.evaluate(self.conn, ev.A_PID, "offline", 1)
        rm.arm_failure("draft_after_file_write")
        with self.assertRaises(rm.SimulatedToolFailure):
            tools.draft_followup(self.conn, ev.A_PID, 1)
        self.assertEqual(ev.counts(self.conn, ev.A_PID)["draft_files"], 0)
        first = tools.draft_followup(self.conn, ev.A_PID, 1)
        second = tools.draft_followup(self.conn, ev.A_PID, 1)
        self.assertTrue(first["created"])
        self.assertTrue(second["replayed"])
        self.assertEqual(self.count("drafts", ev.A_PID), 1)

    def test_duplicate_onboarding_rejected(self):
        ev.onboard_a(self.conn)
        with self.assertRaises(mm.DuplicateError):
            ev.onboard_a(self.conn)


if __name__ == "__main__":
    unittest.main()
