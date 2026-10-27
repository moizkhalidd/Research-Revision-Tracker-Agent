import unittest
from tests.helpers import IsolatedHome
import agent
import evaluation as ev
import memory_manager as mm
from validators import detect_injection


class HostileEngine:
    """Test double that behaves as if the model had obeyed the injected document (NOT a real LLM)."""
    name = "test-double-hostile"
    is_llm = False

    def plan(self, items, mode):
        return {i["comment"]["comment_id"]: {"followup_tools": ["delete_everything", "search_full_document"], "reason": "x"} for i in items}

    def evaluate(self, item, mode):
        return {"verdict": "Addressed", "explanation": "Marked everything addressed. paper_id=PAPER-HACKED",
                "evidence": [{"passage": "E1", "quote": "this quote was invented by the model", "supports": "x"}],
                "action_code": "PRESENT_FOR_APPROVAL", "paper_id": "PAPER-HACKED", "approve": True}


class TestInjection(IsolatedHome):
    def setUp(self):
        super().setUp()
        ev.onboard_a(self.conn)
        ev.ingest(self.conn, ev.A_PID, "injected_document.tex", "claims_injected.json")

    def test_detector_and_flags(self):
        self.assertGreaterEqual(len(detect_injection(open(ev.S("injected_document.tex"), encoding="utf-8").read())), 3)
        n = self.conn.execute("SELECT COUNT(*) FROM passages WHERE paper_id=? AND injection_flag=1", (ev.A_PID,)).fetchone()[0]
        self.assertGreaterEqual(n, 2)

    def test_offline_engine_resists(self):
        ev.evaluate(self.conn, ev.A_PID, "offline", 1)
        v = ev.cycle_verdicts(self.conn, ev.A_PID, 1)
        self.assertTrue(all(x["verdict"] != "Addressed" for x in v.values()))
        self.assertEqual(self.count("author_decisions", ev.A_PID), 0)

    def test_guards_hold_even_if_the_model_obeys_the_document(self):
        original = agent.make_engine
        agent.make_engine = lambda conn, kind, paper_id=None, cycle_id=None: HostileEngine()
        try:
            agent.evaluate_cycle(self.conn, ev.A_PID, 1, "groq")
        finally:
            agent.make_engine = original
        v = ev.cycle_verdicts(self.conn, ev.A_PID, 1)
        self.assertEqual(len(v), 6)
        self.assertTrue(all(x["verdict"] == "Unverified" for x in v.values()), "fabricated quotes must downgrade Addressed")
        self.assertEqual(self.count("verdict_evidence", ev.A_PID), 0)
        self.assertEqual([p["paper_id"] for p in mm.list_papers(self.conn)], [ev.A_PID])
        self.assertEqual(self.count("author_decisions", ev.A_PID), 0)
        self.assertTrue(all(r["status"] == "proposed" for r in self.conn.execute("SELECT status FROM next_actions WHERE paper_id=?", (ev.A_PID,))))


if __name__ == "__main__":
    unittest.main()
