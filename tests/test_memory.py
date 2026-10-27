import json
import subprocess
import sys
import os
import unittest
from tests.helpers import IsolatedHome, PROJECT
import evaluation as ev
import memory_manager as mm


class TestMemory(IsolatedHome):
    def test_restart_restores_state_without_reupload(self):
        ev.onboard_a(self.conn)
        ev.ingest(self.conn, ev.A_PID, "paper_v2.tex", "claims_v2.json")
        ev.evaluate(self.conn, ev.A_PID, "offline", 1)
        before = {s["comment_id"]: s["effective_verdict"] for s in mm.current_status(self.conn, ev.A_PID)}
        self.conn.close()
        env = dict(os.environ, RRT_HOME=str(self.home), PYTHONIOENCODING="utf-8")
        p = subprocess.run([sys.executable, str(PROJECT / "main.py"), "status", "--paper", ev.A_PID, "--json"],
                           capture_output=True, text=True, env=env, encoding="utf-8")
        self.assertEqual(p.returncode, 0, p.stderr)
        after = {s["comment_id"]: s["effective_verdict"] for s in json.loads(p.stdout)["comments"]}
        self.assertEqual(before, after)
        from database import connect
        self.conn = connect()
        self.assertEqual(self.count("documents", ev.A_PID), 2)

    def test_baseline_text_and_source_copy_are_recoverable(self):
        info = ev.onboard_a(self.conn)
        self.assertTrue((self.home / info["stored_path"]).exists())
        text = self.conn.execute("SELECT full_text FROM documents WHERE paper_id=? AND version_no=1", (ev.A_PID,)).fetchone()[0]
        self.assertIn("isolation forest", text.lower())

    def test_suggestions_never_create_decisions(self):
        ev.onboard_a(self.conn)
        ev.ingest(self.conn, ev.A_PID, "paper_v2.tex", "claims_v2.json")
        ev.evaluate(self.conn, ev.A_PID, "offline", 1)
        self.assertEqual(self.count("author_decisions", ev.A_PID), 0)
        self.assertTrue(all(s["approval_state"] == "pending" for s in mm.current_status(self.conn, ev.A_PID)))

    def test_regression_needs_memory(self):
        ev.onboard_a(self.conn)
        ev.ingest(self.conn, ev.A_PID, "paper_v2.tex", "claims_v2.json")
        ev.evaluate(self.conn, ev.A_PID, "offline", 1)
        ev.setup_decision(self.conn, ev.A_PID, "R2.2", "Addressed")
        ev.ingest(self.conn, ev.A_PID, "paper_v3.tex", "claims_v3.json")
        on = ev.evaluate(self.conn, ev.A_PID, "offline", 2, only=["R2.2"], persist=False)["results"][0]
        off = ev.evaluate(self.conn, ev.A_PID, "offline", 2, only=["R2.2"], memory=False, persist=False)["results"][0]
        self.assertTrue(on.regression)
        self.assertEqual(on.action_code, "FLAG_REGRESSION")
        self.assertFalse(off.regression)
        self.assertEqual(self.count("verdicts", ev.A_PID), 6)  # dry runs saved nothing


if __name__ == "__main__":
    unittest.main()
