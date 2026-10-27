import tempfile
import unittest
from pathlib import Path
from tests.helpers import IsolatedHome
import evaluation as ev


class TestEvaluationRunner(IsolatedHome):
    def _removed(self):
        rep = ev.not_run_report()
        self.assertEqual({s["result"] for s in rep["scenarios"]}, {"NOT RUN"})

    def test_all_scenarios_offline(self):
        root = Path(tempfile.mkdtemp(prefix="rrt_eval_"))
        rep = ev.run_all("offline", save=False, root=root, progress=None)
        failed = [(s["id"], s["observed"]) for s in rep["scenarios"] if s["result"] != "PASS"]
        self.assertEqual(failed, [])
        self.assertEqual(rep["metrics"]["duplicate_side_effects_in_retry_tests"], 0)

    def test_memory_comparison_offline(self):
        root = Path(tempfile.mkdtemp(prefix="rrt_cmp_"))
        out = ev.run_compare("offline", save=False, root=root, progress=None)
        self.assertEqual(out["memory_on_as_expected"], "3/3")
        self.assertEqual(out["memory_off_as_expected"], "3/3")


if __name__ == "__main__":
    unittest.main()
