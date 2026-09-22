"""CPU-only checks for the grasp diagnostic's success reporting."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


class GraspDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/validate_warp_tasks.py"
        spec = importlib.util.spec_from_file_location("warp_task_validation", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_planner_done_does_not_imply_confirmed_success(self):
        session = SimpleNamespace(is_done=True, failure_reason=None)
        row = self.module.outcome("cube", 7, 10, False, False,
                                  {"is_success": True}, session)
        self.assertFalse(row["success"])

    def test_confirmed_success_preserves_contact_diagnostics(self):
        session = SimpleNamespace(is_done=True, failure_reason=None)
        info = {"is_success": True, "success_counter": 10,
                "left_contact": True, "right_contact": True}
        row = self.module.outcome("cube", 7, 96, True, False, info, session)
        self.assertTrue(row["success"])
        self.assertEqual(row["success_counter"], 10)
        self.assertTrue(row["left_contact"] and row["right_contact"])

    def test_step_limit_is_not_reported_as_success(self):
        session = SimpleNamespace(is_done=False, failure_reason=None)
        row = self.module.outcome("cube", 7, 420, False, False, {}, session)
        self.assertFalse(row["success"])
        self.assertEqual(row["failure_reason"], "step limit")


if __name__ == "__main__":
    unittest.main()
