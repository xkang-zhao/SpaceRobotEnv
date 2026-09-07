import importlib.util
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "auto_grasp.py"


def load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "auto_grasp_cli",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AutoGraspCliTests(unittest.TestCase):
    def test_unified_cli_exposes_every_supported_task(self):
        module = load_cli_module()

        self.assertEqual(
            set(module.TASK_NAMES),
            {
                "cube",
                "satellite_handle",
                "satellite_left_antenna_panel",
                "satellite2_left_truss_connection",
                "satellite3_left_antenna_panel",
                "satellite3_upper_rod",
                "debris_antenna_panel",
                "debris_truss",
            },
        )

    def test_top_level_help_lists_tasks(self):
        module = load_cli_module()
        output = StringIO()

        with redirect_stdout(output):
            return_code = module.main(["--help"])

        self.assertEqual(return_code, 0)
        self.assertIn("satellite_handle", output.getvalue())
        self.assertIn("debris_truss", output.getvalue())

    def test_unknown_task_is_rejected(self):
        module = load_cli_module()

        with self.assertRaisesRegex(SystemExit, "Unknown task"):
            module.main(["unknown_task"])


if __name__ == "__main__":
    unittest.main()
