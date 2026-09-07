"""Tests for the trajectory-level base-pose consistency figure."""

from __future__ import annotations

import importlib.util
import json
import math
import unittest
from pathlib import Path

import numpy as np
import pinocchio as pin


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "reports"
    / "analysis"
    / "plot_base_pose_consistency.py"
)
SPEC = importlib.util.spec_from_file_location(
    "plot_base_pose_consistency",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
base_plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base_plot)


class TestBasePoseConsistency(unittest.TestCase):
    def test_scenario_set_has_twelve_symmetric_targets(self) -> None:
        specs = base_plot.scenario_specs()
        self.assertEqual(len(specs), 12)
        self.assertEqual(
            sum(spec["kind"] == "translation" for spec in specs),
            6,
        )
        self.assertEqual(
            sum(spec["kind"] == "rotation" for spec in specs),
            6,
        )

    def test_quintic_progress_has_correct_endpoints(self) -> None:
        self.assertEqual(base_plot.quintic_progress(0.0, 2.0), 0.0)
        self.assertEqual(base_plot.quintic_progress(2.0, 2.0), 1.0)
        self.assertEqual(base_plot.quintic_progress(3.0, 2.0), 1.0)
        values = [
            base_plot.quintic_progress(time_s, 2.0)
            for time_s in np.linspace(0.0, 2.0, 41)
        ]
        self.assertTrue(
            all(right >= left for left, right in zip(values, values[1:]))
        )

    def test_path_sampling_preserves_endpoints(self) -> None:
        kin = base_plot.Kinematics(
            str(REPO_ROOT / "mjcf" / "arm.xml"),
            "left_attachment",
        )
        first = base_plot.DEFAULT_INITIAL_Q.copy()
        velocity = np.zeros(kin.model.nv)
        velocity[6:] = [0.1, -0.05, 0.04, 0.03, -0.02, 0.01]
        second = pin.integrate(kin.model, first, velocity)
        path = [first, second]
        arclength = base_plot.arm_path_arclength(path)
        np.testing.assert_allclose(
            base_plot.sample_configuration_path(
                kin.model,
                path,
                arclength,
                0.0,
            ),
            first,
        )
        np.testing.assert_allclose(
            base_plot.sample_configuration_path(
                kin.model,
                path,
                arclength,
                1.0,
            ),
            second,
        )

    def test_summary_uses_median_and_p95_per_time(self) -> None:
        scenarios = [
            {
                "trace": [
                    {"time_s": 0.0, "metric": 0.0},
                    {"time_s": 0.05, "metric": value},
                ]
            }
            for value in (1.0, 2.0, 3.0)
        ]
        curve = base_plot._curve_statistics(scenarios, "metric")
        self.assertEqual(curve[0]["median"], 0.0)
        self.assertEqual(curve[1]["median"], 2.0)
        self.assertTrue(math.isclose(curve[1]["p95"], 2.9))

    def test_generated_assets_and_results_are_complete(self) -> None:
        results_path = (
            REPO_ROOT
            / "reports"
            / "ik_validation"
            / "base_pose_consistency.json"
        )
        png_path = REPO_ROOT / "assets" / "base_pose_consistency.png"
        svg_path = REPO_ROOT / "assets" / "base_pose_consistency.svg"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        base_plot.validate_results(results)
        self.assertEqual(len(results["scenarios"]), 12)
        self.assertEqual(len(results["scenarios"][0]["trace"]), 61)
        self.assertTrue(png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        svg = svg_path.read_text(encoding="utf-8")
        self.assertIn("基座位置漂移（目标与实际）", svg)
        self.assertIn("基座姿态误差（目标–实际）", svg)
        self.assertIn("目标基座姿态变化（中位数）", svg)
        self.assertNotIn("预测误差", svg)


if __name__ == "__main__":
    unittest.main()
