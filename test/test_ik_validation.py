"""Focused tests for the reproducible IK validation suite."""

from __future__ import annotations

import importlib.util
import json
import math
import re
import unittest
from pathlib import Path

import numpy as np
import pinocchio as pin


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "reports" / "analysis" / "ik_validation.py"
SPEC = importlib.util.spec_from_file_location("ik_validation", MODULE_PATH)
assert SPEC and SPEC.loader
ikv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ikv)

PLOT_MODULE_PATH = (
    REPO_ROOT / "reports" / "analysis" / "plot_ik_convergence.py"
)
PLOT_SPEC = importlib.util.spec_from_file_location(
    "plot_ik_convergence", PLOT_MODULE_PATH
)
assert PLOT_SPEC and PLOT_SPEC.loader
ik_plot = importlib.util.module_from_spec(PLOT_SPEC)
PLOT_SPEC.loader.exec_module(ik_plot)


class TestIKValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kin = ikv.Kinematics(
            str(REPO_ROOT / "mjcf" / "arm.xml"), "left_attachment"
        )

    def test_target_set_is_complete_and_reproducible(self) -> None:
        first = ikv.generate_target_specs(42)
        second = ikv.generate_target_specs(42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 48)
        self.assertEqual(
            {"translation": 18, "rotation": 18, "combined": 12},
            {
                kind: sum(item["kind"] == kind for item in first)
                for kind in ("translation", "rotation", "combined")
            },
        )
        self.assertNotEqual(first, ikv.generate_target_specs(43))

    def test_se3_error_is_separated_by_unit(self) -> None:
        q = ikv.initial_configurations()[0]
        pin.framesForwardKinematics(self.kin.model, self.kin.data, q)
        frame = self.kin.model.getFrameId("left_attachment")
        current = pin.SE3(self.kin.data.oMf[frame])
        target = pin.SE3(
            current.rotation @ pin.exp3(np.array([0.0, 0.0, math.radians(2.0)])),
            current.translation + np.array([0.003, 0.004, 0.0]),
        )
        position, rotation = ikv.pose_error(self.kin, q, target)
        self.assertAlmostEqual(position, 0.005, places=10)
        self.assertAlmostEqual(math.degrees(rotation), 2.0, places=10)

    def test_quaternion_shortest_angle_handles_sign(self) -> None:
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        self.assertAlmostEqual(
            ikv.quaternion_angle_rad(identity, -identity), 0.0, places=12
        )
        quarter_turn = np.array(
            [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
        )
        self.assertAlmostEqual(
            ikv.quaternion_angle_rad(identity, quarter_turn),
            math.pi / 2,
            places=12,
        )

    def test_methods_share_iteration_budget(self) -> None:
        q = ikv.initial_configurations()[0]
        pin.framesForwardKinematics(self.kin.model, self.kin.data, q)
        start = pin.SE3(
            self.kin.data.oMf[self.kin.model.getFrameId("left_attachment")]
        )
        target = pin.SE3(start.rotation, start.translation + [0.02, 0.0, 0.0])
        generalized = ikv.solve_equal_budget(
            self.kin, q, target, "generalized", trace=False
        )
        fixed = ikv.solve_equal_budget(
            self.kin, q, target, "fixed", trace=False
        )
        for result in (generalized, fixed):
            self.assertGreaterEqual(result["iterations"], 200)
            self.assertLessEqual(result["iterations"], 250)

    def test_generalized_velocity_satisfies_momentum_constraint(self) -> None:
        q = ikv.initial_configurations()[0]
        pin.framesForwardKinematics(self.kin.model, self.kin.data, q)
        start = pin.SE3(
            self.kin.data.oMf[self.kin.model.getFrameId("left_attachment")]
        )
        target = pin.SE3(start.rotation, start.translation + [0.03, 0.0, 0.0])
        residual = ikv.normalized_momentum_residual(
            self.kin, q, target, damping=0.05
        )
        self.assertLess(residual, 1e-10)

    def test_curated_dataset_statistics(self) -> None:
        summary = ikv.summarize_datasets(REPO_ROOT / "dataset")
        self.assertEqual(summary["task_count"], 8)
        self.assertEqual(summary["episode_count"], 16)
        self.assertEqual(summary["frame_count"], 1751)
        self.assertTrue(
            all(
                row["terminal_success"]
                for row in summary["episode_rows"]
            )
        )
        self.assertIn("不能据此计算", summary["interpretation"])

    def test_generated_results_have_complete_experiment_counts(self) -> None:
        results = json.loads(
            (REPO_ROOT / "reports" / "ik_validation" / "results.json").read_text(
                encoding="utf-8"
            )
        )
        ikv.validate_results(results)
        self.assertEqual(len(results["numerical_ik"]["rows"]), 480)
        self.assertEqual(len(results["dynamic_validation"]["rows"]), 24)
        self.assertEqual(len(results["damping_ablation"]["rows"]), 300)

    def test_report_is_self_contained_and_matches_results(self) -> None:
        report = (
            REPO_ROOT / "reports" / "ik_validation" / "index.html"
        ).read_text(encoding="utf-8")
        results = json.loads(
            (REPO_ROOT / "reports" / "ik_validation" / "results.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report.count("<figure>"), 8)
        self.assertEqual(report.count("<figcaption>"), 8)
        self.assertEqual(report.count("<table>"), 6)
        self.assertEqual(report.count("data:image/jpeg;base64,"), 8)
        self.assertNotRegex(report, r'<(?:script|img)[^>]+src=["\']https?://')
        self.assertNotRegex(report, r'<link[^>]+href=["\']https?://')
        chinese_characters = re.findall(r"[\u4e00-\u9fff]", report)
        self.assertGreaterEqual(len(chinese_characters), 7000)
        self.assertLessEqual(len(chinese_characters), 10000)
        for row in results["numerical_ik"]["summary_by_method"]:
            expected = f"{row['success_rate'] * 100:.2f}%"
            self.assertIn(expected, report)

    def test_standalone_convergence_assets_and_timings(self) -> None:
        results = json.loads(
            (REPO_ROOT / "reports" / "ik_validation" / "results.json").read_text(
                encoding="utf-8"
            )
        )
        rows = ik_plot.select_generalized_combined_rows(results)
        timing = ik_plot.solve_time_statistics(rows)
        self.assertEqual(timing["n"], 60)
        self.assertGreater(timing["median_ms"], 0.0)
        curve = ik_plot.median_trace(rows, "position_error_mm")
        self.assertEqual(len(curve), len(ik_plot.CHECKPOINTS))
        self.assertAlmostEqual(curve[-1][1], curve[-2][1], places=12)

        png_path = REPO_ROOT / "assets" / "ik_convergence.png"
        svg_path = REPO_ROOT / "assets" / "ik_convergence.svg"
        timing_path = REPO_ROOT / "assets" / "ik_solve_times.csv"
        self.assertTrue(png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        svg = svg_path.read_text(encoding="utf-8")
        self.assertIn(f"{timing['median_ms']:.2f} ms", svg)
        self.assertIn(f"P95 = {timing['p95_ms']:.2f} ms", svg)
        self.assertIn("位置误差 / mm", svg)
        self.assertIn("姿态误差 / (°)", svg)
        self.assertNotIn("n = 60", svg)
        self.assertNotIn("插值阶段", svg)
        self.assertNotIn("精调阶段", svg)
        self.assertEqual(
            len(timing_path.read_text(encoding="utf-8-sig").splitlines()), 61
        )


if __name__ == "__main__":
    unittest.main()
