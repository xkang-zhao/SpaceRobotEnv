#!/usr/bin/env python3
"""Validate and plot target-versus-actual free-floating base pose.

The experiment records the complete generalized-IK configuration path, maps
that geometric path to a smooth physical-time reference, and executes only the
arm-joint part in a contact-free MuJoCo model.  The generalized-Jacobian base
pose is treated as the expected target pose and compared with the passive
MuJoCo base motion.

Outputs:

* ``reports/ik_validation/base_pose_consistency.json`` — raw trajectories and
  aggregate curves;
* ``assets/base_pose_consistency.png`` — 300 dpi publication figure;
* ``assets/base_pose_consistency.svg`` — vector publication figure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
import pinocchio as pin


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
ANALYSIS_ROOT = Path(__file__).resolve().parent
for import_root in (SRC_ROOT, ANALYSIS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from ik_validation import (  # noqa: E402
    DEFAULT_INITIAL_Q,
    Kinematics,
    make_target,
    pose_error,
    quaternion_angle_rad,
)


ACTUAL_COLOR = "#0072B2"
TARGET_COLOR = "#D55E00"
ERROR_COLOR = "#0072B2"
P95_COLOR = "#8C8C8C"
TRAJECTORY_END_COLOR = "#666666"


def scenario_specs() -> list[dict[str, Any]]:
    """Return 12 contact-free ±50 mm / ±10° Cartesian targets."""
    specs: list[dict[str, Any]] = []
    for axis_index, axis_name in enumerate("xyz"):
        for sign in (-1, 1):
            translation = np.zeros(3)
            translation[axis_index] = sign * 0.05
            specs.append(
                {
                    "id": f"DT-{axis_name}-{sign:+d}",
                    "kind": "translation",
                    "translation_m": translation.tolist(),
                    "rotation_vector_rad": [0.0, 0.0, 0.0],
                }
            )
    for axis_index, axis_name in enumerate("xyz"):
        for sign in (-1, 1):
            rotation = np.zeros(3)
            rotation[axis_index] = math.radians(sign * 10.0)
            specs.append(
                {
                    "id": f"DR-{axis_name}-{sign:+d}",
                    "kind": "rotation",
                    "translation_m": [0.0, 0.0, 0.0],
                    "rotation_vector_rad": rotation.tolist(),
                }
            )
    return specs


def quintic_progress(time_s: float, duration_s: float) -> float:
    """Fifth-order time scaling with zero endpoint velocity/acceleration."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    tau = float(np.clip(time_s / duration_s, 0.0, 1.0))
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def solve_generalized_path(
    kin: Kinematics,
    q_initial: np.ndarray,
    target: pin.SE3,
    *,
    interpolation_points: int = 20,
    iterations_per_point: int = 10,
    fine_iterations: int = 50,
    damping: float = 0.05,
    dt: float = 0.05,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Run the production generalized-IK step while retaining every state."""
    frame_id = kin.model.getFrameId(kin.frame_name)
    q = np.asarray(q_initial, dtype=float).copy()
    pin.framesForwardKinematics(kin.model, kin.data, q)
    start = pin.SE3(kin.data.oMf[frame_id])
    path = [q.copy()]

    for point_index in range(interpolation_points):
        alpha = (point_index + 1) / interpolation_points
        waypoint = pin.SE3(
            pin.Quaternion(start.rotation)
            .slerp(alpha, pin.Quaternion(target.rotation))
            .matrix(),
            start.translation + alpha * (target.translation - start.translation),
        )
        for _ in range(iterations_per_point):
            q, _ = kin.step_ik(
                q,
                waypoint,
                frame_id,
                damping=damping,
                dt=dt,
            )
            path.append(q.copy())

    for _ in range(fine_iterations):
        position, rotation = pose_error(kin, q, target)
        if position <= 0.001 and rotation <= math.radians(0.1):
            break
        q, _ = kin.step_ik(
            q,
            target,
            frame_id,
            damping=damping,
            dt=dt,
        )
        path.append(q.copy())

    position, rotation = pose_error(kin, q, target)
    return path, {
        "iterations": float(len(path) - 1),
        "final_position_error_mm": position * 1000.0,
        "final_rotation_error_deg": math.degrees(rotation),
    }


def arm_path_arclength(path: Iterable[np.ndarray]) -> np.ndarray:
    """Return normalized cumulative arm-joint arclength for a full-q path."""
    states = [np.asarray(state, dtype=float) for state in path]
    if not states:
        raise ValueError("path must contain at least one state")
    cumulative = np.zeros(len(states), dtype=float)
    for index in range(1, len(states)):
        cumulative[index] = cumulative[index - 1] + np.linalg.norm(
            states[index][7:] - states[index - 1][7:]
        )
    if cumulative[-1] <= 1e-15:
        return cumulative
    return cumulative / cumulative[-1]


def sample_configuration_path(
    model: pin.Model,
    path: list[np.ndarray],
    normalized_arclength: np.ndarray,
    progress: float,
) -> np.ndarray:
    """Interpolate a configuration path at normalized joint-path progress."""
    progress = float(np.clip(progress, 0.0, 1.0))
    if len(path) == 1 or normalized_arclength[-1] <= 1e-15:
        return np.asarray(path[-1], dtype=float).copy()
    upper = int(np.searchsorted(normalized_arclength, progress, side="right"))
    if upper <= 0:
        return np.asarray(path[0], dtype=float).copy()
    if upper >= len(path):
        return np.asarray(path[-1], dtype=float).copy()
    lower = upper - 1
    span = normalized_arclength[upper] - normalized_arclength[lower]
    alpha = 0.0 if span <= 1e-15 else (
        progress - normalized_arclength[lower]
    ) / span
    return np.asarray(
        pin.interpolate(model, path[lower], path[upper], float(alpha)),
        dtype=float,
    )


def _set_mujoco_configuration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_pin: np.ndarray,
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = q_pin[:3]
    data.qpos[3:7] = [q_pin[6], q_pin[3], q_pin[4], q_pin[5]]
    data.qpos[7:13] = q_pin[7:]
    data.ctrl[:6] = q_pin[7:]
    mujoco.mj_forward(model, data)


def _mujoco_base_quaternion_xyzw(data: mujoco.MjData) -> np.ndarray:
    return np.asarray(
        [data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]],
        dtype=float,
    )


def execute_scenario(
    kin: Kinematics,
    mujoco_model: mujoco.MjModel,
    q_initial: np.ndarray,
    spec: dict[str, Any],
    *,
    trajectory_duration_s: float,
    hold_duration_s: float,
    control_period_s: float,
) -> dict[str, Any]:
    """Compare the expected target base pose with passive MuJoCo motion."""
    frame_id = kin.model.getFrameId(kin.frame_name)
    pin.framesForwardKinematics(kin.model, kin.data, q_initial)
    start = pin.SE3(kin.data.oMf[frame_id])
    target = make_target(start, spec)
    path, solve_summary = solve_generalized_path(kin, q_initial, target)
    arclength = arm_path_arclength(path)

    data = mujoco.MjData(mujoco_model)
    _set_mujoco_configuration(mujoco_model, data, q_initial)
    actual_start_position = np.asarray(data.qpos[:3], dtype=float).copy()
    actual_start_quaternion = _mujoco_base_quaternion_xyzw(data)
    target_start_position = np.asarray(path[0][:3], dtype=float).copy()
    target_start_quaternion = np.asarray(path[0][3:7], dtype=float).copy()

    total_duration_s = trajectory_duration_s + hold_duration_s
    sample_count = int(round(total_duration_s / control_period_s)) + 1
    times = np.linspace(0.0, total_duration_s, sample_count)
    substeps_float = control_period_s / float(mujoco_model.opt.timestep)
    substeps = int(round(substeps_float))
    if not math.isclose(
        substeps * float(mujoco_model.opt.timestep),
        control_period_s,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("control_period_s must be an integer MuJoCo step multiple")

    trace: list[dict[str, float]] = []
    for sample_index, time_s in enumerate(times):
        progress = quintic_progress(time_s, trajectory_duration_s)
        target_configuration = sample_configuration_path(
            kin.model,
            path,
            arclength,
            progress,
        )
        if sample_index:
            data.ctrl[:6] = target_configuration[7:]
            for _ in range(substeps):
                mujoco.mj_step(mujoco_model, data)

        actual_position = np.asarray(data.qpos[:3], dtype=float).copy()
        actual_quaternion = _mujoco_base_quaternion_xyzw(data)
        target_position = np.asarray(target_configuration[:3], dtype=float)
        target_quaternion = np.asarray(
            target_configuration[3:7],
            dtype=float,
        )
        actual_arm = np.asarray(data.qpos[7:13], dtype=float)
        trace.append(
            {
                "time_s": float(time_s),
                "progress": progress,
                "target_base_drift_mm": float(
                    np.linalg.norm(target_position - target_start_position)
                    * 1000.0
                ),
                "actual_base_drift_mm": float(
                    np.linalg.norm(actual_position - actual_start_position)
                    * 1000.0
                ),
                "target_base_rotation_deg": math.degrees(
                    quaternion_angle_rad(
                        target_start_quaternion,
                        target_quaternion,
                    )
                ),
                "actual_base_rotation_deg": math.degrees(
                    quaternion_angle_rad(
                        actual_start_quaternion,
                        actual_quaternion,
                    )
                ),
                "base_position_error_mm": float(
                    np.linalg.norm(actual_position - target_position) * 1000.0
                ),
                "base_rotation_error_deg": math.degrees(
                    quaternion_angle_rad(
                        actual_quaternion,
                        target_quaternion,
                    )
                ),
                "arm_tracking_error_l2_rad": float(
                    np.linalg.norm(actual_arm - target_configuration[7:])
                ),
            }
        )

    return {
        "scenario": spec["id"],
        "kind": spec["kind"],
        "target": spec,
        "ik": solve_summary,
        "trace": trace,
    }


def _curve_statistics(
    scenarios: list[dict[str, Any]],
    metric: str,
) -> list[dict[str, float]]:
    values = np.asarray(
        [
            [point[metric] for point in scenario["trace"]]
            for scenario in scenarios
        ],
        dtype=float,
    )
    times = [point["time_s"] for point in scenarios[0]["trace"]]
    return [
        {
            "time_s": float(time_s),
            "median": float(np.median(values[:, index])),
            "p95": float(np.quantile(values[:, index], 0.95)),
            "max": float(np.max(values[:, index])),
        }
        for index, time_s in enumerate(times)
    ]


def summarize_scenarios(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    if not scenarios:
        raise ValueError("at least one scenario is required")
    metrics = (
        "target_base_drift_mm",
        "actual_base_drift_mm",
        "target_base_rotation_deg",
        "actual_base_rotation_deg",
        "base_position_error_mm",
        "base_rotation_error_deg",
        "arm_tracking_error_l2_rad",
    )
    return {
        "scenario_count": len(scenarios),
        "curves": {
            metric: _curve_statistics(scenarios, metric)
            for metric in metrics
        },
    }


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    model_path = (REPO_ROOT / args.model).resolve()
    scene_path = (REPO_ROOT / args.scene).resolve()
    kin = Kinematics(str(model_path), args.frame)
    mujoco_model = mujoco.MjModel.from_xml_path(str(scene_path))
    mujoco_model.opt.timestep = args.physics_timestep
    mujoco_model.geom_contype[:] = 0
    mujoco_model.geom_conaffinity[:] = 0
    q_initial = DEFAULT_INITIAL_Q.copy()
    scenarios = []
    started = time.perf_counter()
    for index, spec in enumerate(scenario_specs(), start=1):
        print(f"[{index:02d}/12] {spec['id']}", flush=True)
        scenarios.append(
            execute_scenario(
                kin,
                mujoco_model,
                q_initial,
                spec,
                trajectory_duration_s=args.trajectory_duration,
                hold_duration_s=args.hold_duration,
                control_period_s=args.control_period,
            )
        )
    elapsed_s = time.perf_counter() - started
    return {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pinocchio": pin.__version__,
            "mujoco": mujoco.__version__,
            "git": _git_metadata(),
        },
        "experiment_config": {
            "model": str(model_path.relative_to(REPO_ROOT)),
            "scene": str(scene_path.relative_to(REPO_ROOT)),
            "frame": args.frame,
            "gravity": mujoco_model.opt.gravity.tolist(),
            "physics_timestep_s": float(mujoco_model.opt.timestep),
            "control_period_s": args.control_period,
            "trajectory_duration_s": args.trajectory_duration,
            "hold_duration_s": args.hold_duration,
            "ik": {
                "interpolation_points": 20,
                "iterations_per_point": 10,
                "fine_iterations": 50,
                "damping": 0.05,
                "dt": 0.05,
            },
            "initial_configuration": q_initial.tolist(),
            "contact_free": True,
        },
        "runtime_s": elapsed_s,
        "scenarios": scenarios,
        "summary": summarize_scenarios(scenarios),
    }


def _configure_matplotlib() -> Any:
    cache_dir = Path(tempfile.gettempdir()) / "my_simulation_matplotlib"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt

    chinese_path = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
    chinese_name = "Songti SC"
    if chinese_path.exists():
        font_manager.fontManager.addfont(str(chinese_path))
        chinese_name = font_manager.FontProperties(
            fname=str(chinese_path)
        ).get_name()
    matplotlib.rcParams.update(
        {
            "font.family": [
                "Times New Roman",
                chinese_name,
                "serif",
            ],
            "font.size": 10.5,
            "axes.unicode_minus": False,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": False,
            "ytick.right": False,
            "svg.fonttype": "none",
        }
    )
    return plt


def _curve_xy(summary: dict[str, Any], metric: str, field: str) -> tuple[list[float], list[float]]:
    rows = summary["curves"][metric]
    return (
        [float(row["time_s"]) for row in rows],
        [float(row[field]) for row in rows],
    )


def plot_results(
    results: dict[str, Any],
    png_output: Path,
    svg_output: Path,
) -> None:
    plt = _configure_matplotlib()
    summary = results["summary"]
    trajectory_end = float(
        results["experiment_config"]["trajectory_duration_s"]
    )
    panels = [
        (
            "基座位置漂移（目标与实际）",
            "基座位置漂移 / mm",
            "target_base_drift_mm",
            "actual_base_drift_mm",
            "（a）基座位置响应",
            "目标基座位置漂移",
            "实际基座位置漂移",
        ),
        (
            "基座姿态变化（目标与实际）",
            "基座姿态变化 / (°)",
            "target_base_rotation_deg",
            "actual_base_rotation_deg",
            "（b）基座姿态响应",
            "目标基座姿态变化",
            "实际基座姿态变化",
        ),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))
    figure.patch.set_facecolor("white")

    for axis, (
        title,
        ylabel,
        target_metric,
        actual_metric,
        label,
        target_legend,
        actual_legend,
    ) in zip(
        axes[0],
        panels,
    ):
        x, target_values = _curve_xy(summary, target_metric, "median")
        _, actual = _curve_xy(summary, actual_metric, "median")
        axis.plot(
            x,
            target_values,
            color=TARGET_COLOR,
            linewidth=2.0,
            linestyle=(0, (5, 2)),
            label=f"{target_legend}（中位数）",
        )
        axis.plot(
            x,
            actual,
            color=ACTUAL_COLOR,
            linewidth=2.1,
            label=f"{actual_legend}（中位数）",
        )
        axis.set_title(title, pad=8)
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False, loc="best")
        axis.text(
            0.5,
            -0.25,
            label,
            transform=axis.transAxes,
            ha="center",
            va="top",
        )

    error_panels = [
        (
            axes[1, 0],
            "基座位置误差（目标–实际）",
            "基座位置误差 / mm",
            "base_position_error_mm",
            "（c）基座位置误差",
        ),
        (
            axes[1, 1],
            "基座姿态误差（目标–实际）",
            "基座姿态误差 / (°)",
            "base_rotation_error_deg",
            "（d）基座姿态误差",
        ),
    ]
    for axis, title, ylabel, metric, label in error_panels:
        x, median = _curve_xy(summary, metric, "median")
        _, p95 = _curve_xy(summary, metric, "p95")
        axis.plot(
            x,
            median,
            color=ERROR_COLOR,
            linewidth=2.1,
            label="中位数",
        )
        axis.plot(
            x,
            p95,
            color=P95_COLOR,
            linewidth=1.5,
            linestyle=(0, (4, 2)),
            label="P95",
        )
        axis.fill_between(
            x,
            median,
            p95,
            color=ERROR_COLOR,
            alpha=0.10,
            linewidth=0,
        )
        axis.set_title(title, pad=8)
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False, loc="best")
        axis.text(
            0.5,
            -0.25,
            label,
            transform=axis.transAxes,
            ha="center",
            va="top",
        )

    for axis in axes.flat:
        axis.axvline(
            trajectory_end,
            color=TRAJECTORY_END_COLOR,
            linewidth=1.0,
            linestyle=(0, (3, 3)),
        )
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.65)
        axis.set_xlabel("仿真时间 / s")
        axis.set_xlim(left=0.0)
        axis.set_ylim(bottom=0.0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0, 0].annotate(
        "轨迹结束",
        xy=(trajectory_end, axes[0, 0].get_ylim()[1]),
        xytext=(4, -5),
        textcoords="offset points",
        ha="left",
        va="top",
        color=TRAJECTORY_END_COLOR,
        fontsize=9,
    )
    figure.suptitle(
        "自由漂浮基座目标位姿与实际位姿对比",
        fontsize=15,
        y=0.995,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.92,
        bottom=0.10,
        wspace=0.27,
        hspace=0.48,
    )
    png_output.parent.mkdir(parents=True, exist_ok=True)
    svg_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        png_output,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        svg_output,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def validate_results(results: dict[str, Any]) -> None:
    if results["summary"]["scenario_count"] != 12:
        raise ValueError("Expected exactly 12 dynamic scenarios")
    sample_counts = {
        len(scenario["trace"]) for scenario in results["scenarios"]
    }
    if len(sample_counts) != 1:
        raise ValueError("All scenarios must share the same sample count")
    for scenario in results["scenarios"]:
        first = scenario["trace"][0]
        for key in (
            "target_base_drift_mm",
            "actual_base_drift_mm",
            "target_base_rotation_deg",
            "actual_base_rotation_deg",
            "base_position_error_mm",
            "base_rotation_error_deg",
        ):
            if abs(float(first[key])) > 1e-9:
                raise ValueError(
                    f"{scenario['scenario']} does not start at zero for {key}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mjcf/arm.xml")
    parser.add_argument("--scene", default="mjcf/1_cube_scene.xml")
    parser.add_argument("--frame", default="left_attachment")
    parser.add_argument("--physics-timestep", type=float, default=0.0005)
    parser.add_argument("--control-period", type=float, default=0.05)
    parser.add_argument("--trajectory-duration", type=float, default=2.0)
    parser.add_argument("--hold-duration", type=float, default=1.0)
    parser.add_argument(
        "--json-output",
        default="reports/ik_validation/base_pose_consistency.json",
    )
    parser.add_argument(
        "--png-output",
        default="assets/base_pose_consistency.png",
    )
    parser.add_argument(
        "--svg-output",
        default="assets/base_pose_consistency.svg",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiment(args)
    validate_results(results)
    json_output = (REPO_ROOT / args.json_output).resolve()
    png_output = (REPO_ROOT / args.png_output).resolve()
    svg_output = (REPO_ROOT / args.svg_output).resolve()
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_results(results, png_output, svg_output)
    print(f"results: {json_output}")
    print(f"figure:  {png_output}")
    print(f"vector:  {svg_output}")


if __name__ == "__main__":
    main()
