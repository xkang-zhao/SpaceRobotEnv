#!/usr/bin/env python3
"""Reproducible validation suite for the free-floating space-robot IK.

This module intentionally does not change the production solver.  It exercises
``src/mujoco_robot/robot_ik.py`` through its public single-step methods, applies
an identical iteration budget to both compared methods, recomputes final SE(3)
errors, performs non-rendering MuJoCo checks, and summarizes existing curated
LeRobot datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
import pinocchio as pin
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mujoco_robot.robot_ik import Kinematics  # noqa: E402


DEFAULT_INITIAL_Q = np.array(
    [
        0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0,
        -1.029673126491, -0.347832443690, 1.723581883836,
        1.754792332365, -0.535514027765, -1.566808473225,
    ],
    dtype=float,
)
ARM_OFFSETS = np.array(
    [
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.15, -0.20, 0.25, -0.15, 0.10, -0.10],
        [-0.18, 0.16, -0.22, 0.18, -0.12, 0.14],
        [0.10, -0.30, 0.18, 0.22, -0.20, 0.12],
        [-0.12, 0.24, -0.16, -0.25, 0.18, -0.15],
    ],
    dtype=float,
)
CHECKPOINTS = tuple(range(10, 251, 10))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status and status != "unavailable"),
        "status_porcelain": status.splitlines(),
    }


def quaternion_angle_rad(q1: Iterable[float], q2: Iterable[float]) -> float:
    """Shortest angular distance between xyzw quaternions."""
    a = np.asarray(q1, dtype=float)
    b = np.asarray(q2, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return 2.0 * math.acos(dot)


def pose_error(kin: Kinematics, q: np.ndarray, target: pin.SE3) -> tuple[float, float]:
    """Return final FK position error [m] and rotation error [rad]."""
    frame_id = kin.model.getFrameId(kin.frame_name)
    pin.framesForwardKinematics(kin.model, kin.data, q)
    current = kin.data.oMf[frame_id]
    position = float(np.linalg.norm(target.translation - current.translation))
    rotation = float(np.linalg.norm(pin.log3(current.rotation.T @ target.rotation)))
    return position, rotation


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    return vector / np.linalg.norm(vector)


def generate_target_specs(seed: int = 42) -> list[dict[str, Any]]:
    """Generate the fixed 48-target set independent of the initial pose."""
    specs: list[dict[str, Any]] = []
    axes = np.eye(3)
    axis_names = "xyz"
    for axis_index, axis in enumerate(axes):
        for magnitude_mm in (20, 50, 100):
            for sign in (-1, 1):
                specs.append(
                    {
                        "id": f"T-{axis_names[axis_index]}-{sign:+d}-{magnitude_mm:03d}",
                        "kind": "translation",
                        "translation_m": (axis * sign * magnitude_mm / 1000.0).tolist(),
                        "rotation_vector_rad": [0.0, 0.0, 0.0],
                    }
                )
    for axis_index, axis in enumerate(axes):
        for magnitude_deg in (5, 10, 15):
            for sign in (-1, 1):
                specs.append(
                    {
                        "id": f"R-{axis_names[axis_index]}-{sign:+d}-{magnitude_deg:02d}",
                        "kind": "rotation",
                        "translation_m": [0.0, 0.0, 0.0],
                        "rotation_vector_rad": (
                            axis * math.radians(sign * magnitude_deg)
                        ).tolist(),
                    }
                )
    rng = np.random.default_rng(seed)
    for index in range(12):
        distance = rng.uniform(0.03, 0.10)
        angle = math.radians(rng.uniform(4.0, 15.0))
        specs.append(
            {
                "id": f"C-{index + 1:02d}",
                "kind": "combined",
                "translation_m": (_unit_vector(rng) * distance).tolist(),
                "rotation_vector_rad": (_unit_vector(rng) * angle).tolist(),
            }
        )
    assert len(specs) == 48
    return specs


def make_target(start: pin.SE3, spec: dict[str, Any]) -> pin.SE3:
    translation = start.translation + np.asarray(spec["translation_m"], dtype=float)
    rotation = start.rotation @ pin.exp3(np.asarray(spec["rotation_vector_rad"], dtype=float))
    return pin.SE3(rotation, translation)


def initial_configurations() -> list[np.ndarray]:
    configs = []
    for arm_offset in ARM_OFFSETS:
        q = DEFAULT_INITIAL_Q.copy()
        q[7:] += arm_offset
        configs.append(q)
    return configs


def normalized_momentum_residual(
    kin: Kinematics, q: np.ndarray, target: pin.SE3, damping: float
) -> float:
    """Residual of M_bb v_b + M_bm v_m normalized by coupling momentum."""
    frame_id = kin.model.getFrameId(kin.frame_name)
    pin.framesForwardKinematics(kin.model, kin.data, q)
    current = kin.data.oMf[frame_id]
    error = np.r_[
        target.translation - current.translation,
        current.rotation @ pin.log3(current.rotation.T @ target.rotation),
    ]
    jacobian, m_bb_inverse, m_bm = kin.compute_matrices(q, frame_id)
    velocity_arm = np.linalg.solve(
        jacobian.T @ jacobian + damping**2 * np.eye(6),
        jacobian.T @ error,
    )
    velocity_base = -m_bb_inverse @ m_bm @ velocity_arm
    pin.crba(kin.model, kin.data, q)
    m_bb = np.asarray(kin.data.M[:6, :6])
    residual = m_bb @ velocity_base + m_bm @ velocity_arm
    denominator = np.linalg.norm(m_bb @ velocity_base) + np.linalg.norm(
        m_bm @ velocity_arm
    )
    return float(np.linalg.norm(residual) / max(denominator, 1e-15))


def solve_equal_budget(
    kin: Kinematics,
    q_initial: np.ndarray,
    target: pin.SE3,
    method: str,
    *,
    interpolation_points: int = 20,
    iterations_per_point: int = 10,
    fine_iterations: int = 50,
    damping: float = 0.05,
    dt: float = 0.05,
    trace: bool = True,
) -> dict[str, Any]:
    """Run either production single-step method under one identical budget."""
    frame_id = kin.model.getFrameId(kin.frame_name)
    step = kin.step_ik if method == "generalized" else kin.step_ik_fixed_base
    q = q_initial.copy()
    pin.framesForwardKinematics(kin.model, kin.data, q)
    start = pin.SE3(kin.data.oMf[frame_id])
    trace_rows: list[dict[str, float]] = []
    iteration = 0

    def record() -> None:
        if trace and (iteration in CHECKPOINTS or iteration == 1):
            position, rotation = pose_error(kin, q, target)
            trace_rows.append(
                {
                    "iteration": iteration,
                    "position_error_mm": position * 1000.0,
                    "rotation_error_deg": math.degrees(rotation),
                }
            )

    started = time.perf_counter_ns()
    for point_index in range(interpolation_points):
        alpha = (point_index + 1) / interpolation_points
        waypoint = pin.SE3(
            pin.Quaternion(start.rotation)
            .slerp(alpha, pin.Quaternion(target.rotation))
            .matrix(),
            start.translation + alpha * (target.translation - start.translation),
        )
        for _ in range(iterations_per_point):
            q, _ = step(q, waypoint, frame_id, damping=damping, dt=dt)
            iteration += 1
            record()
    for _ in range(fine_iterations):
        position, rotation = pose_error(kin, q, target)
        if position <= 0.001 and rotation <= math.radians(0.1):
            break
        q, _ = step(q, target, frame_id, damping=damping, dt=dt)
        iteration += 1
        record()
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    position, rotation = pose_error(kin, q, target)
    if trace and (not trace_rows or trace_rows[-1]["iteration"] != iteration):
        trace_rows.append(
            {
                "iteration": iteration,
                "position_error_mm": position * 1000.0,
                "rotation_error_deg": math.degrees(rotation),
            }
        )
    base_translation = float(np.linalg.norm(q[:3] - q_initial[:3]))
    base_rotation = quaternion_angle_rad(q[3:7], q_initial[3:7])
    return {
        "q_final": q,
        "position_error_mm": position * 1000.0,
        "rotation_error_deg": math.degrees(rotation),
        "success": bool(position <= 0.001 and rotation <= math.radians(0.1)),
        "iterations": iteration,
        "solve_time_ms": elapsed_ms,
        "joint_motion_l2_rad": float(np.linalg.norm(q[7:] - q_initial[7:])),
        "max_joint_change_rad": float(np.max(np.abs(q[7:] - q_initial[7:]))),
        "predicted_base_translation_mm": base_translation * 1000.0,
        "predicted_base_rotation_deg": math.degrees(base_rotation),
        "momentum_residual": (
            normalized_momentum_residual(kin, q, target, damping)
            if method == "generalized"
            else None
        ),
        "trace": trace_rows,
    }


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    return {
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def summarize_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        group_key = tuple(row[key] for key in keys)
        groups.setdefault(group_key, []).append(row)
    summaries = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        summary = {key: value for key, value in zip(keys, group_key)}
        summary.update(
            {
                "n": len(group_rows),
                "success_count": sum(row["success"] for row in group_rows),
                "success_rate": float(np.mean([row["success"] for row in group_rows])),
                "position_error_mm": _quantiles(
                    row["position_error_mm"] for row in group_rows
                ),
                "rotation_error_deg": _quantiles(
                    row["rotation_error_deg"] for row in group_rows
                ),
                "iterations": _quantiles(row["iterations"] for row in group_rows),
                "solve_time_ms": _quantiles(
                    row["solve_time_ms"] for row in group_rows
                ),
                "joint_motion_l2_rad": _quantiles(
                    row["joint_motion_l2_rad"] for row in group_rows
                ),
                "max_joint_change_rad": _quantiles(
                    row["max_joint_change_rad"] for row in group_rows
                ),
            }
        )
        if (
            group_rows[0]["method"] == "generalized"
            and "predicted_base_translation_mm" in group_rows[0]
        ):
            summary["predicted_base_translation_mm"] = _quantiles(
                row["predicted_base_translation_mm"] for row in group_rows
            )
            summary["predicted_base_rotation_deg"] = _quantiles(
                row["predicted_base_rotation_deg"] for row in group_rows
            )
            summary["momentum_residual"] = _quantiles(
                row["momentum_residual"] for row in group_rows
            )
        summaries.append(summary)
    return summaries


def run_numerical_experiment(
    kin: Kinematics, specs: list[dict[str, Any]], timing_repeats: int
) -> dict[str, Any]:
    rows = []
    for config_index, q_initial in enumerate(initial_configurations()):
        pin.framesForwardKinematics(kin.model, kin.data, q_initial)
        start = pin.SE3(kin.data.oMf[kin.model.getFrameId(kin.frame_name)])
        for spec in specs:
            target = make_target(start, spec)
            for method in ("generalized", "fixed"):
                measured = solve_equal_budget(
                    kin, q_initial, target, method, trace=True
                )
                timings = [measured["solve_time_ms"]]
                for _ in range(max(0, timing_repeats - 1)):
                    repeated = solve_equal_budget(
                        kin, q_initial, target, method, trace=False
                    )
                    timings.append(repeated["solve_time_ms"])
                row = {
                    "configuration": config_index + 1,
                    "target_id": spec["id"],
                    "kind": spec["kind"],
                    "method": method,
                    "position_error_mm": measured["position_error_mm"],
                    "rotation_error_deg": measured["rotation_error_deg"],
                    "success": measured["success"],
                    "iterations": measured["iterations"],
                    "solve_time_ms": float(np.median(timings)),
                    "joint_motion_l2_rad": measured["joint_motion_l2_rad"],
                    "max_joint_change_rad": measured["max_joint_change_rad"],
                    "predicted_base_translation_mm": measured[
                        "predicted_base_translation_mm"
                    ],
                    "predicted_base_rotation_deg": measured[
                        "predicted_base_rotation_deg"
                    ],
                    "momentum_residual": measured["momentum_residual"],
                    "trace": measured["trace"],
                }
                rows.append(row)
    return {
        "target_count": len(specs) * len(ARM_OFFSETS),
        "solve_count": len(rows),
        "rows": rows,
        "summary_by_method": summarize_rows(rows, ("method",)),
        "summary_by_method_and_kind": summarize_rows(rows, ("method", "kind")),
        "threshold_failures": [
            {key: value for key, value in row.items() if key != "trace"}
            for row in rows
            if not row["success"]
        ],
    }


def _mujoco_body_pose(model: mujoco.MjModel, data: mujoco.MjData) -> pin.SE3:
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "left_attachment"
    )
    return pin.SE3(
        np.asarray(data.xmat[body_id]).reshape(3, 3).copy(),
        np.asarray(data.xpos[body_id]).copy(),
    )


def _set_mujoco_q(model: mujoco.MjModel, data: mujoco.MjData, q_pin: np.ndarray) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = q_pin[:3]
    data.qpos[3:7] = [q_pin[6], q_pin[3], q_pin[4], q_pin[5]]
    data.qpos[7:13] = q_pin[7:]
    data.ctrl[:6] = q_pin[7:]
    mujoco.mj_forward(model, data)


def run_dynamic_validation(
    kin: Kinematics, scene_path: Path
) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    specs = []
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
            rotation[axis_index] = math.radians(sign * 10)
            specs.append(
                {
                    "id": f"DR-{axis_name}-{sign:+d}",
                    "kind": "rotation",
                    "translation_m": [0.0, 0.0, 0.0],
                    "rotation_vector_rad": rotation.tolist(),
                }
            )
    rows = []
    q_initial = initial_configurations()[0]
    for spec in specs:
        reference_data = mujoco.MjData(model)
        _set_mujoco_q(model, reference_data, q_initial)
        start = _mujoco_body_pose(model, reference_data)
        target = make_target(start, spec)
        for method in ("generalized", "fixed"):
            solved = solve_equal_budget(
                kin, q_initial, target, method, trace=False
            )
            data = mujoco.MjData(model)
            _set_mujoco_q(model, data, q_initial)
            base_start_pos = np.asarray(data.qpos[:3]).copy()
            base_start_quat = np.asarray(
                [data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]]
            )
            data.ctrl[:6] = solved["q_final"][7:]
            for _ in range(100):
                mujoco.mj_step(model, data)
            actual = _mujoco_body_pose(model, data)
            position = float(np.linalg.norm(target.translation - actual.translation))
            rotation = float(
                np.linalg.norm(pin.log3(actual.rotation.T @ target.rotation))
            )
            base_end_quat = np.asarray(
                [data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]]
            )
            rows.append(
                {
                    "scenario": spec["id"],
                    "kind": spec["kind"],
                    "method": method,
                    "position_error_mm": position * 1000.0,
                    "rotation_error_deg": math.degrees(rotation),
                    "actual_base_translation_mm": float(
                        np.linalg.norm(data.qpos[:3] - base_start_pos) * 1000.0
                    ),
                    "actual_base_rotation_deg": math.degrees(
                        quaternion_angle_rad(base_start_quat, base_end_quat)
                    ),
                    "predicted_base_translation_mm": solved[
                        "predicted_base_translation_mm"
                    ],
                    "predicted_base_rotation_deg": solved[
                        "predicted_base_rotation_deg"
                    ],
                    "joint_motion_l2_rad": solved["joint_motion_l2_rad"],
                }
            )
    summaries = []
    for method in ("generalized", "fixed"):
        selected = [row for row in rows if row["method"] == method]
        summaries.append(
            {
                "method": method,
                "n": len(selected),
                "position_error_mm": _quantiles(
                    row["position_error_mm"] for row in selected
                ),
                "rotation_error_deg": _quantiles(
                    row["rotation_error_deg"] for row in selected
                ),
                "actual_base_translation_mm": _quantiles(
                    row["actual_base_translation_mm"] for row in selected
                ),
                "actual_base_rotation_deg": _quantiles(
                    row["actual_base_rotation_deg"] for row in selected
                ),
                "joint_motion_l2_rad": _quantiles(
                    row["joint_motion_l2_rad"] for row in selected
                ),
            }
        )
    return {
        "scenario_count": len(specs),
        "solve_count": len(rows),
        "mujoco_steps_per_command": 100,
        "simulated_seconds_per_command": 100 * float(model.opt.timestep),
        "rows": rows,
        "summary_by_method": summaries,
    }


def run_damping_ablation(
    kin: Kinematics, specs: list[dict[str, Any]]
) -> dict[str, Any]:
    combined = [spec for spec in specs if spec["kind"] == "combined"]
    rows = []
    for damping in (0.01, 0.03, 0.05, 0.10, 0.20):
        for config_index, q_initial in enumerate(initial_configurations()):
            pin.framesForwardKinematics(kin.model, kin.data, q_initial)
            start = pin.SE3(kin.data.oMf[kin.model.getFrameId(kin.frame_name)])
            for spec in combined:
                target = make_target(start, spec)
                solved = solve_equal_budget(
                    kin,
                    q_initial,
                    target,
                    "generalized",
                    damping=damping,
                    trace=False,
                )
                rows.append(
                    {
                        "damping": damping,
                        "configuration": config_index + 1,
                        "target_id": spec["id"],
                        "method": "generalized",
                        "position_error_mm": solved["position_error_mm"],
                        "rotation_error_deg": solved["rotation_error_deg"],
                        "success": solved["success"],
                        "iterations": solved["iterations"],
                        "solve_time_ms": solved["solve_time_ms"],
                        "joint_motion_l2_rad": solved["joint_motion_l2_rad"],
                        "max_joint_change_rad": solved["max_joint_change_rad"],
                    }
                )
    return {
        "target_count_per_damping": len(combined) * len(ARM_OFFSETS),
        "solve_count": len(rows),
        "rows": rows,
        "summary_by_damping": summarize_rows(rows, ("damping",)),
    }


def _parquet_list_column(table: Any, name: str) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_pylist(), dtype=float)


def summarize_datasets(dataset_root: Path) -> dict[str, Any]:
    """Summarize saved demonstrations without inferring an attempt success rate."""
    task_rows = []
    episode_rows = []
    for dataset_dir in sorted(dataset_root.glob("*_lerobot_v21")):
        parquet_paths = sorted(dataset_dir.glob("data/**/*.parquet"))
        task_episodes = []
        for parquet_path in parquet_paths:
            table = pq.read_table(parquet_path)
            state = _parquet_list_column(table, "observation.state")
            timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=float)
            success = np.asarray(table["is_success"].to_pylist(), dtype=bool)
            joints = state[:, :6]
            ee_position = state[:, 6:9]
            ee_quaternion = state[:, 9:13]
            position_steps = np.linalg.norm(np.diff(ee_position, axis=0), axis=1)
            rotation_steps = [
                quaternion_angle_rad(a, b)
                for a, b in zip(ee_quaternion[:-1], ee_quaternion[1:])
            ]
            duration = (
                float(timestamps[-1] - timestamps[0] + np.median(np.diff(timestamps)))
                if len(timestamps) > 1
                else 0.0
            )
            episode = {
                "task": dataset_dir.name.removesuffix("_lerobot_v21"),
                "episode_file": str(parquet_path.relative_to(dataset_root)),
                "frames": table.num_rows,
                "duration_s": duration,
                "ee_path_length_m": float(np.sum(position_steps)),
                "max_position_step_mm": (
                    float(np.max(position_steps) * 1000.0)
                    if len(position_steps)
                    else 0.0
                ),
                "max_rotation_step_deg": (
                    math.degrees(max(rotation_steps)) if rotation_steps else 0.0
                ),
                "joint_range_rad": np.ptp(joints, axis=0).tolist(),
                "terminal_success": bool(success[-1]) if len(success) else False,
                "success_labeled_frame_count": int(np.count_nonzero(success)),
            }
            episode_rows.append(episode)
            task_episodes.append(episode)
        if not task_episodes:
            continue
        task_rows.append(
            {
                "task": task_episodes[0]["task"],
                "episodes": len(task_episodes),
                "frames": sum(row["frames"] for row in task_episodes),
                "duration_s": sum(row["duration_s"] for row in task_episodes),
                "ee_path_length_m": sum(
                    row["ee_path_length_m"] for row in task_episodes
                ),
                "max_joint_range_rad": np.max(
                    np.asarray([row["joint_range_rad"] for row in task_episodes]),
                    axis=0,
                ).tolist(),
                "max_position_step_mm": max(
                    row["max_position_step_mm"] for row in task_episodes
                ),
                "max_rotation_step_deg": max(
                    row["max_rotation_step_deg"] for row in task_episodes
                ),
            }
        )
    return {
        "interpretation": (
            "这些记录是采集流程筛选并保存的成功示范；缺少失败尝试分母，"
            "因此不能据此计算规划器或控制器成功率。"
        ),
        "task_count": len(task_rows),
        "episode_count": len(episode_rows),
        "frame_count": sum(row["frames"] for row in episode_rows),
        "total_duration_s": sum(row["duration_s"] for row in episode_rows),
        "task_rows": task_rows,
        "episode_rows": episode_rows,
        "episode_duration_s": _quantiles(row["duration_s"] for row in episode_rows),
        "episode_path_length_m": _quantiles(
            row["ee_path_length_m"] for row in episode_rows
        ),
        "max_position_step_mm": _quantiles(
            row["max_position_step_mm"] for row in episode_rows
        ),
        "max_rotation_step_deg": _quantiles(
            row["max_rotation_step_deg"] for row in episode_rows
        ),
    }


def build_results(args: argparse.Namespace) -> dict[str, Any]:
    model_path = (REPO_ROOT / args.model).resolve()
    scene_path = (REPO_ROOT / args.scene).resolve()
    dataset_root = (REPO_ROOT / args.dataset_root).resolve()
    kin = Kinematics(str(model_path), args.frame)
    specs = generate_target_specs(args.seed)

    # Warm both code paths before timing.
    q_warm = initial_configurations()[0]
    pin.framesForwardKinematics(kin.model, kin.data, q_warm)
    warm_start = pin.SE3(kin.data.oMf[kin.model.getFrameId(args.frame)])
    warm_target = make_target(warm_start, specs[0])
    solve_equal_budget(kin, q_warm, warm_target, "generalized", trace=False)
    solve_equal_budget(kin, q_warm, warm_target, "fixed", trace=False)

    results = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git": git_metadata(REPO_ROOT),
            "model_sha256": sha256_file(model_path),
            "scene_sha256": sha256_file(scene_path),
            "python_version": platform.python_version(),
            "pinocchio_version": pin.__version__,
            "mujoco_version": mujoco.__version__,
            "numpy_version": np.__version__,
            "pyarrow_version": __import__("pyarrow").__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported",
        },
        "experiment_config": {
            "model_path": str(model_path.relative_to(REPO_ROOT)),
            "scene_path": str(scene_path.relative_to(REPO_ROOT)),
            "dataset_root": str(dataset_root.relative_to(REPO_ROOT)),
            "end_effector_frame": args.frame,
            "nq": kin.model.nq,
            "nv": kin.model.nv,
            "arm_dof": 6,
            "seed": args.seed,
            "initial_configurations": [q.tolist() for q in initial_configurations()],
            "target_definition": {
                "translation": "3 axes × 3 magnitudes × 2 signs = 18",
                "rotation": "3 axes × 3 magnitudes × 2 signs = 18",
                "combined": "12 fixed-seed random SE(3) offsets",
            },
            "budget": {
                "interpolation_points": 20,
                "iterations_per_point": 10,
                "fine_iterations": 50,
                "dt": 0.05,
                "damping": 0.05,
                "velocity_norm_limit": 0.2,
                "timing_repeats": args.timing_repeats,
            },
            "success_threshold": {
                "position_error_mm": 1.0,
                "rotation_error_deg": 0.1,
                "logical_operator": "and",
            },
            "target_specs": specs,
        },
    }
    print("[1/4] running 240-target numerical IK comparison", flush=True)
    results["numerical_ik"] = run_numerical_experiment(
        kin, specs, args.timing_repeats
    )
    print("[2/4] running 12-scenario non-rendering MuJoCo validation", flush=True)
    results["dynamic_validation"] = run_dynamic_validation(kin, scene_path)
    print("[3/4] running 60-target damping ablation", flush=True)
    results["damping_ablation"] = run_damping_ablation(kin, specs)
    print("[4/4] summarizing curated LeRobot demonstrations", flush=True)
    results["task_summary"] = summarize_datasets(dataset_root)
    return results


def validate_results(results: dict[str, Any]) -> None:
    required = {
        "metadata",
        "experiment_config",
        "numerical_ik",
        "dynamic_validation",
        "damping_ablation",
        "task_summary",
    }
    if set(results) != required:
        raise ValueError(f"Unexpected top-level result keys: {set(results)}")
    if results["numerical_ik"]["target_count"] != 240:
        raise ValueError("Expected exactly 240 numerical targets.")
    if results["numerical_ik"]["solve_count"] != 480:
        raise ValueError("Expected 480 method-target numerical solves.")
    if results["dynamic_validation"]["scenario_count"] != 12:
        raise ValueError("Expected exactly 12 dynamic scenarios.")
    if results["dynamic_validation"]["solve_count"] != 24:
        raise ValueError("Expected 24 dynamic method-scenario solves.")
    if results["damping_ablation"]["target_count_per_damping"] != 60:
        raise ValueError("Expected exactly 60 targets per damping value.")
    if results["damping_ablation"]["solve_count"] != 300:
        raise ValueError("Expected exactly 300 damping-ablation solves.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mjcf/arm.xml")
    parser.add_argument("--frame", default="left_attachment")
    parser.add_argument("--scene", default="mjcf/1_cube_scene.xml")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument(
        "--output", default="reports/ik_validation/results.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = build_results(args)
    validate_results(results)
    output = (REPO_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"validated results written to {output}")


if __name__ == "__main__":
    main()
