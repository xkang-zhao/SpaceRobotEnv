"""Synchronous fixed-control replay benchmark, independent of IK and rendering."""

from __future__ import annotations

import gc
import os
import subprocess
from dataclasses import dataclass
from time import perf_counter

import mujoco
import numpy as np


@dataclass
class ControlReplay:
    model: mujoco.MjModel
    initial: mujoco.MjData
    controls: np.ndarray
    substeps: int = 100

    def __post_init__(self):
        self.controls = np.asarray(self.controls, dtype=np.float64)
        if self.controls.ndim != 2 or self.controls.shape[1] != self.model.nu:
            raise ValueError(f"controls must have shape (control_steps, {self.model.nu})")
        if not len(self.controls) or not np.isfinite(self.controls).all():
            raise ValueError("controls must be nonempty and finite")
        if self.substeps <= 0:
            raise ValueError("substeps must be positive")


def record_cube_controls(seed: int = 7, max_steps: int = 400) -> ControlReplay:
    """Generate the teacher trace once, entirely outside benchmark timing."""
    import gymnasium as gym
    import envs  # noqa: F401
    from planners.auto_grasp.tasks import AutoGraspSession

    env = gym.make("SpaceUR10e-Cube-v0", observation_mode="state")
    try:
        base = env.unwrapped
        session = AutoGraspSession("cube", env)
        obs = session.reset(seed)
        initial = mujoco.MjData(base.model)
        mujoco.mj_copyData(initial, base.model, base.data)
        controls = []
        for _ in range(max_steps):
            action = session.compute_action({**obs, "target_pose": base._get_target_pose()})
            transition = env.step(action)
            controls.append(base.data.ctrl.copy())
            session.update(*transition)
            obs = transition[0]
            if session.is_done:
                break
        if not session.is_success:
            raise RuntimeError(f"Cube teacher seed={seed}: {session.failure_reason or 'step limit'}")
        return ControlReplay(base.model, initial, np.asarray(controls))
    finally:
        env.close()


def replay_cpu(trace: ControlReplay) -> tuple[dict, np.ndarray, np.ndarray]:
    """Single-core native MuJoCo baseline, with identical warmup and reset."""
    data = mujoco.MjData(trace.model)
    mujoco.mj_copyData(data, trace.model, trace.initial)
    data.ctrl[:] = trace.controls[0]
    mujoco.mj_step(trace.model, data, nstep=trace.substeps)
    mujoco.mj_copyData(data, trace.model, trace.initial)
    qpos, qvel = [], []
    physics_seconds = 0.0
    for ctrl in trace.controls:
        data.ctrl[:] = ctrl
        start = perf_counter()
        mujoco.mj_step(trace.model, data, nstep=trace.substeps)
        physics_seconds += perf_counter() - start
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise RuntimeError("CPU fixed-control replay produced non-finite state")
    return {
        "backend": "mujoco", "nworld": 1, "cpu_threads": 1,
        "physics_seconds": physics_seconds,
        "physics_steps_per_second": len(trace.controls) * trace.substeps / physics_seconds,
        "mean_control_physics_ms": physics_seconds * 1000 / len(trace.controls),
    }, np.asarray(qpos), np.asarray(qvel)


def _process_gpu_memory_mib() -> float | None:
    """NVML process allocation, including CUDA context and cached allocations."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        values = [float(memory) for pid, memory in
                  (line.split(",") for line in result.stdout.splitlines())
                  if int(pid.strip()) == os.getpid()]
        return sum(values) if values else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def compare_states(qpos, qvel, expected_qpos, expected_qvel) -> dict:
    """Compare every world to CPU, and replicas to world zero."""
    return {
        "finite": bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),
        "max_qpos_abs_error": float(np.max(np.abs(qpos - expected_qpos))),
        "max_qvel_abs_error": float(np.max(np.abs(qvel - expected_qvel))),
        "max_world_qpos_spread": float(np.max(np.abs(qpos - qpos[0]))),
        "max_world_qvel_spread": float(np.max(np.abs(qvel - qvel[0]))),
    }


def replay_warp_batch(trace: ControlReplay, nworld: int, expected_qpos: np.ndarray,
                      expected_qvel: np.ndarray, *, device: str = "cuda:0",
                      nconmax: int = 128, njmax: int = 512,
                      qpos_atol: float = 0.005, qvel_atol: float = 0.1) -> dict:
    """Replicate one initial state; upload fixed controls and read all worlds.

    Timing phases synchronize explicitly. Compilation, graph capture, reset,
    allocation, validation, and memory queries are excluded from throughput.
    Memory measurements include retained Warp pool allocations in this process.
    """
    if nworld <= 0 or nconmax <= 0 or njmax <= 0:
        raise ValueError("nworld, nconmax and njmax must be positive")
    if qpos_atol <= 0 or qvel_atol <= 0:
        raise ValueError("state tolerances must be positive")
    expected_shape = (len(trace.controls), trace.model.nq)
    if expected_qpos.shape != expected_shape or expected_qvel.shape != (len(trace.controls), trace.model.nv):
        raise ValueError("CPU reference dimensions do not match the control trace")
    import mujoco_warp as mjw
    import warp as wp

    cuda = wp.get_device(device)
    if not cuda.is_cuda:
        raise ValueError(f"Batch replay requires CUDA, got {device!r}")
    m = d = graph = capture = None
    setup_start = perf_counter()
    try:
        with wp.ScopedDevice(cuda):
            m = mjw.put_model(trace.model)
            d = mjw.put_data(trace.model, trace.initial, nworld=nworld,
                             nconmax=nconmax, njmax=njmax)
            # Load specialized kernels and run one disposable group before timing.
            mjw.step(m, d)
            wp.synchronize_device(cuda)
            with wp.ScopedCapture() as capture:
                for _ in range(trace.substeps):
                    mjw.step(m, d)
            graph = capture.graph
            wp.capture_launch(graph)
            wp.synchronize_device(cuda)
            # Reset all arrays in place, preserving captured device pointers.
            fresh = mjw.put_data(trace.model, trace.initial, nworld=nworld,
                                 nconmax=nconmax, njmax=njmax)
            _copy_warp_arrays(d, fresh, wp)
            wp.synchronize_device(cuda)
            del fresh
            gc.collect()
            setup_seconds = perf_counter() - setup_start
            host_controls = np.repeat(trace.controls[:, None, :], nworld, axis=1).astype(np.float32)
            timings = {"upload": 0.0, "physics": 0.0, "readback": 0.0}
            errors = {key: 0.0 for key in (
                "max_qpos_abs_error", "max_qvel_abs_error",
                "max_world_qpos_spread", "max_world_qvel_spread")}
            for index, ctrl in enumerate(host_controls):
                start = perf_counter()
                d.ctrl.assign(ctrl)
                wp.synchronize_device(cuda)
                timings["upload"] += perf_counter() - start
                start = perf_counter()
                wp.capture_launch(graph)
                wp.synchronize_device(cuda)
                timings["physics"] += perf_counter() - start
                start = perf_counter()
                qpos, qvel, overflow = d.qpos.numpy(), d.qvel.numpy(), d.overflow.numpy()
                timings["readback"] += perf_counter() - start
                if np.any(overflow):
                    worlds = np.flatnonzero(overflow)
                    raise RuntimeError(f"Warp nworld={nworld}, control={index}: overflow in worlds {worlds[:10].tolist()}")
                comparison = compare_states(qpos, qvel, expected_qpos[index], expected_qvel[index])
                if not comparison.pop("finite"):
                    raise RuntimeError(f"Warp nworld={nworld}, control={index}: non-finite state")
                for key, value in comparison.items():
                    errors[key] = max(errors[key], value)
            physical_steps = nworld * len(trace.controls) * trace.substeps
            total_seconds = sum(timings.values())
            result = {
                "backend": "warp", "nworld": nworld, "device": cuda.name,
                "setup_seconds": setup_seconds,
                "physics_steps_per_second": physical_steps / timings["physics"],
                "steps_per_second_including_transfers": physical_steps / total_seconds,
                "mean_batch_control_ms": total_seconds * 1000 / len(trace.controls),
                "mean_phase_ms": {key: val * 1000 / len(trace.controls) for key, val in timings.items()},
                "process_gpu_memory_mib": _process_gpu_memory_mib(),
                "warp_pool_used_mib": wp.get_mempool_used_mem_current(cuda) / 2**20,
                "state_errors": errors,
                "state_within_tolerance": bool(errors["max_qpos_abs_error"] <= qpos_atol
                                               and errors["max_qvel_abs_error"] <= qvel_atol),
                "qpos_atol": qpos_atol, "qvel_atol": qvel_atol,
                "finite": True, "overflow": False,
            }
            return result
    finally:
        wp.synchronize_device(cuda)
        m = d = graph = capture = None
        gc.collect()


def _copy_warp_arrays(destination, source, wp):
    """Restore a dataclass tree without replacing graph-referenced buffers."""
    from dataclasses import fields, is_dataclass
    for field in fields(source):
        src, dst = getattr(source, field.name), getattr(destination, field.name)
        if isinstance(src, wp.array):
            wp.copy(dst, src)
        elif is_dataclass(src):
            _copy_warp_arrays(dst, src, wp)
