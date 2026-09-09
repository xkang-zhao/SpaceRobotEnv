"""Benchmark fixed-control Cube replay on one CPU core and batched CUDA worlds."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mujoco_robot.batch_physics_benchmark import record_cube_controls, replay_cpu, replay_warp_batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", nargs="+", type=int, default=[1, 32, 128, 512])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warp-device", default="cuda:0")
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=512)
    parser.add_argument("--qpos-atol", type=float, default=0.005)
    parser.add_argument("--qvel-atol", type=float, default=0.1)
    parser.add_argument("--output", type=Path, help="Write JSON to a new file; refuses to overwrite.")
    args = parser.parse_args(argv)
    if any(n <= 0 for n in args.worlds) or min(args.nconmax, args.njmax, args.qpos_atol, args.qvel_atol) <= 0:
        parser.error("world counts, capacities and tolerances must be positive")
    if args.output and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    print("Generating fixed Cube controls on CPU (outside timing)...", file=sys.stderr, flush=True)
    trace = record_cube_controls(args.seed)
    cpu, qpos, qvel = replay_cpu(trace)
    results = [cpu]
    for nworld in args.worlds:
        print(f"Benchmarking Warp nworld={nworld}...", file=sys.stderr, flush=True)
        result = replay_warp_batch(trace, nworld, qpos, qvel, device=args.warp_device,
                                   nconmax=args.nconmax, njmax=args.njmax,
                                   qpos_atol=args.qpos_atol, qvel_atol=args.qvel_atol)
        result["physics_speedup_vs_one_cpu_core"] = result["physics_steps_per_second"] / cpu["physics_steps_per_second"]
        results.append(result)
    report = {
        "workload": "Identical initial Cube state and fixed controls in every world; no IK or rendering in timing",
        "seed": args.seed, "control_steps": len(trace.controls), "substeps": trace.substeps,
        "timestep": trace.model.opt.timestep,
        "controls_sha256": hashlib.sha256(trace.controls.tobytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in ("mujoco", "mujoco-warp", "warp-lang", "numpy")},
        "platform": platform.platform(),
        "capacities_per_world": {"nconmax": args.nconmax, "njmax": args.njmax},
        "results": results,
    }
    payload = json.dumps(report, indent=2, allow_nan=False)
    print(payload)
    if args.output:
        with args.output.open("x", encoding="utf-8") as file:
            file.write(payload + "\n")
    return 0 if all(row.get("state_within_tolerance", True) for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
