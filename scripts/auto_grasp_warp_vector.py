"""Run CUDA Cube grasps with CPU, GPU-with-fallback, or shadow IK."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from planners.auto_grasp.warp_cube import run_vector_cube


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ik-backend", choices=("cpu", "gpu", "shadow"), default="cpu")
    args = parser.parse_args(argv)
    if args.num_envs <= 0 or args.max_steps <= 0 or args.seed < 0:
        parser.error("num-envs/max-steps must be positive and seed nonnegative")
    report = run_vector_cube(args.num_envs, seed=args.seed, max_steps=args.max_steps, device=args.device, ik_backend=args.ik_backend)
    print(json.dumps(report, indent=2))
    return 0 if all(row["success"] for row in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
