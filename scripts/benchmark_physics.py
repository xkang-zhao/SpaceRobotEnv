"""Compare headless Cube grasp episodes on CPU MuJoCo and CUDA Warp."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mujoco_robot.physics_benchmark import benchmark_cube


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mujoco", "warp", "both"), default="both")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--warp-device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    backends = ("mujoco", "warp") if args.backend == "both" else (args.backend,)
    results = [benchmark_cube(backend, seed=args.seed, max_steps=args.max_steps,
                              device=args.warp_device) for backend in backends]
    print(json.dumps(results, indent=2))
    return 0 if all(result["success"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
