#!/usr/bin/env python3
"""Collect Cube LeRobot v2.1 demonstrations with CUDA Warp batch simulation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_collection.autograb_v21 import CollectionConfig, build_camera_map
from data_collection.gpuwarp_autograb_v21 import collect_gpuwarp_dataset, validate_gpuwarp_config
from planners.auto_grasp import get_task_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_name", choices=("cube",), help="GPU Warp currently supports Cube only.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=32, help="CUDA VectorEnv world count.")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts-per-episode", type=int, default=10)
    parser.add_argument("--camera", action="append", default=None, metavar="NAME=OBS_KEY")
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ik-backend", choices=("gpu", "cpu", "shadow"), default="gpu")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--debug-planner", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> tuple[CollectionConfig, str, str]:
    args = build_parser().parse_args(argv)
    task = get_task_spec(args.task_name)
    output = args.output_dir or PROJECT_ROOT / "data" / "local" / f"{args.task_name}_gpuwarp_lerobot_v21"
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    try:
        config = CollectionConfig(
            task_name=args.task_name, output_dir=output, episodes=args.episodes,
            num_envs=args.num_envs, instruction=args.instruction or task.instruction,
            seed=args.seed, max_attempts_per_episode=args.max_attempts_per_episode,
            camera_map=build_camera_map(args.camera), keep_failed=args.keep_failed,
            overwrite=args.overwrite, debug_planner=args.debug_planner,
        )
        validate_gpuwarp_config(config, ik_backend=args.ik_backend)
    except ValueError as exc:
        build_parser().error(str(exc))
    return config, args.ik_backend, args.device


def main(argv: list[str] | None = None) -> int:
    config, ik_backend, device = parse_args(argv)
    started = time.perf_counter()
    try:
        summary = collect_gpuwarp_dataset(config, ik_backend=ik_backend, device=device)
    except KeyboardInterrupt:
        print("[GPUWARP-LEROBOT21] interrupted; no partial dataset was committed.")
        return 130
    print(
        "[GPUWARP-LEROBOT21] finished: "
        f"saved_episodes={summary['saved_episodes']}, successful_episodes={summary['successful_episodes']}, "
        f"attempts={summary['attempts']}, total_frames={summary['total_frames']}, "
        f"gpu_accepted_world_steps={summary['gpu_accepted_world_steps']}, "
        f"cpu_fallback_world_steps={summary['cpu_fallback_world_steps']}, "
        f"ik_backend={ik_backend}, elapsed_s={time.perf_counter() - started:.2f}, "
        f"output={summary['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
