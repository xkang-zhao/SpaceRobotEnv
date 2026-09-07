#!/usr/bin/env python3
"""Collect automatic-grasp demonstrations in LeRobot v2.1 format."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from data_collection.autograb_v21 import (  # noqa: E402
    CollectionConfig,
    build_camera_map,
    collect_dataset,
    validate_collection_config,
)
from planners.auto_grasp import TASK_NAMES, get_task_spec  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect calibrated automatic grasps directly in "
            "LeRobot v2.1 format."
        )
    )
    parser.add_argument("task_name", choices=TASK_NAMES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Dataset root. Defaults to "
            "data/local/<task>_lerobot_v21."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to save.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of independent spawned MuJoCo workers.",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Language instruction stored in tasks.jsonl.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-attempts-per-episode",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        metavar="NAME=OBS_KEY",
        help=(
            "Video feature and observation mapping. Repeat to select "
            "multiple cameras. Defaults to cam1/cam2/cam3."
        ),
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Accept failed attempts as dataset episodes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing dataset only after collection succeeds.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open the interactive viewer; requires --num-envs 1.",
    )
    parser.add_argument(
        "--enable-rerun",
        action="store_true",
        help="Log frames to Rerun; requires --num-envs 1.",
    )
    parser.add_argument("--debug-planner", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> CollectionConfig:
    task = get_task_spec(args.task_name)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "data"
            / "local"
            / f"{args.task_name}_lerobot_v21"
        )
    elif not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    try:
        camera_map = build_camera_map(args.camera)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return CollectionConfig(
        task_name=args.task_name,
        output_dir=output_dir,
        episodes=args.episodes,
        num_envs=args.num_envs,
        instruction=args.instruction or task.instruction,
        seed=args.seed,
        max_attempts_per_episode=args.max_attempts_per_episode,
        camera_map=camera_map,
        keep_failed=args.keep_failed,
        overwrite=args.overwrite,
        render=args.render,
        enable_rerun=args.enable_rerun,
        debug_planner=args.debug_planner,
    )


def parse_args(argv: list[str] | None = None) -> CollectionConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        validate_collection_config(config)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    return config


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    started_at = time.perf_counter()
    try:
        summary = collect_dataset(config)
    except KeyboardInterrupt:
        print("[LEROBOT21] interrupted; no partial dataset was committed.")
        return 130
    print(
        "[LEROBOT21] finished: "
        f"saved_episodes={summary['saved_episodes']}, "
        f"successful_episodes={summary['successful_episodes']}, "
        f"attempts={summary['attempts']}, "
        f"total_frames={summary['total_frames']}, "
        f"workers={summary['worker_pids']}, "
        f"elapsed_s={time.perf_counter() - started_at:.2f}, "
        f"output={summary['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
