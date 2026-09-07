#!/usr/bin/env python3
"""Record presentation videos for the automatic-grasp task suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from reports.record.autograb_video import (  # noqa: E402
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TASK_ORDER,
    VideoRecordingConfig,
    record_video_batch,
    validate_recording_config,
)
from planners.auto_grasp import TASK_NAMES  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        action="append",
        choices=TASK_NAMES,
        default=None,
        help=(
            "Task to record. Repeat to select multiple tasks. "
            "Defaults to all eight tasks."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "assets/videos/auto_grasp",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a task video only after its new recording succeeds.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> VideoRecordingConfig:
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    task_names = tuple(args.task or DEFAULT_TASK_ORDER)
    return VideoRecordingConfig(
        output_dir=output_dir.resolve(),
        task_names=task_names,
        seed=args.seed,
        max_attempts=args.max_attempts,
        overwrite=args.overwrite,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    try:
        validate_recording_config(config)
        manifest = record_video_batch(config)
    except (FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"[VIDEO] finished: tasks={len(manifest['tasks'])}, "
        f"output={config.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
