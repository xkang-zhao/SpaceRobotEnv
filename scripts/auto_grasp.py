"""Unified command-line entry point for automatic grasp tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from planners.auto_grasp import (  # noqa: E402
    LATERAL_GRASP_PROFILES,
    TASK_NAMES,
)


def _print_top_level_help() -> None:
    parser = argparse.ArgumentParser(
        description="Run a calibrated automatic grasp task."
    )
    parser.add_argument("task", choices=TASK_NAMES)
    parser.print_help()
    print("\nRun `python scripts/auto_grasp.py TASK --help` for task options.")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_top_level_help()
        return 0

    task_name = args.pop(0)
    if task_name not in TASK_NAMES:
        choices = ", ".join(TASK_NAMES)
        raise SystemExit(f"Unknown task {task_name!r}. Choose one of: {choices}")
    if task_name == "cube":
        from planners.auto_grasp.cube_cli import main as cube_main

        return cube_main(args)
    if task_name == "satellite_handle":
        from planners.auto_grasp.handle_cli import main as handle_main

        return handle_main(args)

    from planners.auto_grasp.geometry import parse_float_list
    from planners.auto_grasp.lateral import (
        LeftAntennaPanelGrabConfig,
        LeftAntennaPanelGrabPlanner,
    )
    from planners.auto_grasp.lateral_cli import main_for_profile

    profile = LATERAL_GRASP_PROFILES[task_name]
    return main_for_profile(
        profile,
        LeftAntennaPanelGrabConfig,
        LeftAntennaPanelGrabPlanner,
        parse_float_list,
        args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
