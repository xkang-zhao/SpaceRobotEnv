"""Standalone runner for the satellite-handle grasp planner."""

from __future__ import annotations

import argparse

from .geometry import parse_float_list
from .handle import AutograbConfig, AutograbPlanner
from .runner import run_environment_planner


DEFAULT_HANDLE_OFFSET = AutograbConfig().handle_offset


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-grasp satellite handle.")
    parser.add_argument(
        "handle_offset",
        type=parse_float_list,
        nargs="?",
        default=DEFAULT_HANDLE_OFFSET,
        help="Grasp offset [X, Y, Z] in satellite body frame.",
    )
    parser.add_argument(
        "--env",
        default=AutograbConfig.env_id,
        help="Gymnasium environment id.",
    )
    parser.add_argument(
        "--render_mode",
        choices=("none", "human", "rgb_array"),
        default="human",
        help="MuJoCo viewer mode. Camera observations remain enabled in all modes.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace the standalone demo at the environment control frequency.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def build_config_from_args(args: argparse.Namespace) -> AutograbConfig:
    render_mode = None if args.render_mode == "none" else args.render_mode
    return AutograbConfig(
        handle_offset=args.handle_offset,
        env_id=args.env,
        render_mode=render_mode,
        debug=args.debug,
    )


def run_planner(
    config: AutograbConfig,
    *,
    realtime: bool = False,
) -> bool:
    return run_environment_planner(
        config,
        AutograbPlanner,
        "satellite handle grasp confirmed",
        realtime=realtime,
    )


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    config = build_config_from_args(args)
    return 0 if run_planner(config, realtime=args.realtime) else 1
