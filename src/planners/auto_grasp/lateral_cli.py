"""Shared CLI and runner for calibrated lateral grasp tasks."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from .profiles import LateralGraspTaskProfile
from .runner import run_environment_planner


def make_lateral_parser(
    profile: LateralGraspTaskProfile,
    config_type,
    parse_float_list: Callable[[str], list[float]],
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=profile.description)
    parser.add_argument(
        "panel_offset",
        type=parse_float_list,
        nargs="?",
        default=list(profile.grasp_offset_body),
        help=(
            "Deprecated positional grasp offset [X, Y, Z] in "
            f"{profile.object_label} body frame."
        ),
    )
    parser.add_argument(
        "--env",
        default=profile.env_id,
        help=f"Gymnasium environment id. Defaults to {profile.env_id}.",
    )
    parser.add_argument(
        "--grasp_offset",
        type=parse_float_list,
        default=None,
        help=(
            "Calibrated grasp point offset [X, Y, Z] in "
            f"{profile.object_label} body frame. Overrides the positional value."
        ),
    )
    parser.add_argument(
        "--target_frame",
        choices=("ee", "pinch"),
        default=config_type.target_frame,
        help=(
            "Which robot frame should reach grasp_offset. "
            "Default ee matches get_offet.py calibration."
        ),
    )
    parser.add_argument(
        "--approach_axis_body",
        type=parse_float_list,
        default=[0.0, 1.0, 0.0],
        help=(
            "Final grasp distance direction in "
            f"{profile.object_label} body frame."
        ),
    )
    parser.add_argument(
        "--world_left_axis",
        type=parse_float_list,
        default=[0.0, 1.0, 0.0],
        help=(
            "World-frame axis used by move_left. "
            "Default [0, 1, 0] means only world Y changes."
        ),
    )

    # Kept as deprecated compatibility options until old commands are migrated.
    parser.add_argument(
        "--world_up_axis",
        type=parse_float_list,
        default=[0.0, 0.0, 1.0],
        help="Deprecated compatibility option; currently ignored.",
    )
    parser.add_argument(
        "--forward_axis_local",
        type=parse_float_list,
        default=[0.0, 0.0, 1.0],
        help="Deprecated compatibility option; currently ignored.",
    )
    parser.add_argument(
        "--grasp_distance",
        type=float,
        default=config_type.grasp_distance,
        help="Final offset from the calibrated point along the approach axis.",
    )
    parser.add_argument(
        "--pregrasp_distance",
        type=float,
        default=config_type.pregrasp_distance,
        help="Deprecated compatibility option; currently ignored.",
    )
    parser.add_argument(
        "--escape_distance",
        type=float,
        default=config_type.escape_distance,
        help="Deprecated compatibility option; currently ignored.",
    )
    parser.add_argument(
        "--corridor_distance",
        type=float,
        default=config_type.corridor_distance,
        help="Deprecated compatibility option; currently ignored.",
    )
    parser.add_argument(
        "--overpass_height",
        type=float,
        default=config_type.overpass_height,
        help="Deprecated compatibility option; currently ignored.",
    )
    parser.add_argument(
        "--path_step_size",
        type=float,
        default=config_type.path_step_size,
        help="Nominal Cartesian step size for smooth waypoint interpolation.",
    )
    parser.add_argument(
        "--min_segment_steps",
        type=int,
        default=config_type.min_segment_steps,
        help="Minimum number of interpolation steps for each motion segment.",
    )
    if profile.expose_tool_z_rotation:
        parser.add_argument(
            "--tool_z_rotation",
            type=float,
            default=profile.tool_z_rotation,
            help="Local-Z rotation applied to the end-effector target, in radians.",
        )
    if profile.expose_base_orientation:
        parser.add_argument(
            "--base_grasp_orientation_rpy",
            type=parse_float_list,
            default=list(config_type.base_grasp_orientation_rpy),
            help=(
                "Base grasp orientation before the local-Z rotation, "
                "as [roll, pitch, yaw]."
            ),
        )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=config_type.max_steps,
        help="Maximum planner steps before failing.",
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


def build_lateral_config(args, profile: LateralGraspTaskProfile, config_type):
    grasp_offset = (
        args.grasp_offset if args.grasp_offset is not None else args.panel_offset
    )
    tool_z_rotation = getattr(
        args,
        "tool_z_rotation",
        profile.tool_z_rotation,
    )
    base_orientation = getattr(
        args,
        "base_grasp_orientation_rpy",
        config_type.base_grasp_orientation_rpy,
    )
    render_mode = None if args.render_mode == "none" else args.render_mode
    return config_type(
        panel_offset=grasp_offset,
        approach_axis_body=args.approach_axis_body,
        world_left_axis=args.world_left_axis,
        world_up_axis=args.world_up_axis,
        forward_axis_local=args.forward_axis_local,
        target_frame=args.target_frame,
        env_id=args.env,
        grasp_distance=args.grasp_distance,
        pregrasp_distance=args.pregrasp_distance,
        escape_distance=args.escape_distance,
        corridor_distance=args.corridor_distance,
        overpass_height=args.overpass_height,
        path_step_size=args.path_step_size,
        min_segment_steps=args.min_segment_steps,
        base_grasp_orientation_rpy=tuple(base_orientation),
        tool_z_rotation=tool_z_rotation,
        rotation_mode=profile.rotation_mode,
        position_tolerance=profile.position_tolerance,
        grasp_position_tolerance=profile.grasp_position_tolerance,
        max_steps=args.max_steps,
        render_mode=render_mode,
        debug=args.debug,
    )


def config_from_profile(profile: LateralGraspTaskProfile, config_type):
    return config_type(
        panel_offset=list(profile.grasp_offset_body),
        env_id=profile.env_id,
        tool_z_rotation=profile.tool_z_rotation,
        rotation_mode=profile.rotation_mode,
        position_tolerance=profile.position_tolerance,
        grasp_position_tolerance=profile.grasp_position_tolerance,
    )


def run_lateral_planner(
    config,
    planner_type,
    success_label: str,
    *,
    realtime: bool = False,
) -> bool:
    return run_environment_planner(
        config,
        planner_type,
        success_label,
        realtime=realtime,
    )


def main_for_profile(
    profile: LateralGraspTaskProfile,
    config_type,
    planner_type,
    parse_float_list: Callable[[str], list[float]],
    argv: list[str] | None = None,
) -> int:
    parser = make_lateral_parser(profile, config_type, parse_float_list)
    args = parser.parse_args(argv)
    config = build_lateral_config(args, profile, config_type)
    return (
        0
        if run_lateral_planner(
            config,
            planner_type,
            profile.success_label,
            realtime=args.realtime,
        )
        else 1
    )
