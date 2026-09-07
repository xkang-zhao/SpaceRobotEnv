"""Standalone runner for the cube automatic grasp planner."""

from __future__ import annotations

import argparse
import time

from .cube import (
    FPS,
    CubeAutoGraspConfig,
    CubeAutoGraspPlanner,
    action_dict_to_vector,
    parse_float_list,
    require_obs_keys,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-grasp cube without LeRobot dataset recording."
    )
    parser.add_argument(
        "--env",
        default=CubeAutoGraspConfig.env_id,
        help="Gymnasium environment id.",
    )
    parser.add_argument(
        "--front_offset_world",
        type=parse_float_list,
        default=[-0.12, 0.0, 0.0],
    )
    parser.add_argument(
        "--grasp_offset_world",
        type=parse_float_list,
        default=[-0.02, 0.0, 0.0],
    )
    parser.add_argument(
        "--target_frame",
        choices=("pinch", "ee"),
        default=CubeAutoGraspConfig.target_frame,
        help="Which frame should reach the cube target.",
    )
    parser.add_argument(
        "--max_delta_xyz",
        type=parse_float_list,
        default=[0.01, 0.01, 0.01],
    )
    parser.add_argument("--motion_scale", type=float, default=0.3)
    parser.add_argument("--front_position_tolerance", type=float, default=0.03)
    parser.add_argument("--grasp_position_tolerance", type=float, default=0.015)
    parser.add_argument("--close_steps", type=int, default=25)
    parser.add_argument("--success_hold_steps", type=int, default=10)
    parser.add_argument("--episode_timeout_steps", type=int, default=400)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument(
        "--render_mode",
        default="human",
        choices=["none", "human", "rgb_array"],
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace the standalone demo at --fps.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def build_config_from_args(args: argparse.Namespace) -> CubeAutoGraspConfig:
    return CubeAutoGraspConfig(
        env_id=args.env,
        front_offset_world=args.front_offset_world,
        grasp_offset_world=args.grasp_offset_world,
        target_frame=args.target_frame,
        max_delta_xyz=args.max_delta_xyz,
        motion_scale=args.motion_scale,
        front_position_tolerance=args.front_position_tolerance,
        grasp_position_tolerance=args.grasp_position_tolerance,
        close_steps=args.close_steps,
        success_hold_steps=args.success_hold_steps,
        episode_timeout_steps=args.episode_timeout_steps,
        fps=args.fps,
        render_mode=args.render_mode,
        debug=args.debug,
    )


def run_planner(
    config: CubeAutoGraspConfig,
    *,
    realtime: bool = False,
) -> bool:
    import gymnasium as gym
    import envs  # noqa: F401  # Register Gymnasium environments.

    render_mode = None if config.render_mode == "none" else config.render_mode
    env = gym.make(config.env_id, render_mode=render_mode)
    planner = CubeAutoGraspPlanner(config)
    planner.bind_env_base(env.unwrapped)
    obs, _ = env.reset()
    require_obs_keys(obs, ("ee_pose", "target_pose"))

    try:
        while not planner.should_end_episode() and not planner.should_rerecord_episode():
            viewer = getattr(env.unwrapped, "viewer", None)
            if viewer and not viewer.is_running():
                break

            started_at = time.perf_counter()
            action_dict = planner.get_action(
                obs,
                episode_success=bool(env.unwrapped.is_success()),
            )
            action = action_dict_to_vector(action_dict)
            if config.debug and planner.step_index % 25 == 0:
                print(
                    f"[{planner.step_index:04d}] "
                    f"stage={planner.stage}, action={action}"
                )

            obs, _, terminated, truncated, info = env.step(action)
            require_obs_keys(obs, ("ee_pose", "target_pose"))
            if terminated or info.get("is_success", False):
                print(
                    f"[{planner.step_index:04d}] "
                    "SUCCESS: cube grasp confirmed."
                )
                return True
            if truncated:
                distance = info.get("distance", float("nan"))
                print(
                    "FAILED: environment truncated, "
                    f"distance={distance:.4f}"
                )
                return False
            if realtime:
                elapsed = time.perf_counter() - started_at
                time.sleep(max(0.0, 1.0 / config.fps - elapsed))

        if planner.should_end_episode():
            print(f"[{planner.step_index:04d}] SUCCESS: cube grasp confirmed.")
            return True
        print(
            f"FAILED: planner stopped at stage={planner.stage}, "
            f"step={planner.step_index}."
        )
        return False
    except KeyboardInterrupt:
        print("Interrupted.")
        return False
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    config = build_config_from_args(args)
    return 0 if run_planner(config, realtime=args.realtime) else 1
