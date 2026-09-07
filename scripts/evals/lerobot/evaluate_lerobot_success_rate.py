import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from .lerobot_eval_common import (
        EvalConfig,
        run_serial,
        run_worker,
        split_episode_indices,
        validate_eval_config,
    )
except ImportError:
    from lerobot_eval_common import (
        EvalConfig,
        run_serial,
        run_worker,
        split_episode_indices,
        validate_eval_config,
    )


def run_multiprocessing(
    num_episodes: int,
    num_envs: int,
    config: EvalConfig,
    start_method: str,
):
    episode_buckets = split_episode_indices(num_episodes, num_envs)
    ctx = mp.get_context(start_method)
    results = []

    with ProcessPoolExecutor(max_workers=len(episode_buckets), mp_context=ctx) as executor:
        futures = [
            executor.submit(run_worker, worker_id, episode_indices, config)
            for worker_id, episode_indices in enumerate(episode_buckets)
        ]
        for future in as_completed(futures):
            results.extend(future.result())

    return sorted(results, key=lambda item: item.episode_index)


def parse_args() -> argparse.Namespace:
    start_methods = mp.get_all_start_methods()
    default_start_method = "spawn" if "spawn" in start_methods else start_methods[0]

    parser = argparse.ArgumentParser(
        description="Evaluate one local LeRobot policy checkpoint in a SpaceUR10e Gymnasium environment."
    )
    parser.add_argument("--model-path", required=True, help="Local pretrained_model directory or checkpoint path.")
    parser.add_argument(
        "--policy-type",
        default="pi0",
        choices=("pi0", "pi05", "smolvla", "act", "myvla"),
        help="Policy class preset to load from the local checkpoint.",
    )
    parser.add_argument("--policy-class", default=None, help="Optional custom policy class in module:ClassName format.")
    parser.add_argument("--device", default="cuda", help="Torch device for policy inference, such as cuda or cpu.")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps-per-episode", type=int, default=500, help="Maximum control steps per episode.")
    parser.add_argument("--action-chunk-size", type=int, default=1, help="Actions to execute before refreshing obs.")
    parser.add_argument("--task", default="Grasp this red cube in space.", help="Language task passed to the policy.")
    parser.add_argument("--robot-type", default="", help="Robot type string passed to LeRobot inference frame.")
    parser.add_argument("--env-id", default="SpaceUR10e-Cube-v0", help="Gymnasium environment id.")
    parser.add_argument(
        "--render-mode",
        choices=("human", "rgb_array", "none"),
        default="rgb_array",
        help="Use rgb_array/none for headless evaluation. human is best kept for single-env debugging.",
    )
    parser.add_argument("--scene-path", default=None, help="Optional MuJoCo scene XML override.")
    parser.add_argument("--arm-path", default=None, help="Optional robot arm XML override used by IK.")
    parser.add_argument("--frame-name", default=None, help="Optional IK frame name override.")
    parser.add_argument("--no-depth", dest="use_depth", action="store_false", help="Disable depth observation rendering.")
    parser.set_defaults(use_depth=True)
    parser.add_argument("--seed", type=int, default=None, help="Optional base seed for repeatable evaluation.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environment workers.")
    parser.add_argument(
        "--use-multiprocessing",
        action="store_true",
        help="Enable multi-process evaluation when --num-envs is greater than 1.",
    )
    parser.add_argument(
        "--mp-start-method",
        choices=tuple(start_methods),
        default=default_start_method,
        help="Multiprocessing start method. spawn is safest with CUDA and Windows.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be greater than 0.")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be greater than 0.")
    if args.use_multiprocessing and args.num_envs > 1 and args.render_mode == "human":
        raise ValueError("Multiprocessing with --render-mode human is not supported. Use --render-mode rgb_array or none.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    render_mode = None if args.render_mode == "none" else args.render_mode
    config = EvalConfig(
        model_path=os.path.abspath(args.model_path),
        policy_type=args.policy_type,
        policy_class=args.policy_class,
        device=args.device,
        task=args.task,
        robot_type=args.robot_type,
        action_chunk_size=args.action_chunk_size,
        max_steps_per_episode=args.max_steps_per_episode,
        render_mode=render_mode,
        env_id=args.env_id,
        scene_path=None if args.scene_path is None else os.path.abspath(args.scene_path),
        arm_path=None if args.arm_path is None else os.path.abspath(args.arm_path),
        frame_name=args.frame_name,
        use_depth=args.use_depth,
        seed=args.seed,
    )
    validate_eval_config(config)

    use_parallel = args.use_multiprocessing and args.num_envs > 1
    num_workers = min(args.num_envs, args.num_episodes)

    print(
        "Starting evaluation: "
        f"policy_type={args.policy_type}, model_path={args.model_path}, env_id={args.env_id}, "
        f"episodes={args.num_episodes}, max_steps={args.max_steps_per_episode}, "
        f"action_chunk_size={args.action_chunk_size}, device={args.device}, "
        f"multiprocessing={use_parallel}, num_envs={num_workers}, render_mode={args.render_mode}"
    )

    if use_parallel:
        results = run_multiprocessing(args.num_episodes, num_workers, config, args.mp_start_method)
    else:
        if args.num_envs > 1:
            print("Note: --num-envs is greater than 1, but --use-multiprocessing was not set. Running serially.")
        results = run_serial(args.num_episodes, config)

    success_count = sum(result.success for result in results)
    success_rate = success_count / len(results) if results else 0.0

    for result in results:
        distance_text = "nan" if result.final_distance is None else f"{result.final_distance:.4f}"
        print(
            f"Episode {result.episode_index}/{args.num_episodes} finished: "
            f"worker={result.worker_id}, reason={result.reason}, steps={result.steps}, "
            f"success={result.success}, final_distance={distance_text}"
        )

    print(
        "Evaluation summary: "
        f"episodes={len(results)}, successes={success_count}, failures={len(results) - success_count}, "
        f"success_rate={success_rate:.2%}"
    )


if __name__ == "__main__":
    main()
