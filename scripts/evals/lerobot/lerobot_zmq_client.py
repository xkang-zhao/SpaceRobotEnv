from __future__ import annotations

import argparse

try:
    from .lerobot_eval_common import EvalConfig, make_env
    from .lerobot_zmq_protocol import make_infer_request, parse_action_reply
except ImportError:
    from lerobot_eval_common import EvalConfig, make_env
    from lerobot_zmq_protocol import make_infer_request, parse_action_reply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MuJoCo evaluation client against a ZMQ policy server.")
    parser.add_argument("--connect", default="tcp://127.0.0.1:5555", help="ZMQ REQ server address.")
    parser.add_argument("--env-id", default="SpaceUR10e-Cube-v0", help="Gymnasium environment id.")
    parser.add_argument(
        "--render-mode",
        choices=("human", "rgb_array", "none"),
        default="human",
        help="MuJoCo render mode. Use none for headless client execution.",
    )
    parser.add_argument("--scene-path", default=None, help="Optional MuJoCo scene XML override.")
    parser.add_argument("--arm-path", default=None, help="Optional robot arm XML override used by IK.")
    parser.add_argument("--frame-name", default=None, help="Optional IK frame name override.")
    parser.add_argument("--no-depth", dest="use_depth", action="store_false", help="Disable depth observations.")
    parser.set_defaults(use_depth=True)
    parser.add_argument("--task", default=None, help="Optional per-request task override.")
    parser.add_argument("--robot-type", default=None, help="Optional per-request robot type override.")
    parser.add_argument("--action-chunk-size", type=int, default=None, help="Optional per-request action chunk size.")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--max-steps-per-episode", type=int, default=500, help="Maximum control steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Optional base seed for repeatable evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = None if args.render_mode == "none" else args.render_mode
    env_config = EvalConfig(
        model_path="unused-by-zmq-client",
        render_mode=render_mode,
        env_id=args.env_id,
        scene_path=args.scene_path,
        arm_path=args.arm_path,
        frame_name=args.frame_name,
        use_depth=args.use_depth,
        seed=args.seed,
    )

    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(args.connect)
    env = make_env(env_config)

    success_count = 0
    completed_episodes = 0
    try:
        for episode_idx in range(1, args.num_episodes + 1):
            obs, _ = env.reset(seed=None if args.seed is None else args.seed + episode_idx)
            steps = 0
            success = False
            reason = "max_steps"

            while steps < args.max_steps_per_episode:
                socket.send_pyobj(
                    make_infer_request(
                        observation=obs,
                        task=args.task,
                        robot_type=args.robot_type,
                        action_chunk_size=args.action_chunk_size,
                    )
                )
                actions = parse_action_reply(socket.recv_pyobj())

                for env_action in actions:
                    obs, _reward, terminated, truncated, info = env.step(env_action)
                    steps += 1

                    if terminated or info.get("is_success", False):
                        success = True
                        reason = "success"
                        break
                    if truncated:
                        reason = "truncated"
                        break
                    if steps >= args.max_steps_per_episode:
                        break

                if success or reason == "truncated" or steps >= args.max_steps_per_episode:
                    break

            completed_episodes += 1
            success_count += int(success)
            print(
                f"Episode {episode_idx}/{args.num_episodes} finished: "
                f"reason={reason}, steps={steps}, success={success}"
            )
    finally:
        env.close()
        socket.close(linger=0)
        context.term()

    success_rate = success_count / completed_episodes if completed_episodes else 0.0
    print(
        f"Client summary: episodes={completed_episodes}, successes={success_count}, "
        f"success_rate={success_rate:.2%}"
    )


if __name__ == "__main__":
    main()
