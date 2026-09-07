from __future__ import annotations

import argparse
import os

try:
    from .lerobot_eval_common import EvalConfig, LeRobotPolicyRunner, validate_eval_config
    from .lerobot_zmq_protocol import (
        INFER,
        PING,
        PONG,
        SHUTDOWN,
        make_action_reply,
        make_error_reply,
    )
except ImportError:
    from lerobot_eval_common import EvalConfig, LeRobotPolicyRunner, validate_eval_config
    from lerobot_zmq_protocol import (
        INFER,
        PING,
        PONG,
        SHUTDOWN,
        make_action_reply,
        make_error_reply,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a ZMQ LeRobot policy inference server.")
    parser.add_argument("--bind", default="tcp://127.0.0.1:5555", help="ZMQ REP bind address.")
    parser.add_argument("--model-path", required=True, help="Local pretrained_model directory or checkpoint path.")
    parser.add_argument(
        "--policy-type",
        default="pi0",
        choices=("pi0", "pi05", "smolvla", "act", "myvla"),
        help="Policy class preset to load from the local checkpoint.",
    )
    parser.add_argument("--policy-class", default=None, help="Optional custom policy class in module:ClassName format.")
    parser.add_argument("--device", default="cuda", help="Torch device for policy inference, such as cuda or cpu.")
    parser.add_argument("--task", default="Grasp this red cube in space.", help="Default language task.")
    parser.add_argument("--robot-type", default="", help="Default LeRobot robot type string.")
    parser.add_argument("--action-chunk-size", type=int, default=8, help="Maximum number of actions per reply.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvalConfig(
        model_path=os.path.abspath(args.model_path),
        policy_type=args.policy_type,
        policy_class=args.policy_class,
        device=args.device,
        task=args.task,
        robot_type=args.robot_type,
        action_chunk_size=args.action_chunk_size,
    )
    validate_eval_config(config)

    import zmq

    runner = LeRobotPolicyRunner(config)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(args.bind)
    print(f"LeRobot ZMQ server listening on {args.bind}")

    try:
        while True:
            request = socket.recv_pyobj()
            request_type = request.get("type")

            if request_type == PING:
                socket.send_pyobj({"type": PONG})
                continue
            if request_type == SHUTDOWN:
                socket.send_pyobj({"type": "shutdown_ack"})
                break
            if request_type != INFER:
                socket.send_pyobj(make_error_reply(f"Unsupported request type: {request_type!r}"))
                continue

            try:
                actions = runner.select_action_chunk(
                    request["observation"],
                    task=request.get("task"),
                    robot_type=request.get("robot_type"),
                    action_chunk_size=request.get("action_chunk_size"),
                )
                socket.send_pyobj(make_action_reply(actions))
            except Exception as exc:
                socket.send_pyobj(make_error_reply(str(exc)))
    finally:
        socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
