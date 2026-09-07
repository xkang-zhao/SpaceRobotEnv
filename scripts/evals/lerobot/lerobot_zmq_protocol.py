from __future__ import annotations

from typing import Any

import numpy as np


INFER = "infer"
PING = "ping"
PONG = "pong"
SHUTDOWN = "shutdown"
ACTION_CHUNK = "action_chunk"
ERROR = "error"


def make_infer_request(
    observation: dict[str, Any],
    task: str | None = None,
    robot_type: str | None = None,
    action_chunk_size: int | None = None,
) -> dict[str, Any]:
    return {
        "type": INFER,
        "observation": observation,
        "task": task,
        "robot_type": robot_type,
        "action_chunk_size": action_chunk_size,
    }


def make_action_reply(actions: Any) -> dict[str, Any]:
    actions_array = np.asarray(actions, dtype=np.float32)
    if actions_array.ndim != 2 or actions_array.shape[1] != 7:
        raise ValueError(f"Action chunk must have shape (N, 7), got {actions_array.shape}.")
    return {
        "type": ACTION_CHUNK,
        "num_actions": int(actions_array.shape[0]),
        "actions": actions_array,
    }


def parse_action_reply(reply: dict[str, Any]) -> np.ndarray:
    if reply.get("type") == ERROR:
        raise RuntimeError(reply.get("message", "ZMQ policy server returned an error."))
    if reply.get("type") != ACTION_CHUNK:
        raise ValueError(f"Expected action_chunk reply, got {reply.get('type')!r}.")
    return np.asarray(reply["actions"], dtype=np.float32)


def make_error_reply(message: str) -> dict[str, str]:
    return {"type": ERROR, "message": message}
