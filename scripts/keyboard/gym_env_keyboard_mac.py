"""macOS-safe keyboard teleoperation using MuJoCo viewer key callbacks.

Run with:
    mjpython scripts/keyboard/gym_env_keyboard_mac.py

This avoids pynput because pynput's macOS backend can crash inside mjpython's
AppKit process when it queries input sources from a background thread.
"""

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import glfw
import mujoco
import mujoco.viewer
import numpy as np

try:
    import Quartz
except ImportError:  # pragma: no cover - non-macOS fallback
    Quartz = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import envs  # noqa: E402,F401  # Registers Gymnasium environments.


TRAN_STEP = 0.003
FPS = 20

KEY_ESCAPE = 256

# GLFW keypad key codes.
KEY_KP_1 = 321
KEY_KP_4 = 324
KEY_KP_5 = 325
KEY_KP_6 = 326
KEY_KP_7 = 327
KEY_KP_8 = 328
KEY_KP_9 = 329

# macOS virtual key codes used by Quartz.CGEventSourceKeyState.
MAC_KEY_ESCAPE = 53
MAC_KEY_1 = 18
MAC_KEY_4 = 21
MAC_KEY_5 = 23
MAC_KEY_6 = 22
MAC_KEY_7 = 26
MAC_KEY_8 = 28
MAC_KEY_9 = 25
MAC_KEY_KP_1 = 83
MAC_KEY_KP_4 = 86
MAC_KEY_KP_5 = 87
MAC_KEY_KP_6 = 88
MAC_KEY_KP_7 = 89
MAC_KEY_KP_8 = 91
MAC_KEY_KP_9 = 92

KEY_TO_NAME = {
    ord("1"): "1",
    ord("4"): "4",
    ord("5"): "5",
    ord("6"): "6",
    ord("7"): "7",
    ord("8"): "8",
    ord("9"): "9",
    KEY_KP_1: "1",
    KEY_KP_4: "4",
    KEY_KP_5: "5",
    KEY_KP_6: "6",
    KEY_KP_7: "7",
    KEY_KP_8: "8",
    KEY_KP_9: "9",
}

MAC_KEY_TO_NAME = {
    MAC_KEY_1: "1",
    MAC_KEY_4: "4",
    MAC_KEY_5: "5",
    MAC_KEY_6: "6",
    MAC_KEY_7: "7",
    MAC_KEY_8: "8",
    MAC_KEY_9: "9",
    MAC_KEY_KP_1: "1",
    MAC_KEY_KP_4: "4",
    MAC_KEY_KP_5: "5",
    MAC_KEY_KP_6: "6",
    MAC_KEY_KP_7: "7",
    MAC_KEY_KP_8: "8",
    MAC_KEY_KP_9: "9",
}

MOVEMENT_KEYS = {"4", "5", "6", "7", "8", "9"}


def movement_action_from_keys(keys: set[str]) -> np.ndarray:
    translation = np.zeros(3, dtype=np.float32)
    if "8" in keys:
        translation[0] += TRAN_STEP
    if "5" in keys:
        translation[0] -= TRAN_STEP
    if "6" in keys:
        translation[1] += TRAN_STEP
    if "4" in keys:
        translation[1] -= TRAN_STEP
    if "7" in keys:
        translation[2] += TRAN_STEP
    if "9" in keys:
        translation[2] -= TRAN_STEP
    return translation


class KeyboardTeleopState:
    """State updated by MuJoCo's viewer key callback and consumed by the loop."""

    def __init__(self):
        self.running = True
        self.gripper_signal = 0.0
        self._pressed_keys: set[str] = set()
        self._pending_translation = np.zeros(3, dtype=np.float32)

    def handle_key(self, key: int) -> None:
        if key == KEY_ESCAPE:
            self.running = False
            return

        key_name = KEY_TO_NAME.get(key)
        if key_name is None:
            return

        if key_name == "1":
            self.gripper_signal = -1.0 if self.gripper_signal > 0.0 else 1.0
            print(f"夹爪翻转信号: {self.gripper_signal:+.0f}")
            return

        if key_name in MOVEMENT_KEYS:
            self._pending_translation += movement_action_from_keys({key_name})

    def set_pressed_keys(self, keys: set[str]) -> None:
        self._pressed_keys = set(keys) & MOVEMENT_KEYS

    def consume_action(self) -> np.ndarray:
        action = np.zeros(7, dtype=np.float32)
        action[:3] = movement_action_from_keys(self._pressed_keys) + self._pending_translation
        action[6] = self.gripper_signal
        self._pending_translation[:] = 0.0
        return action


def _viewer_glfw_window(viewer):
    for attr in ("_window", "window", "_glfw_window", "glfw_window"):
        window = getattr(viewer, attr, None)
        if window is not None:
            return window

    sim_ref = getattr(viewer, "_sim", None)
    sim = sim_ref() if callable(sim_ref) else sim_ref
    if sim is None:
        return None

    for attr in ("_window", "window", "_glfw_window", "glfw_window"):
        window = getattr(sim, attr, None)
        if window is not None:
            return window
    return None


def poll_pressed_keys(viewer, state: KeyboardTeleopState) -> None:
    window = _viewer_glfw_window(viewer)
    if window is None:
        poll_quartz_pressed_keys(state)
        return

    if glfw.get_key(window, KEY_ESCAPE) == glfw.PRESS:
        state.running = False
        return

    pressed = set()
    for key_code, key_name in KEY_TO_NAME.items():
        if key_name not in MOVEMENT_KEYS:
            continue
        if glfw.get_key(window, key_code) in (glfw.PRESS, glfw.REPEAT):
            pressed.add(key_name)
    state.set_pressed_keys(pressed)


def poll_quartz_pressed_keys(state: KeyboardTeleopState, key_state_getter=None) -> bool:
    if Quartz is None and key_state_getter is None:
        return False

    source = Quartz.kCGEventSourceStateHIDSystemState if Quartz is not None else None
    getter = key_state_getter or Quartz.CGEventSourceKeyState

    if getter(source, MAC_KEY_ESCAPE):
        state.running = False
        return True

    pressed = set()
    for key_code, key_name in MAC_KEY_TO_NAME.items():
        if key_name not in MOVEMENT_KEYS:
            continue
        if getter(source, key_code):
            pressed.add(key_name)
    state.set_pressed_keys(pressed)
    return True


def apply_camera(viewer, env_id: str) -> None:
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.fixedcamid = -1
        viewer.cam.trackbodyid = -1
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = -60
        viewer.cam.elevation = -30
        viewer.cam.lookat[:] = (1.6, 0.0, 2.65)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="macOS-safe MuJoCo viewer keyboard teleoperation")
    parser.add_argument("--env", type=str, default="SpaceUR10e-Cube-v0", help="Gym environment name")
    return parser


def main() -> None:
    args = make_parser().parse_args()

    env = gym.make(args.env, render_mode=None)
    obs, _ = env.reset()

    state = KeyboardTeleopState()
    viewer = mujoco.viewer.launch_passive(
        env.unwrapped.model,
        env.unwrapped.data,
        key_callback=state.handle_key,
    )
    apply_camera(viewer, args.env)

    print("Gym 环境已启动，macOS MuJoCo viewer 键盘控制如下：")
    print("8/5: x 轴正/负方向")
    print("4/6: y 轴负/正方向")
    print("7/9: z 轴正/负方向")
    print("1: 翻转夹爪开合")
    print("ESC 或关闭 MuJoCo 窗口: 退出")
    print("移动键支持长按连续输入。")
    print("请先点击 MuJoCo viewer 窗口，让窗口获得键盘焦点。")

    try:
        camera_apply_count = 0
        while state.running and viewer.is_running():
            poll_pressed_keys(viewer, state)
            action = state.consume_action()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                reason = "success" if terminated else "truncated"
                print(
                    f"环境已重置: reason={reason}, "
                    f"distance={info.get('distance', 0.0):.4f}, reward={reward:.4f}"
                )
                obs, _ = env.reset()

            if camera_apply_count < 5:
                apply_camera(viewer, args.env)
                camera_apply_count += 1
            viewer.sync()
            time.sleep(1.0 / FPS)
    except KeyboardInterrupt:
        print("收到退出信号，正在关闭环境...")
    finally:
        if viewer.is_running():
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
