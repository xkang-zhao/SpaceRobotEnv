"""macOS-safe offset calibration using MuJoCo viewer key polling.

Run with:
    mjpython scripts/get_offet_mac.py --env SpaceUR10e-Satellite2-v0

This avoids pynput because pynput can crash under mjpython/AppKit on macOS.
"""

from __future__ import annotations

import argparse
import math
import os
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.chdir(PROJECT_ROOT)

import envs  # noqa: E402,F401  # Registers Gymnasium environments.


TRAN_STEP = 0.003
ROT_STEP = 0.05
FPS = 20

KEY_ESCAPE = 256
KEY_D = ord("D")
KEY_MINUS = ord("-")
KEY_KP_SUBTRACT = 333
KEY_KP_4 = 324
KEY_KP_5 = 325
KEY_KP_6 = 326
KEY_KP_7 = 327
KEY_KP_8 = 328
KEY_KP_9 = 329

MAC_KEY_ESCAPE = 53
MAC_KEY_D = 2
MAC_KEY_MINUS = 27
MAC_KEY_4 = 21
MAC_KEY_5 = 23
MAC_KEY_6 = 22
MAC_KEY_7 = 26
MAC_KEY_8 = 28
MAC_KEY_9 = 25
MAC_KEY_KP_SUBTRACT = 78
MAC_KEY_KP_4 = 86
MAC_KEY_KP_5 = 87
MAC_KEY_KP_6 = 88
MAC_KEY_KP_7 = 89
MAC_KEY_KP_8 = 91
MAC_KEY_KP_9 = 92

KEY_TO_NAME = {
    ord("4"): "4",
    ord("5"): "5",
    ord("6"): "6",
    ord("7"): "7",
    ord("8"): "8",
    ord("9"): "9",
    KEY_KP_4: "4",
    KEY_KP_5: "5",
    KEY_KP_6: "6",
    KEY_KP_7: "7",
    KEY_KP_8: "8",
    KEY_KP_9: "9",
}

MAC_KEY_TO_NAME = {
    MAC_KEY_4: "4",
    MAC_KEY_5: "5",
    MAC_KEY_6: "6",
    MAC_KEY_7: "7",
    MAC_KEY_8: "8",
    MAC_KEY_9: "9",
    MAC_KEY_KP_4: "4",
    MAC_KEY_KP_5: "5",
    MAC_KEY_KP_6: "6",
    MAC_KEY_KP_7: "7",
    MAC_KEY_KP_8: "8",
    MAC_KEY_KP_9: "9",
}

MOVEMENT_KEYS = {"4", "5", "6", "7", "8", "9"}
TOGGLE_KEY_CODES = {KEY_MINUS, KEY_KP_SUBTRACT}
MAC_TOGGLE_KEY_CODES = {MAC_KEY_MINUS, MAC_KEY_KP_SUBTRACT}


def calibration_action_from_keys(keys: set[str], is_rotation_mode: bool) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    if is_rotation_mode:
        if "8" in keys:
            action[3] += ROT_STEP
        if "5" in keys:
            action[3] -= ROT_STEP
        if "6" in keys:
            action[4] += ROT_STEP
        if "4" in keys:
            action[4] -= ROT_STEP
        if "7" in keys:
            action[5] += ROT_STEP
        if "9" in keys:
            action[5] -= ROT_STEP
    else:
        if "8" in keys:
            action[0] += TRAN_STEP
        if "5" in keys:
            action[0] -= TRAN_STEP
        if "6" in keys:
            action[1] += TRAN_STEP
        if "4" in keys:
            action[1] -= TRAN_STEP
        if "7" in keys:
            action[2] += TRAN_STEP
        if "9" in keys:
            action[2] -= TRAN_STEP
    return action


def movement_action_from_keys(keys: set[str]) -> np.ndarray:
    return calibration_action_from_keys(keys, is_rotation_mode=False)


def format_xyz(values: np.ndarray) -> str:
    return np.array2string(np.asarray(values, dtype=float), precision=6, suppress_small=False)


def compute_offsets(satellite_xyz: np.ndarray, ee_xyz: np.ndarray, sat_rot: np.ndarray):
    world_offset = np.asarray(ee_xyz, dtype=float) - np.asarray(satellite_xyz, dtype=float)
    body_offset = np.asarray(sat_rot, dtype=float).reshape(3, 3).T @ world_offset
    return world_offset, body_offset


def rotation_matrix_to_rpy(rot: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rot, dtype=float).reshape(3, 3)
    pitch = math.asin(-float(matrix[2][0]))

    if abs(math.cos(pitch)) > 1e-6:
        yaw = math.atan2(float(matrix[1][0]), float(matrix[0][0]))
        roll = math.atan2(float(matrix[2][1]), float(matrix[2][2]))
    else:
        yaw = 0.0
        roll = math.atan2(-float(matrix[0][1]), float(matrix[1][1]))

    return np.asarray([roll, pitch, yaw], dtype=float)


def compute_relative_rotation(target_rot: np.ndarray, ee_rot: np.ndarray) -> np.ndarray:
    target_matrix = np.asarray(target_rot, dtype=float).reshape(3, 3)
    ee_matrix = np.asarray(ee_rot, dtype=float).reshape(3, 3)
    return target_matrix.T @ ee_matrix


class OffsetCalibrationState:
    def __init__(self):
        self.running = True
        self.print_and_exit = False
        self.rotation_mode = False
        self._pressed_keys: set[str] = set()
        self._pending_action = np.zeros(7, dtype=np.float32)
        self._toggle_key_down = False

    def handle_key(self, key: int) -> None:
        if key == KEY_ESCAPE:
            self.running = False
            return
        if key == KEY_D:
            self.print_and_exit = True
            self.running = False
            return

        if key in TOGGLE_KEY_CODES:
            self.toggle_rotation_mode()
            self._toggle_key_down = True
            return

        key_name = KEY_TO_NAME.get(key)
        if key_name in MOVEMENT_KEYS:
            self._pending_action += calibration_action_from_keys({key_name}, self.rotation_mode)

    def toggle_rotation_mode(self) -> None:
        self.rotation_mode = not self.rotation_mode
        mode_name = "旋转模式" if self.rotation_mode else "平移模式"
        print(f"已切换到 {mode_name}")

    def set_toggle_key_pressed(self, pressed: bool) -> None:
        if pressed and not self._toggle_key_down:
            self.toggle_rotation_mode()
        self._toggle_key_down = pressed

    def set_pressed_keys(self, keys: set[str]) -> None:
        self._pressed_keys = set(keys) & MOVEMENT_KEYS

    def consume_action(self) -> np.ndarray:
        action = calibration_action_from_keys(self._pressed_keys, self.rotation_mode) + self._pending_action
        self._pending_action[:] = 0.0
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


def poll_pressed_keys(viewer, state: OffsetCalibrationState) -> None:
    window = _viewer_glfw_window(viewer)
    if window is None:
        poll_quartz_pressed_keys(state)
        return

    if glfw.get_key(window, KEY_ESCAPE) == glfw.PRESS:
        state.running = False
        return
    if glfw.get_key(window, KEY_D) == glfw.PRESS:
        state.print_and_exit = True
        state.running = False
        return

    toggle_pressed = any(glfw.get_key(window, key_code) in (glfw.PRESS, glfw.REPEAT) for key_code in TOGGLE_KEY_CODES)
    state.set_toggle_key_pressed(toggle_pressed)

    pressed = set()
    for key_code, key_name in KEY_TO_NAME.items():
        if glfw.get_key(window, key_code) in (glfw.PRESS, glfw.REPEAT):
            pressed.add(key_name)
    state.set_pressed_keys(pressed)


def poll_quartz_pressed_keys(state: OffsetCalibrationState, key_state_getter=None) -> bool:
    if Quartz is None and key_state_getter is None:
        return False

    source = Quartz.kCGEventSourceStateHIDSystemState if Quartz is not None else None
    getter = key_state_getter or Quartz.CGEventSourceKeyState

    if getter(source, MAC_KEY_ESCAPE):
        state.running = False
        return True
    if getter(source, MAC_KEY_D):
        state.print_and_exit = True
        state.running = False
        return True

    toggle_pressed = any(getter(source, key_code) for key_code in MAC_TOGGLE_KEY_CODES)
    state.set_toggle_key_pressed(toggle_pressed)

    pressed = set()
    for key_code, key_name in MAC_KEY_TO_NAME.items():
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
    parser = argparse.ArgumentParser(
        description="macOS-safe keyboard teleop for printing satellite and end-effector offsets."
    )
    parser.add_argument("--env", type=str, default="SpaceUR10e-Satellite-v0", help="Gymnasium environment id.")
    return parser


def print_offsets(env, obs) -> None:
    satellite_xyz = np.asarray(obs["target_pose"][:3], dtype=float)
    satellite_quat = np.asarray(obs["target_pose"][3:7], dtype=float)
    ee_xyz = np.asarray(obs["ee_pose"][:3], dtype=float)
    ee_quat = np.asarray(obs["ee_pose"][3:7], dtype=float)
    sat_rot = np.asarray(
        env.unwrapped.data.xmat[env.unwrapped.target_body_id],
        dtype=float,
    ).reshape(3, 3)
    ee_rot = np.asarray(
        env.unwrapped.data.site_xmat[env.unwrapped.ee_site_id],
        dtype=float,
    ).reshape(3, 3)
    relative_rot = compute_relative_rotation(sat_rot, ee_rot)
    world_offset, body_offset = compute_offsets(satellite_xyz, ee_xyz, sat_rot)
    print(f"卫星中心点世界坐标: {format_xyz(satellite_xyz)}")
    print(f"机器人末端坐标: {format_xyz(ee_xyz)}")
    print(f"世界坐标系 offset: {format_xyz(world_offset)}")
    print(f"卫星 body 坐标系 offset: {format_xyz(body_offset)}")
    print(f"卫星姿态 quat(wxyz): {format_xyz(satellite_quat)}")
    print(f"机器人末端姿态 quat(wxyz): {format_xyz(ee_quat)}")
    print(f"卫星姿态 RPY(rad): {format_xyz(rotation_matrix_to_rpy(sat_rot))}")
    print(f"机器人末端姿态 RPY(rad): {format_xyz(rotation_matrix_to_rpy(ee_rot))}")
    print(f"相对姿态 RPY(rad, satellite->ee): {format_xyz(rotation_matrix_to_rpy(relative_rot))}")


def main() -> None:
    args = make_parser().parse_args()

    env = gym.make(args.env, render_mode=None)
    obs, _ = env.reset()
    state = OffsetCalibrationState()
    viewer = mujoco.viewer.launch_passive(
        env.unwrapped.model,
        env.unwrapped.data,
        key_callback=state.handle_key,
    )
    apply_camera(viewer, args.env)

    print("macOS 标定控制已启动。")
    print("-: 切换 平移模式 / 旋转模式")
    print("平移模式: 8/5→末端x, 6/4→末端y, 7/9→末端z")
    print("旋转模式: 8/5→绕末端x轴, 6/4→绕末端y轴, 7/9→绕末端z轴")
    print("D: 打印 offset 后退出")
    print("ESC 或关闭 MuJoCo 窗口: 退出")
    print("移动键支持长按连续输入。")

    try:
        camera_apply_count = 0
        while state.running and viewer.is_running():
            poll_pressed_keys(viewer, state)
            if state.print_and_exit:
                break

            action = state.consume_action()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

            if camera_apply_count < 5:
                apply_camera(viewer, args.env)
                camera_apply_count += 1
            viewer.sync()
            time.sleep(1.0 / FPS)

        if state.print_and_exit:
            print_offsets(env, obs)
    except KeyboardInterrupt:
        print("收到 Ctrl+C，正在退出。")
    finally:
        if viewer.is_running():
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
