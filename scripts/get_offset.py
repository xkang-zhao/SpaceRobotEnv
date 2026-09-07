import argparse
import os
import sys
import time
from pathlib import Path
from threading import Lock

import numpy as np
from pynput import keyboard


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.chdir(PROJECT_ROOT)

import gymnasium as gym
import envs  # noqa: F401


TRAN_STEP = 0.003
ROT_STEP = 0.05
FPS = 20
KEYPAD_VK_MAP = {
    97: "1",
    100: "4",
    101: "5",
    102: "6",
    103: "7",
    104: "8",
    105: "9",
    65457: "1",
    65460: "4",
    65461: "5",
    65462: "6",
    65463: "7",
    65464: "8",
    65465: "9",
}


def normalize_key(key) -> str | None:
    if key == keyboard.Key.esc:
        return "esc"

    if isinstance(key, keyboard.KeyCode):
        if key.char is not None:
            return key.char.lower()
        return KEYPAD_VK_MAP.get(key.vk)

    return None


def build_action(pressed_keys: set[str], is_rotation_mode: bool) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)

    if is_rotation_mode:
        if "8" in pressed_keys:
            action[3] += ROT_STEP
        if "5" in pressed_keys:
            action[3] -= ROT_STEP

        if "6" in pressed_keys:
            action[4] += ROT_STEP
        if "4" in pressed_keys:
            action[4] -= ROT_STEP

        if "7" in pressed_keys:
            action[5] += ROT_STEP
        if "9" in pressed_keys:
            action[5] -= ROT_STEP
    else:
        if "8" in pressed_keys:
            action[0] += TRAN_STEP
        if "5" in pressed_keys:
            action[0] -= TRAN_STEP

        if "6" in pressed_keys:
            action[1] += TRAN_STEP
        if "4" in pressed_keys:
            action[1] -= TRAN_STEP

        if "7" in pressed_keys:
            action[2] += TRAN_STEP
        if "9" in pressed_keys:
            action[2] -= TRAN_STEP

    return action


def format_xyz(values: np.ndarray) -> str:
    return np.array2string(np.asarray(values, dtype=float), precision=6, suppress_small=False)


def main():
    parser = argparse.ArgumentParser(
        description="Keyboard teleop for printing satellite center and robot end-effector coordinates."
    )
    parser.add_argument("--env", type=str, default="SpaceUR10e-Satellite-v0", help="Gymnasium environment id.")
    args = parser.parse_args()

    env = gym.make(args.env, render_mode="human")
    obs, _ = env.reset()

    pressed_keys: set[str] = set()
    lock = Lock()
    state = {"running": True, "print_and_exit": False, "rotation_mode": False}

    def on_press(key):
        key_name = normalize_key(key)
        if key_name is None:
            return None

        with lock:
            if key_name == "esc":
                state["running"] = False
                return False
            if key_name == "d":
                state["print_and_exit"] = True
                state["running"] = False
                return False
            if key_name == "1":
                state["rotation_mode"] = not state["rotation_mode"]
                mode_name = "旋转模式" if state["rotation_mode"] else "平移模式"
                print(f"已切换到 {mode_name}")
                return None
            pressed_keys.add(key_name)

        return None

    def on_release(key):
        key_name = normalize_key(key)
        if key_name is None:
            return None

        with lock:
            pressed_keys.discard(key_name)

        return None

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("键盘控制已启动。")
    print("1: 切换 平移模式 / 旋转模式")
    print("平移模式: 8/5→末端x, 6/4→末端y, 7/9→末端z")
    print("旋转模式: 8/5→绕末端x轴, 6/4→绕末端y轴, 7/9→绕末端z轴")
    print("D: 打印卫星中心世界坐标和机器人末端坐标，然后退出")
    print("ESC 或关闭 MuJoCo 窗口: 退出")

    try:
        while True:
            if env.unwrapped.viewer and not env.unwrapped.viewer.is_running():
                break

            with lock:
                should_print = bool(state["print_and_exit"])
                running = bool(state["running"])
                is_rotation = bool(state["rotation_mode"])
                action = build_action(set(pressed_keys), is_rotation)

            if should_print:
                break
            if not running:
                break

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

            time.sleep(1.0 / FPS)

        with lock:
            should_print = bool(state["print_and_exit"])

        if should_print:
            satellite_xyz = np.asarray(obs["target_pose"][:3], dtype=float)
            ee_xyz = np.asarray(obs["ee_pose"][:3], dtype=float)
            world_offset = ee_xyz - satellite_xyz
            sat_rot = np.asarray(
                env.unwrapped.data.xmat[env.unwrapped.target_body_id],
                dtype=float,
            ).reshape(3, 3)
            body_offset = sat_rot.T @ world_offset
            print(f"卫星中心点世界坐标: {format_xyz(satellite_xyz)}")
            print(f"机器人末端坐标: {format_xyz(ee_xyz)}")
            print(f"世界坐标系 offset: {format_xyz(world_offset)}")
            print(f"卫星 body 坐标系 offset: {format_xyz(body_offset)}")
    except KeyboardInterrupt:
        print("收到 Ctrl+C，正在退出。")
    finally:
        state["running"] = False
        listener.stop()
        listener.join(timeout=1.0)
        env.close()


if __name__ == "__main__":
    main()
