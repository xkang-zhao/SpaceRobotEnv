import sys
import time
from pathlib import Path
from threading import Lock
import argparse

import numpy as np
from pynput import keyboard
import rerun as rr

# Add project roots
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import gymnasium as gym
import envs

from lerobot.datasets.lerobot_dataset import LeRobotDataset

TRAN_STEP = 0.003
# One env.step advances 0.05 s of simulation time.
FPS = 20
DEFAULT_CAM1_KEY = "third_left_camera"
DEFAULT_CAM2_KEY = "third_right_camera"
DEFAULT_CAM3_KEY = "left_wrist_camera"
KEYPAD_VK_MAP = {
    97: "1", 100: "4", 101: "5", 102: "6", 103: "7", 104: "8", 105: "9",
    65457: "1", 65460: "4", 65461: "5", 65462: "6", 65463: "7", 65464: "8", 65465: "9",
}

def normalize_key(key) -> str | None:
    if key == keyboard.Key.esc: return "esc"
    # Added backspace for dropping current episode
    if key == keyboard.Key.backspace: return "backspace" 
    if key == keyboard.Key.left: return "left"
    if key == keyboard.Key.right: return "right"

    if isinstance(key, keyboard.KeyCode):
        if key.char is not None:
            return key.char
        return KEYPAD_VK_MAP.get(key.vk)
    return None

def build_action(pressed_keys: set[str], gripper_signal: float) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    if "8" in pressed_keys: action[0] += TRAN_STEP
    if "5" in pressed_keys: action[0] -= TRAN_STEP
    if "6" in pressed_keys: action[1] += TRAN_STEP
    if "4" in pressed_keys: action[1] -= TRAN_STEP
    if "7" in pressed_keys: action[2] += TRAN_STEP
    if "9" in pressed_keys: action[2] -= TRAN_STEP
    action[6] = gripper_signal
    return action

def require_obs_keys(obs: dict, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in obs]
    if not missing:
        return

    available = ", ".join(sorted(obs.keys()))
    raise KeyError(
        f"Missing observation key(s): {missing}. Available keys: {available}"
    )

def main():
    parser = argparse.ArgumentParser(description="Keyboard Teleop for Data Collection (Pure LeRobot Dataset)")
    parser.add_argument("--env", type=str, default="SpaceUR10e-Cube-v0", help="Gym Env name")
    parser.add_argument("--repo_id", type=str, default="local/spaceur10e_keyboard_cube", help="LeRobot Dataset ID")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to record")
    parser.add_argument("--enable_rerun", action="store_true", help="Enable Rerun visualization")
    parser.add_argument("--task", type=str, default="Grab the object", help="Task description")
    parser.add_argument("--cam1_key", type=str, default=DEFAULT_CAM1_KEY, help="Observation key for dataset cam1")
    parser.add_argument("--cam2_key", type=str, default=DEFAULT_CAM2_KEY, help="Observation key for dataset cam2")
    parser.add_argument("--cam3_key", type=str, default=DEFAULT_CAM3_KEY, help="Observation key for dataset cam3")
    args = parser.parse_args()

    # Create environment
    env = gym.make(args.env, render_mode="human")
    obs, _ = env.reset()
    require_obs_keys(obs, ("ee_pose", "joint_pos", args.cam1_key, args.cam2_key, args.cam3_key))

    # Features map definition
    features = {
        "action": {"dtype": "float32", "shape": (7,), "names": ["dx", "dy", "dz", "wx", "wy", "wz", "gripper"]},
        "observation.state": {"dtype": "float32", "shape": (13,), "names": [
            "j1", "j2", "j3", "j4", "j5", "j6", "x", "y", "z", "qw", "qx", "qy", "qz"
        ]},
        "observation.images.cam1": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
        "observation.images.cam2": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
        "observation.images.cam3": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
    }

    print(f"Creating/Loading dataset {args.repo_id}...")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=PROJECT_ROOT / "data" / args.repo_id,
        fps=FPS,
        features=features,
        use_videos=True,
        # Set to larger values only if FPS drops significantly
        image_writer_processes=0, 
        image_writer_threads=8, 
        batch_encoding_size=1
    )

    if args.enable_rerun:
        rr.init("keyboard_teleop_collection", spawn=True)

    pressed_keys: set[str] = set()
    lock = Lock()
    state = {"running": True, "gripper_signal": 0.0, "drop_episode": False, "save_episode": False}

    def on_press(key):
        key_name = normalize_key(key)
        if key_name is None: return None
        with lock:
            if key_name == "esc":
                state["running"] = False
                return False
            if key_name == "backspace" or key_name == "left":
                state["drop_episode"] = True
            if key_name == "right":
                state["save_episode"] = True
            if key_name == "1" and key_name not in pressed_keys:
                state["gripper_signal"] = -1.0 if state["gripper_signal"] > 0.0 else 1.0
            pressed_keys.add(key_name)
        return None

    def on_release(key):
        key_name = normalize_key(key)
        if key_name is None: return None
        with lock:
            pressed_keys.discard(key_name)
        return None

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("================== RECORDING STARTED ==================")
    print("Controls:")
    print("8/5: x axis positive/negative")
    print("4/6: y axis negative/positive")
    print("7/9: z axis positive/negative")
    print("1: toggle gripper")
    print("Left Arrow / Backspace: DROP current episode and reset")
    print("Right Arrow: SUCCESS & SAVE current episode")
    print("ESC / Close Window: Quit")
    
    recorded_episodes = dataset.num_episodes

    try:
        while state["running"] and recorded_episodes < args.num_episodes:
            if env.unwrapped.viewer and not env.unwrapped.viewer.is_running():
                break

            with lock:
                action = build_action(set(pressed_keys), state["gripper_signal"])
                drop_requested = state["drop_episode"]
                save_requested = state["save_episode"]
                if drop_requested:
                    state["drop_episode"] = False
                if save_requested:
                    state["save_episode"] = False

            if drop_requested:
                print("--- Episode DROPPED! Buffer cleared. ---")
                dataset.clear_episode_buffer()
                obs, _ = env.reset()
                require_obs_keys(obs, ("ee_pose", "joint_pos", args.cam1_key, args.cam2_key, args.cam3_key))
                continue

            if save_requested:
                print(f"--- Task SUCCESS (Manual)! Saving episode {recorded_episodes + 1}... ---")
                dataset.save_episode()
                recorded_episodes += 1
                obs, _ = env.reset()
                require_obs_keys(obs, ("ee_pose", "joint_pos", args.cam1_key, args.cam2_key, args.cam3_key))
                continue

            # Record frame data BEFORE stepping (so obs matches action) 
            # Note: ee_pose is extracted directly for observation.state
            robot_state = np.concatenate([obs["joint_pos"], obs["ee_pose"]]).astype(np.float32)
            frame = {
                "action": action,
                "observation.state": robot_state,
                "observation.images.cam1": obs[args.cam1_key],
                "observation.images.cam2": obs[args.cam2_key],
                "observation.images.cam3": obs[args.cam3_key],
                "task": args.task
            }
            dataset.add_frame(frame)

            if args.enable_rerun:
                rr.log("action", rr.BarChart(action))
                rr.log("observation/images/cam1", rr.Image(obs[args.cam1_key]))
                rr.log("observation/images/cam2", rr.Image(obs[args.cam2_key]))
                rr.log("observation/images/cam3", rr.Image(obs[args.cam3_key]))

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = next_obs
            
            # Save episode if successfully grasped
            if terminated:
                print(f"Task SUCCESS! Saving episode {recorded_episodes + 1}...")
                dataset.save_episode()
                recorded_episodes += 1
                obs, _ = env.reset()
                require_obs_keys(obs, ("ee_pose", "joint_pos", args.cam1_key, args.cam2_key, args.cam3_key))
            # Drop episode if object flew away / timed out / failed
            elif truncated:
                print("Task Failed (truncated). Dropping episode...")
                dataset.clear_episode_buffer()
                obs, _ = env.reset()
                require_obs_keys(obs, ("ee_pose", "joint_pos", args.cam1_key, args.cam2_key, args.cam3_key))

            time.sleep(1.0 / FPS)

    except KeyboardInterrupt:
        print("Interrupted by user...")
    finally:
        state["running"] = False
        listener.stop()
        listener.join(timeout=1.0)
        
        print(f"Finalizing dataset... (Saved {recorded_episodes} episodes)")
        dataset.finalize()
        env.close()

if __name__ == "__main__":
    main()
