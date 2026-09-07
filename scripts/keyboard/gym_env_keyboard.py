# 标准库：系统路径、时间控制、路径处理、线程锁
import sys
import time
from pathlib import Path
from threading import Lock

# 第三方库：数值计算、键盘监听
import numpy as np
from pynput import keyboard

# 将项目根目录和源码目录加入 sys.path，确保 import 正常
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# 导入 Gym 环境注册
import gymnasium as gym
import envs
import argparse

# 每次按键的平动步长（米）
TRAN_STEP = 0.003
# 控制循环帧率（Hz）
FPS = 20
# 小键盘虚拟键码 -> 数字字符映射表
# 97~105: 主键盘小键盘 1~9，65457~65465: NumLock 下的小键盘 1~9
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
    """将 pynput 按键事件统一转换成可识别的字符串名称"""
    # ESC 特殊键，直接返回固定名称
    if key == keyboard.Key.esc:
        return "esc"

    # 普通字符键：优先用 char 属性，否则查小键盘映射表
    if isinstance(key, keyboard.KeyCode):
        if key.char is not None:
            return key.char
        return KEYPAD_VK_MAP.get(key.vk)

    return None


def build_action(pressed_keys: set[str], gripper_signal: float) -> np.ndarray:
    """根据当前按下的键和夹爪信号构造 7 维动作向量"""
    # action[0..2]: 末端在 x/y/z 方向上的平动增量
    # action[3..5]: 末端旋转（此处不使用，保持为 0）
    # action[6]: 夹爪开关信号
    action = np.zeros(7, dtype=np.float32)

    # 8/5 控制 x 轴（正/负）
    if "8" in pressed_keys:
        action[0] += TRAN_STEP
    if "5" in pressed_keys:
        action[0] -= TRAN_STEP

    # 6/4 控制 y 轴（正/负）
    if "6" in pressed_keys:
        action[1] += TRAN_STEP
    if "4" in pressed_keys:
        action[1] -= TRAN_STEP

    # 7/9 控制 z 轴（正/负）
    if "7" in pressed_keys:
        action[2] += TRAN_STEP
    if "9" in pressed_keys:
        action[2] -= TRAN_STEP

    # 夹爪信号：由外部状态传入，仅在按键触发时切换
    action[6] = gripper_signal
    # 这里只有平移，没有旋转
    return action


def main():
    """主函数：创建 Gym 环境并通过键盘控制机械臂运动和夹爪开合"""

    # 命令行参数解析
    parser = argparse.ArgumentParser(description="键盘控制 SpaceUR10e 机械臂")
    parser.add_argument("--env", type=str, default="SpaceUR10e-Cube-v0", help="Gym 环境名称")
    args = parser.parse_args()

    # 通过 gym.make 创建环境（使用注册的默认参数）
    env = gym.make(args.env, render_mode="human")

    # 重置环境到初始状态
    env.reset()
    # 当前按下的键集合
    pressed_keys: set[str] = set()
    # 线程锁，保护 pressed_keys 和 state 的并发访问
    lock = Lock()
    # 共享状态：running 控制主循环，gripper_signal 跟踪夹爪当前信号
    state = {"running": True, "gripper_signal": 0.0}

    def on_press(key):
        """按键按下回调：更新按下的键集合，处理 ESC 退出和夹爪切换"""
        key_name = normalize_key(key)
        if key_name is None:
            return None

        with lock:
            # ESC 键：标记退出
            if key_name == "esc":
                state["running"] = False
                return False

            # 数字 1 键：首次按下时翻转夹爪信号（正->负 或 负->正）
            if key_name == "1" and key_name not in pressed_keys:
                state["gripper_signal"] = -1.0 if state["gripper_signal"] > 0.0 else 1.0
                print(f"夹爪翻转信号: {state['gripper_signal']:+.0f}")

            pressed_keys.add(key_name)

        return None

    def on_release(key):
        """按键释放回调：从按下集合中移除该键"""
        key_name = normalize_key(key)
        if key_name is None:
            return None

        with lock:
            pressed_keys.discard(key_name)

        return None

    # 启动键盘监听线程
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # 打印控制说明
    print("Gym 环境已启动，键盘控制如下：")
    print("8/5: x 轴正/负方向")
    print("4/6: y 轴负/正方向")
    print("7/9: z 轴正/负方向")
    print("1: 翻转夹爪开合")
    print("ESC 或关闭 MuJoCo 窗口: 退出")
    print("支持主键盘数字键和小键盘数字键。")

    try:
        # 主控制循环：按固定帧率执行 step
        while state["running"]:
            # 如果 MuJoCo 窗口被用户关闭，退出循环
            if env.unwrapped.viewer and not env.unwrapped.viewer.is_running():
                break

            # 根据当前按下的键和夹爪信号构造动作
            with lock:
                action = build_action(set(pressed_keys), state["gripper_signal"])

            # 执行一步仿真，获取新的观测、奖励和终止标志
            _, reward, terminated, truncated, info = env.step(action)

            # 如果任务成功或超时，自动重置环境
            if terminated or truncated:
                reason = "success" if terminated else "truncated"
                print(
                    f"环境已重置: reason={reason}, "
                    f"distance={info.get('distance', 0.0):.4f}, reward={reward:.4f}"
                )
                env.reset()

            # 维持固定帧率
            time.sleep(1.0 / FPS)
    except KeyboardInterrupt:
        print("收到退出信号，正在关闭环境...")
    finally:
        # 清理资源：停止监听、关闭环境
        state["running"] = False
        listener.stop()
        listener.join(timeout=1.0)
        env.close()


if __name__ == "__main__":
    main()
