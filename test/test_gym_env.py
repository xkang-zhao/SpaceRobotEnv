import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from envs.space_ur10e_env import SpaceUR10eEnv

def test_random_actions():
    # 创建环境
    # 这些路径相对于项目根目录。
    # 如果你在项目根目录下运行 "python test/test_gym_env.py" 
    # 那么默认工作目录就是项目根文件夹，路径是有效的。
    env = SpaceUR10eEnv(
        render_mode="human", # "rgb_array" 或 "human"
        scene_path='./mjcf/1_cube_scene.xml',
        arm_path='./mjcf/arm.xml',
        frame_name="left_attachment"
    )

    obs, info = env.reset()
    print("Gym 环境已启动。正在进行随机动作测试...")

    try:
        while True:
            # 1. 采样随机动作 (Gym 环境会自动根据动作空间的定义采样)
            # action 为 [-1, 1] 范围内的 7 维浮点数数组
            action = env.action_space.sample()
            action = np.clip(env.action_space.sample(), -0.005, 0.005)

            # 2. 向环境发送动作并执行一步仿真
            obs, reward, terminated, truncated, info = env.step(action)

            # 3. 如果环境已终止 (抓取成功) 或超时/物体飞出 (截断)，则重置环境
            if terminated or truncated:
                if terminated:
                    print("抓取成功！重置中...")
                else:
                    print("由于距离过远或超时导致截断，重置中...")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\n捕获到退出信号，正在关闭环境...")
    finally:
        env.close()
        print("测试完成。已清理环境。")

if __name__ == "__main__":
    test_random_actions()
