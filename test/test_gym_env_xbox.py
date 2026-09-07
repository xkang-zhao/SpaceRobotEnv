
# Xbox 手柄交互测试

from utils.xbox_controller import XboxController
from envs.space_ur10e_env import SpaceUR10eEnv
import numpy as np

env = SpaceUR10eEnv(render_mode="human", # "rgb_array" or "human"
                    scene_path='./mjcf/1_cube_scene.xml',
                    arm_path='./mjcf/arm.xml', frame_name="left_attachment"
                    )

obs, info = env.reset()

xbox = XboxController()
print("Gym 环境已启动。尝试连接 Xbox 手柄...")

try:
    while True:
        if xbox.is_connected():
            # 获取手柄输入 (Deltas)
            action = xbox.handle_input()
            if action is None:
                action = np.zeros(7) 
            # print(action)
            # 限制范围
            action = np.clip(action, -1.0, 1.0)
        else:
            action = np.zeros(7)

        obs, reward, terminated, truncated, info = env.step(action)
        
        # if terminated or truncated:
        #     obs, info = env.reset()
            
except KeyboardInterrupt:
    print("退出...")
finally:

    xbox.disconnect()
    env.close()
