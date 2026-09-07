import pygame
import sys
import time

'''
A: 0
B: 1
X: 2
Y: 3
back: 6
start: 7
'''
def main():
    # 1. 初始化 Pygame
    pygame.init()
    pygame.joystick.init()

    # 2. 检测手柄数量
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("未检测到手柄，请检查 USB 连接！")
        return

    # 3. 初始化第一个手柄
    # 如果你有多个手柄，这里索引可以是 0, 1, 2...
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"已连接手柄: {joystick.get_name()}")
    print(f"该手柄有 {joystick.get_numaxes()} 个轴 (摇杆/扳机)")
    print(f"该手柄有 {joystick.get_numbuttons()} 个按钮")
    print("-" * 30)
    print("开始监听输入 (按 Ctrl+C 退出)...")

    try:
        # 4. 进入主循环监听事件
        while True:
            # Pygame 需要不断处理事件队列
            for event in pygame.event.get():
                
                # --- 摇杆/轴移动事件 (Axis Motion) ---
                # 通常：
                # axis 0: 左摇杆左右 (-1 左 ~ 1 右)
                # axis 1: 左摇杆上下 (-1 上 ~ 1 下)
                # axis 2/3/4/5: 右摇杆或扳机键 (视手柄型号而定)
                if event.type == pygame.JOYAXISMOTION:
                    axis_value = event.value
                    # 设置一个“死区”(Deadzone)，防止摇杆漂移产生的微小数值干扰
                    if abs(axis_value) > 0.1: 
                        print(f"摇杆轴 {event.axis} 移动: {axis_value:.2f}")

                # --- 按钮按下事件 (Button Down) ---
                elif event.type == pygame.JOYBUTTONDOWN:
                    print(f"按钮 {event.button} 被按下")

                # --- 按钮松开事件 (Button Up) ---
                elif event.type == pygame.JOYBUTTONUP:
                    print(f"按钮 {event.button} 松开")

                # --- 方向键/苦力帽事件 (Hat Motion) ---
                # value 通常是元组 (x, y)，例如 (-1, 0) 代表左
                elif event.type == pygame.JOYHATMOTION:
                    print(f"方向键(Hat) {event.hat} 状态: {event.value}")

            # 稍微休眠一下避免 CPU 100% 占用
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n程序已退出。")
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()