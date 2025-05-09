# SPDX-License-Identifier: BSD-3-Clause
# 
# Copyright (c) 2025 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import math
import numpy as np
import mujoco
import mujoco_viewer
from tqdm import tqdm
import argparse
import time


class AirbotJointController:
    """控制器类，用于AirBot关节空间的轨迹控制"""
    
    def __init__(self, model_path, dt=0.001, render=True):
        """
        初始化控制器
        
        Args:
            model_path: MuJoCo模型路径
            dt: 仿真时间步长
            render: 是否渲染可视化
        """
        # 加载模型和初始化数据
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)
        
        # 仿真设置
        self.dt = dt
        self.render_enabled = render
        if render:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        
        # 机器人设置
        self.num_joints = 6  # AirBot有6个关节
        
        # 关节限制
        self.joint_lower_limits = np.array([-3.14, -2.96, -0.087, -2.96, -1.74, -3.14], dtype=np.float64)
        self.joint_upper_limits = np.array([2.09, 0.17, 3.14, 2.96, 1.74, 3.14], dtype=np.float64)
        
        # 关节名称（如果需要）
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        
        # 初始化机器人状态
        mujoco.mj_resetData(self.model, self.data)
        self.step_simulation()
        
        # 获取末端执行器（link6）的ID，用于可视化
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link6")
            
        # 打印初始位置
        self.print_current_state()
    
    def step_simulation(self):
        """执行一步仿真"""
        mujoco.mj_step(self.model, self.data)
        if self.render_enabled:
            self.viewer.render()
    
    def get_joint_positions(self):
        """获取当前关节位置"""
        return self.data.qpos[-self.num_joints:].copy()
    
    def get_end_effector_pose(self):
        """获取末端执行器的位置和四元数（用于显示信息）"""
        if self.end_effector_id != -1:
            pos = self.data.xpos[self.end_effector_id].copy()
            quat = self.data.xquat[self.end_effector_id].copy()
            return pos, quat
        else:
            return np.zeros(3), np.zeros(4)
    
    def set_joint_positions(self, q):
        """
        设置关节位置
        
        Args:
            q: 目标关节角度数组
        """
        q_clipped = np.clip(q, self.joint_lower_limits, self.joint_upper_limits)
        self.data.ctrl[:self.num_joints] = q_clipped
    
    def move_to_joint_position(self, target_q, duration=1.0, dt=0.01):
        """
        将机器人移动到指定的关节位置
        
        Args:
            target_q: 目标关节角度
            duration: 移动持续时间
            dt: 控制步长
        """
        # 获取当前关节位置
        start_q = self.get_joint_positions()
        
        # 计算步数
        steps = int(duration / dt)
        
        # 执行轨迹
        for i in range(steps):
            # 计算插值因子
            alpha = min(1.0, i / (steps - 1)) if steps > 1 else 1.0
            
            # 线性插值计算当前目标关节位置
            current_target_q = start_q + alpha * (target_q - start_q)
            
            # 设置关节位置
            self.set_joint_positions(current_target_q)
            
            # 执行仿真步进
            step_count = max(1, int(dt / self.dt))
            for _ in range(step_count):
                self.step_simulation()
        
        # 确保到达最终位置
        self.set_joint_positions(target_q)
        for _ in range(10):  # 多执行几步确保稳定
            self.step_simulation()
    
    def execute_joint_trajectory(self, traj_func, duration, dt=0.01):
        """
        执行关节空间轨迹
        
        Args:
            traj_func: 轨迹生成函数，接受时间t，返回关节角度q
            duration: 轨迹持续时间(秒)
            dt: 轨迹控制步长(秒)
        """
        # 计算步数
        steps = int(duration / dt)
        
        # 主循环
        for i in tqdm(range(steps), desc="执行轨迹..."):
            t = i * dt
            
            # 获取当前时刻的目标关节角度
            target_q = traj_func(t)
            
            # 设置关节控制
            self.set_joint_positions(target_q)
            
            # 执行仿真步进
            step_count = max(1, int(dt / self.dt))
            for _ in range(step_count):
                self.step_simulation()
                
            # 每隔一段时间打印状态
            if i % 100 == 0:
                self.print_current_state()
    
    def print_current_state(self):
        """打印当前机器人状态"""
        curr_q = self.get_joint_positions()
        curr_pos, curr_quat = self.get_end_effector_pose()
        
        print(f"关节角度: [{', '.join([f'{q:.3f}' for q in curr_q])}]")
        print(f"末端位置: [{', '.join([f'{p:.3f}' for p in curr_pos])}]")
        print("-" * 40)
    
    def close(self):
        """关闭控制器和可视化"""
        if self.render_enabled:
            self.viewer.close()


# 关节轨迹生成函数

def joint_sin_trajectory(home_q, amplitudes, frequencies, phases, duration):
    """
    生成正弦关节轨迹
    
    Args:
        home_q: 基准关节位置
        amplitudes: 各关节的振幅
        frequencies: 各关节的频率
        phases: 各关节的相位
        duration: 轨迹持续时间
        
    Returns:
        轨迹函数，接受时间t，返回关节角度q
    """
    def traj(t):
        # 计算周期时间因子 (0 到 1)
        t_norm = t % duration / duration
        
        # 为每个关节生成正弦波形
        q = np.copy(home_q)
        for i in range(len(home_q)):
            q[i] += amplitudes[i] * math.sin(2 * math.pi * frequencies[i] * t_norm + phases[i])
        
        return q
    
    return traj


def joint_line_trajectory(start_q, end_q, duration):
    """
    生成关节空间的直线轨迹
    
    Args:
        start_q: 起始关节角度
        end_q: 结束关节角度
        duration: 轨迹持续时间
        
    Returns:
        轨迹函数，接受时间t，返回关节角度q
    """
    def traj(t):
        alpha = min(t / duration, 1.0)  # 归一化时间因子
        q = start_q + alpha * (end_q - start_q)
        return q
    
    return traj


def joint_circle_trajectory(home_q, joint_pairs, radius, frequency, duration):
    """
    生成两个关节协同运动形成的圆形轨迹
    
    Args:
        home_q: 基准关节位置
        joint_pairs: 要协同运动的关节对列表，例如[(0,1), (2,3)]
        radius: 圆的半径（弧度）
        frequency: 圆的频率
        duration: 轨迹持续时间
        
    Returns:
        轨迹函数，接受时间t，返回关节角度q
    """
    def traj(t):
        # 计算周期时间因子
        t_norm = t % duration / duration
        angle = 2 * math.pi * frequency * t_norm
        
        # 复制基准位置
        q = np.copy(home_q)
        
        # 为每对关节生成圆形轨迹
        for joint1, joint2 in joint_pairs:
            q[joint1] += radius * math.cos(angle)
            q[joint2] += radius * math.sin(angle)
        
        return q
    
    return traj


def joint_wave_trajectory(home_q, duration):
    """
    生成一个波浪状关节轨迹，使各关节依次摆动
    
    Args:
        home_q: 基准关节位置
        duration: 轨迹持续时间
        
    Returns:
        轨迹函数，接受时间t，返回关节角度q
    """
    num_joints = len(home_q)
    
    def traj(t):
        # 计算周期时间因子
        t_norm = t % duration / duration
        
        # 复制基准位置
        q = np.copy(home_q)
        
        # 为每个关节生成波形，时间上有相位差
        for i in range(num_joints):
            phase = 2 * math.pi * i / num_joints
            q[i] += 0.3 * math.sin(2 * math.pi * t_norm + phase)
        
        return q
    
    return traj


def main():
    parser = argparse.ArgumentParser(description='AirBot关节空间轨迹控制')
    parser.add_argument('--model_path', type=str, 
                      default='/home/yurong/humanoid-gym/resources/robots/airbot/airbot_play_v3_0_gripper.xml',
                      help='MuJoCo XML模型文件路径')
    parser.add_argument('--trajectory', type=str, default='line',
                      choices=['sin', 'line', 'circle', 'wave'],
                      help='要执行的关节轨迹类型: sin, line, circle, wave')
    parser.add_argument('--duration', type=float, default=10.0,
                      help='轨迹执行总时长(秒)')
    parser.add_argument('--dt', type=float, default=0.01,
                      help='轨迹控制步长(秒)')
    args = parser.parse_args()
    
    # 初始化控制器
    controller = AirbotJointController(args.model_path, dt=0.001, render=True)
    
    try:
        # 获取当前关节位置作为参考
        home_q = controller.get_joint_positions()
        print(f"初始关节位置: {home_q}")
        
        # 根据选择的轨迹类型创建轨迹函数
        if args.trajectory == 'sin':
            # 正弦轨迹: 每个关节做不同频率的正弦运动
            amplitudes = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])  # 振幅
            frequencies = np.array([1.0, 0.5, 0.75, 1.5, 1.0, 0.5])  # 频率
            phases = np.array([0, math.pi/2, math.pi, math.pi*3/2, 0, math.pi/2])  # 相位
            traj_func = joint_sin_trajectory(home_q, amplitudes, frequencies, phases, args.duration)
            
        elif args.trajectory == 'line':
            # 直线轨迹: 从当前位置到另一个位置
            start_q = home_q.copy()
            # 计算一个安全的目标位置，确保在关节限制范围内
            end_q = home_q + np.array([0.5, -0.5, 0.3, 0.5, -0.5, 0.3])
            end_q = np.clip(end_q, controller.joint_lower_limits, controller.joint_upper_limits)
            traj_func = joint_line_trajectory(start_q, end_q, args.duration / 2)
            
        elif args.trajectory == 'circle':
            # 圆形轨迹: 选择两对关节，让它们形成圆周运动
            joint_pairs = [(0, 1), (2, 3)]  # 第1,2关节一组，第3,4关节一组
            radius = 0.3  # 圆的半径（弧度）
            frequency = 1.0  # 频率
            traj_func = joint_circle_trajectory(home_q, joint_pairs, radius, frequency, args.duration)
            
        else:  # wave
            # 波浪轨迹: 所有关节依次做波浪运动
            traj_func = joint_wave_trajectory(home_q, args.duration)
        
        # 执行轨迹
        print(f"开始执行{args.trajectory}关节轨迹，持续{args.duration}秒...")
        controller.execute_joint_trajectory(traj_func, args.duration, dt=args.dt)
        
        # 执行完毕后返回初始位置
        print("轨迹执行完毕，返回初始位置...")
        controller.move_to_joint_position(home_q, duration=2.0)
        
        # 保持一段时间
        print("返回初始位置完成，保持2秒...")
        time.sleep(2)
        
    finally:
        # 确保在退出时关闭控制器
        controller.close()
    

if __name__ == "__main__":
    main()