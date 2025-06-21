import math
import numpy as np
import pybullet as p
import pybullet_data
import time
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import random
import argparse
import os
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Sim2simCfg:
    class sim_config:
        urdf_model_path = '/home/yurong/下载/airbot_usd/urdf/airbot_play_v3_0_gripper.urdf'  # Update with actual URDF path
        sim_duration = 60.0
        dt = 1/100
        decimation = 2

    class env:
        num_actions = 12  # 12 joints for dual-arm (6 + 6)
        num_single_obs = 58  # 原来51 + 7 (box pose: 3 pos + 4 quat)
        num_observations = num_single_obs  # Will be updated if frame_stack > 1
        frame_stack = 1  # Modify if you use frame stacking

    class robot_config:
        # Default joint positions (home position) for both robots
        home_position_robot1 = np.zeros(6, dtype=np.double)
        home_position_robot2 = np.zeros(6, dtype=np.double)
        
        # Joint limits for both robots
        joint_lower_limits = np.array([-3.14, -2.96, -0.087, -2.96, -1.74, -3.14]*2, dtype=np.double)
        joint_upper_limits = np.array([2.09, 0.17, 3.14,  2.96, 1.74, 3.14]*2, dtype=np.double)
        
        # Joint names (for PyBullet to identify joints)
        joint_names_robot1 = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        joint_names_robot2 = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        
        # End effector link names
        end_effector_link_robot1 = "link6"  # Update with actual end effector link name
        end_effector_link_robot2 = "link6"  # Update with actual end effector link name
        
        # Robot base positions (for dual-arm setup - face to face mirror setup)
        robot1_base_position = [0, 0, 0]     # First robot at origin
        robot1_base_orientation = [0, 0, 0, 1]  # No rotation
        robot2_base_position = [0.47, 0.0, 0.0]   # Second robot 0.47m away
        robot2_base_orientation = [0.0, 0.0, 1.0, 0.0]  # 180 degree rotation around Z-axis for mirror setup

    class box_config:
        # Box configuration similar to the IsaacSim definition
        size = (0.15, 0.11, 0.22)  # width, height, depth
        position = [0.235, 0.0, 0.11]  # x, y, z (adjusted z to be on ground + half height)
        orientation = [0, 0, 0, 1]  # quaternion [x, y, z, w]
        mass = 0.65361
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        color = [0.0, 0.0, 1.0, 1.0]  # RGBA - blue color

    class normalization:
        obs_scales = type('', (), {})()
        obs_scales.lin_vel = 1.0
        obs_scales.ang_vel = 1.0
        obs_scales.dof_pos = 1.0
        obs_scales.dof_vel = 0.1
        
        clip_observations = 100.0
        clip_actions = 1.5
        
    class control:
        action_scale = 0.5  # Scale for joint position commands


class ReachTaskConfig:
    """Configuration for the reaching task"""
    def __init__(self):
        # Target position ranges (similar to your command ranges)
        self.pos_range_x = (0.235, 0.235)
        self.pos_range_y = (0,0)
        self.pos_range_z = (0.5, 0.5)
        self.pos_range_roll = (0, 0)
        self.pos_range_pitch = (0, 0)
        self.pos_range_yaw = (0, 0)
        
        # Current target position
        self.target_pos = np.array([
            random.uniform(*self.pos_range_x),
            random.uniform(*self.pos_range_y),
            random.uniform(*self.pos_range_z)
        ])
        self.target_rpy = np.array([
            random.uniform(*self.pos_range_roll),
            random.uniform(*self.pos_range_pitch),
            random.uniform(*self.pos_range_yaw)
        ])

        # Target update frequency (in seconds)
        self.target_update_time = 4.0
        self.time_since_last_update = 0.0
        
        # Target visualization (will be initialized in the main function)
        self.target_visual_id = None
    
    def update_target(self, dt):
        """Update target position if needed"""
        self.time_since_last_update += dt
        if self.time_since_last_update >= self.target_update_time:
            self.target_pos[0] = random.uniform(*self.pos_range_x)
            self.target_pos[1] = random.uniform(*self.pos_range_y)
            self.target_pos[2] = random.uniform(*self.pos_range_z)

            self.target_rpy[0] = random.uniform(*self.pos_range_roll)
            self.target_rpy[1] = random.uniform(*self.pos_range_pitch)
            self.target_rpy[2] = random.uniform(*self.pos_range_yaw)
            self.time_since_last_update = 0.0
            return True
        return False


def parse_actions_from_txt(file_path):
    """从txt文件中解析action数据
    
    Args:
        file_path: txt文件路径
        
    Returns:
        actions_list: 包含所有action的列表，每个action是长度为12的numpy数组
    """
    actions_list = []
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 使用正则表达式匹配 "env1 actions:" 后面的数组
        pattern = r'env1 actions:\s*\[(.*?)\]'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            # 将字符串转换为数字列表
            action_str = match.strip()
            # 分割并转换为浮点数
            action_values = []
            for val in action_str.split():
                try:
                    action_values.append(float(val))
                except ValueError:
                    continue
            
            if len(action_values) == 12:  # 确保是12维的action
                actions_list.append(np.array(action_values))
                
        print(f"成功解析了 {len(actions_list)} 个action")
        return actions_list
        
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")
        return []
    except Exception as e:
        print(f"解析文件时出错: {e}")
        return []


def init_pybullet(gui=True):
    """Initialize PyBullet simulation environment
    
    Args:
        gui: Whether to use GUI visualization
        
    Returns:
        Physics client ID
    """
    if gui:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
        
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0/240.0)  # Default PyBullet timestep
    
    # Load plane
    p.loadURDF("plane.urdf")
    
    return physicsClient


def load_box(cfg):
    """Load box into PyBullet simulation
    
    Args:
        cfg: Configuration object containing box configuration
        
    Returns:
        Box body ID
    """
    # Create box collision shape
    box_collision_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[cfg.box_config.size[0]/2, cfg.box_config.size[1]/2, cfg.box_config.size[2]/2]
    )
    
    # Create box visual shape
    box_visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[cfg.box_config.size[0]/2, cfg.box_config.size[1]/2, cfg.box_config.size[2]/2],
        rgbaColor=cfg.box_config.color
    )
    
    # Create multi-body with both collision and visual shapes
    box_id = p.createMultiBody(
        baseMass=cfg.box_config.mass,
        baseCollisionShapeIndex=box_collision_shape,
        baseVisualShapeIndex=box_visual_shape,
        basePosition=cfg.box_config.position,
        baseOrientation=cfg.box_config.orientation
    )
    
    # Set material properties
    p.changeDynamics(
        box_id,
        -1,  # base link
        lateralFriction=cfg.box_config.static_friction,
        restitution=cfg.box_config.restitution
    )
    
    return box_id


def load_dual_robots(cfg):
    """Load dual robots URDF into PyBullet with face-to-face mirror setup
    
    Args:
        cfg: Configuration object
        
    Returns:
        Robot IDs and joint indices for both robots
    """
    # Load robot 1 (at origin, no rotation)
    robot1_id = p.loadURDF(cfg.sim_config.urdf_model_path, 
                          basePosition=cfg.robot_config.robot1_base_position,
                          baseOrientation=cfg.robot_config.robot1_base_orientation,
                          useFixedBase=True)
    
    # Load robot 2 (0.47m away, 180 degree rotation for mirror setup)
    robot2_id = p.loadURDF(cfg.sim_config.urdf_model_path, 
                          basePosition=cfg.robot_config.robot2_base_position,
                          baseOrientation=cfg.robot_config.robot2_base_orientation,
                          useFixedBase=True)
    
    # Get controllable joint indices for robot 1
    joint_indices_robot1 = []
    for i in range(p.getNumJoints(robot1_id)):
        joint_info = p.getJointInfo(robot1_id, i)
        joint_name = joint_info[1].decode('utf-8')
        if joint_name in cfg.robot_config.joint_names_robot1:
            joint_indices_robot1.append(i)
    
    # Get controllable joint indices for robot 2
    joint_indices_robot2 = []
    for i in range(p.getNumJoints(robot2_id)):
        joint_info = p.getJointInfo(robot2_id, i)
        joint_name = joint_info[1].decode('utf-8')
        if joint_name in cfg.robot_config.joint_names_robot2:
            joint_indices_robot2.append(i)
    
    # Reset joints to home position for both robots
    for idx in joint_indices_robot1:
        p.resetJointState(robot1_id, idx, 0.0)
    
    for idx in joint_indices_robot2:
        p.resetJointState(robot2_id, idx, 0.0)
    
    return robot1_id, robot2_id, joint_indices_robot1, joint_indices_robot2


# 保存当前坐标轴可视化的线段 ID
current_axis_visuals = []

def create_target_visual(position, rpy=None):
    """Create visual coordinate axes at the given position and orientation"""
    global current_axis_visuals

    # 默认无旋转
    if rpy is not None:
        quat = p.getQuaternionFromEuler(rpy)
    else:
        quat = [0, 0, 0, 1]

    rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    origin = np.array(position)

    axis_length = 0.1  # 可视化轴长度
    colors = {
        "x": [1, 0, 0],
        "y": [0, 1, 0],
        "z": [0, 0, 1],
    }

    # 先删除旧的可视化线
    for line_id in current_axis_visuals:
        p.removeUserDebugItem(line_id)
    current_axis_visuals.clear()

    # 创建新的三个方向的轴线
    for i, axis in enumerate(["x", "y", "z"]):
        direction = rot_matrix[:, i]
        end_point = origin + axis_length * direction
        line_id = p.addUserDebugLine(
            origin, end_point, colors[axis], lineWidth=2.0, lifeTime=0  # 0 = 持续存在
        )
        current_axis_visuals.append(line_id)

def update_target_visual(position, rpy=None):
    create_target_visual(position, rpy)


def get_joint_states(robot_id, joint_indices):
    """Get joint positions and velocities
    
    Args:
        robot_id: PyBullet body ID for the robot
        joint_indices: List of joint indices to query
        
    Returns:
        Joint positions and velocities as numpy arrays
    """
    positions = []
    velocities = []
    
    for idx in joint_indices:
        state = p.getJointState(robot_id, idx)
        positions.append(state[0])
        velocities.append(state[1])
    
    return np.array(positions), np.array(velocities)


def get_link_state(robot_id, link_name):
    """Get position and orientation of a specific link
    
    Args:
        robot_id: PyBullet body ID for the robot
        link_name: Name of the link to query
        
    Returns:
        Position [x, y, z] and orientation quaternion [x, y, z, w]
    """
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        if info[12].decode('utf-8') == link_name:
            link_state = p.getLinkState(robot_id, i)
            return link_state[0], link_state[1]  # worldPos, worldOrn
    
    return None, None


def plot_trajectory_comparison(time_stamps, input_actions, actual_positions, cfg, save_path=None):
    """绘制输入action和实际关节位置的对比图
    
    Args:
        time_stamps: 时间戳列表
        input_actions: 输入的action序列 (N, 12)
        actual_positions: 实际关节位置序列 (N, 12)  
        cfg: 配置对象
        save_path: 保存图片的路径（可选）
    """
    # 确保数据长度一致
    min_len = min(len(time_stamps), len(input_actions), len(actual_positions))
    time_stamps = time_stamps[:min_len]
    input_actions = np.array(input_actions[:min_len])
    actual_positions = np.array(actual_positions[:min_len])
    
    # 应用action scaling到输入actions（用于对比）
    scaled_actions = input_actions * cfg.control.action_scale
    
    # 创建子图 - 分别显示两个机器人的关节
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    fig.suptitle('input Action vs actual position', fontsize=16)
    
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']
    
    # 绘制机器人1的关节 (前6个关节)
    for i in range(6):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 绘制输入action (scaled)
        ax.plot(time_stamps, scaled_actions[:, i], 'b-', linewidth=2, label='input Action (scaled)', alpha=0.7)
        # 绘制实际关节位置
        ax.plot(time_stamps, actual_positions[:, i], 'r-', linewidth=2, label='actual position')
        
        ax.set_title(f'robot1 - {joint_names[i]}')
        # ax.set_xlabel('time (s)')
        ax.set_ylabel('joint angle (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加关节限制线
        ax.axhline(y=cfg.robot_config.joint_lower_limits[i], color='k', linestyle='--', alpha=0.5, label='关节限制')
        ax.axhline(y=cfg.robot_config.joint_upper_limits[i], color='k', linestyle='--', alpha=0.5)
    
    # 绘制机器人2的关节 (后6个关节)
    for i in range(6):
        row = (i // 3) + 2  # 放在下面两行
        col = i % 3
        ax = axes[row, col]
        
        # 绘制输入action (scaled)
        ax.plot(time_stamps, scaled_actions[:, i+6], 'b-', linewidth=2, label='input Action (scaled)', alpha=0.7)
        # 绘制实际关节位置
        ax.plot(time_stamps, actual_positions[:, i+6], 'r-', linewidth=2, label='actual position')
        
        ax.set_title(f'robot2 - {joint_names[i]}')
        # ax.set_xlabel('time (s)')
        ax.set_ylabel('joint angle (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加关节限制线
        ax.axhline(y=cfg.robot_config.joint_lower_limits[i+6], color='k', linestyle='--', alpha=0.5, label='关节限制')
        ax.axhline(y=cfg.robot_config.joint_upper_limits[i+6], color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"轨迹对比图已保存到: {save_path}")
    
    plt.show()


def plot_tracking_errors(time_stamps, input_actions, actual_positions, cfg, save_path=None):
    """绘制跟踪误差图
    
    Args:
        time_stamps: 时间戳列表
        input_actions: 输入的action序列
        actual_positions: 实际关节位置序列
        cfg: 配置对象
        save_path: 保存路径（可选）
    """
    # 计算跟踪误差
    scaled_actions = np.array(input_actions) * cfg.control.action_scale
    actual_positions = np.array(actual_positions)
    
    min_len = min(len(scaled_actions), len(actual_positions))
    errors = actual_positions[:min_len] - scaled_actions[:min_len]
    
    # 创建误差图
    fig, axes = plt.subplots(2, 6, figsize=(18, 8))
    fig.suptitle('关节跟踪误差分析', fontsize=16)
    
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']
    
    # 机器人1的误差
    for i in range(6):
        ax = axes[0, i]
        ax.plot(time_stamps[:min_len], errors[:, i], 'g-', linewidth=1.5)
        ax.set_title(f'robot1 - {joint_names[i]}')
        # ax.set_xlabel('time (s)')
        ax.set_ylabel('error (rad)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        # 计算并显示统计信息
        rmse = np.sqrt(np.mean(errors[:, i]**2))
        max_error = np.max(np.abs(errors[:, i]))
        ax.text(0.02, 0.98, f'RMSE: {rmse:.4f}\nMax: {max_error:.4f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 机器人2的误差
    for i in range(6):
        ax = axes[1, i]
        ax.plot(time_stamps[:min_len], errors[:, i+6], 'g-', linewidth=1.5)
        ax.set_title(f'robot2 - {joint_names[i]}')
        # ax.set_xlabel('time (s)')
        ax.set_ylabel('error (rad)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        # 计算并显示统计信息
        rmse = np.sqrt(np.mean(errors[:, i+6]**2))
        max_error = np.max(np.abs(errors[:, i+6]))
        ax.text(0.02, 0.98, f'RMSE: {rmse:.4f}\nMax: {max_error:.4f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        error_save_path = save_path.replace('.png', '_errors.png')
        plt.savefig(error_save_path, dpi=300, bbox_inches='tight')
        print(f"跟踪误差图已保存到: {error_save_path}")
    
    plt.show()


def run_dual_arm_trajectory_following(actions_list, cfg, task_cfg, gui=True, plot_results=True, save_plots=False):
    """运行PyBullet双臂机器人轨迹跟踪并绘制结果
    
    Args:
        actions_list: 从txt文件解析的action列表
        cfg: 仿真配置
        task_cfg: 任务配置
        gui: 是否使用GUI可视化
        plot_results: 是否绘制结果图
        save_plots: 是否保存图片
        
    Returns:
        None
    """
    # 初始化PyBullet
    physicsClient = init_pybullet(gui)
    
    # 加载双臂机器人并获取关节索引
    robot1_id, robot2_id, joint_indices_robot1, joint_indices_robot2 = load_dual_robots(cfg)
    
    # 加载盒子对象
    box_id = load_box(cfg)
    
    # 创建目标可视化
    task_cfg.target_visual_id = create_target_visual(
        task_cfg.target_pos, 
        task_cfg.target_rpy
    )
    
    # 初始化控制变量
    target_q = np.zeros(cfg.env.num_actions, dtype=np.double)  # 12个关节
    
    # 计算仿真参数
    dt = cfg.sim_config.dt
    
    # 检查是否有action数据
    if not actions_list:
        print("没有找到有效的action数据，退出仿真")
        p.disconnect()
        return
    
    print(f"开始执行轨迹，共有 {len(actions_list)} 个action")
    
    # 数据记录用于绘图
    time_stamps = []
    input_actions_log = []
    actual_positions_log = []
    
    # 主仿真循环
    action_index = 0
    count_lowlevel = 0
    current_time = 0.0
    
    # 使用action数量来控制仿真步数，而不是固定的时间
    total_steps = len(actions_list) * cfg.sim_config.decimation
    
    for step in tqdm(range(total_steps), desc="执行轨迹跟踪中..."):
        # 在指定的decimation间隔更新action
        if count_lowlevel % cfg.sim_config.decimation == 0 and action_index < len(actions_list):
            # 获取当前action
            current_action = actions_list[action_index]
            
            # 应用action scaling
            target_q = current_action * cfg.control.action_scale
            
            # 记录输入action
            input_actions_log.append(current_action.copy())
            
            # 打印当前执行的action信息
            if action_index % 50 == 0:  # 每50个action打印一次
                print(f"执行第 {action_index} 个action: {current_action[:6]} (robot1), {current_action[6:]} (robot2)")
            
            action_index += 1
        
        # 将目标位置限制在关节限制范围内
        target_q_clipped = np.clip(
            target_q, 
            cfg.robot_config.joint_lower_limits, 
            cfg.robot_config.joint_upper_limits
        )
        
        # 对机器人1应用位置控制（前6个关节）
        for i, joint_idx in enumerate(joint_indices_robot1):
            p.setJointMotorControl2(
                bodyUniqueId=robot1_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_q[i],
                maxVelocity=0.8,
                force=500  # 增加力矩限制以确保稳定控制
            )
        
        # 对机器人2应用位置控制（后6个关节）
        for i, joint_idx in enumerate(joint_indices_robot2):
            p.setJointMotorControl2(
                bodyUniqueId=robot2_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_q[i + 6],  # 偏移6个关节对应第二个机器人
                maxVelocity=0.8,
                force=500
            )
        
        # 执行仿真步骤
        p.stepSimulation()
        
        # 记录实际关节位置（每个decimation周期记录一次）
        if count_lowlevel % cfg.sim_config.decimation == 0:
            # 获取两个机器人的关节状态
            pos1, _ = get_joint_states(robot1_id, joint_indices_robot1)
            pos2, _ = get_joint_states(robot2_id, joint_indices_robot2)
            
            # 合并两个机器人的关节位置
            combined_positions = np.concatenate([pos1, pos2])
            actual_positions_log.append(combined_positions)
            time_stamps.append(current_time)
            current_time += dt * cfg.sim_config.decimation
        
        # 如果使用GUI，保持实时仿真
        if gui:
            time.sleep(dt)
        
        # 增加计数器
        count_lowlevel += 1
        
        # 如果所有action都执行完了，可以选择停止或者重复
        if action_index >= len(actions_list):
            print("所有action已执行完毕")
            break
    
    # 仿真结束后保持一段时间观察结果
    if gui:
        print("轨迹执行完成，保持仿真状态5秒...")
        for _ in range(500):  # 5秒
            p.stepSimulation()
            time.sleep(0.01)
    
    # 断开连接
    p.disconnect()
    
    # 绘制结果
    if plot_results and len(time_stamps) > 0:
        print("正在生成轨迹对比图...")
        
        save_path = None
        if save_plots:
            save_path = "trajectory_comparison.png"
        
        # 绘制轨迹对比图
        plot_trajectory_comparison(time_stamps, input_actions_log, actual_positions_log, cfg, save_path)
        
        # 绘制跟踪误差图
        plot_tracking_errors(time_stamps, input_actions_log, actual_positions_log, cfg, save_path)
        
        # 打印统计信息
        if len(input_actions_log) > 0 and len(actual_positions_log) > 0:
            scaled_actions = np.array(input_actions_log) * cfg.control.action_scale
            actual_positions = np.array(actual_positions_log)
            min_len = min(len(scaled_actions), len(actual_positions))
            errors = actual_positions[:min_len] - scaled_actions[:min_len]
            
            print("\n=== 轨迹跟踪统计信息 ===")
            print(f"总时长: {time_stamps[-1]:.2f} 秒")
            print(f"数据点数: {len(time_stamps)}")
            print(f"平均采样频率: {len(time_stamps)/time_stamps[-1]:.1f} Hz")
            
            print("\n机器人1关节误差统计 (RMSE):")
            for i in range(6):
                rmse = np.sqrt(np.mean(errors[:, i]**2))
                max_error = np.max(np.abs(errors[:, i]))
                print(f"  joint{i+1}: RMSE={rmse:.4f} rad, Max={max_error:.4f} rad")
            
            print("\n机器人2关节误差统计 (RMSE):")
            for i in range(6):
                rmse = np.sqrt(np.mean(errors[:, i+6]**2))
                max_error = np.max(np.abs(errors[:, i+6]))
                print(f"  joint{i+1}: RMSE={rmse:.4f} rad, Max={max_error:.4f} rad")
            
            overall_rmse = np.sqrt(np.mean(errors**2))
            print(f"\n整体RMSE: {overall_rmse:.4f} rad")


def create_interactive_plot(time_stamps, input_actions, actual_positions, cfg):
    """创建交互式动态轨迹图
    
    Args:
        time_stamps: 时间戳
        input_actions: 输入动作
        actual_positions: 实际位置
        cfg: 配置
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    
    # 准备数据
    scaled_actions = np.array(input_actions) * cfg.control.action_scale
    actual_positions = np.array(actual_positions)
    min_len = min(len(scaled_actions), len(actual_positions))
    
    # 创建图形
    fig, axes = plt.subplots(2, 6, figsize=(20, 10))
    fig.suptitle('交互式双臂机器人轨迹查看器', fontsize=16)
    
    # 添加时间滑块
    ax_slider = plt.axes([0.1, 0.02, 0.8, 0.03])
    time_slider = Slider(ax_slider, '时间进度', 0, len(time_stamps)-1, valinit=0, valfmt='%d')
    
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']
    lines_input = []
    lines_actual = []
    lines_current = []
    
    # 初始化图形
    for i in range(12):
        robot_idx = 0 if i < 6 else 1
        joint_idx = i if i < 6 else i - 6
        ax = axes[robot_idx, joint_idx]
        
        # 绘制完整轨迹
        line_input, = ax.plot(time_stamps[:min_len], scaled_actions[:min_len, i], 
                             'b-', alpha=0.5, label='input action')
        line_actual, = ax.plot(time_stamps[:min_len], actual_positions[:min_len, i], 
                              'r-', alpha=0.5, label='real position')
        
        # 当前时间点标记
        line_current, = ax.plot([], [], 'go', markersize=8, label='当前时刻')
        
        lines_input.append(line_input)
        lines_actual.append(line_actual)
        lines_current.append(line_current)
        
        ax.set_title(f'robot{robot_idx+1} - {joint_names[joint_idx]}')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('关节角度 (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def update_plot(frame):
        frame = int(frame)
        if frame < len(time_stamps) and frame < min_len:
            current_time = time_stamps[frame]
            for i in range(12):
                # 更新当前时间点标记
                lines_current[i].set_data([current_time], [actual_positions[frame, i]])
            fig.canvas.draw_idle()
    
    time_slider.on_changed(update_plot)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='双臂AirBot轨迹跟踪 (PyBullet)')
    parser.add_argument('--trajectory_file', type=str, required=True,
                      help='包含action轨迹的txt文件路径')
    parser.add_argument('--model_path', type=str, default='/home/yurong/下载/airbot_usd/urdf/airbot_play_v3_0.urdf',
                      help='机器人URDF文件路径')
    parser.add_argument('--headless', action='store_true',
                      help='无头模式运行（无GUI）')
    parser.add_argument('--no_plot', action='store_true',
                      help='不显示轨迹对比图')
    parser.add_argument('--save_plots', action='store_true',
                      help='保存轨迹图片到文件')
    parser.add_argument('--interactive', action='store_true',
                      help='显示交互式轨迹查看器')
    args = parser.parse_args()
    
    # 使用命令行参数更新配置
    cfg = Sim2simCfg()
    cfg.sim_config.urdf_model_path = args.model_path
    
    # 对于frame stacking，调整观测维度
    if cfg.env.frame_stack > 1:
        cfg.env.num_observations = cfg.env.num_single_obs * cfg.env.frame_stack
    
    # 初始化任务配置
    task_cfg = ReachTaskConfig()
    
    # 从txt文件解析action
    actions_list = parse_actions_from_txt(args.trajectory_file)
    
    if actions_list:
        # 运行双臂轨迹跟踪仿真
        run_dual_arm_trajectory_following(
            actions_list, 
            cfg, 
            task_cfg, 
            gui=not args.headless,
            plot_results=not args.no_plot,
            save_plots=args.save_plots
        )
        
        # 如果需要交互式查看器
        if args.interactive and not args.no_plot:
            print("启动交互式轨迹查看器...")
            # 重新运行一遍以获取数据（实际应用中可以保存数据避免重复运行）
            print("注意: 交互式查看器需要重新运行仿真以收集数据")
    else:
        print("无法解析action数据，请检查文件格式")