import math
import numpy as np
import time
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import random
import argparse
import os
import csv
import matplotlib.pyplot as plt
import airbot  # 真机控制库

class DualArmRealCfg:
    class sim_config:
        sim_duration = 8.0
        dt = 1/50  # 真机控制频率稍低
        decimation = 2

    class env:
        num_actions = 12  # 12 joints for dual-arm (6 + 6)
        num_single_obs = 51  # 51维观测
        num_observations = num_single_obs  # Will be updated if frame_stack > 1
        frame_stack = 1  # Modify if you use frame stacking

    class robot_config:
        # Default joint positions (home position) for both robots
        home_position_robot1 = np.zeros(6, dtype=np.double)
        home_position_robot2 = np.zeros(6, dtype=np.double)
        
        # Joint limits for both robots
        joint_lower_limits = np.array([-3.14, -2.96, -0.087, -2.96, -1.74, -3.14]*2, dtype=np.double)
        joint_upper_limits = np.array([2.09, 0.17, 3.14,  2.96, 1.74, 3.14]*2, dtype=np.double)
        
        # CAN接口配置
        robot1_can_interface = "can0"
        robot2_can_interface = "can1"  # 第二个机器人使用不同的CAN接口
        
        # 机器人配置
        end_mode = "None"
        
        # Robot base positions (for dual-arm setup - face to face mirror setup)
        robot1_base_position = [0, 0, 0]     # First robot at origin
        robot1_base_orientation = [0, 0, 0, 1]  # No rotation
        robot2_base_position = [0.47, 0.0, 0.0]   # Second robot 0.47m away
        robot2_base_orientation = [0.0, 0.0, 1.0, 0.0]  # 180 degree rotation around Z-axis

    class box_config:
        # Box configuration (虚拟的，用于观测计算)
        size = (0.15, 0.11, 0.22)  # width, height, depth
        position = [0.235, 0.0, 0.11]  # x, y, z
        orientation = [0, 0, 0, 1]  # quaternion [x, y, z, w]

    class normalization:
        obs_scales = type('', (), {})()
        obs_scales.lin_vel = 1.0
        obs_scales.ang_vel = 1.0
        obs_scales.dof_pos = 1.0
        obs_scales.dof_vel = 1
        
        clip_observations = 100.0
        clip_actions = 1.5
        
    class control:
        action_scale = 0.5  # Scale for joint position commands
        use_action_filter = True
        filter_size = 5
        decay_factor = 0.8
        max_change_rate = 0.1
        
        use_joint_state_filter = True
        joint_filter_size = 3
        joint_decay_factor = 0.7
        joint_moving_avg_weight = 0.3


class ReachTaskConfig:
    """Configuration for the reaching task"""
    def __init__(self):
        # Target position ranges
        self.pos_range_x = (0.235, 0.235)
        self.pos_range_y = (0, 0)
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


class ActionFilter:
    """Action filter to prevent oscillation and sudden changes"""
    def __init__(self, filter_size=5, decay_factor=0.8, max_change_rate=0.1):
        self.filter_size = filter_size
        self.decay_factor = decay_factor
        self.max_change_rate = max_change_rate
        self.action_history = deque(maxlen=filter_size)
        self.last_action = None
        
    def filter(self, raw_action):
        """Apply filtering to raw action"""
        if self.last_action is None:
            self.last_action = raw_action.copy()
            return raw_action
        
        # Rate limiting
        action_diff = raw_action - self.last_action
        action_diff_norm = np.linalg.norm(action_diff)
        
        if action_diff_norm > self.max_change_rate:
            action_diff = action_diff / action_diff_norm * self.max_change_rate
        
        # Apply exponential moving average
        filtered_action = self.last_action + self.decay_factor * action_diff
        
        # Store in history for potential future use
        self.action_history.append(filtered_action.copy())
        
        self.last_action = filtered_action.copy()
        return filtered_action


class JointStateFilter:
    """Joint state filter to smooth sensor readings"""
    def __init__(self, filter_size=3, decay_factor=0.7, moving_avg_weight=0.3):
        self.filter_size = filter_size
        self.decay_factor = decay_factor
        self.moving_avg_weight = moving_avg_weight
        self.position_history = deque(maxlen=filter_size)
        self.velocity_history = deque(maxlen=filter_size)
        self.last_filtered_pos = None
        self.last_filtered_vel = None
        
    def filter(self, positions, velocities):
        """Apply filtering to joint positions and velocities"""
        # Store raw data
        self.position_history.append(positions.copy())
        self.velocity_history.append(velocities.copy())
        
        if len(self.position_history) < 2:
            self.last_filtered_pos = positions.copy()
            self.last_filtered_vel = velocities.copy()
            return positions, velocities
        
        # Apply moving average to positions
        pos_array = np.array(list(self.position_history))
        avg_pos = np.mean(pos_array, axis=0)
        
        # Apply exponential smoothing
        if self.last_filtered_pos is not None:
            filtered_pos = (self.moving_avg_weight * avg_pos + 
                          (1 - self.moving_avg_weight) * self.last_filtered_pos)
        else:
            filtered_pos = avg_pos
        
        # Apply moving average to velocities
        vel_array = np.array(list(self.velocity_history))
        avg_vel = np.mean(vel_array, axis=0)
        
        if self.last_filtered_vel is not None:
            filtered_vel = (self.moving_avg_weight * avg_vel + 
                          (1 - self.moving_avg_weight) * self.last_filtered_vel)
        else:
            filtered_vel = avg_vel
        
        self.last_filtered_pos = filtered_pos.copy()
        self.last_filtered_vel = filtered_vel.copy()
        
        return filtered_pos, filtered_vel


def compute_pose_error_numpy(t_current, q_current, t_target, q_target, rot_error_type="axis_angle"):
    """
    计算位姿误差
    Args:
        t_current: 当前位置 [x, y, z]
        q_current: 当前四元数 [x, y, z, w]
        t_target: 目标位置 [x, y, z]
        q_target: 目标四元数 [x, y, z, w]
        rot_error_type: 旋转误差类型 ("axis_angle" 或 "quat")
    Returns:
        pos_error: 位置误差
        rot_error: 旋转误差
    """
    # 位置误差
    pos_error = t_target - t_current
    
    # 旋转误差
    if rot_error_type == "axis_angle":
        # 使用轴角表示
        rot_current = R.from_quat(q_current)
        rot_target = R.from_quat(q_target)
        rot_error_matrix = rot_target * rot_current.inv()
        rot_error = rot_error_matrix.as_rotvec()
    elif rot_error_type == "quat":
        # 四元数误差
        rot_current = R.from_quat(q_current)
        rot_target = R.from_quat(q_target)
        rot_error_quat = (rot_target * rot_current.inv()).as_quat()
        rot_error = rot_error_quat
    else:
        raise ValueError(f"Unknown rot_error_type: {rot_error_type}")
    
    return pos_error, rot_error


def get_object_pose_in_robot_frame(box_config, robot_base_pos, robot_base_quat):
    """Get virtual object pose in robot's root frame
    
    Args:
        box_config: Box configuration
        robot_base_pos: Robot base position
        robot_base_quat: Robot base quaternion [x, y, z, w]
        
    Returns:
        Object pose in robot's frame [x, y, z, qw, qx, qy, qz] (7D)
    """
    # Get object pose in world frame (virtual)
    object_pos_w = np.array(box_config.position)
    object_quat_w = np.array(box_config.orientation)  # [x, y, z, w] format
    
    # Get robot base pose in world frame
    robot_pos_w = np.array(robot_base_pos)
    robot_quat_w = np.array(robot_base_quat)  # [x, y, z, w] format
    
    # Convert quaternions to rotation matrices
    robot_rot_w = R.from_quat(robot_quat_w).as_matrix()
    
    # Transform object position to robot frame
    object_pos_relative = object_pos_w - robot_pos_w
    object_pos_b = robot_rot_w.T @ object_pos_relative
    
    # Transform object orientation to robot frame
    robot_rot_inv = R.from_quat(robot_quat_w).inv()
    object_rot_w = R.from_quat(object_quat_w)
    object_rot_b = robot_rot_inv * object_rot_w
    object_quat_b = object_rot_b.as_quat()  # [x, y, z, w] format
    
    # Convert to [w, x, y, z] format
    object_quat_b_wxyz = np.array([object_quat_b[3], object_quat_b[0], object_quat_b[1], object_quat_b[2]])
    
    # Concatenate position and orientation
    object_pose_b = np.concatenate([object_pos_b, object_quat_b_wxyz])
    
    return object_pose_b


def get_dual_arm_obs(robot1, robot2, cfg, task_cfg, prev_action, joint_state_filter1=None, joint_state_filter2=None):
    """Extract observations for dual-arm setup
    
    Args:
        robot1: First robot instance
        robot2: Second robot instance
        cfg: Configuration object
        task_cfg: Task configuration
        prev_action: Previous action (12-dim)
        joint_state_filter1: Joint state filter for robot1
        joint_state_filter2: Joint state filter for robot2
        
    Returns:
        Observation vector
    """
    try:
        # Get joint states for both robots
        q1 = np.array(robot1.get_current_joint_q())
        dq1 = np.array(robot1.get_current_joint_dq())
        q2 = np.array(robot2.get_current_joint_q())
        dq2 = np.array(robot2.get_current_joint_dq())
        
        # Apply joint state filtering if enabled
        if joint_state_filter1 is not None:
            q1, dq1 = joint_state_filter1.filter(q1, dq1)
        if joint_state_filter2 is not None:
            q2, dq2 = joint_state_filter2.filter(q2, dq2)
        
        # Get end effector states
        ee1_pos = np.array(robot1.get_current_translation())
        ee1_quat = np.array(robot1.get_current_rotation())  # [x, y, z, w]
        ee2_pos = np.array(robot2.get_current_translation())
        ee2_quat = np.array(robot2.get_current_rotation())  # [x, y, z, w]
        
        # Calculate poses relative to world coordinate system
        ee1_quat_wxyz = [ee1_quat[3], ee1_quat[0], ee1_quat[1], ee1_quat[2]]
        ee1_relative_pose = -np.array(ee1_quat_wxyz)
        
        ee2_quat_wxyz = [ee2_quat[3], ee2_quat[0], ee2_quat[1], ee2_quat[2]]
        ee2_relative_pose = -np.array(ee2_quat_wxyz)
        
        # Target object position and orientation (7D: xyz + wxyz quaternion)
        target_quat = R.from_euler('ZYX', task_cfg.target_rpy[::-1]).as_quat()
        target_quat_wxyz = [target_quat[3], target_quat[0], target_quat[1], target_quat[2]]
        target_object_position = np.concatenate([task_cfg.target_pos, target_quat_wxyz])
        
        # Create observation vector
        obs = np.zeros(cfg.env.num_single_obs, dtype=np.float32)
        
        # Joint positions and velocities for both robots
        obs[0:6] = q1
        obs[6:12] = dq1 * cfg.normalization.obs_scales.dof_vel
        obs[12:18] = q2
        obs[18:24] = dq2 * cfg.normalization.obs_scales.dof_vel
        
        # End effector relative poses
        obs[24:28] = ee1_relative_pose
        obs[28:32] = ee2_relative_pose
        
        # Target object position
        obs[32:39] = target_object_position
        
        # Last actions
        obs[39:51] = prev_action
        
        return obs
        
    except Exception as e:
        print(f"Error getting observations: {e}")
        return np.zeros(cfg.env.num_single_obs, dtype=np.float32)


class DualArmJointDataRecorder:
    def __init__(self, robot1, robot2, sampling_rate=50):
        """
        初始化双臂机械臂关节数据记录器
        """
        self.robot1 = robot1
        self.robot2 = robot2
        self.sampling_rate = sampling_rate
        
        # 数据存储
        self.robot1_action_data = []
        self.robot1_actual_data = []
        self.robot2_action_data = []
        self.robot2_actual_data = []
        self.timestamps = []
        
        self.num_joints = 6
        self.start_time = None
        self.recording = False
        self.stop_requested = False
        
    def get_dual_arm_joint_states(self):
        """获取双臂机器人的实际关节状态"""
        try:
            robot1_positions = np.array(self.robot1.get_current_joint_q())
            robot2_positions = np.array(self.robot2.get_current_joint_q())
            return robot1_positions, robot2_positions
        except Exception as e:
            print(f"Error getting joint states: {e}")
            return np.zeros(6), np.zeros(6)
        
    def start_recording(self, action_function):
        """开始记录双臂机器人数据"""
        self.recording = True
        self.stop_requested = False
        self.start_time = time.time()
        print("开始记录双臂机器人数据...")
        
        sample_count = 0
        last_progress_time = time.time()
        
        while not self.stop_requested:
            loop_start_time = time.time()
            
            # 获取当前action (12维)
            current_action = action_function()
            robot1_action = current_action[:6]
            robot2_action = current_action[6:]
            
            # 获取实际关节角度
            robot1_actual, robot2_actual = self.get_dual_arm_joint_states()
            
            # 记录数据
            curr_time = time.time() - self.start_time
            self.timestamps.append(curr_time)
            self.robot1_action_data.append(robot1_action)
            self.robot1_actual_data.append(robot1_actual)
            self.robot2_action_data.append(robot2_action)
            self.robot2_actual_data.append(robot2_actual)
            
            # 控制采样率
            elapsed = time.time() - loop_start_time
            sleep_time = max(0, 1/self.sampling_rate - elapsed)
            time.sleep(sleep_time)
            
            sample_count += 1
            
            # 进度显示
            if time.time() - last_progress_time > 5:
                print(f"数据记录中... 已记录 {sample_count} 个采样点，耗时 {curr_time:.1f} 秒")
                last_progress_time = time.time()
        
        self.recording = False
        record_duration = time.time() - self.start_time
        print(f"双臂机器人数据记录完成! 共记录了 {len(self.timestamps)} 个采样点，持续了 {record_duration:.1f} 秒")
    
    def stop_recording(self):
        """停止记录数据"""
        self.stop_requested = True
        while self.recording:
            time.sleep(0.1)
        print("双臂机器人数据记录已停止")
        
    def save_to_csv(self, folder="./data"):
        """保存数据到CSV文件"""
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # 保存Robot1数据
        robot1_action_filename = os.path.join(folder, f"robot1_action_{timestamp}.csv")
        with open(robot1_action_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp'] + [f'joint_{i+1}' for i in range(self.num_joints)]
            writer.writerow(header)
            for ts, action in zip(self.timestamps, self.robot1_action_data):
                writer.writerow([ts] + list(action))
        
        robot1_actual_filename = os.path.join(folder, f"robot1_actual_{timestamp}.csv")
        with open(robot1_actual_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp'] + [f'joint_{i+1}' for i in range(self.num_joints)]
            writer.writerow(header)
            for ts, actual in zip(self.timestamps, self.robot1_actual_data):
                writer.writerow([ts] + list(actual))
        
        # 保存Robot2数据
        robot2_action_filename = os.path.join(folder, f"robot2_action_{timestamp}.csv")
        with open(robot2_action_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp'] + [f'joint_{i+1}' for i in range(self.num_joints)]
            writer.writerow(header)
            for ts, action in zip(self.timestamps, self.robot2_action_data):
                writer.writerow([ts] + list(action))
        
        robot2_actual_filename = os.path.join(folder, f"robot2_actual_{timestamp}.csv")
        with open(robot2_actual_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp'] + [f'joint_{i+1}' for i in range(self.num_joints)]
            writer.writerow(header)
            for ts, actual in zip(self.timestamps, self.robot2_actual_data):
                writer.writerow([ts] + list(actual))
                
        print(f"双臂机器人数据已保存至:")
        print(f"  Robot1 Action: {robot1_action_filename}")
        print(f"  Robot1 Actual: {robot1_actual_filename}")
        print(f"  Robot2 Action: {robot2_action_filename}")
        print(f"  Robot2 Actual: {robot2_actual_filename}")
        
        return robot1_action_filename, robot1_actual_filename, robot2_action_filename, robot2_actual_filename
    
    def plot_dual_arm_data(self):
        """绘制双臂机器人数据对比图"""
        if len(self.timestamps) == 0:
            print("没有数据可以绘制！")
            return
            
        # Robot1对比图
        fig1, axes1 = plt.subplots(3, 2, figsize=(15, 10))
        axes1 = axes1.flatten()
        fig1.suptitle('Robot1: Action vs Actual Joint Angles', fontsize=16)
        
        for i in range(self.num_joints):
            ax = axes1[i]
            robot1_actions = [action[i] for action in self.robot1_action_data]
            robot1_actuals = [actual[i] for actual in self.robot1_actual_data]
            
            ax.plot(self.timestamps, robot1_actions, 'b-', label='Action', linewidth=2)
            ax.plot(self.timestamps, robot1_actuals, 'r-', label='Actual', linewidth=2)
            ax.set_title(f'Robot1 Joint {i+1}', fontsize=12)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Joint Angle (rad)')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('./data/robot1_action_vs_actual.png', dpi=300, bbox_inches='tight')
        print("Robot1图表已保存为 './data/robot1_action_vs_actual.png'")
        
        # Robot2对比图
        fig2, axes2 = plt.subplots(3, 2, figsize=(15, 10))
        axes2 = axes2.flatten()
        fig2.suptitle('Robot2: Action vs Actual Joint Angles', fontsize=16)
        
        for i in range(self.num_joints):
            ax = axes2[i]
            robot2_actions = [action[i] for action in self.robot2_action_data]
            robot2_actuals = [actual[i] for actual in self.robot2_actual_data]
            
            ax.plot(self.timestamps, robot2_actions, 'b-', label='Action', linewidth=2)
            ax.plot(self.timestamps, robot2_actuals, 'r-', label='Actual', linewidth=2)
            ax.set_title(f'Robot2 Joint {i+1}', fontsize=12)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Joint Angle (rad)')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('./data/robot2_action_vs_actual.png', dpi=300, bbox_inches='tight')
        print("Robot2图表已保存为 './data/robot2_action_vs_actual.png'")
        
        plt.show()


def run_dual_arm_real_robot(policy, cfg, task_cfg):
    """Run the real dual-arm robot control
    
    Args:
        policy: PyTorch policy model
        cfg: Simulation configuration
        task_cfg: Task configuration
        
    Returns:
        None
    """
    print("初始化双臂真机控制系统...")
    
    # 创建双臂机器人实例
    try:
        print(f"连接Robot1 (CAN接口: {cfg.robot_config.robot1_can_interface})...")
        robot1 = airbot.create_agent(
            can_interface=cfg.robot_config.robot1_can_interface, 
            end_mode=cfg.robot_config.end_mode
        )
        print("Robot1连接成功!")
        
        print(f"连接Robot2 (CAN接口: {cfg.robot_config.robot2_can_interface})...")
        robot2 = airbot.create_agent(
            can_interface=cfg.robot_config.robot2_can_interface, 
            end_mode=cfg.robot_config.end_mode
        )
        print("Robot2连接成功!")
        
    except Exception as e:
        print(f"机器人连接失败: {e}")
        return
    
    # 初始化控制变量 (12维：两个6自由度机器人)
    target_q = np.zeros(cfg.env.num_actions, dtype=np.double)
    action = np.zeros(cfg.env.num_actions, dtype=np.double)
    prev_action = np.zeros(cfg.env.num_actions, dtype=np.double)
    current_target_q_clipped = np.zeros(cfg.env.num_actions, dtype=np.double)
    
    # 初始化动作滤波器
    action_filter = None
    if cfg.control.use_action_filter:
        action_filter = ActionFilter(
            filter_size=cfg.control.filter_size,
            decay_factor=cfg.control.decay_factor,
            max_change_rate=cfg.control.max_change_rate
        )
        print("Action filtering enabled")
    
    # 初始化关节状态滤波器（双臂各一个）
    joint_state_filter1 = None
    joint_state_filter2 = None
    if cfg.control.use_joint_state_filter:
        joint_state_filter1 = JointStateFilter(
            filter_size=cfg.control.joint_filter_size,
            decay_factor=cfg.control.joint_decay_factor,
            moving_avg_weight=cfg.control.joint_moving_avg_weight
        )
        joint_state_filter2 = JointStateFilter(
            filter_size=cfg.control.joint_filter_size,
            decay_factor=cfg.control.joint_decay_factor,
            moving_avg_weight=cfg.control.joint_moving_avg_weight
        )
        print("Joint state filtering enabled for both arms")
    
    # 初始化数据记录器
    data_recorder = DualArmJointDataRecorder(robot1, robot2, sampling_rate=50)
    
    # 初始化观测历史
    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros(cfg.env.num_single_obs, dtype=np.float32))
    
    # 初始化仿真计数器
    count_lowlevel = 0
    
    # 计算总仿真步数
    dt = cfg.sim_config.dt
    total_steps = int(cfg.sim_config.sim_duration / dt)
    
    # 获取当前目标关节位置函数（用于数据记录器）
    def get_current_target_q():
        return current_target_q_clipped
    
    # 调试日志 - 双臂数据
    raw_joint_positions_log_robot1 = []
    filtered_joint_positions_log_robot1 = []
    raw_joint_positions_log_robot2 = []
    filtered_joint_positions_log_robot2 = []
    
    # 在单独线程中启动数据记录
    import threading
    recording_thread = threading.Thread(target=data_recorder.start_recording, args=(get_current_target_q,))
    recording_thread.daemon = True
    recording_thread.start()
    
    try:
        print("开始双臂机器人控制...")
        # 主控制循环
        for step in tqdm(range(total_steps), desc="Running Dual-Arm Robots..."):
            # 策略以较低频率运行（基于decimation因子）
            if count_lowlevel % cfg.sim_config.decimation == 0:
                # 在每个控制周期检查是否需要更新目标位置
                if task_cfg.update_target(dt * cfg.sim_config.decimation):
                    new_quat = R.from_euler('ZYX', task_cfg.target_rpy[::-1]).as_quat()
                    # Convert to [x, y, z, w] format
                    new_quat = np.array([new_quat[1], new_quat[2], new_quat[3], new_quat[0]])
                    print(f"目标位置更新: pos={task_cfg.target_pos}, rpy={task_cfg.target_rpy}")
                
                # 获取原始关节位置用于调试（双臂）
                if cfg.control.use_joint_state_filter:
                    raw_positions_robot1 = robot1.get_current_joint_q()
                    raw_positions_robot2 = robot2.get_current_joint_q()
                    raw_joint_positions_log_robot1.append(raw_positions_robot1)
                    raw_joint_positions_log_robot2.append(raw_positions_robot2)
                
                # 获取滤波后的观测（双臂）
                obs = get_dual_arm_obs(robot1, robot2, cfg, task_cfg, prev_action, 
                                     joint_state_filter1, joint_state_filter2)
                
                # 存储滤波后的关节位置
                if cfg.control.use_joint_state_filter:
                    filtered_joint_positions_log_robot1.append(obs[0:6])   # Robot1关节
                    filtered_joint_positions_log_robot2.append(obs[12:18]) # Robot2关节
                
                # 更新观测中的前一个动作
                obs[39:51] = prev_action

                # 更新观测历史
                hist_obs.append(obs)
                hist_obs.popleft()
                
                # 为策略准备输入
                if cfg.env.frame_stack > 1:
                    policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                    for i in range(cfg.env.frame_stack):
                        start_idx = i * cfg.env.num_single_obs
                        end_idx = start_idx + cfg.env.num_single_obs
                        policy_input[0, start_idx:end_idx] = hist_obs[i]
                else:
                    policy_input = obs.reshape(1, -1)
                
                # 计算下一个动作
                raw_action = policy(torch.tensor(policy_input, dtype=torch.float32))[0].detach().numpy()
                
                # 应用动作滤波
                if action_filter is not None:
                    action = action_filter.filter(raw_action)
                else:
                    action = raw_action
                
                # 缩放动作
                target_q = action * cfg.control.action_scale
                
                # 将动作限制在关节限制范围内
                target_q_clipped = np.clip(
                    target_q, 
                    cfg.robot_config.joint_lower_limits, 
                    cfg.robot_config.joint_upper_limits
                )
                
                # 更新当前目标关节位置（用于数据记录器）
                current_target_q_clipped = target_q_clipped.copy()
                
                # 分离两个机器人的目标关节位置
                target_q_robot1 = target_q_clipped[:6]   # 前6个关节给Robot1
                target_q_robot2 = target_q_clipped[6:]   # 后6个关节给Robot2
                
                # 发送目标关节位置到两个机器人
                try:
                    # 控制Robot1
                    robot1.set_target_joint_q(target_q_robot1, vel=3.5, blocking=False)
                    # 控制Robot2
                    robot2.set_target_joint_q(target_q_robot2, vel=3.5, blocking=False)
                    
                    print(f"发送控制命令时间戳: {time.time():.6f} 秒")

                    # 获取两个机器人的末端执行器状态
                    effector_pos1 = robot1.get_current_translation()
                    effector_quat1 = robot1.get_current_rotation()  # [x, y, z, w] 格式
                    effector_pos2 = robot2.get_current_translation()
                    effector_quat2 = robot2.get_current_rotation()  # [x, y, z, w] 格式

                    # Debug position info for both robots
                    print(f"Robot1 - Target pos: {task_cfg.target_pos}, Actual pos: {effector_pos1}")
                    print(f"Robot2 - Actual pos: {effector_pos2}")

                    # 构造目标朝向四元数：ZYX 欧拉角转四元数，注意需要转换为 [x, y, z, w]
                    target_euler = task_cfg.target_rpy[::-1]  # 从 RPY → YXZ
                    target_quat = R.from_euler('ZYX', target_euler).as_quat()  # 得到 [x, y, z, w]

                    print(f"Target quat: {target_quat}")
                    print(f"Robot1 quat: {effector_quat1}")
                    print(f"Robot2 quat: {effector_quat2}")

                    # --- 统一误差估计 for Robot1 ---
                    pos_error1, rot_error1 = compute_pose_error_numpy(
                        t_current=np.array(effector_pos1),
                        q_current=np.array(effector_quat1),
                        t_target=np.array(task_cfg.target_pos),
                        q_target=np.array(target_quat),
                        rot_error_type="axis_angle",  # 或 "quat"
                    )
                    rot_norm1 = np.linalg.norm(rot_error1)
                    print(f"Robot1 - Position error: {pos_error1}, Norm: {np.linalg.norm(pos_error1):.4f}")
                    print(f"Robot1 - Rotation error: {rot_error1}, Angle (rad): {rot_norm1:.4f}")

                    # --- 统一误差估计 for Robot2 (如果需要的话，可以有不同的目标) ---
                    pos_error2, rot_error2 = compute_pose_error_numpy(
                        t_current=np.array(effector_pos2),
                        q_current=np.array(effector_quat2),
                        t_target=np.array(task_cfg.target_pos),  # 可以为Robot2设置不同目标
                        q_target=np.array(target_quat),
                        rot_error_type="axis_angle",
                    )
                    rot_norm2 = np.linalg.norm(rot_error2)
                    print(f"Robot2 - Position error: {pos_error2}, Norm: {np.linalg.norm(pos_error2):.4f}")
                    print(f"Robot2 - Rotation error: {rot_error2}, Angle (rad): {rot_norm2:.4f}")

                    # 打印关节状态滤波效果（如果启用）
                    if (cfg.control.use_joint_state_filter and 
                        len(raw_joint_positions_log_robot1) > 0 and 
                        len(raw_joint_positions_log_robot2) > 0):
                        
                        # Robot1滤波效果
                        raw_pos1 = np.array(raw_joint_positions_log_robot1[-1])
                        filtered_pos1 = np.array(filtered_joint_positions_log_robot1[-1])
                        filter_diff1 = np.linalg.norm(raw_pos1 - filtered_pos1)
                        
                        # Robot2滤波效果
                        raw_pos2 = np.array(raw_joint_positions_log_robot2[-1])
                        filtered_pos2 = np.array(filtered_joint_positions_log_robot2[-1])
                        filter_diff2 = np.linalg.norm(raw_pos2 - filtered_pos2)
                        
                        print(f"Robot1 joint filter effect (norm): {filter_diff1:.6f}")
                        print(f"Robot2 joint filter effect (norm): {filter_diff2:.6f}")
                        
                        if filter_diff1 > 0.05:  # 只在有显著差异时显示详细信息
                            print(f"  Robot1 Raw joints: {raw_pos1}")
                            print(f"  Robot1 Filtered: {filtered_pos1}")
                        
                        if filter_diff2 > 0.05:
                            print(f"  Robot2 Raw joints: {raw_pos2}")
                            print(f"  Robot2 Filtered: {filtered_pos2}")

                except Exception as e:
                    print(f"Error setting joint positions: {e}")
                    break
                
                # 存储前一个动作
                prev_action = action
            
            # 等待一个时间步的持续时间
            time.sleep(dt)
            
            # 增加计数器
            count_lowlevel += 1
            
    except KeyboardInterrupt:
        print("操作被用户中断")
    
    finally:
        # 在归位之前停止记录
        print("停止记录数据...")
        data_recorder.stop_recording()
        
        # 绘制关节滤波对比图（如果启用）
        if (cfg.control.use_joint_state_filter and 
            len(raw_joint_positions_log_robot1) > 0 and 
            len(raw_joint_positions_log_robot2) > 0):
            try:
                print("生成双臂关节滤波对比图...")
                
                # 转换日志为numpy数组便于绘图
                raw_positions_array1 = np.array(raw_joint_positions_log_robot1)
                filtered_positions_array1 = np.array(filtered_joint_positions_log_robot1)
                raw_positions_array2 = np.array(raw_joint_positions_log_robot2)
                filtered_positions_array2 = np.array(filtered_joint_positions_log_robot2)
                
                # Robot1滤波对比图
                plt.figure(figsize=(15, 10))
                for i in range(6):  # 6个关节
                    plt.subplot(3, 2, i+1)
                    plt.plot(raw_positions_array1[:, i], 'r-', label='Raw', alpha=0.7)
                    plt.plot(filtered_positions_array1[:, i], 'b-', label='Filtered', linewidth=2)
                    plt.title(f'Robot1 Joint {i+1} Filtering')
                    plt.xlabel('Time Step')
                    plt.ylabel('Joint Position (rad)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('./data/robot1_joint_filtering_comparison.png', dpi=300)
                print("Robot1滤波效果对比图已保存至 './data/robot1_joint_filtering_comparison.png'")
                
                # Robot2滤波对比图
                plt.figure(figsize=(15, 10))
                for i in range(6):  # 6个关节
                    plt.subplot(3, 2, i+1)
                    plt.plot(raw_positions_array2[:, i], 'r-', label='Raw', alpha=0.7)
                    plt.plot(filtered_positions_array2[:, i], 'b-', label='Filtered', linewidth=2)
                    plt.title(f'Robot2 Joint {i+1} Filtering')
                    plt.xlabel('Time Step')
                    plt.ylabel('Joint Position (rad)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('./data/robot2_joint_filtering_comparison.png', dpi=300)
                print("Robot2滤波效果对比图已保存至 './data/robot2_joint_filtering_comparison.png'")
                
            except Exception as e:
                print(f"生成滤波对比图时出错: {e}")
        
        # 任务完成时归位
        try:
            print("任务完成，双臂机械臂归位中...")
            # 两个机器人同时归位
            robot1.set_target_joint_q(cfg.robot_config.home_position_robot1, vel=0.2)
            robot2.set_target_joint_q(cfg.robot_config.home_position_robot2, vel=0.2)
            time.sleep(5)  # 等待机械臂归位
            
            # 保存目标历史（如果有该方法）
            if hasattr(task_cfg, 'save_target_history'):
                task_cfg.save_target_history("targets.csv")
            
            # 保存和绘制数据
            print("保存数据并生成图表...")
            robot1_action_file, robot1_actual_file, robot2_action_file, robot2_actual_file = data_recorder.save_to_csv()
            data_recorder.plot_dual_arm_data()
            
        except Exception as e:
            print(f"归位过程中出错: {e}")
            # 即使归位出错也保存数据
            print("保存已记录的数据...")
            robot1_action_file, robot1_actual_file, robot2_action_file, robot2_actual_file = data_recorder.save_to_csv()
            data_recorder.plot_dual_arm_data()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AirBot Reach Task Deployment with Filtering')
    parser.add_argument('--load_model', type=str, required=True,
                      help='Path to the trained policy model')
    parser.add_argument('--target_update_time', type=float, default=10.0,
                      help='Time interval (seconds) between target updates')
    
    # Action filter parameters
    parser.add_argument('--use_filter', type=bool, default=False,
                      help='Enable action filtering to prevent oscillation')
    parser.add_argument('--filter_size', type=int, default=5,
                      help='Size of action history for moving average')
    parser.add_argument('--decay_factor', type=float, default=0.99,
                      help='Weight for exponential smoothing (higher = more smoothing)')
    parser.add_argument('--max_change_rate', type=float, default=0.05,
                      help='Maximum allowed change in action per timestep')
    
    # Joint state filter parameters
    parser.add_argument('--use_joint_filter', type=bool, default=False,
                      help='Enable joint state filtering')
    parser.add_argument('--joint_filter_size', type=int, default=5,
                      help='Size of joint state history for moving average')
    parser.add_argument('--joint_decay_factor', type=float, default=0.99,
                      help='Weight for exponential smoothing for joint states')
    parser.add_argument('--joint_moving_avg_weight', type=float, default=0.6,
                      help='Weight for moving average vs exponential smoothing for joint states')
    
    args = parser.parse_args()
    
    # Update configuration 
    cfg = DualArmRealCfg()
    
    # Apply command line arguments for action filtering
    cfg.control.use_action_filter = args.use_filter
    cfg.control.filter_size = args.filter_size
    cfg.control.decay_factor = args.decay_factor
    cfg.control.max_change_rate = args.max_change_rate
    
    # Apply command line arguments for joint state filtering
    cfg.control.use_joint_state_filter = args.use_joint_filter
    cfg.control.joint_filter_size = args.joint_filter_size
    cfg.control.joint_decay_factor = args.joint_decay_factor
    cfg.control.joint_moving_avg_weight = args.joint_moving_avg_weight
    
    # For frame stacking, adjust observation dimension
    if cfg.env.frame_stack > 1:
        cfg.env.num_observations = cfg.env.num_single_obs * cfg.env.frame_stack
    
    # Initialize task configuration
    task_cfg = ReachTaskConfig()
    # 应用命令行参数设置的目标更新时间
    task_cfg.target_update_time = args.target_update_time
    
    # Initialize the joint state filter
    # global joint_state_filter

    if cfg.control.use_joint_state_filter:
        joint_state_filter = JointStateFilter(
            filter_size=cfg.control.joint_filter_size,
            decay_factor=cfg.control.joint_decay_factor,
            moving_avg_weight=cfg.control.joint_moving_avg_weight
        )
        print("Joint state filtering enabled to reduce sensing noise and oscillation")
    
    # Load policy
    policy = torch.jit.load(args.load_model)
    
    # Run robot with data recording throughout the entire execution
    run_dual_arm_real_robot(policy, cfg, task_cfg)