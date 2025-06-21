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
import csv
import matplotlib.pyplot as plt

class Sim2simCfg:
    class sim_config:
        urdf_model_path = '/home/yurong/下载/airbot_usd/urdf/airbot_play_v3_0_gripper.urdf'  # Update with actual URDF path
        sim_duration = 8.0
        dt = 1/100
        decimation = 2

    class env:
        num_actions = 12  # 12 joints for dual-arm (6 + 6)
        num_single_obs = 51  # 原来51 + 7 (box pose: 3 pos + 4 quat)
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
        obs_scales.dof_vel = 1
        
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

link6_axis_visuals = []

def draw_axes(position, rpy=None, axis_length=0.1, visuals_list=None):
    """清除旧的三轴线段并绘制新的坐标轴"""
    if visuals_list is not None:
        # 清除所有旧线段
        for line_id in visuals_list:
            try:
                p.removeUserDebugItem(line_id)
            except:
                pass  # 如果已经被清除，忽略错误
        visuals_list.clear()

    # 构建姿态变换矩阵
    quat = p.getQuaternionFromEuler(rpy) if rpy is not None else [0, 0, 0, 1]
    rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    origin = np.array(position)
    axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # x, y, z

    # 添加新线段
    new_line_ids = []
    for i in range(3):  # x/y/z 三轴
        direction = rot_matrix[:, i]
        end = origin + axis_length * direction
        line_id = p.addUserDebugLine(origin, end, axis_colors[i], lineWidth=2.0, lifeTime=0)
        new_line_ids.append(line_id)

    if visuals_list is not None:
        visuals_list.extend(new_line_ids)


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


def get_object_pose_in_robot_frame(box_id, robot_id):
    """Get object pose in robot's root frame (equivalent to IsaacSim's object_pose_in_robot_root_frame)
    
    Args:
        box_id: PyBullet body ID for the box
        robot_id: PyBullet body ID for the robot
        
    Returns:
        Object pose in robot's frame [x, y, z, qw, qx, qy, qz] (7D)
    """
    # Get object pose in world frame
    object_pos_w, object_quat_w = p.getBasePositionAndOrientation(box_id)
    object_pos_w = np.array(object_pos_w)
    object_quat_w = np.array(object_quat_w)  # [x, y, z, w] format
    
    # Get robot base pose in world frame
    robot_pos_w, robot_quat_w = p.getBasePositionAndOrientation(robot_id)
    robot_pos_w = np.array(robot_pos_w)
    robot_quat_w = np.array(robot_quat_w)  # [x, y, z, w] format
    
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
    
    # Convert to [w, x, y, z] format to match IsaacSim convention
    object_quat_b_wxyz = np.array([object_quat_b[3], object_quat_b[0],object_quat_b[1], object_quat_b[2]])
    
    # Concatenate position and orientation
    object_pose_b = np.concatenate([object_pos_b, object_quat_b_wxyz])
    
    return object_pose_b


def get_dual_arm_obs(robot1_id, robot2_id, box_id, joint_indices_robot1, joint_indices_robot2, 
                     end_effector_link_robot1, end_effector_link_robot2, cfg, task_cfg, prev_action):
    """Extract observations for dual-arm setup with box object
    
    Args:
        robot1_id: PyBullet body ID for robot 1
        robot2_id: PyBullet body ID for robot 2
        box_id: PyBullet body ID for the box
        joint_indices_robot1: List of joint indices for robot 1
        joint_indices_robot2: List of joint indices for robot 2
        end_effector_link_robot1: Name of the end effector link for robot 1
        end_effector_link_robot2: Name of the end effector link for robot 2
        cfg: Configuration object
        task_cfg: Task configuration
        prev_action: Previous action (12-dim)
        
    Returns:
        Observation vector [joint_pos_robot1(6), joint_vel_robot1(6), 
                           joint_pos_robot2(6), joint_vel_robot2(6), 
                           ee1_relative_pose(4), ee2_relative_pose(4),
                           target_object_position(7), box_pose_in_robot1_frame(7),
                           last_actions(12)] = 58 dims
    """
    # Get joint positions and velocities for both robots
    q1, dq1 = get_joint_states(robot1_id, joint_indices_robot1)
    q2, dq2 = get_joint_states(robot2_id, joint_indices_robot2)
    
    # Get end effector states
    ee1_pos, ee1_quat = get_link_state(robot1_id, end_effector_link_robot1)
    ee2_pos, ee2_quat = get_link_state(robot2_id, end_effector_link_robot2)

    # Calculate poses relative to world coordinate system (robot1's baselink)
    if ee1_quat is not None:
        # ee1_relative_pose: position(3) + quaternion(1) = 4D (wxyz format)
        ee1_quat_wxyz = [ee1_quat[3], ee1_quat[0], ee1_quat[1], ee1_quat[2]]
        ee1_relative_pose = -np.array(ee1_quat_wxyz)
    else:
        ee1_relative_pose = np.zeros(4)
    
    if ee2_quat is not None:
        # Transform ee2 pose to world coordinate system
        ee2_quat_wxyz = [ee2_quat[3], ee2_quat[0], ee2_quat[1], ee2_quat[2]]
        ee2_relative_pose = -np.array(ee2_quat_wxyz)
    else:
        ee2_relative_pose = np.zeros(4)
    
    # Target object position and orientation (7D: xyz + wxyz quaternion)
    target_quat = R.from_euler('ZYX', task_cfg.target_rpy[::-1]).as_quat()
    # Convert to [w, x, y, z] format
    target_quat_wxyz = [target_quat[3], target_quat[0],target_quat[1], target_quat[2]]
    target_object_position = np.concatenate([task_cfg.target_pos, target_quat_wxyz])
    
    # Get box pose in robot1's frame (7D: xyz + wxyz quaternion)
    box_pose_in_robot1_frame = get_object_pose_in_robot_frame(box_id, robot1_id)
    
    # Create observation vector (58 dims total)
    obs = np.zeros(cfg.env.num_single_obs, dtype=np.float32)
    
    # 1. Joint positions robot 1 (6 dims, normalized)
    obs[0:6] = q1
    
    # 2. Joint velocities robot 1 (6 dims, normalized)
    obs[6:12] = dq1 * cfg.normalization.obs_scales.dof_vel
    
    # 3. Joint positions robot 2 (6 dims, normalized)
    obs[12:18] = q2 
    
    # 4. Joint velocities robot 2 (6 dims, normalized)
    obs[18:24] = dq2 * cfg.normalization.obs_scales.dof_vel
    
    # 5. End effector 1 relative pose (4 dims: xyz + w from wxyz quaternion)
    obs[24:28] = ee1_relative_pose
    
    # 6. End effector 2 relative pose (4 dims: xyz + w from wxyz quaternion)
    obs[28:32] = ee2_relative_pose
    
    
    # 8. Box pose in robot1 frame (7 dims: xyz + wxyz quaternion)
    obs[32:39] = target_object_position
    
    # 9. Last actions (12 dims)
    obs[39:51] = prev_action

    # # 7. Target object position (7 dims: xyz + wxyz quaternion)
    # obs[32:39] = box_pose_in_robot1_frame
    
    # # 8. Box pose in robot1 frame (7 dims: xyz + wxyz quaternion)
    # obs[39:46] = target_object_position
    
    # # 9. Last actions (12 dims)
    # obs[46:58] = prev_action

    print(f"obs: {obs}")
    return obs


class DualArmJointDataRecorder:
    def __init__(self, robot1_id, robot2_id, joint_indices_robot1, joint_indices_robot2, sampling_rate=50):
        """
        初始化双臂机械臂关节数据记录器
        
        参数:
        robot1_id: 机器人1的PyBullet ID
        robot2_id: 机器人2的PyBullet ID
        joint_indices_robot1: 机器人1的关节索引列表
        joint_indices_robot2: 机器人2的关节索引列表
        sampling_rate: 采样率(Hz)
        """
        self.robot1_id = robot1_id
        self.robot2_id = robot2_id
        self.joint_indices_robot1 = joint_indices_robot1
        self.joint_indices_robot2 = joint_indices_robot2
        self.sampling_rate = sampling_rate
        
        # 数据存储
        self.robot1_action_data = []
        self.robot1_actual_data = []
        self.robot2_action_data = []
        self.robot2_actual_data = []
        self.timestamps = []
        
        self.num_joints = 6  # 每个机器人6个关节
        self.start_time = None
        self.recording = False
        self.stop_requested = False
        
    def get_dual_arm_joint_states(self):
        """获取双臂机器人的实际关节状态"""
        # 获取机器人1的关节状态
        robot1_positions = []
        for idx in self.joint_indices_robot1:
            state = p.getJointState(self.robot1_id, idx)
            robot1_positions.append(state[0])
        
        # 获取机器人2的关节状态
        robot2_positions = []
        for idx in self.joint_indices_robot2:
            state = p.getJointState(self.robot2_id, idx)
            robot2_positions.append(state[0])
            
        return np.array(robot1_positions), np.array(robot2_positions)
        
    def start_recording(self, action_function):
        """
        开始记录双臂机器人的action和实际关节角度数据
        
        参数:
        action_function: 返回当前action (12维: robot1的6个关节 + robot2的6个关节)的函数
        """
        self.recording = True
        self.stop_requested = False
        self.start_time = time.time()
        print("开始记录双臂机器人数据，将持续到机械臂运动结束...")
        
        sample_count = 0
        last_progress_time = time.time()
        
        while not self.stop_requested:
            loop_start_time = time.time()
            
            # 获取当前action (12维)
            current_action = action_function()
            robot1_action = current_action[:6]  # 前6个是robot1的action
            robot2_action = current_action[6:]  # 后6个是robot2的action
            
            # 获取实际关节角度
            robot1_actual, robot2_actual = self.get_dual_arm_joint_states()
            
            # 记录时间戳和数据
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
            
            # 每隔几秒显示一次进度
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
            time.sleep(0.1)  # 等待记录线程停止
        print("双臂机器人数据记录已停止")
        
    def save_to_csv(self, folder="./data"):
        """将记录的数据保存为CSV文件"""
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # 保存Robot1的action和actual数据
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
        
        # 保存Robot2的action和actual数据
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
        """
        绘制双臂机器人的action和实际关节角度对比图
        生成两幅图：Robot1和Robot2
        """
        if len(self.timestamps) == 0:
            print("没有数据可以绘制！")
            return
            
        # 创建Robot1的对比图
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
        print("Robot1图表已生成并保存为 './data/robot1_action_vs_actual.png'")
        
        # 创建Robot2的对比图
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
        print("Robot2图表已生成并保存为 './data/robot2_action_vs_actual.png'")
        
        plt.show()


def run_dual_arm_pybullet(policy, cfg, task_cfg, gui=True):
    """Run the PyBullet simulation with dual-arm setup and box object
    
    Args:
        policy: PyTorch policy model
        cfg: Simulation configuration
        task_cfg: Task configuration
        gui: Whether to use GUI visualization
        
    Returns:
        None
    """
    # Initialize PyBullet
    physicsClient = init_pybullet(gui)
    
    # Load dual robots and get joint indices
    robot1_id, robot2_id, joint_indices_robot1, joint_indices_robot2 = load_dual_robots(cfg)
    
    # Load box object
    box_id = load_box(cfg)
    
    # Create target visual
    task_cfg.target_visual_id = create_target_visual(
        task_cfg.target_pos, 
        task_cfg.target_rpy
    )
    
    # Initialize control variables for both robots
    target_q = np.zeros(cfg.env.num_actions, dtype=np.double)  # 12 joints total
    action = np.zeros(cfg.env.num_actions, dtype=np.double)
    prev_action = np.zeros(cfg.env.num_actions, dtype=np.double)
    current_action = np.zeros(cfg.env.num_actions, dtype=np.double)
    
    # Initialize observation history for frame stacking if used
    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros(cfg.env.num_single_obs, dtype=np.float32))
    
    # Initialize simulation counter
    count_lowlevel = 0
    
    # Calculate total simulation steps
    dt = cfg.sim_config.dt
    total_steps = int(cfg.sim_config.sim_duration / dt)

    def get_current_action():
        """返回当前的action (12维)"""
        return current_action
    
    # 创建双臂数据记录器
    data_recorder = DualArmJointDataRecorder(
        robot1_id, robot2_id, 
        joint_indices_robot1, joint_indices_robot2, 
        sampling_rate=50
    )
    
    # Start recording data in a separate thread
    import threading
    recording_thread = threading.Thread(target=data_recorder.start_recording, args=(get_current_action,))
    recording_thread.daemon = True  # 设为守护线程，主程序结束时会自动终止
    recording_thread.start()
    
    # Main simulation loop
    for step in tqdm(range(total_steps), desc="Simulating dual-arm with box..."):
        # Update target if needed
        if task_cfg.update_target(dt):
            ee1_pos = get_link_state(robot1_id, cfg.robot_config.end_effector_link_robot1)[0]
            ee2_pos = get_link_state(robot2_id, cfg.robot_config.end_effector_link_robot2)[0]
            box_pos, _ = p.getBasePositionAndOrientation(box_id)
            print(f"Robot1 EE pos: {ee1_pos}")
            print(f"Robot2 EE pos: {ee2_pos}")
            print(f"Box pos: {box_pos}")
            print(f"New target: {task_cfg.target_pos}")
            update_target_visual(
                task_cfg.target_pos, 
                task_cfg.target_rpy
            )
        
        # Policy runs at lower frequency (based on decimation factor)
        if count_lowlevel % cfg.sim_config.decimation == 0:
            # Get full observation for dual-arm setup with box
            obs = get_dual_arm_obs(
                robot1_id, robot2_id, box_id,
                joint_indices_robot1, joint_indices_robot2,
                cfg.robot_config.end_effector_link_robot1,
                cfg.robot_config.end_effector_link_robot2,
                cfg, task_cfg, prev_action
            )
            
            # Update observation history
            hist_obs.append(obs)
            hist_obs.popleft()
            
            # Prepare input for policy
            if cfg.env.frame_stack > 1:
                policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                for i in range(cfg.env.frame_stack):
                    start_idx = i * cfg.env.num_single_obs
                    end_idx = start_idx + cfg.env.num_single_obs
                    policy_input[0, start_idx:end_idx] = hist_obs[i]
            else:
                policy_input = obs.reshape(1, -1)
                
            # Get action from policy
            action = policy(torch.tensor(policy_input, dtype=torch.float32))[0].detach().numpy()
            target_q = action * cfg.control.action_scale
            
            # 更新当前action用于记录
            current_action = target_q.copy()
            
            print(f"Action: {action}")
        
        # Apply position control to both robots
        target_q_clipped = np.clip(
            target_q, 
            cfg.robot_config.joint_lower_limits, 
            cfg.robot_config.joint_upper_limits
        )
        
        # Apply position control to robot 1 (first 6 joints)
        for i, joint_idx in enumerate(joint_indices_robot1):
            p.setJointMotorControl2(
                bodyUniqueId=robot1_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_q_clipped[i],
                maxVelocity=0.8
            )
        
        # Apply position control to robot 2 (next 6 joints)
        for i, joint_idx in enumerate(joint_indices_robot2):
            p.setJointMotorControl2(
                bodyUniqueId=robot2_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_q_clipped[i + 6],  # Offset by 6 for second robot
                maxVelocity=0.8
            )
        
        prev_action = action
        
        # Step simulation
        p.stepSimulation()
        
        # If using GUI, maintain real-time simulation
        if gui:
            time.sleep(dt)
        
        # Increment counter
        count_lowlevel += 1

  
    print("停止记录双臂机器人数据...")
    data_recorder.stop_recording()
    
    # 保存数据到CSV文件
    data_recorder.save_to_csv()
    
    # 绘制双臂机器人的对比图
    data_recorder.plot_dual_arm_data()
        
    # Disconnect when done
    p.disconnect()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual-Arm AirBot Task Deployment (PyBullet)')
    parser.add_argument('--load_model', type=str, required=True,
                      help='Path to the trained policy model')
    parser.add_argument('--model_path', type=str, default='/home/yurong/下载/airbot_usd/urdf/airbot_play_v3_0.urdf',
                      help='Path to the robot URDF file')
    parser.add_argument('--switch_interval', type=float, default=3.0,
                      help='Time interval (seconds) between switching actions')
    parser.add_argument('--headless', action='store_true',
                      help='Run in headless mode (no GUI)')
    args = parser.parse_args()
    
    # Update configuration with command line arguments
    cfg = Sim2simCfg()
    cfg.sim_config.urdf_model_path = args.model_path
    
    # For frame stacking, adjust observation dimension
    if cfg.env.frame_stack > 1:
        cfg.env.num_observations = cfg.env.num_single_obs * cfg.env.frame_stack
    
    # Initialize task configuration
    task_cfg = ReachTaskConfig()
    
    # Load policy
    policy = torch.jit.load(args.load_model)
    
    # Run dual-arm simulation with box
    run_dual_arm_pybullet(policy, cfg, task_cfg, gui=not args.headless)
