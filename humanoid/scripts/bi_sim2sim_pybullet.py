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

class Sim2simCfg:
    class sim_config:
        urdf_model_path = '/home/yurong/下载/airbot_usd/urdf/airbot_play_v3_0_gripper.urdf'  # Update with actual URDF path
        sim_duration = 60.0
        dt = 1/100
        decimation = 2

    class env:
        num_actions = 12  # 12 joints for dual-arm (6 + 6)
        num_single_obs = 51  
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

    class normalization:
        obs_scales = type('', (), {})()
        obs_scales.lin_vel = 1.0
        obs_scales.ang_vel = 1.0
        obs_scales.dof_pos = 1.0
        obs_scales.dof_vel = 0.05
        
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


def get_dual_arm_obs(robot1_id, robot2_id, joint_indices_robot1, joint_indices_robot2, 
                     end_effector_link_robot1, end_effector_link_robot2, cfg, task_cfg, prev_action):
    """Extract observations for dual-arm setup
    
    Args:
        robot1_id: PyBullet body ID for robot 1
        robot2_id: PyBullet body ID for robot 2
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
                           target_object_position(7), last_actions(12)] = 51 dims
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
    # print(f"relative pose: {ee1_relative_pose}")
    if ee2_quat is not None:
        # Transform ee2 pose to world coordinate system
        ee2_quat_wxyz = [ee2_quat[3], ee2_quat[0], ee2_quat[1], ee2_quat[2]]
        
        ee2_relative_pose = -np.array(ee2_quat_wxyz)
    else:
        ee2_relative_pose = np.zeros(4)
    # print(f"relative pose: {ee2_relative_pose}")
    # Target object position and orientation (7D: xyz + wxyz quaternion)
    target_quat = R.from_euler('ZYX', task_cfg.target_rpy[::-1]).as_quat()
    # Convert to [w, x, y, z] format
    target_quat_wxyz = [target_quat[1], target_quat[2],target_quat[3], target_quat[0]]
    target_object_position = np.concatenate([task_cfg.target_pos, target_quat_wxyz])
    
    # Create observation vector (51 dims total)
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
    
    # 7. Target object position (7 dims: xyz + wxyz quaternion)
    obs[32:39] = target_object_position
    
    # 8. Last actions (12 dims)
    obs[39:51] = prev_action
    
    print(f"obs: {obs}")
    return obs


def run_dual_arm_pybullet(policy, cfg, task_cfg, gui=True):
    """Run the PyBullet simulation with dual-arm setup
    
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
    
    # Create target visual
    task_cfg.target_visual_id = create_target_visual(
        task_cfg.target_pos, 
        task_cfg.target_rpy
    )
    
    # Initialize control variables for both robots
    target_q = np.zeros(cfg.env.num_actions, dtype=np.double)  # 12 joints total
    action = np.zeros(cfg.env.num_actions, dtype=np.double)
    prev_action = np.zeros(cfg.env.num_actions, dtype=np.double)
    
    # Initialize observation history for frame stacking if used
    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros(cfg.env.num_single_obs, dtype=np.float32))
    
    # Initialize simulation counter
    count_lowlevel = 0
    
    # Calculate total simulation steps
    dt = cfg.sim_config.dt
    total_steps = int(cfg.sim_config.sim_duration / dt)
    
    # Main simulation loop
    for step in tqdm(range(total_steps), desc="Simulating dual-arm..."):
        # Update target if needed
        if task_cfg.update_target(dt):
            ee1_pos = get_link_state(robot1_id, cfg.robot_config.end_effector_link_robot1)[0]
            ee2_pos = get_link_state(robot2_id, cfg.robot_config.end_effector_link_robot2)[0]
            print(f"Robot1 EE pos: {ee1_pos}")
            print(f"Robot2 EE pos: {ee2_pos}")
            print(f"New target: {task_cfg.target_pos}")
            update_target_visual(
                task_cfg.target_pos, 
                task_cfg.target_rpy
            )
        
        # Policy runs at lower frequency (based on decimation factor)
        if count_lowlevel % cfg.sim_config.decimation == 0:
            # Get full observation for dual-arm setup
            obs = get_dual_arm_obs(
                robot1_id, robot2_id, 
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
                maxVelocity=5.0
            )
        
        # Apply position control to robot 2 (next 6 joints)
        for i, joint_idx in enumerate(joint_indices_robot2):
            p.setJointMotorControl2(
                bodyUniqueId=robot2_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_q_clipped[i + 6],  # Offset by 6 for second robot
                maxVelocity=5.0
            )
        
        prev_action = action
        
        # Step simulation
        p.stepSimulation()
        
        # If using GUI, maintain real-time simulation
        if gui:
            time.sleep(dt)
        
        # Increment counter
        count_lowlevel += 1
    
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
    
    # Run dual-arm simulation
    run_dual_arm_pybullet(policy, cfg, task_cfg, gui=not args.headless)