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


class Sim2simCfg:
    class sim_config:
        urdf_model_path = '/home/yurong/下载/airbot_usd/urdf/airbot_play_v3_0_gripper.urdf'  # Update with actual URDF path
        sim_duration = 60.0
        dt = 1/60
        decimation = 2

    class env:
        num_actions = 6  # 6 joints for the airbot
        num_single_obs = 25  # Adapt based on your observation space
        num_observations = num_single_obs  # Will be updated if frame_stack > 1
        frame_stack = 1  # Modify if you use frame stacking

    class robot_config:
        # Default joint positions (home position)
        home_position = np.zeros(6, dtype=np.double)
        # Joint limits
        joint_lower_limits = np.array([-3.14, -2.96, -0.087, -2.96, -1.74, -3.14], dtype=np.double)
        joint_upper_limits = np.array([2.09, 0.17, 3.14,  2.96, 1.74, 3.14], dtype=np.double)
        # Joint names (for PyBullet to identify joints)
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        # End effector link name
        end_effector_link = "link6"  # Update with actual end effector link name

    class normalization:
        obs_scales = type('', (), {})()
        obs_scales.lin_vel = 1.0
        obs_scales.ang_vel = 0.25
        obs_scales.dof_pos = 1.0
        obs_scales.dof_vel = 0.05
        
        clip_observations = 100.0
        clip_actions = 1.5
        
    class control:
        action_scale = 0.5  # Scale for joint position commands


class ReachTaskConfig:
    """Configuration for the reaching task with proper blocking behavior"""
    def __init__(self, block_duration=5.0):
        # Target position ranges 
        self.pos_range_x = (0.5, 0.5)
        self.pos_range_y = (0, 0)
        self.pos_range_z = (0.3, 0.3)
        self.pos_range_roll = (math.pi/2, math.pi/2)
        self.pos_range_pitch = (math.pi, math.pi)
        self.pos_range_yaw = (math.pi/2, math.pi/2)
        
        # State management
        self.block_duration = block_duration
        self.blocked_time = 0.0
        self.is_first_action = True
        self.is_blocking = False
        
        # Target positions
        self.target_pos_1 = np.array([
            random.uniform(*self.pos_range_x),
            random.uniform(*self.pos_range_y),
            random.uniform(*self.pos_range_z)
        ])
        self.target_rpy_1 = np.array([
            random.uniform(*self.pos_range_roll),
            random.uniform(*self.pos_range_pitch),
            random.uniform(*self.pos_range_yaw)
        ])
        
        self.target_pos_2 = np.array([
            random.uniform(*self.pos_range_x),
            random.uniform(*self.pos_range_y),
            random.uniform(*self.pos_range_z)
        ])
        self.target_rpy_2 = np.array([
            random.uniform(*self.pos_range_roll),
            random.uniform(*self.pos_range_pitch),
            random.uniform(*self.pos_range_yaw)
        ])
        
        # Current target (will be updated)
        self.current_target_pos = self.target_pos_1
        self.current_target_rpy = self.target_rpy_1
        
        # Target visualization
        self.target_visual_id = None
    
    def update_task_state(self, dt):
        """Manage task state with proper blocking"""
        if self.is_blocking:
            self.blocked_time += dt
            if self.blocked_time >= self.block_duration:
                # Switch to next target and reset blocking
                self.is_blocking = False
                self.blocked_time = 0.0
                self.is_first_action = False
                
                # Switch to second target
                self.current_target_pos = self.target_pos_2
                self.current_target_rpy = self.target_rpy_2
                
                return True  # Indicates a target change
        
        return False


def init_pybullet(gui=True):
    """Initialize PyBullet simulation environment"""
    if gui:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
        
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0/240.0)  # Default PyBullet timestep
    
    # Load plane
    p.loadURDF("plane.urdf")
    


def load_robot(cfg):
    """Load robot URDF into PyBullet
    
    Args:
        cfg: Configuration object
        
    Returns:
        Robot ID, list of controllable joint indices
    """
    robot_id = p.loadURDF(cfg.sim_config.urdf_model_path, 
                         basePosition=[0, 0, 0],
                         useFixedBase=True)
    
    # Get controllable joint indices
    joint_indices = []
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        if joint_name in cfg.robot_config.joint_names:
            joint_indices.append(i)
    
    # Reset joints to home position
    for idx in joint_indices:
        p.resetJointState(robot_id, idx, 0.0)
    
    return robot_id, joint_indices


# def create_target_visual(position, rpy=None):
#     """Create visual marker for target position
    
#     Args:
#         position: Target position [x, y, z]
#         rpy: Target orientation as roll, pitch, yaw (optional)
        
#     Returns:
#         Visual shape ID
#     """
#     if rpy is not None:
#         quat = p.getQuaternionFromEuler([rpy[0], rpy[1], rpy[2]])
#     else:
#         quat = [0, 0, 0, 1]
        
#     visual_id = p.createVisualShape(
#         shapeType=p.GEOM_SPHERE,
#         radius=0.02,
#         rgbaColor=[1, 0, 0, 0.7]
#     )
    
#     body_id = p.createMultiBody(
#         baseMass=0,
#         baseVisualShapeIndex=visual_id,
#         basePosition=position,
#         baseOrientation=quat
#     )
    
#     return body_id
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




# def update_target_visual(visual_id, position, rpy=None):
#     """Update the position of the target visual marker
    
#     Args:
#         visual_id: ID of the visual marker
#         position: New position [x, y, z]
#         rpy: New orientation as roll, pitch, yaw (optional)
#     """
#     if rpy is not None:
#         quat = p.getQuaternionFromEuler([rpy[0], rpy[1], rpy[2]])
#     else:
#         quat = [0, 0, 0, 1]
        
#     p.resetBasePositionAndOrientation(visual_id, position, quat)




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


def get_obs(robot_id, joint_indices, end_effector_link, cfg, task_cfg):
    """Extract observations matching the format used in training
    
    Args:
        robot_id: PyBullet body ID for the robot
        joint_indices: List of joint indices to query
        end_effector_link: Name of the end effector link
        cfg: Configuration object
        task_cfg: Task configuration
        
    Returns:
        Observation vector
    """
    # Get joint positions and velocities
    q, dq = get_joint_states(robot_id, joint_indices)
    
    # Create observation vector
    obs = np.zeros(cfg.env.num_single_obs, dtype=np.float32)
    
    # 1. Joint positions (normalized)
    obs[0:6] = q * cfg.normalization.obs_scales.dof_pos
    
    # 2. Joint velocities (normalized)
    obs[6:12] = dq * cfg.normalization.obs_scales.dof_vel
    
    # 3. Target position and orientation
    obs[12:15] = task_cfg.target_pos_1
    quat = R.from_euler('ZYX', task_cfg.target_rpy_1[::-1]).as_quat()
    # Convert to [x, y, z, w] format
    quat = np.array([quat[1], quat[2], quat[3], quat[0]])
    obs[15:19] = quat
    
    # 4. Previous action (will be updated in the main loop)
    obs[19:25] = np.zeros(6)
    
    return obs

def run_pybullet(policy, cfg, task_cfg, gui=True):
    """Run the PyBullet simulation with proper blocking behavior"""
    # Initialize PyBullet
    physicsClient = init_pybullet(gui)
    
    # Load robot and get joint indices
    robot_id, joint_indices = load_robot(cfg)
    
    # Create initial target visual
    task_cfg.target_visual_id = create_target_visual(
        task_cfg.current_target_pos, 
        task_cfg.current_target_rpy
    )
    
    # Initialize control variables
    target_q = np.zeros(cfg.env.num_actions, dtype=np.double)
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
    for step in tqdm(range(total_steps), desc="Simulating..."):
        # Update task state and check for target change
        target_updated = task_cfg.update_task_state(dt)
        
        if target_updated:
            # Update visual and print target info when target changes
            print(f"link6_pos: {get_link_state(robot_id, cfg.robot_config.end_effector_link)[0]}")
            print(f"New target: {task_cfg.current_target_pos}")
            update_target_visual(
                task_cfg.current_target_pos, 
                task_cfg.current_target_rpy
            )
        
        # Policy runs at lower frequency (based on decimation factor)
        if count_lowlevel % cfg.sim_config.decimation == 0:
            # Get full observation
            obs = get_obs(
                robot_id, 
                joint_indices, 
                cfg.robot_config.end_effector_link, 
                cfg, 
                task_cfg
            )

            # Update the previous action in observation
            obs[19:25] = prev_action
            
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
            print(f"Policy input: {policy_input}")
            # Compute action based on task state
            if not task_cfg.is_blocking:
                action = policy(torch.tensor(policy_input, dtype=torch.float32))[0].detach().numpy()
                target_q = action * cfg.control.action_scale
                
                # If this is the first action, start blocking after computing it
                if task_cfg.is_first_action:
                    task_cfg.is_blocking = True
                    task_cfg.blocked_time = 0.0
     
                prev_action = action
        
        # Apply position control to the joints
        target_q_clipped = np.clip(
            target_q, 
            cfg.robot_config.joint_lower_limits, 
            cfg.robot_config.joint_upper_limits
        )
        print(f"Target joint positions: {target_q_clipped}")
        # Apply pure position control to each joint
        for i, joint_idx in enumerate(joint_indices):
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_q_clipped[i],
                maxVelocity=5.0  # Optional: limit velocity of the movement
            )
            print(f"Joint target position: {target_q_clipped}")
        
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
    parser = argparse.ArgumentParser(description='AirBot Reach Task Deployment (PyBullet)')
    parser.add_argument('--load_model', type=str, required=True,
                      help='Path to the trained policy model')
    parser.add_argument('--model_path', type=str, default='/home/yurong/下载/airbot_usd/urdf/airbot_play_v3_0_gripper.urdf',
                      help='Path to the robot URDF file')
    parser.add_argument('--block_duration', type=float, default=10.0,
                      help='Duration to stop at each target position (seconds)')
    parser.add_argument('--headless', action='store_true',
                      help='Run in headless mode (no GUI)')
    args = parser.parse_args()
    
    # Update configuration with command line arguments
    cfg = Sim2simCfg()
    cfg.sim_config.urdf_model_path = args.model_path
    
    # For frame stacking, adjust observation dimension
    if cfg.env.frame_stack > 1:
        cfg.env.num_observations = cfg.env.num_single_obs * cfg.env.frame_stack
    
    # Initialize task configuration with blocking duration
    task_cfg = ReachTaskConfig(block_duration=args.block_duration)
    
    # Load policy
    policy = torch.jit.load(args.load_model)
    
    # Run simulation
    run_pybullet(policy, cfg, task_cfg, gui=not args.headless)