import math
import numpy as np
import time
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import random
import argparse
import airbot

# Create robot instance
robot = airbot.create_agent(can_interface="can0", end_mode="gripper")

def compute_pose_error_numpy(
    t_current: np.ndarray,
    q_current: np.ndarray,
    t_target: np.ndarray,
    q_target: np.ndarray,
    rot_error_type: str = "axis_angle",
):
    """
    Compute position and rotation error between current and target pose using numpy.

    Args:
        t_current: Current position, shape (3,)
        q_current: Current orientation as quaternion [x, y, z, w]
        t_target: Target position, shape (3,)
        q_target: Target orientation as quaternion [x, y, z, w]
        rot_error_type: "quat" or "axis_angle"

    Returns:
        pos_error: (3,)
        rot_error: (3,) if "axis_angle", (4,) if "quat"
    """
    # Position error
    pos_error = t_target - t_current

    # Normalize input quaternions
    q_current = q_current / np.linalg.norm(q_current)
    q_target = q_target / np.linalg.norm(q_target)

    # Compute inverse of current quaternion
    q_current_inv = np.array([ -q_current[0], -q_current[1], -q_current[2], q_current[3] ])  # conjugate

    # Quaternion multiplication: q_error = q_target * q_current_inv
    r_target = R.from_quat(q_target)
    r_current_inv = R.from_quat(q_current_inv)
    r_error = r_target * r_current_inv

    if rot_error_type == "quat":
        rot_error = r_error.as_quat()  # [x, y, z, w]
    elif rot_error_type == "axis_angle":
        rot_error = r_error.as_rotvec()  # [rx, ry, rz], length is angle in radians
    else:
        raise ValueError("Unsupported rot_error_type. Use 'quat' or 'axis_angle'.")

    return pos_error, rot_error

class Sim2simCfg:
    class sim_config:
        sim_duration = 60.0
        dt = 1/35
        decimation = 2

    class env:
        num_actions = 6  # 6 joints for the airbot
        num_single_obs = 25  # Adapt based on your observation space
        num_observations = num_single_obs  # Will be updated if frame_stack > 1
        frame_stack = 1  # Modify if you use frame stacking

    class robot_config:
        # Joint limits (from previous configuration)
        joint_lower_limits = np.array([-3.14, -2.96, -0.087, -2.96, -1.74, -3.14], dtype=np.double)
        joint_upper_limits = np.array([2.09, 0.17, 3.14,  2.96, 1.74, 3.14], dtype=np.double)
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        end_effector_link = "link6"

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
    """Configuration for the reaching task"""
    def __init__(self):
        # Target position ranges (similar to your command ranges)
        self.pos_range_x = (0.35, 0.65)
        self.pos_range_y = (-0.2,0.2)
        self.pos_range_z = (0.15, 0.5)
        self.pos_range_roll = (math.pi/2, math.pi/2)
        self.pos_range_pitch = (math.pi/2, math.pi*3/2)
        self.pos_range_yaw = (math.pi/2, math.pi/2)
        # self.pos_range_yaw = (0, 0)
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


def get_joint_states(robot):
    """Get joint positions and velocities using AirBot API
    
    Args:
        robot: AirBot robot instance
        
    Returns:
        Joint positions and velocities as numpy arrays
    """
    # Get current joint positions
    positions = robot.get_current_joint_q()

    # Get current joint velocities
    velocities = robot.get_current_joint_v()
    return np.array(positions), np.array(velocities)


def get_obs(robot, cfg, task_cfg):
    """Extract observations matching the format used in training
    
    Args:
        robot: AirBot robot instance
        cfg: Configuration object
        task_cfg: Task configuration
        
    Returns:
        Observation vector
    """
    # Get joint positions and velocities
    q, dq = get_joint_states(robot)
    
    # Create observation vector
    obs = np.zeros(cfg.env.num_single_obs, dtype=np.float32)
    
    # 1. Joint positions (normalized)
    obs[0:6] = q * cfg.normalization.obs_scales.dof_pos
    
    # 2. Joint velocities (normalized)
    obs[6:12] = dq * cfg.normalization.obs_scales.dof_vel
    
    # 3. Target position and orientation
    obs[12:15] = task_cfg.target_pos
    quat = R.from_euler('ZYX', task_cfg.target_rpy[::-1]).as_quat()
    # Convert to [x, y, z, w] format
    quat = np.array([quat[1], quat[2], quat[3], quat[0]])
    obs[15:19] = quat
    
    # 4. Previous action (will be updated in the main loop)
    obs[19:25] = np.zeros(6)

    return obs


def run_robot(policy, cfg, task_cfg):
    """Run the robot simulation with the provided policy
    
    Args:
        policy: PyTorch policy model
        cfg: Simulation configuration
        task_cfg: Task configuration
        
    Returns:
        None
    """
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
    
    # Print initial target information
    # print(f"Initial Target Position: {task_cfg.target_pos}")
    quati = R.from_euler('ZYX', task_cfg.target_rpy[::-1]).as_quat()
    # Convert to [x, y, z, w] format
    quati = np.array([quati[1], quati[2], quati[3], quati[0]])
    # print(f"Initial Target Orientation (XYZW): {quati}")
    
    # Main simulation loop
    for step in tqdm(range(total_steps), desc="Running Robot..."):
        # Policy runs at lower frequency (based on decimation factor)
        if count_lowlevel % cfg.sim_config.decimation == 0:
            # 在每个控制周期检查是否需要更新目标位置
            # Check if target needs to be updated (passing the actual elapsed time)
            if task_cfg.update_target(dt * cfg.sim_config.decimation):
                # print(f"New Target Position: {task_cfg.target_pos}")
                new_quat = R.from_euler('ZYX', task_cfg.target_rpy[::-1]).as_quat()
                # Convert to [x, y, z, w] format
                new_quat = np.array([new_quat[1], new_quat[2], new_quat[3], new_quat[0]])
                # print(f"New Target Orientation (XYZW): {new_quat}")
            
            # Get full observation
            obs = get_obs(robot, cfg, task_cfg)
            
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
            
            # 打印policy输入中的target位置，用于调试
            # print(f"Target in policy input: Pos={policy_input[0, 12:15]}, Quat={policy_input[0, 15:19]}")
            
            # Calculate next action
            action = policy(torch.tensor(policy_input, dtype=torch.float32))[0].detach().numpy()
            target_q = action * cfg.control.action_scale
            
            # Clip actions to joint limits
            target_q_clipped = np.clip(
                target_q, 
                cfg.robot_config.joint_lower_limits, 
                cfg.robot_config.joint_upper_limits
            )
            # print(f"Target Joint Positions: {target_q_clipped}")
            # Send target joint positions to robot
            try:
                robot.set_target_joint_q(target_q_clipped, vel=0.8, blocking=False)

                effector_pos = robot.get_current_translation()
                effector_quat = robot.get_current_rotation()  # [x, y, z, w] 格式

                print(f"End Effector target pos: {task_cfg.target_pos}")
                print(f"End Effector actual pos: {effector_pos}")

                # 构造目标朝向四元数：ZYX 欧拉角转四元数，注意需要转换为 [x, y, z, w]
                target_euler = task_cfg.target_rpy[::-1]  # 从 RPY → YXZ
                target_quat = R.from_euler('ZYX', target_euler).as_quat()  # 得到 [x, y, z, w]

                print(f"End Effector target quat: {target_quat}")
                print(f"End Effector actual quat: {effector_quat}")

                # --- 统一误差估计 ---
                pos_error, rot_error = compute_pose_error_numpy(
                    t_current=np.array(effector_pos),
                    q_current=np.array(effector_quat),
                    t_target=np.array(task_cfg.target_pos),
                    q_target=np.array(target_quat),
                    rot_error_type="axis_angle",  # 或 "quat"
                )
                rot_norm = np.linalg.norm(rot_error)-math.pi
                print(f"Position error: {pos_error}, Norm: {np.linalg.norm(pos_error):.4f}")
                print(f"Rotation error: {rot_error}, Angle (rad): {np.linalg.norm(rot_norm):.4f}")

            except Exception as e:
                print(f"Error setting joint positions: {e}")
                break
            
            # Store previous action
            prev_action = action
        
        # Wait for the duration of one timestep
        time.sleep(dt)
        
        # Increment counter
        count_lowlevel += 1
    
    # Optional: Move to home position when done
    try:
        print("Task completed, moving to home position...")
        robot.set_target_joint_q(np.zeros(6), vel=0.2)
        time.sleep(10)
    except Exception as e:
        print(f"Error moving to home position: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AirBot Reach Task Deployment')
    parser.add_argument('--load_model', type=str, required=True,
                      help='Path to the trained policy model')
    parser.add_argument('--target_update_time', type=float, default=10.0,
                      help='Time interval (seconds) between target updates')
    args = parser.parse_args()
    
    # Update configuration 
    cfg = Sim2simCfg()
    
    # For frame stacking, adjust observation dimension
    if cfg.env.frame_stack > 1:
        cfg.env.num_observations = cfg.env.num_single_obs * cfg.env.frame_stack
    
    # Initialize task configuration
    task_cfg = ReachTaskConfig()
    # 应用命令行参数设置的目标更新时间
    task_cfg.target_update_time = args.target_update_time
    
    # Load policy
    policy = torch.jit.load(args.load_model)
    
    # Run robot simulation
    run_robot(policy, cfg, task_cfg)