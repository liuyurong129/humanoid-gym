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
import matplotlib.pyplot as plt
import os
import csv
# Create robot instance
robot = airbot.create_agent(can_interface="can0", end_mode="none")

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

class JointDataRecorder:
    def __init__(self, robot, sampling_rate=100, recording_time=10):
        """
        初始化关节数据记录器
        
        参数:
        robot: 机械臂对象，需要有get_current_joint_q方法
        sampling_rate: 采样率(Hz)
        recording_time: 记录时间(秒)
        """
        self.robot = robot
        self.sampling_rate = sampling_rate
        self.recording_time = recording_time
        self.target_q_data = []
        self.actual_q_data = []
        self.timestamps = []
        self.num_joints = 6  # 假设是6自由度机械臂
        
    def record_data(self, target_q_function):
        """
        记录目标关节角度和实际关节角度数据
        
        参数:
        target_q_function: 返回target_q_clipped的函数
        """
        print(f"开始记录数据，将持续 {self.recording_time} 秒...")
        num_samples = int(self.sampling_rate * self.recording_time)
        
        for i in range(num_samples):
            start_time = time.time()
            
            # 获取目标关节角度和实际关节角度
            target_q = target_q_function()
            actual_q = self.robot.get_current_joint_q()
            
            # 记录时间戳和数据
            self.timestamps.append(i / self.sampling_rate)
            self.target_q_data.append(target_q)
            self.actual_q_data.append(actual_q)
            
            # 控制采样率
            elapsed = time.time() - start_time
            sleep_time = max(0, 1/self.sampling_rate - elapsed)
            time.sleep(sleep_time)
            
            # 显示进度
            if i % (self.sampling_rate // 2) == 0:
                print(f"进度: {i/num_samples*100:.1f}%")
        
        print("数据记录完成!")
    
    def save_to_csv(self, folder="./data"):
        """将记录的数据保存为CSV文件"""
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # 保存目标关节角度
        target_filename = os.path.join(folder, f"target_q_{timestamp}.csv")
        with open(target_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp'] + [f'joint_{i+1}' for i in range(self.num_joints)]
            writer.writerow(header)
            
            for i, (ts, q) in enumerate(zip(self.timestamps, self.target_q_data)):
                writer.writerow([ts] + list(q))
        
        # 保存实际关节角度
        actual_filename = os.path.join(folder, f"actual_q_{timestamp}.csv")
        with open(actual_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp'] + [f'joint_{i+1}' for i in range(self.num_joints)]
            writer.writerow(header)
            
            for i, (ts, q) in enumerate(zip(self.timestamps, self.actual_q_data)):
                writer.writerow([ts] + list(q))
                
        print(f"数据已保存至: {target_filename} 和 {actual_filename}")
        return target_filename, actual_filename
    
    def plot_data(self, target_file=None, actual_file=None):
        """
        将目标关节角度和实际关节角度数据绘制成图表
        
        参数:
        target_file: 目标关节角度CSV文件名，如果为None则使用内存中的数据
        actual_file: 实际关节角度CSV文件名，如果为None则使用内存中的数据
        """
        target_data = self.target_q_data
        actual_data = self.actual_q_data
        timestamps = self.timestamps
        
        # 如果提供了文件名，则从文件中读取数据
        if target_file and actual_file:
            timestamps = []
            target_data = [[] for _ in range(self.num_joints)]
            actual_data = [[] for _ in range(self.num_joints)]
            
            # 读取目标关节角度数据
            with open(target_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    timestamps.append(float(row[0]))
                    for i in range(self.num_joints):
                        target_data[i].append(float(row[i+1]))
            
            # 读取实际关节角度数据
            with open(actual_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    for i in range(self.num_joints):
                        actual_data[i].append(float(row[i+1]))
        else:
            # 将数据重新组织为每个关节一个列表
            target_data = list(zip(*target_data))
            actual_data = list(zip(*actual_data))
        
        # 创建6张图，每个关节一张图
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(self.num_joints):
            ax = axes[i]
            if target_file and actual_file:
                ax.plot(timestamps, target_data[i], 'b-', label='Target')
                ax.plot(timestamps, actual_data[i], 'r-', label='Actual')
            else:
                ax.plot(timestamps, [q[i] for q in target_data], 'b-', label='Target')
                ax.plot(timestamps, [q[i] for q in actual_data], 'r-', label='Actual')
                
            ax.set_title(f'Joint {i+1}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Joint Angle (rad)')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('./data/joint_angles_comparison.png', dpi=300)
        plt.show()
        print("图表已生成并保存为 './data/joint_angles_comparison.png'")

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
                robot.set_target_joint_q(target_q_clipped, vel=0.5, blocking=False)

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