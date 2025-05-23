# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import random
import argparse
import time


class Sim2simCfg:
    class sim_config:
        mujoco_model_path = '/home/yurong/humanoid-gym/resources/robots/airbot/airbot_play_v3_0_gripper.xml'  # Update with your actual path
        sim_duration = 600.0
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
        self.pos_range_y = (-0.2, 0.2)
        self.pos_range_z = (0.15, 0.5)
        self.pos_range_roll=(0,0)
        self.pos_range_pitch=(math.pi,math.pi)
        self.pos_range_yaw=(-math.pi/2, math.pi/2)
        
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
        self.target_update_time = 100.0
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


def get_obs(robot, cfg, task_cfg):
    """Extract observations from the robot interface
    
    Args:
        robot: Robot interface object
        cfg: Configuration object
        task_cfg: Task configuration
        
    Returns:
        Observation vector matching the training environment's format
    """
    # Extract joint positions and velocities from robot interface
    q = robot.get_current_joint_q().astype(np.double)
    dq = robot.get_current_joint_v().astype(np.double)
    
    # Create observation vector
    obs = np.zeros(cfg.env.num_single_obs, dtype=np.float32)
    
    # 1. Joint positions (normalized)
    obs[0:6] = q * cfg.normalization.obs_scales.dof_pos
    
    # 2. Joint velocities (normalized)
    obs[6:12] = dq * cfg.normalization.obs_scales.dof_vel
    
    # 3. end-effector position (command)
    obs[12:15] = task_cfg.target_pos
    quat = R.from_euler('ZYX', task_cfg.target_rpy[::-1]).as_quat()
    obs[15:19] = quat
    
    # 6. Previous action (will be updated in the main loop)
    obs[19:25] = np.zeros(6)  # Will store the previous action
    
    return obs


def run_robot(policy, cfg, task_cfg, robot):
    """Run the robot control with the provided policy
    
    Args:
        policy: PyTorch policy model
        cfg: Configuration object
        task_cfg: Task configuration
        robot: Robot interface object
        
    Returns:
        None
    """
    # Initialize control variables
    target_q = np.zeros(cfg.env.num_actions, dtype=np.double)
    action = np.zeros(cfg.env.num_actions, dtype=np.double)
    prev_action = np.zeros(cfg.env.num_actions, dtype=np.double)
    
    # 初始化目标切换的计时器
    action_timer = 0.0
    switch_interval = 5.0  
    
    # Initialize observation history for frame stacking if used
    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros(cfg.env.num_single_obs, dtype=np.float32))
    
    # Initialize simulation counter
    count_lowlevel = 0
    first_action = True 
    
    # 用于存储下一个目标位置
    target_next = np.zeros(cfg.env.num_actions, dtype=np.double)
    
    # Main control loop
    total_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)
    for step in tqdm(range(total_steps), desc="Running robot..."):
        dt = cfg.sim_config.dt
        
        # 更新动作计时器
        action_timer += dt
        
        # Update target if needed
        if task_cfg.update_target(dt):
            print(f"New target: {task_cfg.target_pos}")

        # Policy runs at lower frequency (100Hz)
        if count_lowlevel % cfg.sim_config.decimation == 0:
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
                
            # 始终计算下一个目标位置，即使不立即使用
            new_action = policy(torch.tensor(policy_input, dtype=torch.float32))[0].detach().numpy()
            target_next = new_action * cfg.control.action_scale
            
            if first_action:
                action = new_action
                target_q = action * cfg.control.action_scale
                prev_action = action.copy()
                first_action = False 
                # 重置计时器
                action_timer = 0.0
        
        # 检查是否需要切换到下一个目标
        if action_timer >= switch_interval:
            print(f"动作已执行 {switch_interval} 秒，切换到新动作")
            new_action = policy(torch.tensor(policy_input, dtype=torch.float32))[0].detach().numpy()
            target_next = new_action * cfg.control.action_scale
            target_q = target_next.copy()  # 将下一个目标作为当前目标
            action_timer = 0.0  # 重置计时器
        
        # Apply position control to the robot
        # Clip targets to joint limits for safety
        target_q_clipped = np.clip(
            target_q, 
            cfg.robot_config.joint_lower_limits, 
            cfg.robot_config.joint_upper_limits
        )
        
        if step % 60 == 0 or action_timer < 0.1:  # 每秒或切换后立即打印
            print(f"当前动作计时: {action_timer:.2f}秒")
            print(f"当前目标: {target_q}")
            print(f"下一目标: {target_next}")
        
        # Send position command to robot
        robot.set_joint_positions(target_q_clipped)
        
        # Sleep to maintain control frequency
        time.sleep(dt)
        
        # Increment counter
        count_lowlevel += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AirBot Reach Task Deployment')
    parser.add_argument('--load_model', type=str, required=True,
                      help='Path to the trained policy model')
    parser.add_argument('--switch_interval', type=float, default=3.0,
                      help='Time interval (seconds) between switching actions')
    args = parser.parse_args()
    
    # Update configuration with command line arguments
    cfg = Sim2simCfg()
    
    # For frame stacking, adjust observation dimension
    if cfg.env.frame_stack > 1:
        cfg.env.num_observations = cfg.env.num_single_obs * cfg.env.frame_stack
    
    # Initialize task configuration
    task_cfg = ReachTaskConfig()
    
    # Load policy
    policy = torch.jit.load(args.load_model)
    
    # Initialize robot interface
    robot = RobotInterface()  # 需要实现这个类
    
    # Run robot control
    run_robot(policy, cfg, task_cfg, robot)