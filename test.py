import argparse
import numpy as np
import mujoco
import os
from scipy.spatial.transform import Rotation as R


def compute_forward_kinematics(model_path, joint_angles):
    """
    计算给定关节角度下机器人末端执行器(link6)的位置和姿态
    
    参数:
        model_path: MuJoCo模型XML文件路径
        joint_angles: 长度为6的数组，表示六个关节的角度
        
    返回:
        position: 末端执行器(link6)的位置坐标 [x, y, z]
        orientation_quat: 末端执行器的四元数姿态 [w, x, y, z]
        orientation_euler: 末端执行器的欧拉角姿态 [roll, pitch, yaw] (弧度)
    """
    # 加载MuJoCo模型
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # 获取关节索引
    joint_indices = []
    joint_found = True
    
    # 尝试几种可能的关节命名模式
    joint_name_patterns = [
        ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        ["shoulder", "upper_arm", "elbow", "forearm", "wrist1", "wrist2"],
        ["q1", "q2", "q3", "q4", "q5", "q6"]
    ]
    
    for pattern in joint_name_patterns:
        found_indices = []
        all_found = True
        for joint_name in pattern:
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if idx == -1:
                all_found = False
                break
            found_indices.append(idx)
        
        if all_found:
            print(f"找到关节索引，使用命名模式: {pattern}")
            joint_indices = found_indices
            joint_found = True
            break
    
    # 如果无法通过名称找到关节，尝试使用后6个关节
    if not joint_found:
        print("警告: 无法通过名称找到关节索引。使用模型中的最后6个关节。")
        joint_indices = list(range(model.njnt - 6, model.njnt))
    
    # 获取关节位置在qpos中的地址
    joint_qpos_addr = []
    for idx in joint_indices:
        addr = model.jnt_qposadr[idx]
        joint_qpos_addr.append(addr)
    
    print(f"关节位置地址: {joint_qpos_addr}")
    
    # 设置关节角度
    for i, angle in enumerate(joint_angles):
        data.qpos[joint_qpos_addr[i]] = angle
    
    # 正向动力学计算（更新机器人构型）
    mujoco.mj_forward(model, data)
    
    # 获取link6的ID
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    if link6_id == -1:
        print("错误: 模型中找不到'link6'")
        return None, None, None
    
    # 获取link6的位置和姿态
    position = data.xpos[link6_id].copy()
    orientation_quat = data.xquat[link6_id].copy()  # MuJoCo四元数格式: [w, x, y, z]
    
    # 计算欧拉角（使用MuJoCo内置函数）
    euler = np.zeros(3)
    quat_xyzw = [orientation_quat[1], orientation_quat[2], orientation_quat[3], orientation_quat[0]]
    r = R.from_quat(quat_xyzw)
    euler = r.as_euler('xyz', degrees=False)
    # mujoco.mju_quat2Euler(euler, orientation_quat)
    
    return position, orientation_quat, euler


def main():
    parser = argparse.ArgumentParser(description='计算机器人末端执行器位置和姿态')
    parser.add_argument('--model_path', type=str, required=True,
                      help='MuJoCo模型XML文件路径')
    parser.add_argument('--joint_angles', type=float, nargs=6, required=True,
                      help='6个关节角度值，空格分隔')
    parser.add_argument('--degrees', action='store_true',
                      help='输入角度是否为角度制（默认为弧度制）')
    
    args = parser.parse_args()
    
    # 如果输入是角度制，转换为弧度制
    joint_angles = np.array(args.joint_angles)
    if args.degrees:
        joint_angles = np.radians(joint_angles)
    
    # 确保模型路径存在
    if not os.path.exists(args.model_path):
        print(f"错误: 找不到模型文件 {args.model_path}")
        return
    
    # 计算正向运动学
    position, orientation_quat, orientation_euler = compute_forward_kinematics(
        args.model_path, joint_angles)
    
    if position is None:
        return
    
    # 打印结果
    print("\n=== 正向运动学结果 ===")
    print(f"输入关节角度 (弧度): {joint_angles}")
    if args.degrees:
        print(f"输入关节角度 (角度): {args.joint_angles}")
    
    print(f"\n末端执行器位置:")
    print(f"X: {position[0]:.6f}")
    print(f"Y: {position[1]:.6f}")
    print(f"Z: {position[2]:.6f}")
    
    print(f"\n末端执行器姿态 (四元数) [w, x, y, z]:")
    print(f"w: {orientation_quat[0]:.6f}")
    print(f"x: {orientation_quat[1]:.6f}")
    print(f"y: {orientation_quat[2]:.6f}")
    print(f"z: {orientation_quat[3]:.6f}")
    
    # 转换为角度制的欧拉角
    euler_degrees = np.degrees(orientation_euler)
    print(f"\n末端执行器姿态 (欧拉角) [roll, pitch, yaw]:")
    print(f"Roll  (x轴旋转): {orientation_euler[0]:.6f} rad = {euler_degrees[0]:.2f}°")
    print(f"Pitch (y轴旋转): {orientation_euler[1]:.6f} rad = {euler_degrees[1]:.2f}°")
    print(f"Yaw   (z轴旋转): {orientation_euler[2]:.6f} rad = {euler_degrees[2]:.2f}°")


if __name__ == '__main__':
    main()