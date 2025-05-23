from scipy.spatial.transform import Rotation as R

# 四元数
quat_target = [-0.5, 0.5, -0.5, 0.5]
quat_actual = [0,0,0,1]

# 将四元数转换为旋转对象
rot_target = R.from_quat(quat_target)
rot_actual = R.from_quat(quat_actual)

# 获取欧拉角（默认 ZYX 顺序）
euler_target = rot_target.as_euler('zyx', degrees=True)  # 转换为度数
euler_actual = rot_actual.as_euler('zyx', degrees=True)  # 转换为度数

print(f"Target Euler angles: {euler_target}")
print(f"Actual Euler angles: {euler_actual}")
