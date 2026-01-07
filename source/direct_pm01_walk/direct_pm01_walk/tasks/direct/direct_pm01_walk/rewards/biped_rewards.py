import torch
import math
from isaaclab.utils.math import euler_xyz_from_quat

"""
EngineAI Biped 综合奖励函数库 - IsaacLab 精准对标版
"""

# --- 1. 姿态综合奖励 (对标: reward_orientation) ---
def get_reward_orientation(env):
    roll, pitch, _ = euler_xyz_from_quat(env.robot.data.root_quat_w)
    euler_xy = torch.stack((roll, pitch), dim=-1)
    
    # 欧拉角核函数 (系数 10)
    quat_mismatch = torch.exp(-torch.sum(torch.abs(euler_xy), dim=1) * 10)
    # 重力投影核函数 (系数 20)
    orientation = torch.exp(-torch.norm(env.robot.data.projected_gravity_b[:, :2], dim=1) * 20)
    
    return (quat_mismatch + orientation) / 2.0

# --- 2. 静止维持奖励 (对标: reward_stand_still) ---
def get_reward_stand_still(env):
    dof_deviation = torch.sum(torch.abs(env.robot.data.joint_pos - env.robot.data.default_joint_pos), dim=1)
    commands_norm = torch.norm(env.commands[:, :2], dim=1)
    # 权重在外部 scale 处理，此处仅返回原始偏差
    return dof_deviation * (commands_norm < 0.15)

# --- 3. 足端空中时间 (对标: reward_feet_air_time) ---
def get_reward_feet_air_time(env):
    contact_sensor = env.scene.sensors.get("contact_forces")
    if contact_sensor is None: return torch.zeros(env.num_envs, device=env.device)
    
    first_contact = contact_sensor.compute_first_contact(env.step_dt)
    air_time = contact_sensor.data.last_air_time.clamp(0, 0.5)
    return torch.sum(air_time * first_contact, dim=1)

# --- 4. 接触力保护 (对标: reward_feet_contact_forces) ---
def get_reward_feet_contact_forces(env, max_contact_force: float = 500.0):
    contact_sensor = env.scene.sensors.get("contact_forces")
    if contact_sensor is None: return torch.zeros(env.num_envs, device=env.device)

    net_forces_norm = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
    force_deviation = net_forces_norm - max_contact_force
    # 按照 Gym 逻辑，超出部分 clip 到 0-350
    return torch.sum(force_deviation.clip(0, 350), dim=1)

# --- 5. 动态基座高度 (对标: reward_base_height) ---
def get_reward_base_height(env, target_height: float = 0.8132): # <-- 修正为 Gym 里的 0.8132
    contact_sensor = env.scene.sensors.get("contact_forces")
    if contact_sensor is None or contact_sensor.data.pos_w is None:
        return torch.zeros(env.num_envs, device=env.device)

    net_forces_norm = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
    stance_mask = (net_forces_norm > 1.0).float()
    
    foot_heights = contact_sensor.data.pos_w[:, :, 2]
    num_stance_feet = torch.sum(stance_mask, dim=1)
    
    # 避免腾空时奖励突变：如果没有脚着地，使用世界坐标 z
    measured_avg_foot_height = torch.sum(foot_heights * stance_mask, dim=1) / (num_stance_feet + 1e-6)
    
    # 相对高度逻辑 (含 0.05m 偏移)
    relative_height = env.robot.data.root_pos_w[:, 2] - (measured_avg_foot_height - 0.05)
    
    # 系数 100
    return torch.exp(-torch.abs(relative_height - target_height) * 100)

# --- 6. 补充：线速度追踪 (对标: tracking_lin_vel) ---
def get_reward_tracking_lin_vel(env, sigma: float = 5.0): # <-- 对应 config 中的 tracking_sigma
    lin_vel_error = torch.sum(torch.square(env.commands[:, :2] - env.robot.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / sigma)

# --- 7. 补充：关节限位 (对标: dof_pos_limits) ---
def get_reward_dof_pos_limits(env):
    out_of_limits = -(torch.abs(env.robot.data.joint_pos) - env.robot.data.soft_joint_pos_limits[..., 1]).clamp(min=0.0)
    return torch.sum(torch.square(out_of_limits), dim=1)