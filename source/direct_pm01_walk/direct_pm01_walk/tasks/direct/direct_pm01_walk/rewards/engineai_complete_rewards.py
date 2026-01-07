import torch
import math
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

"""
修正版 - EngineAI 众擎机器人全模块奖励函数库
解决了 feet_clearance 的维度广播报错
"""

# --- 1. 任务追踪 (Base Velocity) ---
def track_lin_vel_xy_exp(env, sigma=5.0):
    error = torch.sum(torch.square(env.commands[:, :2] - env.robot.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-error / sigma)

def track_ang_vel_z_exp(env, sigma=5.0):
    error = torch.square(env.commands[:, 2] - env.robot.data.root_ang_vel_b[:, 2])
    return torch.exp(-error / sigma)

def vel_mismatch_exp(env, sigma=0.5):
    """惩罚垂直速度，维持平面运动一致性"""
    return torch.exp(-torch.abs(env.robot.data.root_lin_vel_b[:, 2]) / sigma)

def low_speed_penalty(env):
    """指令为零时的静止奖励"""
    return (torch.norm(env.commands[:, :2], dim=1) < 0.1).float() * torch.norm(env.robot.data.root_lin_vel_b[:, :2], dim=1)

# --- 2. 姿态与几何 (Base Pos/Ori) ---
def orientation_combined(env):
    roll, pitch, _ = euler_xyz_from_quat(env.robot.data.root_quat_w)
    quat_mismatch = torch.exp(-torch.sum(torch.abs(torch.stack((roll, pitch), dim=-1)), dim=1) * 10)
    orientation = torch.exp(-torch.norm(env.robot.data.projected_gravity_b[:, :2], dim=1) * 20)
    return (quat_mismatch + orientation) / 2.0

def base_height_dynamic(env, target=0.8132):
    contact_sensor = env.scene.sensors["contact_forces"]
    stance_mask = (torch.norm(contact_sensor.data.net_forces_w, dim=-1) > 1.0).float()
    foot_z = contact_sensor.data.pos_w[:, :, 2]
    # 支撑脚平均高度
    avg_foot_z = torch.sum(foot_z * stance_mask, dim=1) / (torch.sum(stance_mask, dim=1) + 1e-6)
    rel_height = env.robot.data.root_pos_w[:, 2] - (avg_foot_z - 0.05)
    return torch.exp(-torch.abs(rel_height - target) * 100)

def base_acc_l2(env):
    """惩罚基座加速度"""
    # 需在 env._prev_root_lin_vel_b 维护上一时刻速度
    return torch.norm(env.robot.data.root_lin_vel_b - env._prev_root_lin_vel_b, dim=1)

# --- 3. 关节控制 (DOF) ---
def dof_ref_pos_diff(env, sigma=0.26):
    """关节参考位置追踪"""
    # 假设 env._ref_joint_pos 由相位发生器实时更新
    return torch.exp(-torch.sum(torch.square(env.robot.data.joint_pos - env._ref_joint_pos), dim=1) / sigma)

def default_joint_pos_l2(env):
    """非核心关节回归默认姿态"""
    return torch.sum(torch.square(env.robot.data.joint_pos - env.robot.data.default_joint_pos), dim=1)

def dof_vel_l2(env):
    return torch.sum(torch.square(env.robot.data.joint_vel), dim=1)

def dof_acc_l2(env):
    """需在 env._prev_joint_vel 维护"""
    return torch.sum(torch.square(env.robot.data.joint_vel - env._prev_joint_vel), dim=1)

# --- 4. 步态与间距 (Gait) ---
def feet_air_time(env):
    contact_sensor = env.scene.sensors["contact_forces"]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)
    return torch.sum(contact_sensor.data.last_air_time.clamp(0, 0.5) * first_contact, dim=1)

def feet_contact_number(env):
    """奖励维持单脚支撑状态"""
    num_contacts = torch.sum((torch.norm(env.scene.sensors["contact_forces"].data.net_forces_w, dim=-1) > 1.0).float(), dim=1)
    return (num_contacts == 1).float()

def feet_clearance(env, target_h=0.1):
    """摆动腿净空高度奖励"""
    contact_sensor = env.scene.sensors["contact_forces"]
    # 关键修复：使用 unsqueeze(-1) 确保 (1024, 2) - (1024, 1) 的正确广播
    root_height_offset = (env.robot.data.root_pos_w[:, 2].unsqueeze(-1) - 0.8132)
    foot_z_rel = contact_sensor.data.pos_w[:, :, 2] - root_height_offset
    
    stance_mask = (torch.norm(contact_sensor.data.net_forces_w, dim=-1) > 1.0)
    # 仅奖励非支撑脚
    return torch.sum((~stance_mask).float() * torch.exp(-torch.abs(foot_z_rel - target_h) * 20), dim=1)

def feet_distance_l2(env, target=0.2):
    pos = env.scene.sensors["contact_forces"].data.pos_w
    dist = torch.norm(pos[:, 0, :2] - pos[:, 1, :2], dim=1)
    return torch.exp(-torch.square(dist - target) / 0.05)

def knee_distance_l2(env, target=0.2):
    # 需预先获取膝盖 body_ids
    pos = env.robot.data.body_pos_w[:, env._knee_ids, :2]
    dist = torch.norm(pos[:, 0, :] - pos[:, 1, :], dim=1)
    return torch.exp(-torch.square(dist - target) / 0.05)

# --- 5. 动作与物理约束 (Action/Contact) ---
def feet_contact_forces_penalty(env, max_force=500.0):
    forces = torch.norm(env.scene.sensors["contact_forces"].data.net_forces_w, dim=-1)
    return torch.sum((forces - max_force).clip(0, 350), dim=1)

def foot_slip_penalty(env):
    contact_sensor = env.scene.sensors["contact_forces"]
    stance_mask = (torch.norm(contact_sensor.data.net_forces_w, dim=-1) > 1.0).float()
    foot_vel = torch.norm(contact_sensor.data.vel_w_raw[:, :, :2], dim=-1)
    return torch.sum(stance_mask * foot_vel, dim=1)

def action_smoothness_l2(env):
    """二阶导平滑惩罚"""
    return torch.sum(torch.square(env.actions - 2 * env.last_actions + env.last_last_actions), dim=1)

def torques_l2(env):
    return torch.sum(torch.square(env.robot.data.joint_effort), dim=1)