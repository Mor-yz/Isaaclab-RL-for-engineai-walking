# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .direct_pm01_walk_env_cfg import DirectPm01WalkEnvCfg
from direct_pm01_walk.tasks.direct.direct_pm01_walk.rewards.rewards import *
from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_rotate_inverse, euler_xyz_from_quat
   


class DirectPm01WalkEnv(DirectRLEnv):
    cfg: DirectPm01WalkEnvCfg

    def __init__(self, cfg: DirectPm01WalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        print("Available sensors:", list(self.scene.sensors.keys()))


        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)

        self._lfoot_ids, _ = self.robot.find_bodies("link_ankle_roll_l")
        self._rfoot_ids, _ = self.robot.find_bodies("link_ankle_roll_r")
        assert len(self._lfoot_ids) == 1 and len(self._rfoot_ids) == 1, "检查脚链路命名是否匹配"
        self._l = self._lfoot_ids[0]
        self._r = self._rfoot_ids[0]

        # IMU 信息缓存
        self._prev_root_lin_vel_b = torch.zeros_like(self.robot.data.root_lin_vel_b)
        self._prev_root_ang_vel_b = torch.zeros_like(self.robot.data.root_ang_vel_b)


        #指令相关
        # 行走指令（机体坐标系下 vx, vy, wz）
        self.control_dt = float(self.cfg.sim.dt * self.cfg.decimation)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self._command_time_left = torch.zeros(self.num_envs, device=self.device)
        # 初始化指令
        self._sample_commands(range(self.num_envs))
        
        #symetry buffer
        self.phase_key_angles = {
            0: None,
            1: None,
            2: None,
            3: None,
        }
        self.phase_threshold = 0.1  # 弧度阈值
        self.phase_refs = torch.tensor([0.0, math.pi/2, math.pi, 3*math.pi/2], device=self.device)

        # 推力相关状态
        self.push_timer = torch.zeros(self.num_envs, device=self.device)
        self.push_cooldown = torch.zeros(self.num_envs, device=self.device)
        self.push_force = torch.zeros(self.num_envs, 3, device=self.device)

        # 随机推力的范围
        self.push_force_range = (-10.0, 10.0)       # 牛顿
        self.push_interval_range = (1.0, 3.0)         # 两次推力间隔（秒）
        self.push_duration_range = (0.2, 0.6)         # 推力持续时间（秒）

        self.robot.set_debug_vis(True)
        print(self.robot.has_debug_vis_implementation)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.scene.robot)
        self.scene.articulations["robot"] = self.robot

    def _apply_push_force(self, force_vec):
        """将当前 push_force 写入到机器人肩部 link。"""

        target_links = [
            "link_torso_yaw",
        ]

        env_ids = torch.arange(self.num_envs, device=self.device)

        # 构造 buffer
        forces = torch.zeros((self.num_envs, self.robot.num_bodies, 3), device=self.device)
        torques = torch.zeros_like(forces)

        for name in target_links:
            body_ids, _ = self.robot.find_bodies(name)
            bid = body_ids[0]
            forces[:, bid, :] = force_vec  # force_vec shape: (num_envs, 3)

        # 写入缓存
        self.robot.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            env_ids=env_ids,
            is_global=True,
        )

    def _update_random_push(self):
        """周期性地随机施加推力 + 随机方向 + 随机持续时间。"""

        dt = self.control_dt
        env_ids = torch.arange(self.num_envs, device=self.device)

        # 更新计时器
        self.push_timer -= dt
        self.push_cooldown -= dt

        # 哪些环境需要开始新的推力？
        start_new = self.push_cooldown <= 0

        if start_new.any():
            # 重新设定 cooldown（下一次推力之前的等待时间）
            new_intervals = torch.rand_like(self.push_cooldown[start_new]) * \
                            (self.push_interval_range[1] - self.push_interval_range[0]) + self.push_interval_range[0]
            self.push_cooldown[start_new] = new_intervals

            # 设置本次推力持续时间
            new_durations = torch.rand_like(self.push_timer[start_new]) * \
                            (self.push_duration_range[1] - self.push_duration_range[0]) + self.push_duration_range[0]
            self.push_timer[start_new] = new_durations

            # 随机方向的推力
            minf, maxf = self.push_force_range
            rand_force = torch.rand((start_new.sum(), 3), device=self.device) * (maxf - minf) + minf
            self.push_force[start_new] = rand_force

        # 哪些环境正在推？
        pushing = self.push_timer > 0

        # 正在推的环境 → 使用 push_force
        current_force = torch.zeros_like(self.push_force)
        current_force[pushing] = self.push_force[pushing]

        # 将力写入缓存（下一步将施加）
        self._apply_push_force(current_force)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        phase_delta = 2 * math.pi * self.cfg.sim.dt / 2.0  #周期为2.0秒
        self.gait_phase = (self.gait_phase + phase_delta) % (2 * math.pi)

        self.actions = actions.clone()

        # 更新指令剩余时间并按需刷新
        self._command_time_left -= self.control_dt
        resample_ids = torch.nonzero(self._command_time_left <= 0.0, as_tuple=False).squeeze(-1)
        if resample_ids.numel() > 0:
            self._sample_commands(resample_ids)
            
            
        # check symetry buffer
        phase = self.gait_phase  # (num_envs,)
        joint_pos = self.robot.data.joint_pos
        for i, ref in enumerate(self.phase_refs):
            near_ref = torch.abs((phase - ref + math.pi) % (2*math.pi) - math.pi) < self.phase_threshold
            if near_ref.any():
                # 记录这些环境的关键相位角度
                self.phase_key_angles[i] = joint_pos.clone().detach()

        #施加随机推力
        self._update_random_push()

    def _apply_action(self) -> None:
        action_scale = 1.0
        joint_target = self.default_joint_pos + self.actions * action_scale
        print('action: %.4f'% joint_target[0][0].item())
        self.robot.set_joint_position_target(joint_target)


 
    def _get_observations(self) -> dict:

        base_quat = self.robot.data.root_quat_w
        base_ang_vel = quat_rotate_inverse(base_quat, self.robot.data.root_ang_vel_w)

        noise_std = 1.0   # 噪声标准差，可按需要调整
        noise = torch.randn_like(base_ang_vel) * noise_std
        base_ang_vel = base_ang_vel + noise

        roll, pitch, yaw = euler_xyz_from_quat(base_quat)
        base_euler_xyz = torch.stack([roll, pitch, yaw], dim=-1)

        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel

        # phase的sin和cos
        phase_sin = torch.sin(self.gait_phase).unsqueeze(-1)
        phase_cos = torch.cos(self.gait_phase).unsqueeze(-1)

        obs = torch.cat(
            [
                base_ang_vel,
                base_euler_xyz,
                joint_pos,
                joint_vel,
                phase_sin,
                phase_cos,
                self.commands,
            ],
            dim=-1,
        )
        print('position: %.4f'% joint_pos[0][0].item())

        return {"policy": obs}


    # 左腿：j00_hip_pitch_l、j01_hip_roll_l、j02_hip_yaw_l、j03_knee_pitch_l、j04_ankle_pitch_l、j05_ankle_roll_l。
    # 右腿：j06_hip_pitch_r、j07_hip_roll_r、j08_hip_yaw_r、j09_knee_pitch_r、j10_ankle_pitch_r、j11_ankle_roll_r。
    # 躯干：j12_waist_yaw。
    # 左臂：j13_shoulder_pitch_l、j14_shoulder_roll_l、j15_shoulder_yaw_l、j16_elbow_pitch_l、j17_elbow_yaw_l。
    # 右臂：j18_shoulder_pitch_r、j19_shoulder_roll_r、j20_shoulder_yaw_r、j21_elbow_pitch_r、j22_elbow_yaw_r。
    # 头部：j23_head_yaw。

    def _get_rewards(self) -> torch.Tensor:
   
        l2 = flat_orientation_l2(self)  # 传入 env
        weight = 1.0
        print("flat_orientation_l2: %.3f \t weighted: %.3f" % (-l2.mean().item(), -l2.mean().item() * weight))
        reward = -l2*weight

        penalty = fall_penalty(self)
        weight = 1.0
        print("fall_penalty: %.3f \t weighted: %.3f" % (-penalty.mean().item(), -penalty.mean().item() * weight))
        reward -= penalty * weight

        joint_pos_limits_penalty = joint_pos_limits(self)
        weight = 0.1
        print("joint_pos_limits_penalty: %.3f \t weighted: %.3f" % (-joint_pos_limits_penalty.mean().item(), -joint_pos_limits_penalty.mean().item() * weight))
        reward -= joint_pos_limits_penalty * weight

        joint_torques_penalty = joint_torques_l2(self)
        weight = 0.01
        print("joint_torques_penalty: %.3f \t weighted: %.3f" % (-joint_torques_penalty.mean().item(), -joint_torques_penalty.mean().item() * weight))
        reward -= joint_torques_penalty * weight

        joint_acc_penalty = joint_acc_l2(self)
        weight = 0.00000001
        print("joint_acc_penalty: %.3f \t weighted: %.3f" % (-joint_acc_penalty.mean().item(), -joint_acc_penalty.mean().item() * weight))
        reward -= joint_acc_penalty * weight

        action_rate_penalty = action_rate_l2(self)
        weight = 0.1
        print("action_rate_penalty: %.3f \t weighted: %.3f" % (-action_rate_penalty.mean().item(), -action_rate_penalty.mean().item() * weight))
        reward -= action_rate_penalty * weight
        
        joint_threashold_reward = joint_angle_threshold_reward(self, ["j03_knee_pitch_l", "j09_knee_pitch_r"], threshold=0.2)
        weight = 1
        print("joint_threashold_reward: %.3f \t weighted: %.3f" % (joint_threashold_reward.mean().item(), joint_threashold_reward.mean().item() * weight))
        reward += joint_threashold_reward * weight

        # action_velocity_continuity_penalty = action_velocity_continuity(self)
        # weight = 0.01
        # print("action_velocity_continuity_penalty: %.3f \t weighted: %.3f" % (-action_velocity_continuity_penalty.mean().item(), -action_velocity_continuity_penalty.mean().item() * weight))
        # reward -= action_velocity_continuity_penalty * weight
       

        lin_vel_z_penalty = lin_vel_z_l2(self)
        weight = 10.0
        print("lin_vel_z_penalty: %.3f \t weighted: %.3f" % (-lin_vel_z_penalty.mean().item(), -lin_vel_z_penalty.mean().item() * weight))
        reward -= lin_vel_z_penalty * weight

        ang_vel_xy_penalty = ang_vel_xy_l2(self)
        weight = 0.00
        print("ang_vel_xy_penalty: %.3f \t weighted: %.3f" % (-ang_vel_xy_penalty.mean().item(), -ang_vel_xy_penalty.mean().item() * weight))
        reward -= ang_vel_xy_penalty * weight

        gait_phase_reward = get_gait_phase_reward(self)
        weight = 20 #逐渐调大权重
        print("gait_phase_reward: %.3f \t weighted: %.3f" % (gait_phase_reward.mean().item(), gait_phase_reward.mean().item() * weight))
        reward += gait_phase_reward * weight
        
        upper_body_deviation_penalty = joint_deviation_l1(self, 
                                                         joint_names=["j12_waist_yaw",
                                                                      "j13_shoulder_pitch_l", "j14_shoulder_roll_l", "j15_shoulder_yaw_l",
                                                                      "j16_elbow_pitch_l", "j17_elbow_yaw_l",
                                                                      "j18_shoulder_pitch_r", "j19_shoulder_roll_r", "j20_shoulder_yaw_r",
                                                                      "j21_elbow_pitch_r", "j22_elbow_yaw_r",
                                                                      "j23_head_yaw"])
        weight = 1
        reward -= upper_body_deviation_penalty * weight
        print("upper_body_deviation_penalty: %.3f \t weighted: %.3f" % (-upper_body_deviation_penalty.mean().item(), -upper_body_deviation_penalty.mean().item() * weight))

        waist_head_deviation_penalty = joint_deviation_l1(self, 
                                                         joint_names=["j12_waist_yaw", "j23_head_yaw"])
        weight = 2.0
        reward -= waist_head_deviation_penalty * weight
        print("waist_head_deviation_penalty: %.3f \t weighted: %.3f" % (-waist_head_deviation_penalty.mean().item(), -waist_head_deviation_penalty.mean().item() * weight))


        hip_deviation_penalty = joint_deviation_l1(self, joint_names=["j02_hip_yaw_l", "j08_hip_yaw_r", "j01_hip_roll_l", "j07_hip_roll_r"])
        weight = 1.0
        reward -= hip_deviation_penalty * weight
        print("hip_deviation_penalty: %.3f \t weighted: %.3f" % (-hip_deviation_penalty.mean().item(), -hip_deviation_penalty.mean().item() * weight))

        hip_deviation_l2_penalty = joint_deviation_l2(self, joint_names=["j02_hip_yaw_l", "j08_hip_yaw_r", "j01_hip_roll_l", "j07_hip_roll_r"])
        weight = 10.0
        reward -= hip_deviation_l2_penalty * weight
        print("hip_deviation_l2_penalty: %.3f \t weighted: %.3f" % (-hip_deviation_l2_penalty.mean().item(), -hip_deviation_l2_penalty.mean().item() * weight))

        leg_deviation_penalty = joint_deviation_l1(self, 
                                                   joint_names=["j00_hip_pitch_l", "j06_hip_pitch_r",
                                                                "j03_knee_pitch_l", "j09_knee_pitch_r",
                                                                "j04_ankle_pitch_l", "j10_ankle_pitch_r"])
        weight = 0.001
        reward -= leg_deviation_penalty * weight
        print("leg_deviation_penalty: %.3f \t weighted: %.3f" % (-leg_deviation_penalty.mean().item(), -leg_deviation_penalty.mean().item() * weight))

        joint_symmetry_penalty = joint_symmetry_l2(self, 
                                                   joint_pairs=[
                                                       ["j00_hip_pitch_l", "j06_hip_pitch_r"],
                                                       ["j03_knee_pitch_l", "j09_knee_pitch_r"],
                                                       ["j04_ankle_pitch_l", "j10_ankle_pitch_r"],
                                                   ])
        weight = 0.0
        reward -= joint_symmetry_penalty * weight
        print("joint_symmetry_penalty: %.3f \t weighted: %.3f" % (-joint_symmetry_penalty.mean().item(), -joint_symmetry_penalty.mean().item() * weight))

        left_leg_sum_penalty = joint_sum_l2(self, joint_names=["j00_hip_pitch_l", "j03_knee_pitch_l", "j04_ankle_pitch_l"])
        weight = 1
        reward -= left_leg_sum_penalty * weight
        print("left_leg_sum_penalty: %.3f \t weighted: %.3f" % (-left_leg_sum_penalty.mean().item(), -left_leg_sum_penalty.mean().item() * weight))

        left_leg_equal_penalty = joint_equal_l2(self, joint_name_a="j00_hip_pitch_l", joint_name_b="j04_ankle_pitch_l")
        weight = 1
        reward -= left_leg_equal_penalty * weight
        print("left_leg_equal_penalty: %.3f \t weighted: %.3f" % (-left_leg_equal_penalty.mean().item(), -left_leg_equal_penalty.mean().item() * weight))

        right_leg_sum_penalty = joint_sum_l2(self, joint_names=["j06_hip_pitch_r", "j09_knee_pitch_r", "j10_ankle_pitch_r"])
        weight = 1
        reward -= right_leg_sum_penalty * weight
        print("right_leg_sum_penalty: %.3f \t weighted: %.3f" % (-right_leg_sum_penalty.mean().item(), -right_leg_sum_penalty.mean().item() * weight))

        right_leg_equal_penalty = joint_equal_l2(self, joint_name_a="j06_hip_pitch_r", joint_name_b="j10_ankle_pitch_r")
        weight = 1
        reward -= right_leg_equal_penalty * weight
        print("right_leg_equal_penalty: %.3f \t weighted: %.3f" % (-right_leg_equal_penalty.mean().item(), -right_leg_equal_penalty.mean().item() * weight))

        #指令跟踪奖励
        command_lin_vel_reward = command_lin_vel_tracking_reward(self)
        weight = 3.0
        reward += command_lin_vel_reward * weight
        print("command_lin_vel_reward: %.3f \t weighted: %.3f" % (command_lin_vel_reward.mean().item(), command_lin_vel_reward.mean().item() * weight))

        command_ang_vel_reward = command_ang_vel_tracking_reward(self)
        weight = 0.5
        reward += command_ang_vel_reward * weight
        print("command_ang_vel_reward: %.3f \t weighted: %.3f" % (command_ang_vel_reward.mean().item(), command_ang_vel_reward.mean().item() * weight))
        
        
        #gait_phase_symmetry_rwd = gait_phase_symmetry_reward(self, [['j00_hip_pitch_l', 'j06_hip_pitch_r'],
        #                                                            ['j13_shoulder_pitch_l', 'j18_shoulder_pitch_r'], 
        #                                                            ['j03_knee_pitch_l', 'j09_knee_pitch_r'],
        #                                                             ['j04_ankle_pitch_l', 'j11_ankle_roll_r']])
        #weight = 0.2
        #reward += gait_phase_symmetry_rwd * weight
        #print("gait_phase_symmetry_rwd: %.3f \t weighted: %.3f" % (gait_phase_symmetry_rwd.mean().item(), gait_phase_symmetry_rwd.mean().item() * weight))
        
        feet_air_time_biped_reward = feet_air_time_biped(self, ['link_ankle_roll_l', 'link_ankle_roll_r'])
        weight = 10
        reward += feet_air_time_biped_reward * weight
        print("feet_air_time_biped_reward: %.3f \t weighted: %.3f" % (feet_air_time_biped_reward.mean().item(), feet_air_time_biped_reward.mean().item() * weight))

        print("total reward: %.3f" % reward.mean().item())
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fallen = self.robot.data.root_pos_w[:, 2] < 0.4
        l2 = flat_orientation_l2(self)
        tilted = l2 > 0.1   # 阈值可根据实际模型重心调
        done = torch.logical_or(fallen, tilted)
        return done, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset selected environments to default state (minimal version)."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # 调用父类逻辑（清理 buffers）
        super()._reset_idx(env_ids)

        # 默认状态
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        #print("joint pos on reset:", joint_pos[0])
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        root_state = self.robot.data.default_root_state[env_ids].clone()
        #print("root state on reset:", root_state[0])

        # 将每个环境放到对应的 origin（env_spacing 控制）
        root_state[:, :3] += self.scene.env_origins[env_ids]

        #print("root state after setting origin:", root_state[0])
        # 重置 IMU 速度缓存为0
        self._prev_root_lin_vel_b[env_ids] = torch.zeros_like(self._prev_root_lin_vel_b[env_ids])
        self._prev_root_ang_vel_b[env_ids] = torch.zeros_like(self._prev_root_ang_vel_b[env_ids])

        # 姿态添加少量随机扰动
        quat = torch.ones((len(env_ids), 4), device=self.device, dtype=torch.float32)
        quat[:, 1:] = 0.0
        noise_axis = torch.randn_like(quat[:, 1:])
        noise_axis = noise_axis / torch.norm(noise_axis, dim=-1, keepdim=True)
        noise_angle = 0.02 * torch.randn(len(env_ids), 1, device=self.device)  # 约9度随机旋转
        sin_half = torch.sin(noise_angle / 2)
        quat_noise = torch.cat([torch.cos(noise_angle / 2), sin_half * noise_axis], dim=-1)
        #root_state[:, 3:7] = quat_noise


        # 写入仿真
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        #self.gait_phase[env_ids] = 0.0
        # 随机初始相位
        self.gait_phase[env_ids] = sample_uniform(0.0, 2 * math.pi, (len(env_ids),), device=self.device)

        self._sample_commands(env_ids)

    def _sample_commands(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        """为指定环境采样新的行走指令。"""

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return

        num_envs = env_ids_t.shape[0]
        cmd_cfg = self.cfg.commands

        self.commands[env_ids_t, 0] = sample_uniform(
            cmd_cfg.lin_vel_x[0], cmd_cfg.lin_vel_x[1], (num_envs,), device=self.device
        )
        self.commands[env_ids_t, 1] = sample_uniform(
            cmd_cfg.lin_vel_y[0], cmd_cfg.lin_vel_y[1], (num_envs,), device=self.device
        )
        self.commands[env_ids_t, 2] = sample_uniform(
            cmd_cfg.ang_vel_yaw[0], cmd_cfg.ang_vel_yaw[1], (num_envs,), device=self.device
        )

        self._command_time_left[env_ids_t] = sample_uniform(
            cmd_cfg.resample_interval_range[0], cmd_cfg.resample_interval_range[1], (num_envs,), device=self.device
        )
