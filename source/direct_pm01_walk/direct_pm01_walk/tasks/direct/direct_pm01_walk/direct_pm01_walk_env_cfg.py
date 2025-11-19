# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from direct_pm01_walk.assets.robots.pm01.pm01 import PM01_CFG
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg


import gymnasium as gym
import numpy as np


@configclass
class Pm01WalkSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            color=(0.9, 0.9, 0.9),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        ),
    )

    # robot
    robot: ArticulationCfg = PM01_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot", 
        spawn=PM01_CFG.spawn.replace(activate_contact_sensors=True),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(link_ankle_roll_l|link_ankle_roll_r)$", 
        history_length=3, 
        track_air_time=True,
        update_period=0.01
    )


@configclass
class CommandCfg:
    """机器人跟随的目标指令范围设置。"""

    # 指令刷新时间间隔范围（秒）
    resample_interval_range = (5.0, 10.0)
    # 机体坐标系下的目标线速度范围（m/s）
    lin_vel_x = (-0.0, 0.3)
    lin_vel_y = (-0.0, 0.0)
    # 机体坐标系下的目标偏航角速度范围（rad/s）
    ang_vel_yaw = (-0.3, 0.3)



@configclass
class DirectPm01WalkEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 150.0

    # - spaces definition
    action_space = 24
    observation_space = 59
    state_space = 2

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, 
                                       render_interval=decimation, 
                                       #gravity=(0.0, 0.0, 0.0),
    )

    # scene
    scene: Pm01WalkSceneCfg = Pm01WalkSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    commands: CommandCfg = CommandCfg()
