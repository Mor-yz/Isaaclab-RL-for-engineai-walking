# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot."""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
#from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

def _resolve_usd_path() -> str:
    """解析 USD 资源路径，并在缺少 Git LFS 资源时给出明确提示。"""

    usd_path = Path(__file__).resolve().parent / "usd" / "pm01.usd"
    if not usd_path.exists():
        raise FileNotFoundError(
            f"未找到 PM01 USD 资源文件：{usd_path}. 请确认仓库克隆完整。"
        )

    try:
        with usd_path.open("rb") as usd_file:
            header = usd_file.read(128)
    except OSError as exc:  # pragma: no cover - 仅在文件系统异常时触发
        raise OSError(f"无法读取 PM01 USD 资源文件：{usd_path}") from exc

    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            "检测到 PM01 USD 资源仍为 Git LFS 指针文件。"
            "请在仓库根目录执行 `git lfs install` 和 `git lfs pull` 后重试。"
        )

    return usd_path.as_posix()
USD_PATH = _resolve_usd_path()

PM01_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            enable_gyroscopic_forces=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85), 
        joint_pos={
            "j14_shoulder_roll_l": 0.3,
            "j19_shoulder_roll_r": -0.3,
            
            "j00_hip_pitch_l": -0.2,   # 左髋前屈
            "j03_knee_pitch_l": 0.45,   # 左膝弯曲
            "j04_ankle_pitch_l": -0.2, # 左踝背屈（脚尖稍下压）

            "j06_hip_pitch_r": -0.2,   # 右髋前屈
            "j09_knee_pitch_r": 0.45,   # 右膝弯曲
            "j10_ankle_pitch_r": -0.2, # 右踝背屈
        }
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=["j04_ankle_pitch_l", "j05_ankle_roll_l", "j10_ankle_pitch_r", "j11_ankle_roll_r",],
            effort_limit_sim=150.0,
            stiffness=20.0,
            damping=0.2,
        ),        
        "hip_knee": ImplicitActuatorCfg(
            joint_names_expr=["j00_hip_pitch_l", "j03_knee_pitch_l", "j06_hip_pitch_r", "j09_knee_pitch_r", ],    
            effort_limit_sim=300.0,    
            stiffness=70.0,           
            damping=7.0,                
        ),

        "others": ImplicitActuatorCfg(
            joint_names_expr=["j01_hip_roll_l", "j02_hip_yaw_l", "j07_hip_roll_r", "j08_hip_yaw_r", 
                            "j12_waist_yaw",
                            "j13_shoulder_pitch_l",
                            "j14_shoulder_roll_l",
                            "j15_shoulder_yaw_l",
                            "j16_elbow_pitch_l",
                            "j17_elbow_yaw_l",
                            "j18_shoulder_pitch_r",
                            "j19_shoulder_roll_r",
                            "j20_shoulder_yaw_r",
                            "j21_elbow_pitch_r",
                            "j22_elbow_yaw_r",
                            "j23_head_yaw",],    
            effort_limit_sim=300.0,    
            stiffness=50.0,           
            damping=5.0,                
        ),


    },
)
