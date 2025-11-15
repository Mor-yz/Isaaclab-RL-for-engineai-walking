import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from interface_protocol.msg import ImuInfo, JointState, JointCommand
import torch
import numpy as np
import math
import time
from scipy.signal import firwin


joint_names = [
    "j00_hip_pitch_l", "j01_hip_roll_l", "j02_hip_yaw_l", "j03_knee_pitch_l", "j04_ankle_pitch_l", "j05_ankle_roll_l",
    "j06_hip_pitch_r", "j07_hip_roll_r", "j08_hip_yaw_r", "j09_knee_pitch_r", "j10_ankle_pitch_r", "j11_ankle_roll_r",
    "j12_waist_yaw",
    "j13_shoulder_pitch_l", "j14_shoulder_roll_l", "j15_shoulder_yaw_l", "j16_elbow_pitch_l", "j17_elbow_yaw_l",
    "j18_shoulder_pitch_r", "j19_shoulder_roll_r", "j20_shoulder_yaw_r", "j21_elbow_pitch_r", "j22_elbow_yaw_r",
    "j23_head_yaw"
]

ros_topic_sequence = [0, 1, 2, 6, 7, 8, 12, 17, 3, 9, 10, 13, 4, 14, 15, 18, 5, 11, 16, 19, 20, 21, 22, 23]
nn_input_sequence = [0, 6, 12, 1, 7, 13, 18, 23, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]

default_offset = [-0.2, 0.0, 0.0, 0.45, -0.2, 0.0, 
                  -0.2, 0.0, 0.0, 0.45, -0.2, 0.0,
                  0.0,
                  0.0, 0.3, 0.0, 0.0, 0.0,
                  0.0, -0.3, 0.0, 0.0, 0.0,
                  0.0]

b = firwin(numtaps=21, cutoff=6.0, fs=200.0, pass_zero="lowpass")

class FIRFilter:
    def __init__(self, b, dim):
        """
        b  : FIR 系数，一维数组
        dim: 信号维度，比如 24（joint_pos_nn），3（w_real/euler_xyz）
        """
        self.b = np.asarray(b, dtype=float)
        self.dim = dim
        self.M = len(self.b)
        # 保存最近 M 个输入样本
        self.buf = np.zeros((self.M, self.dim), dtype=float)

    def filter(self, x):
        """
        输入一帧 x (dim,)，输出一帧 y (dim,)，适合在读每行时调用。
        """
        x = np.asarray(x, dtype=float).reshape(1, self.dim)

        # 滚动缓冲区：往后挪一格，把新数据放到 buf[0]
        self.buf = np.roll(self.buf, 1, axis=0)
        self.buf[0] = x

        # FIR 卷积：沿着时间维做加权和
        # y = sum_k b[k] * buf[k]
        y = np.tensordot(self.b, self.buf, axes=(0, 0))  # (dim,)
        return y

class PolicyInferenceNode(Node):
    def __init__(self):
        super().__init__('policy_inference_node')

        # ======== 加载策略模型 ========
        checkpoint_path = '../logs/rsl_rl/cartpole_direct/2025-11-11_10-21-55/exported/policy.pt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = torch.jit.load(checkpoint_path, map_location=self.device)
        self.policy.eval().to(self.device)
        self.get_logger().info("Loaded TorchScript policy model")
        self.action_filter = FIRFilter(b, dim=24)
        self.imu_filter = FIRFilter(b, dim=3)

        # ======== ROS2 订阅 ========
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  # 发布者使用的是 depth=1
            durability=DurabilityPolicy.VOLATILE
        )
        
        joint_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  # 发布者使用的是 depth=1
            durability=DurabilityPolicy.VOLATILE
        )
        self.sub_imu = self.create_subscription(ImuInfo, '/hardware/imu_info', self.imu_callback, imu_qos)
        self.sub_joint = self.create_subscription(JointState, '/hardware/joint_state', self.joint_callback, joint_qos)

        # ======== 发布 joint_command ========
        cmd_qos = QoSProfile(
            depth=10,  # 控制命令可以设置更大的深度
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.pub_cmd = self.create_publisher(JointCommand, '/hardware/joint_command', cmd_qos)

        # ======== 状态存储 ========
        self.imu_data = None
        self.joint_data = None

        # ======== Gait clock ========
        self.period = 0.8  # 秒
        self.phase_offset = np.random.uniform(0, 2 * math.pi)
        self.start_time = time.time()

        # ======== 控制循环（100 Hz） ========
        self.timer = self.create_timer(0.005, self.update)

        self.get_logger().info("Policy inference node started, 200 Hz")

        self.ros_to_nn_map = nn_input_sequence
        self.nn_to_ros_map = [self.ros_to_nn_map.index(i) for i in range(len(self.ros_to_nn_map))]
        print("ros_to_nn_map:", self.ros_to_nn_map)
        print("nn_to_ros_map:", self.nn_to_ros_map)

    def imu_callback(self, msg: ImuInfo):
        self.imu_data = msg

    def joint_callback(self, msg: JointState):
        self.joint_data = msg

    def update(self):
        if self.imu_data is None or self.joint_data is None:
            return

        # ========== 计算 gait phase ==========
        elapsed = (time.time() - self.start_time)
        phase = (elapsed * 2 * math.pi / self.period + self.phase_offset) % (2 * math.pi)
        phase_sin = math.sin(phase)
        phase_cos = math.cos(phase)

        # ========== 计算旋转矩阵 R_real ==========
        q = self.imu_data.quaternion
        qw, qx, qy, qz = q.w, q.x, q.y, q.z
        R_real = np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ])

        w_real = np.array([
            self.imu_data.angular_velocity.x,
            self.imu_data.angular_velocity.y,
            self.imu_data.angular_velocity.z,
        ])

        # ========== 计算欧拉角（ZYX, roll-pitch-yaw） ==========
        # roll = atan2(R[2,1], R[2,2])
        # pitch = -asin(R[2,0])
        # yaw = atan2(R[1,0], R[0,0])
        euler_xyz = np.array([
            math.atan2(R_real[2,1], R_real[2,2]),
            -math.asin(R_real[2,0]),
            math.atan2(R_real[1,0], R_real[0,0]),
        ])

        # ========== 从 JointState 提取 ==========
        joint_pos_ros = np.array(self.joint_data.position[:24])
        joint_vel_ros = np.array(self.joint_data.velocity[:24])

        joint_pos_nn = joint_pos_ros[self.ros_to_nn_map].copy()
        joint_vel_nn = joint_vel_ros[self.ros_to_nn_map].copy()

        # ========== commands 全 0 ==========
        commands = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # ========== 拼接 observation ==========
        obs = np.concatenate([
            self.imu_filter.filter(w_real),
            euler_xyz,
            joint_pos_nn,
            joint_vel_nn,
            [phase_sin],
            [phase_cos],
            commands,
        ])
        assert obs.shape[0] == 59, f"Observation size mismatch: {obs.shape}"
        print('w_real:', w_real)
        print('euler_xyz:', euler_xyz)
        print('joint_pos_nn', joint_pos_nn)
        print('joint_vel_nn', joint_vel_nn)
        print('phase_sin', phase_sin)
        print('phase_cos', phase_cos)
        print('commands', commands)



        # ========== 转 Tensor 推理 ==========
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.policy(x)
        action = action.squeeze(0).cpu().numpy()
        action_ros = action[self.nn_to_ros_map] + np.array(default_offset)
        action_ros = self.action_filter.filter(action_ros)
        print('action_ros:', action_ros)

        # ========== 生成 JointCommand ==========
        cmd = JointCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.position = [float(a) for a in action_ros[:24]]  # 取前24维或全部
        #cmd.position = [0.0] * 24  # 取前24维或全部
        cmd.velocity = [0.0] * 24
        cmd.torque = [0.0] * 24
        cmd.feed_forward_torque = [0.0] * 24
        cmd.stiffness = [50.0] * 24
        cmd.damping = [5.0] * 24
        cmd.parallel_parser_type = 0

        self.pub_cmd.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()