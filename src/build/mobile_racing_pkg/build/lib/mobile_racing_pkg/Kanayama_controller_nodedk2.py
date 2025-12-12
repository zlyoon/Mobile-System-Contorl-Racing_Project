#!/usr/bin/env python3
import csv
import time
import rclpy
from rclpy.node import Node

import numpy as np
import math

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3Stamped, Twist
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Quaternion

TEAM_NAME = "JDK"


class KanayamaControllerNode(Node):
    def __init__(self):
        super().__init__('kanayama_controller')

        # ============================================================
        #                     PARAMETERS
        # ============================================================
        self.declare_parameter('path_csv', 'optimized_path_eps015.csv')

        # 속도 관련
        self.declare_parameter('v_ref', 12.0)  # 직선 목표속도
        self.declare_parameter('ay_max', 5.0)  # 횡가속도 제한

        # Kanayama gains
        self.declare_parameter('kx', 0.25)
        self.declare_parameter('ky', 1.2)
        self.declare_parameter('kth', 1.4)

        # 경계 충돌 방지 (더 약하게 조정)
        self.declare_parameter('lane_half_width', 4.0)
        self.declare_parameter('rep_margin', 0.6)
        self.declare_parameter('rep_A', 0.2)
        self.declare_parameter('rep_B', 0.5)

        # ROS topics
        self.declare_parameter('ego_topic', '/mobile_system_control/ego_vehicle')
        self.declare_parameter('ctrl_topic', '/mobile_system_control/control_msg')
        self.declare_parameter('debug_odom_topic', '/debug_odom')
        self.declare_parameter('debug_cmd_vel_topic', '/debug_cmd_vel')

        # Load params
        path_csv = self.get_parameter('path_csv').value
        self.v_ref = self.get_parameter('v_ref').value
        self.kx = self.get_parameter('kx').value
        self.ky = self.get_parameter('ky').value
        self.kth = self.get_parameter('kth').value
        self.ay_max = self.get_parameter('ay_max').value

        self.lane_half_width = self.get_parameter('lane_half_width').value
        self.rep_margin = self.get_parameter('rep_margin').value
        self.rep_A = self.get_parameter('rep_A').value
        self.rep_B = self.get_parameter('rep_B').value

        self.base_v = self.v_ref
        self.base_ky = self.ky
        self.base_kth = self.kth

        # ============================================================
        #                     LOAD PATH
        # ============================================================
        self.get_logger().info(f"Loading racing line: {path_csv}")

        data = np.loadtxt(path_csv, delimiter=',', skiprows=1)
        self.path_x = data[:, 0]
        self.path_y = data[:, 1]
        self.num_points = len(self.path_x)

        # heading & curvature (smooth version)
        raw_yaw = self.compute_heading(self.path_x, self.path_y)
        self.path_yaw = self.smooth_signal(raw_yaw, 13)

        raw_kappa = self.compute_curvature(self.path_x, self.path_y)
        self.path_kappa = self.smooth_signal(raw_kappa, 11)

        # ============================================================
        #                     ROS PUBLISHERS
        # ============================================================
        ego_topic = self.get_parameter('ego_topic').value
        ctrl_topic = self.get_parameter('ctrl_topic').value
        debug_odom_topic = self.get_parameter('debug_odom_topic').value
        debug_cmd_vel_topic = self.get_parameter('debug_cmd_vel_topic').value

        self.ctrl_pub = self.create_publisher(Vector3Stamped, ctrl_topic, 10)
        self.ego_sub = self.create_subscription(Float32MultiArray, ego_topic, self.ego_callback, 10)

        self.debug_odom_pub = self.create_publisher(Odometry, debug_odom_topic, 10)
        self.debug_cmd_pub = self.create_publisher(Twist, debug_cmd_vel_topic, 10)

        # RVIZ PATH PUB
        self.path_pub = self.create_publisher(Path, '/global_path', 1)
        self.global_path_msg = self.build_path_msg()
        self.timer = self.create_timer(0.5, self.publish_path)

        # state
        self.idx_ref = None
        self.prev_steer = 0.0
        self.w_prev = 0.0

        # LOG
        timestamp = int(time.time())
        self.log_file = open(
            f'/home/zlyoon/ros2_racing/log_timeattack_{timestamp}.csv', 'w', newline=''
        )
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(
            ["t", "x", "y", "yaw", "v",
             "e_x", "e_y", "e_th",
             "v_cmd", "w_cmd",
             "throttle", "steer", "brake"]
        )

        self.get_logger().info("🔥 Time-Attack Controller Fully Loaded.")

    # ============================================================
    #                   PATH FUNCTIONS
    # ============================================================
    def compute_heading(self, x, y):
        dx = np.gradient(x)
        dy = np.gradient(y)
        return np.arctan2(dy, dx)

    def compute_curvature(self, x, y):
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        return (dx * ddy - dy * ddx) / ((dx*dx + dy*dy)**1.5 + 1e-6)

    def smooth_signal(self, sig, w=9):
        kernel = np.ones(w) / w
        return np.convolve(sig, kernel, mode='same')

    def build_path_msg(self):
        msg = Path()
        msg.header.frame_id = "map"
        for i in range(self.num_points):
            p = PoseStamped()
            p.header.frame_id = "map"
            p.pose.position.x = float(self.path_x[i])
            p.pose.position.y = float(self.path_y[i])
            msg.poses.append(p)
        return msg

    def publish_path(self):
        self.global_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.global_path_msg)

    # ============================================================
    #                   CONTROL CALLBACK
    # ============================================================
    def ego_callback(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) < 5:
            return

        x, y, yaw, v_meas = data[0], data[1], data[2], data[3]

        # 1. nearest point search
        idx = self.find_forward_index(x, y, v_meas)
        xr = self.path_x[idx]
        yr = self.path_y[idx]
        thr = self.path_yaw[idx]
        kappa_r = self.path_kappa[idx]

        # 2. tracking errors
        dx = x - xr
        dy = y - yr

        cos_th = math.cos(thr)
        sin_th = math.sin(thr)

        # 안정적인 e_x, e_y 정의
        e_x = cos_th * dx + sin_th * dy
        e_y = sin_th * dx - cos_th * dy
        e_th = self.normalize_angle(yaw - thr)

        dist_lat = abs(e_y)

        # 3. curvature-based safe speed
        k_safe = abs(kappa_r) + 0.003
        v_curv = math.sqrt(self.ay_max / k_safe)
        v_r = min(self.v_ref, v_curv)

        # lateral-error-based speed shaping
        sigma_lat = 2.0
        v_r = v_r * math.exp(-0.5 * (dist_lat / sigma_lat)**2)
        v_r = max(4.0, min(v_r, self.v_ref))

        w_r = v_r * kappa_r

        # 4. gain adaptation
        scale_v = max(0.6, min((self.base_v / v_r)**0.6, 1.5))
        curv_norm = min(abs(kappa_r) / 0.12, 1.0)
        gain_curv = 1.0 + 0.6 * curv_norm

        ky_eff = self.base_ky * scale_v * gain_curv
        kth_eff = self.base_kth * scale_v * gain_curv

        # 5. repulsive force near boundary
        dist_to_wall = self.lane_half_width - dist_lat
        e_y_rep = e_y

        if dist_to_wall < self.rep_margin:
            d = max(dist_to_wall, 0.0)
            rep = self.rep_A * math.exp(-d / self.rep_B)
            direction = -math.copysign(1.0, e_y)
            e_y_rep += direction * rep

        # 6. Kanayama control
        v = v_r * math.cos(e_th) + self.kx * e_x
        w = w_r + v_r * (ky_eff * e_y_rep + kth_eff * math.sin(e_th))

        # 7. filtering
        w = 0.2 * w + 0.8 * self.w_prev
        self.w_prev = w

        v = min(v, 15.5)
        w = max(min(w, 3.0), -3.0)

        throttle, steer_raw, brake = self.vw_to_control(v, v_meas, w)
        steer = 0.85 * self.prev_steer + 0.15 * steer_raw
        self.prev_steer = steer

        # 8. Publish control
        now = self.get_clock().now()
        cmd = Vector3Stamped()
        cmd.header.frame_id = TEAM_NAME
        cmd.header.stamp = now.to_msg()
        cmd.vector.x = float(throttle)
        cmd.vector.y = float(steer)
        cmd.vector.z = float(brake)
        self.ctrl_pub.publish(cmd)

        self.publish_debug(x, y, yaw, v_meas, v, w)

        # LOG
        t = now.nanoseconds * 1e-9
        self.log_writer.writerow([
            t, x, y, yaw, v_meas,
            e_x, e_y, e_th,
            v, w,
            throttle, steer, brake
        ])

    # ============================================================
    #                   HELPERS
    # ============================================================
    def find_forward_index(self, x, y, v):
        if self.idx_ref is None:
            dx = self.path_x - x
            dy = self.path_y - y
            self.idx_ref = int(np.argmin(dx*dx + dy*dy))
            return self.idx_ref

        window = 80
        indices = (self.idx_ref + np.arange(0, window)) % self.num_points

        dx = self.path_x[indices] - x
        dy = self.path_y[indices] - y
        local = int(np.argmin(dx*dx + dy*dy))

        self.idx_ref = int(indices[local])
        return self.idx_ref

    def normalize_angle(self, a):
        while a > math.pi: a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a

    def vw_to_control(self, v_cmd, v_meas, w_cmd):
        max_speed = 15.5
        v_cmd = max(0.0, min(max_speed, v_cmd))

        v_err = v_cmd - v_meas
        k_th, k_br = 0.22, 0.28

        if v_err >= 0:
            throttle = k_th * v_err
            brake = 0.0
        else:
            throttle = 0.0
            brake = k_br * (-v_err)

        throttle = min(1.0, max(0.0, throttle))
        brake = min(1.0, max(0.0, brake))

        # steering
        wheelbase = 1.023
        max_steer_rad = math.radians(30)

        if v_cmd < 0.1:
            kappa = 0.0
        else:
            kappa = w_cmd / max(v_cmd, 1e-3)

        steer_rad = math.atan(wheelbase * kappa)
        steer_rad = max(-max_steer_rad, min(max_steer_rad, steer_rad))
        steer = steer_rad / max_steer_rad

        return throttle, steer, brake

    def publish_debug(self, x, y, yaw, v, v_cmd, w_cmd):
        msg = Odometry()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        q = Quaternion()
        q.w = math.cos(yaw/2)
        q.z = math.sin(yaw/2)
        msg.pose.pose.orientation = q
        msg.twist.twist.linear.x = v
        self.debug_odom_pub.publish(msg)

        tw = Twist()
        tw.linear.x = float(v_cmd)
        tw.angular.z = float(w_cmd)
        self.debug_cmd_pub.publish(tw)


def main(args=None):
    rclpy.init(args=args)
    node = KanayamaControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
