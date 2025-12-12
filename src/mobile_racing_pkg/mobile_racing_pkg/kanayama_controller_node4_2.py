#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import math
import csv
import os

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3Stamped

TEAM_NAME = "JDK"


class KanayamaControllerNode(Node):
    def __init__(self):
        super().__init__('kanayama_controller')

        # === 기본 파라미터 (YAML 필요 없음) ===
        self.path_csv = "OOOptimized_path_eps015.csv"
        self.v_ref = 12.0
        self.kx = 0.5
        self.ky = 0.8
        self.kth = 1.8

        self.a_lat_max = 7.0
        self.search_window = 50
        self.lookahead = 5
        self.kappa_turn_threshold = 0.25

        self.ego_topic = "/mobile_system_control/ego_vehicle"
        self.ctrl_topic = "/mobile_system_control/control_msg"

        # === 자동 로깅 옵션 ===
        self.enable_log = True
        self.log_file_path = "/home/zlyoon/ros2_logs/kanayama_log.csv"

        # === 경로 로드 ===
        self.get_logger().info(f'Loading path: {self.path_csv}')
        data = np.loadtxt(self.path_csv, delimiter=',', skiprows=1)
        self.path_x = data[:, 0]
        self.path_y = data[:, 1]
        self.num_points = len(self.path_x)

        self.path_yaw = self.compute_heading(self.path_x, self.path_y)
        self.path_kappa = self.compute_curvature(self.path_x, self.path_y)

        self.current_idx = None

        # === 자동 로깅 초기화 ===
        self.log_writer = None
        self.init_logging()

        # === pub/sub ===
        self.ctrl_pub = self.create_publisher(Vector3Stamped, self.ctrl_topic, 10)
        self.ego_sub = self.create_subscription(
            Float32MultiArray, self.ego_topic, self.ego_callback, 10
        )

        self.get_logger().info("KanayamaControllerNode initialized (auto logging enabled).")

    # --------------------------------------------------------------
    # 로깅 초기화 (YAML 필요 없음)
    # --------------------------------------------------------------
    def init_logging(self):
        if not self.enable_log:
            return

        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

        file_exists = os.path.exists(self.log_file_path)
        self.log_file = open(self.log_file_path, 'a', newline='')
        self.log_writer = csv.writer(self.log_file)

        if not file_exists or os.path.getsize(self.log_file_path) == 0:
            self.log_writer.writerow([
                't',
                'x', 'y', 'yaw',
                'v_meas', 'steer_meas',
                'xr', 'yr', 'thr',
                'kappa_r',
                'v_r', 'w_r',
                'e_x', 'e_y', 'e_th',
                'v_cmd', 'w_cmd',
                'throttle', 'steer_norm', 'brake',
                'idx_path'
            ])
            self.get_logger().info(f"Log header created: {self.log_file_path}")

    # --------------------------------------------------------------
    def compute_heading(self, x, y):
        dx = np.gradient(x)
        dy = np.gradient(y)
        return np.arctan2(dy, dx)

    def compute_curvature(self, x, y):
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        return (dx * ddy - dy * ddx) / (np.power(dx*dx + dy*dy, 1.5) + 1e-6)

    # --------------------------------------------------------------
    def ego_callback(self, msg):
        data = msg.data
        if len(data) < 5:
            return

        # === 상태 ===
        x, y, yaw, v_meas, steer_meas = map(float, data)

        # 최근접점
        idx_center = self.find_nearest_index(x, y)
        kappa_center_abs = abs(self.path_kappa[idx_center])

        eff_lookahead = 0 if kappa_center_abs > self.kappa_turn_threshold else self.lookahead
        idx_ref = (idx_center + eff_lookahead) % self.num_points

        xr = self.path_x[idx_ref]
        yr = self.path_y[idx_ref]
        thr = self.path_yaw[idx_ref]
        kappa_r = self.path_kappa[idx_ref]

        # === 곡률 기반 속도 제한 ===
        v_ref_max = self.v_ref
        kappa_abs = abs(kappa_r)

        if kappa_abs < 1e-6:
            v_kappa_limit = v_ref_max
        else:
            v_kappa_limit = math.sqrt(self.a_lat_max / kappa_abs)

        v_r = min(v_ref_max, v_kappa_limit)
        v_min = 3.0
        v_r = max(v_r, v_min)

        w_r = v_r * kappa_r

        # === 오차 (차량 좌표계) ===
        dx = xr - x
        dy = yr - y

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        e_x =  cos_y * dx + sin_y * dy
        e_y = -sin_y * dx + cos_y * dy
        e_th = self.normalize_angle(thr - yaw)

        # === Kanayama 제어 law ===
        v_cmd = v_r * math.cos(e_th) + self.kx * e_x
        v_cmd = max(v_min, min(v_cmd, v_ref_max))

        w_cmd = w_r + v_r * (self.ky * e_y + self.kth * math.sin(e_th))
        w_cmd = max(min(w_cmd, 5.0), -5.0)

        throttle, steer_norm, brake = self.vw_to_control(v_cmd, v_meas, w_cmd)

        # Publish
        msg_out = Vector3Stamped()
        msg_out.header.stamp = self.get_clock().now().to_msg()
        msg_out.header.frame_id = TEAM_NAME
        msg_out.vector.x = throttle
        msg_out.vector.y = steer_norm
        msg_out.vector.z = brake
        self.ctrl_pub.publish(msg_out)

        # === 로그 기록 ===
        if self.enable_log and self.log_writer:
            t = self.get_clock().now().nanoseconds * 1e-9
            self.log_writer.writerow([
                t, x, y, yaw,
                v_meas, steer_meas,
                xr, yr, thr,
                kappa_r,
                v_r, w_r,
                e_x, e_y, e_th,
                v_cmd, w_cmd,
                throttle, steer_norm, brake,
                idx_ref
            ])

    # --------------------------------------------------------------
    def find_nearest_index(self, x, y):
        N = self.num_points

        if self.current_idx is None:
            dx = self.path_x - x
            dy = self.path_y - y
            self.current_idx = int(np.argmin(dx*dx + dy*dy))
            return self.current_idx

        w = max(1, self.search_window)
        local_idx = (np.arange(-w, w+1) + self.current_idx) % N

        dx = self.path_x[local_idx] - x
        dy = self.path_y[local_idx] - y

        best = int(local_idx[np.argmin(dx*dx + dy*dy)])
        self.current_idx = best
        return best

    # --------------------------------------------------------------
    def normalize_angle(self, a):
        while a > math.pi:  a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a

    # --------------------------------------------------------------
    def vw_to_control(self, v_cmd, v_meas, w_cmd):

        max_steer_deg = 20.0
        max_steer_rad = math.radians(max_steer_deg)
        wheelbase = 1.023

        # 속도 제어
        v_error = v_cmd - v_meas
        throttle = max(0.0, min(1.0, 0.2 * v_error))
        brake = 0.0

        if abs(v_cmd) < 0.1:
            kappa = 0.0
        else:
            kappa = w_cmd / v_cmd

        steer_angle = math.atan(wheelbase * kappa)
        steer_angle = max(-max_steer_rad, min(max_steer_rad, steer_angle))

        steer_norm = steer_angle / max_steer_rad
        return throttle, steer_norm, brake


def main(args=None):
    rclpy.init(args=args)
    node = KanayamaControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    if node.enable_log:
        node.log_file.close()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
