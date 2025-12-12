#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import math

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3Stamped

TEAM_NAME = "JDK"


class KanayamaControllerNode(Node):
    def __init__(self):
        super().__init__('kanayama_controller')

        # ===== 파라미터 =====
        self.declare_parameter('path_csv', 'OOOptimized_path_eps015.csv')
        self.declare_parameter('v_ref', 11.0)    # [m/s] 직선 기준 속도
        self.declare_parameter('kx', 0.4)
        self.declare_parameter('ky', 0.8)
        self.declare_parameter('kth', 1.8)

        # 허용 최대 횡가속도
        self.declare_parameter('a_lat_max', 5.0)    # [m/s^2]

        # 토픽 이름
        self.declare_parameter('ego_topic', '/mobile_system_control/ego_vehicle')
        self.declare_parameter('ctrl_topic', '/mobile_system_control/control_msg')

        # ----- 파라미터 읽기 -----
        path_csv = self.get_parameter('path_csv').get_parameter_value().string_value
        self.v_ref = self.get_parameter('v_ref').get_parameter_value().double_value
        self.kx = self.get_parameter('kx').get_parameter_value().double_value
        self.ky = self.get_parameter('ky').get_parameter_value().double_value
        self.kth = self.get_parameter('kth').get_parameter_value().double_value
        self.a_lat_max = self.get_parameter('a_lat_max').get_parameter_value().double_value

        ego_topic = self.get_parameter('ego_topic').get_parameter_value().string_value
        ctrl_topic = self.get_parameter('ctrl_topic').get_parameter_value().string_value

        # ===== 경로 로드 =====
        self.get_logger().info(f'Loading global path from: {path_csv}')

        # 헤더(문자) 유무 자동 판별
        with open(path_csv, 'r') as f:
            first = f.readline()
        skip = 1 if any(c.isalpha() for c in first) else 0

        data = np.loadtxt(path_csv, delimiter=',', skiprows=skip)
        self.path_x = data[:, 0]
        self.path_y = data[:, 1]
        self.num_points = len(self.path_x)

        self.path_yaw = self.compute_heading(self.path_x, self.path_y)
        self.path_kappa = self.compute_curvature(self.path_x, self.path_y)

        # 진행 방향 기반 인덱스 상태
        self.current_idx = 0
        self.has_init_index = False

        # ===== Publisher / Subscriber =====
        self.ctrl_pub = self.create_publisher(Vector3Stamped, ctrl_topic, 10)
        self.ego_sub = self.create_subscription(
            Float32MultiArray,
            ego_topic,
            self.ego_callback,
            10
        )

        self.get_logger().info('KanayamaControllerNode initialized (with lookahead).')

    # --------------------------------------------------------------
    # 경로 heading, curvature 계산
    # --------------------------------------------------------------
    def compute_heading(self, x, y):
        dx = np.gradient(x)
        dy = np.gradient(y)
        yaw = np.arctan2(dy, dx)
        return yaw

    def compute_curvature(self, x, y):
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        kappa = (dx * ddy - dy * ddx) / (np.power(dx*dx + dy*dy, 1.5) + 1e-6)
        return kappa

    # --------------------------------------------------------------
    # 진행 방향 기반 + lookahead target index
    # --------------------------------------------------------------
    def get_target_index(self, x, y, v_meas):
        # 0) lookahead 거리 [m] : 속도에 따라 증가
        v = max(v_meas, 0.0)
        L_min = 3.0
        L_max = 12.0
        Lfh = L_min + 0.5 * v
        Lfh = max(L_min, min(Lfh, L_max))

        # 1) 첫 호출이면 전체에서 최근접점
        if not self.has_init_index:
            nearest = self.find_nearest_index(x, y)
            self.current_idx = nearest
            self.has_init_index = True
        else:
            nearest = self.current_idx

        # 2) 현재 인덱스에서 "앞으로만" 일정 window 탐색해서 최근접점
        search_window = 60  # 앞으로 최대 60 포인트만 탐색
        start = nearest
        end = min(nearest + search_window, self.num_points - 1)

        seg_x = self.path_x[start:end+1]
        seg_y = self.path_y[start:end+1]
        dx = seg_x - x
        dy = seg_y - y
        dist2 = dx*dx + dy*dy
        local_idx = int(np.argmin(dist2))
        base_idx = start + local_idx

        # 3) base_idx에서 arc-length 기준 Lfh만큼 앞을 실제 목표점으로
        idx = base_idx
        dist_sum = 0.0
        while dist_sum < Lfh and idx < self.num_points - 1:
            dx = self.path_x[idx+1] - self.path_x[idx]
            dy = self.path_y[idx+1] - self.path_y[idx]
            dist_sum += math.hypot(dx, dy)
            idx += 1

        self.current_idx = idx
        return idx

    # --------------------------------------------------------------
    # 차량 상태 콜백
    # --------------------------------------------------------------
    def ego_callback(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) < 5:
            self.get_logger().warn('ego_vehicle data length < 5')
            return

        x = float(data[0])
        y = float(data[1])
        yaw = float(data[2])
        v_meas = float(data[3])
        steer_meas = float(data[4])  # 사용 안 해도 됨

        # 1) lookahead가 적용된 목표 인덱스
        idx = self.get_target_index(x, y, v_meas)

        xr = self.path_x[idx]
        yr = self.path_y[idx]
        thr = self.path_yaw[idx]
        kappa_r = self.path_kappa[idx]

        # ----- 곡률 기반 속도 제한 -----
        v_ref_max = self.v_ref
        kappa_abs = abs(kappa_r)

        if kappa_abs < 1e-4:          # 너무 작은 곡률은 직선 취급
            kappa_abs = 1e-4

        v_kappa_limit = math.sqrt(self.a_lat_max / kappa_abs)
        v_kappa_limit = min(v_kappa_limit, v_ref_max)
        v_kappa_limit = max(0.5 * v_ref_max, v_kappa_limit)   # 최소 50% 정도는 유지

        v_r = v_kappa_limit
        w_r = v_r * kappa_r

        # 2) 경로 좌표계 오차
        dx = x - xr
        dy = y - yr
        cos_th = math.cos(thr)
        sin_th = math.sin(thr)

        e_x =  cos_th * dx + sin_th * dy
        e_y = -sin_th * dx + cos_th * dy
        e_th = self.normalize_angle(yaw - thr)

        # 3) Kanayama 제어 law (원래 식 그대로)
        v = v_r * math.cos(e_th) + self.kx * e_x
        w = w_r + v_r * (self.ky * e_y + self.kth * math.sin(e_th))

        # saturation
        v = max(min(v, v_ref_max), 0.0)
        w = max(min(w, 5.0), -5.0)

        # 4) (v, w) → (throttle, steer, brake)
        throttle, steer_norm, brake = self.vw_to_control(v, v_meas, w)

        # 5) publish
        now = self.get_clock().now()
        ctrl_msg = Vector3Stamped()
        ctrl_msg.header.stamp = now.to_msg()
        ctrl_msg.header.frame_id = TEAM_NAME
        ctrl_msg.vector.x = float(throttle)
        ctrl_msg.vector.y = float(steer_norm)
        ctrl_msg.vector.z = float(brake)
        self.ctrl_pub.publish(ctrl_msg)

    # --------------------------------------------------------------
    # 유틸 함수들
    # --------------------------------------------------------------
    def find_nearest_index(self, x, y):
        dx = self.path_x - x
        dy = self.path_y - y
        dist2 = dx*dx + dy*dy
        idx = int(np.argmin(dist2))
        return idx

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def vw_to_control(self, v_cmd, v_meas, w_cmd):
        """
        Kanayama 출력 (v_cmd, w_cmd)을
        차량 제어 입력 (throttle, steer_norm, brake)로 변환.
        """
        max_steer_deg = 20.0
        max_steer_rad = math.radians(max_steer_deg)
        wheelbase = 1.023   # [m]

        # --- 종방향: throttle / brake ---
        v_error = v_cmd - v_meas
        k_v_throttle = 0.2
        k_v_brake = 0.3

        if v_error >= 0.0:
            throttle = k_v_throttle * v_error
            brake = 0.0
        else:
            throttle = 0.0
            brake = k_v_brake * (-v_error)

        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))

        # --- 횡방향: steering ---
        if abs(v_cmd) < 0.1:
            kappa = 0.0
        else:
            kappa = w_cmd / v_cmd

        steer_angle = math.atan(wheelbase * kappa)
        steer_angle = max(-max_steer_rad, min(max_steer_rad, steer_angle))
        steer_norm = steer_angle / max_steer_rad
        steer_norm = max(-1.0, min(1.0, steer_norm))

        return throttle, steer_norm, brake


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
