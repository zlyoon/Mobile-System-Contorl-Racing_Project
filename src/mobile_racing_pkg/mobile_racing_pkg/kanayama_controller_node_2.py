#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import math
import csv
import time

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3Stamped

TEAM_NAME = "JDK"


class KanayamaControllerNode(Node):
    def __init__(self):
        super().__init__('kanayama_controller')

        # === 파라미터 ===
        self.declare_parameter('path_csv', '/home/zlyoon/ros2_racing/path_eps015_margin200.csv')

        self.declare_parameter('v_ref', 11.5)        # [m/s] 직선 기준 속도
        self.declare_parameter('kx', 0.5)
        self.declare_parameter('ky', 0.8)
        self.declare_parameter('kth', 1.8)

        # 허용 최대 횡가속도 (곡률 기반 속도 제한)
        self.declare_parameter('a_lat_max', 5.0)     # [m/s^2]

        # 최근접점 탐색용 (local search 윈도우, lookahead)
        self.declare_parameter('search_window', 50)  # 가까운 인덱스 ±몇 개만 검색
        self.declare_parameter('lookahead', 5)       # 평상시 lookahead index
        # 곡률이 이 값보다 크면 lookahead 끔 (유턴/급코너)
        self.declare_parameter('kappa_turn_threshold', 0.25)  # [1/m] 정도에서 튜닝

        # 입력/출력 topic
        self.declare_parameter('ego_topic', '/mobile_system_control/ego_vehicle')
        self.declare_parameter('ctrl_topic', '/mobile_system_control/control_msg')

        # 파라미터 읽기
        path_csv = self.get_parameter('path_csv').get_parameter_value().string_value
        self.v_ref = self.get_parameter('v_ref').get_parameter_value().double_value
        self.kx = self.get_parameter('kx').get_parameter_value().double_value
        self.ky = self.get_parameter('ky').get_parameter_value().double_value
        self.kth = self.get_parameter('kth').get_parameter_value().double_value
        self.a_lat_max = self.get_parameter('a_lat_max').get_parameter_value().double_value

        self.search_window = self.get_parameter('search_window').get_parameter_value().integer_value
        self.lookahead = self.get_parameter('lookahead').get_parameter_value().integer_value
        self.kappa_turn_threshold = self.get_parameter(
            'kappa_turn_threshold'
        ).get_parameter_value().double_value

        ego_topic = self.get_parameter('ego_topic').get_parameter_value().string_value
        ctrl_topic = self.get_parameter('ctrl_topic').get_parameter_value().string_value

        # === 경로 로드 ===
        self.get_logger().info(f'Loading global path from: {path_csv}')
        data = np.loadtxt(path_csv, delimiter=',', skiprows=1)
        self.path_x = data[:, 0]
        self.path_y = data[:, 1]
        self.num_points = len(self.path_x)

        self.path_yaw = self.compute_heading(self.path_x, self.path_y)
        self.path_kappa = self.compute_curvature(self.path_x, self.path_y)

        # local-search용 현재 인덱스 상태
        self.current_idx = None

        # === Publisher / Subscriber ===
        self.ctrl_pub = self.create_publisher(Vector3Stamped, ctrl_topic, 10)

        self.ego_sub = self.create_subscription(
            Float32MultiArray,
            ego_topic,
            self.ego_callback,
            10
        )

        # === 로그 파일 설정 ===
        timestamp = int(time.time())
        log_path = f'/home/zlyoon/ros2_racing/log_kanayama_{timestamp}.csv'
        self.get_logger().info(f'Logging to: {log_path}')

        self.log_file = open(log_path, 'w', newline='')
        self.log_writer = csv.writer(self.log_file)

        # 🔹 21개 항목 포맷 헤더
        self.log_writer.writerow([
            't',                    # 시각 (sec)
            'x', 'y', 'yaw',
            'v_meas', 'steer_meas',
            'xr', 'yr', 'thr',
            'kappa_r',
            'v_r', 'w_r',
            'e_x', 'e_y', 'e_th',
            'v_cmd', 'w_cmd',
            'throttle', 'steer_norm', 'brake',
            'idx_ref'
        ])

        self.get_logger().info(
            'KanayamaControllerNode initialized '
            '(local nearest, curvature-based lookahead, forward-only v, logging enabled).'
        )

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
        kappa = (dx * ddy - dy * ddx) / (np.power(dx * dx + dy * dy, 1.5) + 1e-6)
        return kappa

    # --------------------------------------------------------------
    # 차량 상태 콜백 (Float32MultiArray)
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
        steer_meas = float(data[4])  # 현재 스티어 값 (사용 안 해도 됨)

        # 1) 경로 위에서 가장 가까운 점 인덱스 (local search)
        idx_center = self.find_nearest_index(x, y)
        kappa_center_abs = abs(self.path_kappa[idx_center])

        # --- 곡률 큰 구간에서는 lookahead 끄기 ---
        if kappa_center_abs > self.kappa_turn_threshold:
            eff_lookahead = 0
        else:
            eff_lookahead = self.lookahead

        # 최종 reference index
        idx_ref = (idx_center + eff_lookahead) % self.num_points

        xr = self.path_x[idx_ref]
        yr = self.path_y[idx_ref]
        thr = self.path_yaw[idx_ref]
        kappa_r = self.path_kappa[idx_ref]

        # ----- 곡률 기반 속도 제한 -----
        v_ref_max = self.v_ref           # 직선에서의 최대 속도
        kappa_abs = abs(kappa_r)

        if kappa_abs < 1e-6:
            v_kappa_limit = v_ref_max
        else:
            v_kappa_limit = math.sqrt(self.a_lat_max / kappa_abs)

        # 기준 속도: 곡률 기반 제한과 직선 최대 속도 중 작은 값
        v_r = min(v_ref_max, v_kappa_limit)

        # 너무 느려지지 않게 최소 속도 보장 (forward-only)
        v_min = 3.0   # 필요하면 2.0~4.0 사이에서 튜닝
        v_r = max(v_r, v_min)

        # 기준 yaw rate
        w_r = v_r * kappa_r

        # 2) 경로 좌표계로 오차 계산
        dx = x - xr
        dy = y - yr

        cos_th = math.cos(thr)
        sin_th = math.sin(thr)

        e_x =  cos_th * dx + sin_th * dy
        e_y = -sin_th * dx + cos_th * dy
        e_th = self.normalize_angle(yaw - thr)

        # 3) Kanayama 제어 law (속도는 전진만, 오차는 w로만 반영)
        v = v_r                          # e_th, e_x는 속도에는 반영하지 않음 (forward-only)
        v = max(v_min, min(v, v_ref_max))

        w = w_r + v_r * (self.ky * e_y + self.kth * math.sin(e_th))

        # 속도/각속도 saturation
        v = max(min(v, v_ref_max), 0.0)   # [m/s]
        w = max(min(w, 5.0), -5.0)        # [rad/s]

        # 4) (v, w) -> (throttle, steer, brake)
        throttle, steer_norm, brake = self.vw_to_control(v, v_meas, w)

        # 5) 제어 명령 publish (Vector3Stamped)
        now = self.get_clock().now()
        ctrl_msg = Vector3Stamped()
        ctrl_msg.header.stamp = now.to_msg()
        ctrl_msg.header.frame_id = TEAM_NAME
        ctrl_msg.vector.x = float(throttle)
        ctrl_msg.vector.y = float(steer_norm)
        ctrl_msg.vector.z = float(brake)
        self.ctrl_pub.publish(ctrl_msg)

        # 6) 로그 기록 (21개 항목 포맷)
        t_sec = now.nanoseconds * 1e-9
        self.log_writer.writerow([
            t_sec,
            x, y, yaw,
            v_meas, steer_meas,
            xr, yr, thr,
            kappa_r,
            v_r, w_r,
            e_x, e_y, e_th,
            v,    # v_cmd
            w,    # w_cmd
            throttle, steer_norm, brake,
            idx_ref
        ])

    # --------------------------------------------------------------
    # 유틸 함수들
    # --------------------------------------------------------------
    def find_nearest_index(self, x, y):
        """
        local search 기반 최근접점 찾기:
        - 첫 호출: 전체 경로에서 글로벌 최소 거리 인덱스
        - 이후: current_idx ± search_window 범위에서만 검색
        """
        N = self.num_points

        # 최초 한 번은 전체 검색
        if self.current_idx is None:
            dx = self.path_x - x
            dy = self.path_y - y
            dist2 = dx * dx + dy * dy
            idx = int(np.argmin(dist2))
            self.current_idx = idx
            return idx

        # 이후에는 current_idx 주변에서만 검색
        w = max(1, int(self.search_window))   # 안전하게 최소 1 이상
        local_indices = (np.arange(-w, w + 1) + self.current_idx) % N

        dx = self.path_x[local_indices] - x
        dy = self.path_y[local_indices] - y
        dist2 = dx * dx + dy * dy

        best_local = int(local_indices[np.argmin(dist2)])
        self.current_idx = best_local
        return best_local

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

        - throttle: 0 ~ 1
        - steer_norm: -1 ~ 1 (정규화된 스티어)
        - brake: 0 ~ 1
        """
        max_steer_deg = 20.0
        max_steer_rad = math.radians(max_steer_deg)
        wheelbase = 1.023             # [m] 축거

        # --- 속도 제어 (throttle only, brake 사용 X)
        v_error = v_cmd - v_meas
        k_v_throttle = 0.2

        throttle = k_v_throttle * v_error
        throttle = max(0.0, min(1.0, throttle))
        brake = 0.0

        # --- 조향 제어 (w_cmd -> steering angle -> 정규화)
        if abs(v_cmd) < 0.1:
            kappa = 0.0
        else:
            kappa = w_cmd / v_cmd

        steer_angle = math.atan(wheelbase * kappa)
        steer_angle = max(-max_steer_rad, min(max_steer_rad, steer_angle))
        steer_norm = steer_angle / max_steer_rad
        steer_norm = max(-1.0, min(1.0, steer_norm))

        return throttle, steer_norm, brake

    def destroy_node(self):
        # 노드 종료 시 로그 파일 안전하게 닫기
        try:
            if hasattr(self, 'log_file') and self.log_file:
                self.log_file.close()
        except Exception as e:
            self.get_logger().warn(f'Error while closing log file: {e}')
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KanayamaControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
