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

        # === 파라미터 ===
        self.declare_parameter('path_csv', 'OOOptimized_path_eps015.csv')
        self.declare_parameter('v_ref', 11.5)    # [m/s] 기준 속도
        self.declare_parameter('kx', 0.5)
        self.declare_parameter('ky', 0.8)
        self.declare_parameter('kth', 1.8)

        # 추가 : 허용 최대 횡가속도
        self.declare_parameter('a_lat_max', 5.0)    # [m/s^2]

        # 입력/출력 topic
        self.declare_parameter('ego_topic', '/mobile_system_control/ego_vehicle')
        self.declare_parameter('ctrl_topic', '/mobile_system_control/control_msg')

        # 로그 관련 파라미터
        self.declare_parameter('enable_log', False)
        self.declare_parameter('log_file', 'kanayama_log.csv')

        # 파라미터 읽기
        path_csv = self.get_parameter('path_csv').get_parameter_value().string_value
        self.v_ref = self.get_parameter('v_ref').get_parameter_value().double_value
        self.kx = self.get_parameter('kx').get_parameter_value().double_value
        self.ky = self.get_parameter('ky').get_parameter_value().double_value
        self.kth = self.get_parameter('kth').get_parameter_value().double_value
        self.a_lat_max = self.get_parameter('a_lat_max').get_parameter_value().double_value

        ego_topic = self.get_parameter('ego_topic').get_parameter_value().string_value
        ctrl_topic = self.get_parameter('ctrl_topic').get_parameter_value().string_value

        self.enable_log = self.get_parameter('enable_log').get_parameter_value().bool_value
        self.log_file_path = self.get_parameter('log_file').get_parameter_value().string_value

        # === 경로 로드 ===
        self.get_logger().info(f'Loading global path from: {path_csv}')
        data = np.loadtxt(path_csv, delimiter=',', skiprows=1)
        self.path_x = data[:, 0]
        self.path_y = data[:, 1]
        self.num_points = len(self.path_x)

        self.path_yaw = self.compute_heading(self.path_x, self.path_y)
        self.path_kappa = self.compute_curvature(self.path_x, self.path_y)

        # === Publisher / Subscriber ===
        # 제어 명령 출력
        self.ctrl_pub = self.create_publisher(Vector3Stamped, ctrl_topic, 10)

        # 차량 상태 입력 (Float32MultiArray)
        self.ego_sub = self.create_subscription(
            Float32MultiArray,
            ego_topic,
            self.ego_callback,
            10
        )

        # === 로그 파일 설정 ===
        self.log_writer = None
        self.log_file_handle = None
        if self.enable_log:
            try:
                # 디렉토리 없으면 생성
                log_dir = os.path.dirname(self.log_file_path)
                if log_dir != '' and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

                file_exists = os.path.exists(self.log_file_path)
                self.log_file_handle = open(self.log_file_path, mode='a', newline='')
                self.log_writer = csv.writer(self.log_file_handle)

                # 새 파일이면 헤더 추가
                if not file_exists or os.path.getsize(self.log_file_path) == 0:
                    self.log_writer.writerow([
                        't',               # 시각 (sec)
                        'x', 'y', 'yaw',   # 현재 차량 자세
                        'v_meas', 'steer_meas',
                        'xr', 'yr', 'thr', # 참조 경로 포인트
                        'kappa_r',
                        'v_r', 'w_r',      # 참조 속도 / 참조 yaw rate
                        'e_x', 'e_y', 'e_th',
                        'v_cmd', 'w_cmd',
                        'throttle', 'steer_norm', 'brake',
                        'idx_path'
                    ])
                self.get_logger().info(f'Logging enabled -> {self.log_file_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to open log file: {e}')
                self.enable_log = False

        self.get_logger().info('KanayamaControllerNode initialized (with logging option).')

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

        # 1) 경로 위에서 가장 가까운 점 인덱스 찾기
        idx = self.find_nearest_index(x, y)

        xr = self.path_x[idx]
        yr = self.path_y[idx]
        thr = self.path_yaw[idx]
        kappa_r = self.path_kappa[idx]

        # ----- 곡률 기반 속도 제한 -----
        v_ref_max = self.v_ref           # 직선에서의 최대 속도
        kappa_abs = abs(kappa_r)

        if kappa_abs < 1e-6:
            # 거의 직선이면 곡률 제한 없이 최대 속도 사용
            v_kappa_limit = v_ref_max
        else:
            # v_max = sqrt(a_lat_max / |kappa|)
            v_kappa_limit = math.sqrt(self.a_lat_max / kappa_abs)

        # 실제로 쓸 기준 속도
        v_r = min(v_ref_max, v_kappa_limit)

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

        # 3) Kanayama 제어 law
        v_cmd = v_r * math.cos(e_th) + self.kx * e_x
        w_cmd = w_r + v_r * (self.ky * e_y + self.kth * math.sin(e_th))

        # 속도/각속도 saturation
        v_cmd = max(min(v_cmd, v_ref_max), 0.0)   # [m/s]
        w_cmd = max(min(w_cmd, 5.0), -5.0)        # [rad/s]

        # 4) (v, w) -> (throttle, steer, brake)
        throttle, steer_norm, brake = self.vw_to_control(v_cmd, v_meas, w_cmd)

        # 5) 제어 명령 publish (Vector3Stamped)
        now = self.get_clock().now()
        ctrl_msg = Vector3Stamped()
        ctrl_msg.header.stamp = now.to_msg()
        ctrl_msg.header.frame_id = TEAM_NAME
        ctrl_msg.vector.x = float(throttle)
        ctrl_msg.vector.y = float(steer_norm)
        ctrl_msg.vector.z = float(brake)
        self.ctrl_pub.publish(ctrl_msg)

        # 6) 로그 저장
        if self.enable_log and self.log_writer is not None:
            try:
                t_sec = now.nanoseconds * 1e-9  # ROS2 clock: nanoseconds from epoch
                self.log_writer.writerow([
                    f'{t_sec:.6f}',
                    x, y, yaw,
                    v_meas, steer_meas,
                    xr, yr, thr,
                    kappa_r,
                    v_r, w_r,
                    e_x, e_y, e_th,
                    v_cmd, w_cmd,
                    throttle, steer_norm, brake,
                    idx
                ])
                # 버퍼 강제 플러시(혹시 모를 크래시 대비)
                self.log_file_handle.flush()
            except Exception as e:
                self.get_logger().error(f'Failed to write log: {e}')
                # 한 번 깨지면 계속 에러 쌓이는 거 방지용
                self.enable_log = False

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

        - throttle: 0 ~ 1
        - steer_norm: -1 ~ 1 (정규화된 스티어)
        - brake: 0 ~ 1
        """
        max_steer_deg = 20.0
        max_steer_rad = math.radians(max_steer_deg)
        wheelbase = 1.023             # [m] 축거

        # --- 속도 제어 (throttle / brake)
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

        # --- 조향 제어 (w_cmd -> steering angle -> 정규화)
        if abs(v_cmd) < 0.1:
            kappa = 0.0
        else:
            kappa = w_cmd / v_cmd

        # bicycle model: delta = atan(L * kappa)
        steer_angle = math.atan(wheelbase * kappa)
        steer_angle = max(-max_steer_rad, min(max_steer_rad, steer_angle))
        steer_norm = steer_angle / max_steer_rad
        steer_norm = max(-1.0, min(1.0, steer_norm))

        return throttle, steer_norm, brake

    def destroy_node(self):
        # 로그 파일 정리
        if self.log_file_handle is not None:
            try:
                self.log_file_handle.close()
            except Exception:
                pass
        super().destroy_node()


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
