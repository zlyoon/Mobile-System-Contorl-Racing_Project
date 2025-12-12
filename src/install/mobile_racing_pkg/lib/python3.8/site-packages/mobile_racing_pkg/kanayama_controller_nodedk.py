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

        # ---------------- 파라미터 선언 ----------------
        self.declare_parameter('path_csv', 'optimized_path_eps015.csv')

        # 기본 기준 속도 (직선에서 최대 쓰려는 속도) [m/s]
        self.declare_parameter('v_ref_base', 13.0)

        # Kanayama gain
        self.declare_parameter('kx', 0.4)
        self.declare_parameter('ky', 1.2)
        self.declare_parameter('kth', 1.8)

        # topic 이름
        self.declare_parameter('ego_topic', '/mobile_system_control/ego_vehicle')
        self.declare_parameter('ctrl_topic', '/mobile_system_control/control_msg')
        self.declare_parameter('debug_odom_topic', '/debug_odom')
        self.declare_parameter('debug_cmd_vel_topic', '/debug_cmd_vel')

        # 파라미터 읽기
        path_csv = self.get_parameter('path_csv').get_parameter_value().string_value
        self.v_ref_base = self.get_parameter('v_ref_base').get_parameter_value().double_value
        self.kx = self.get_parameter('kx').get_parameter_value().double_value
        self.ky = self.get_parameter('ky').get_parameter_value().double_value
        self.kth = self.get_parameter('kth').get_parameter_value().double_value

        ego_topic = self.get_parameter('ego_topic').get_parameter_value().string_value
        ctrl_topic = self.get_parameter('ctrl_topic').get_parameter_value().string_value
        debug_odom_topic = self.get_parameter('debug_odom_topic').get_parameter_value().string_value
        debug_cmd_vel_topic = self.get_parameter('debug_cmd_vel_topic').get_parameter_value().string_value

        # ---------------- 경로 로드 ----------------
        self.get_logger().info(f'Loading global path from: {path_csv}')
        data = np.loadtxt(path_csv, delimiter=',', skiprows=1)
        self.path_x = data[:, 0]
        self.path_y = data[:, 1]
        self.num_points = len(self.path_x)

        self.path_yaw = self.compute_heading(self.path_x, self.path_y)
        self.path_kappa = self.compute_curvature(self.path_x, self.path_y)

        # ---------------- 주행 관련 상수 ----------------
        self.v_max = 15.5          # [m/s] 56 km/h
        self.w_max = 2.0           # [rad/s] yaw rate 제한
        self.mu = 0.9              # 마찰계수(코너 속도용)
        self.g = 9.81

        self.lookahead_gain = 0.3  # lookahead = lookahead_gain * v_meas (point 단위로 변환)
        self.merge_dist = 2.0      # [m] 경로와 이 거리 안으로 들어오면 ALIGN 모드로
        self.align_yaw_deg = 15.0  # [deg] 헤딩 차이가 이 이하이면 TRACK 모드로

        # 모드: MERGE -> ALIGN -> TRACK
        self.mode = "MERGE"

        # steer smoothing
        self.prev_steer = 0.0

        # ---------------- ROS2 Pub/Sub ----------------
        self.ctrl_pub = self.create_publisher(Vector3Stamped, ctrl_topic, 10)

        self.ego_sub = self.create_subscription(
            Float32MultiArray,
            ego_topic,
            self.ego_callback,
            10
        )

        self.debug_odom_pub = self.create_publisher(Odometry, debug_odom_topic, 10)
        self.debug_cmd_pub = self.create_publisher(Twist, debug_cmd_vel_topic, 10)

        # RViz 경로용
        self.path_pub = self.create_publisher(Path, '/global_path', 1)
        self.global_path_msg = self.build_path_msg()
        self.path_timer = self.create_timer(0.5, self.path_timer_callback)

        self.get_logger().info('KanayamaControllerNode (merge + fast lap) initialized.')

        # ---------------- 로그 파일 ----------------
        timestamp = int(time.time())
        self.log_file = open(
            f'/home/zlyoon/ros2_racing/log_kanayama_fast_{timestamp}.csv',
            'w',
            newline=''
        )
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow([
            't', 'mode',
            'x', 'y', 'yaw', 'v_meas',
            'e_x', 'e_y', 'e_th',
            'v_cmd', 'w_cmd',
            'throttle', 'steer_norm', 'brake'
        ])
  # ============================================================
    # 경로 관련 함수
    # ============================================================
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

    def build_path_msg(self):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for i in range(self.num_points):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(self.path_x[i])
            pose.pose.position.y = float(self.path_y[i])
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)
        return path_msg

    def path_timer_callback(self):
        if self.path_pub.get_subscription_count() == 0:
            return
        self.global_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.global_path_msg)

    # ============================================================
     # 시뮬레이터 상태 콜백
    # ============================================================
    def ego_callback(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) < 5:
            self.get_logger().warn('ego_vehicle data length < 5')
            return

        x = float(data[0])
        y = float(data[1])
        yaw = float(data[2])
        v_meas = float(data[3])
        steer_meas = float(data[4])  # 지금은 안 씀

        # 1) 먼저 경로 상 target index (거리 + 헤딩 고려) 계산
        idx_base = self.find_target_index(x, y, yaw)

        # 2) look-ahead index 적용 (빠를수록 더 앞을 봄)
        #    path 점 간 간격이 대략 일정하다고 보고, v_meas*lookahead_gain 만큼 인덱스 앞쪽 사용
        la_step = int(self.lookahead_gain * max(v_meas, 0.0))
        idx = min(idx_base + la_step, self.num_points - 1)

        xr = self.path_x[idx]
        yr = self.path_y[idx]
        thr = self.path_yaw[idx]
        kappa_r = self.path_kappa[idx]

        # 현재 위치와 경로/헤딩 차이
        dist = math.sqrt((x - xr)**2 + (y - yr)**2)
        yaw_err = abs(self.normalize_angle(yaw - thr))

        # ---------------- 모드 전환 로직 ----------------
        if self.mode == "MERGE":
            # 경로와 충분히 가까워지면 ALIGN으로 전환
            if dist < self.merge_dist:
                self.mode = "ALIGN"
            self.merge_mode(x, y, yaw, xr, yr)
            self.log_simple(x, y, yaw, v_meas, mode="MERGE")
            return

        if self.mode == "ALIGN":
            # 헤딩 차이가 충분히 작아지면 TRACK으로 전환
            if yaw_err < math.radians(self.align_yaw_deg):
                self.mode = "TRACK"
            self.align_mode(yaw, thr)
            self.log_simple(x, y, yaw, v_meas, mode="ALIGN")
            return

        # ---------------- TRACK 모드 (Kanayama) ----------------
        e_x, e_y, e_th = self.compute_errors(x, y, yaw, xr, yr, thr)

        # 곡률 기반 adaptive v_ref
        v_curve = math.sqrt(self.mu * self.g / (abs(kappa_r) + 1e-6))
        v_r = min(v_curve, self.v_max)
        # base v_ref와도 비교해서 너무 느려지지 않게 (직선에서 속도 회복)
        v_r = min(max(v_r, min(self.v_ref_base, self.v_max)), self.v_max)

        w_r = v_r * kappa_r

        # 속도 기반 gain scheduling (고속일 때 gain 줄이기)
        ky_eff = self.ky * (5.0 / (v_meas + 5.0))
        kth_eff = self.kth * (5.0 / (v_meas + 5.0))

        # Kanayama 제어 law
        v_cmd = v_r * math.cos(e_th) + self.kx * e_x
        w_cmd = w_r + v_r * (ky_eff * e_y + kth_eff * math.sin(e_th))

        # 속도/각속도 saturation
        v_cmd = max(0.0, min(v_cmd, self.v_max))
        w_cmd = max(-self.w_max, min(w_cmd, self.w_max))

        # v, w → throttle, steer_norm, brake 변환
        throttle, steer_norm, brake = self.vw_to_control(v_cmd, v_meas, w_cmd)

        # steering smoothing (jerk 감소)
        steer_norm = 0.7 * self.prev_steer + 0.3 * steer_norm
        self.prev_steer = steer_norm

        # 제어 publish
        self.publish_control(throttle, steer_norm, brake)

        # 디버그용 publish
        self.publish_debug_odom(x, y, yaw, v_meas)
        self.publish_debug_cmd(v_cmd, w_cmd)

        # 로그 기록
        now = self.get_clock().now().nanoseconds * 1e-9
        self.log_writer.writerow([
            now, self.mode,
            x, y, yaw, v_meas,
            e_x, e_y, e_th,
            v_cmd, w_cmd,
            throttle, steer_norm, brake
        ])

    def destroy_node(self):
        try:
            self.log_file.close()
        except Exception:
            pass
        return super().destroy_node()

    # ============================================================
    # 모드별 동작
    # ============================================================
    def merge_mode(self, x, y, yaw, xr, yr):
        """
        경로 바깥에서 경로 근처까지 진입하는 모드:
        단순히 목표점(xr, yr)을 향해 회전 + 천천히 접근.
        """
        dx = xr - x
        dy = yr - y
        target_yaw = math.atan2(dy, dx)
        steer = self.steer_to_yaw(yaw, target_yaw)
        throttle = 0.3
        brake = 0.0
        self.publish_control(throttle, steer, brake)

    def align_mode(self, yaw, thr):
        """
        경로 근처에서 헤딩만 맞추는 모드:
        thr 방향으로만 돌도록 하고 속도는 낮게.
        """
        steer = self.steer_to_yaw(yaw, thr)
        throttle = 0.2
        brake = 0.0
        self.publish_control(throttle, steer, brake)

    # ============================================================
    # 에러 계산 / 인덱스 선택
    # ============================================================
    def compute_errors(self, x, y, yaw, xr, yr, thr):
        dx = x - xr
        dy = y - yr

        cos_th = math.cos(thr)
        sin_th = math.sin(thr)

        e_x =  cos_th * dx + sin_th * dy
        e_y = -sin_th * dx + cos_th * dy
        e_th = self.normalize_angle(yaw - thr)

        return e_x, e_y, e_th

    def find_target_index(self, x, y, yaw):
        """
        거리 + 헤딩차이까지 고려해서 경로 상 목표 인덱스를 선정.
        """
        dx = self.path_x - x
        dy = self.path_y - y
        dist2 = dx*dx + dy*dy

        # 가장 가까운 20개 후보 중에서 헤딩까지 고려
        nearest = np.argsort(dist2)[:20]
        best_idx = int(nearest[0])
        best_cost = 1e9

        for idx in nearest:
            yaw_r = self.path_yaw[idx]
            yaw_err = abs(self.normalize_angle(yaw - yaw_r))
            cost = dist2[idx] + 2.0 * yaw_err  # 헤딩차이에도 가중치 부여
            if cost < best_cost:
                best_cost = cost
                best_idx = int(idx)

        return best_idx

    # ============================================================
    # 조향 / 제어 변환
    # ============================================================
    def steer_to_yaw(self, yaw, target_yaw):
        err = self.normalize_angle(target_yaw - yaw)
        max_steer_deg = 25.0
        max_rad = math.radians(max_steer_deg)
        steer_angle = max(-max_rad, min(max_rad, err))
        return steer_angle / max_rad

    def vw_to_control(self, v_cmd, v_meas, w_cmd):
        """
        Kanayama 출력 (v_cmd, w_cmd)를
        (throttle, steer_norm, brake)로 변환.
        """
        max_speed = self.v_max
        max_steer_deg = 25.0
        max_steer_rad = math.radians(max_steer_deg)
        wheelbase = 1.023

        # --- 속도 제어 (P 제어)
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

        # --- 조향 제어 (w_cmd -> steering angle)
        if abs(v_cmd) < 0.1:
            steer_angle = 0.0
        else:
            kappa = w_cmd / max(v_cmd, 0.1)
            steer_angle = math.atan(wheelbase * kappa)

        steer_angle = max(-max_steer_rad, min(max_steer_rad, steer_angle))
        steer_norm = steer_angle / max_steer_rad
        steer_norm = max(-1.0, min(1.0, steer_norm))

        return throttle, steer_norm, brake

    def publish_control(self, throttle, steer_norm, brake):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = TEAM_NAME
        msg.vector.x = float(throttle)
        msg.vector.y = float(steer_norm)
        msg.vector.z = float(brake)
        self.ctrl_pub.publish(msg)

    # ============================================================
    # 디버그 publish
    # ============================================================
    def publish_debug_odom(self, x, y, yaw, v):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link'
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        q = self.yaw_to_quaternion(yaw)
        msg.pose.pose.orientation = q
        msg.twist.twist.linear.x = v
        self.debug_odom_pub.publish(msg)

    def publish_debug_cmd(self, v, w):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.debug_cmd_pub.publish(cmd)

    # ============================================================
    # 유틸
    # ============================================================
    def yaw_to_quaternion(self, yaw):
        q = Quaternion()
        q.w = math.cos(yaw / 2.0)
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        return q

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def log_simple(self, x, y, yaw, v_meas, mode="MERGE"):
        now = self.get_clock().now().nanoseconds * 1e-9
        self.log_writer.writerow([
            now, mode,
            x, y, yaw, v_meas,
            0.0, 0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0, 0.0
        ])


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
