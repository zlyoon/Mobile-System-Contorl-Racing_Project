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


class PIDControllerNode(Node):
    def __init__(self):
        super().__init__('pid_controller')

        # === 파라미터 설정 ===
        self.declare_parameter('path_csv', 'optimized_path_eps15_sf3.csv')

        # 종방향 속도 목표 [m/s]
        self.declare_parameter('v_ref', 13.0)   # ≈ 28.8 km/h

        # 종방향 PID 게인
        self.declare_parameter('Kp_v', 0.4)
        self.declare_parameter('Ki_v', 0.05)
        self.declare_parameter('Kd_v', 0.05)

        # 횡방향 PID 게인 (경로 기준 lateral + heading)
        self.declare_parameter('Kp_lat', 0.6)
        self.declare_parameter('Ki_lat', 0.0)
        self.declare_parameter('Kd_lat', 0.15)

        # heading error 비중 (e_lat = e_y + K_heading*e_theta)
        self.declare_parameter('K_heading', 1.5)

        # 시뮬레이터 토픽
        self.declare_parameter('ego_topic', '/mobile_system_control/ego_vehicle')
        self.declare_parameter('ctrl_topic', '/mobile_system_control/control_msg')

        # 디버그 토픽 (RViz)
        self.declare_parameter('debug_odom_topic', '/debug_odom_pid')
        self.declare_parameter('debug_cmd_vel_topic', '/debug_cmd_vel_pid')

        # 파라미터 읽기
        path_csv = self.get_parameter('path_csv').get_parameter_value().string_value
        self.v_ref = self.get_parameter('v_ref').get_parameter_value().double_value

        self.Kp_v = self.get_parameter('Kp_v').get_parameter_value().double_value
        self.Ki_v = self.get_parameter('Ki_v').get_parameter_value().double_value
        self.Kd_v = self.get_parameter('Kd_v').get_parameter_value().double_value

        self.Kp_lat = self.get_parameter('Kp_lat').get_parameter_value().double_value
        self.Ki_lat = self.get_parameter('Ki_lat').get_parameter_value().double_value
        self.Kd_lat = self.get_parameter('Kd_lat').get_parameter_value().double_value

        self.K_heading = self.get_parameter('K_heading').get_parameter_value().double_value

        ego_topic = self.get_parameter('ego_topic').get_parameter_value().string_value
        ctrl_topic = self.get_parameter('ctrl_topic').get_parameter_value().string_value
        debug_odom_topic = self.get_parameter('debug_odom_topic').get_parameter_value().string_value
        debug_cmd_vel_topic = self.get_parameter('debug_cmd_vel_topic').get_parameter_value().string_value

        # === 경로 로드 ===
        self.get_logger().info(f'[PID] Loading global path from: {path_csv}')
        data = np.loadtxt(path_csv, delimiter=',')
        self.path_x = data[:, 0]
        self.path_y = data[:, 1]
        self.num_points = len(self.path_x)

        self.path_yaw = self.compute_heading(self.path_x, self.path_y)

        # === Publisher / Subscriber 설정 ===
        # 시뮬레이터 제어 출력
        self.ctrl_pub = self.create_publisher(Vector3Stamped, ctrl_topic, 10)

        # 시뮬레이터 상태 입력
        self.ego_sub = self.create_subscription(
            Float32MultiArray,
            ego_topic,
            self.ego_callback,
            10
        )

        # 디버깅용
        self.debug_odom_pub = self.create_publisher(Odometry, debug_odom_topic, 10)
        self.debug_cmd_pub = self.create_publisher(Twist, debug_cmd_vel_topic, 10)

        # 경로 publish (RViz용)
        self.path_pub = self.create_publisher(Path, '/global_path_pid', 1)
        self.global_path_msg = self.build_path_msg()
        self.path_timer = self.create_timer(0.5, self.path_timer_callback)

        # === PID 내부 상태 ===
        self.prev_time = None

        # speed PID
        self.int_e_v = 0.0
        self.prev_e_v = 0.0

        # lateral PID
        self.int_e_lat = 0.0
        self.prev_e_lat = 0.0

        self.get_logger().info('[PID] PIDControllerNode initialized.')

        # === 로그 파일 열기 ===
        timestamp = int(time.time())
        self.log_file = open(
            f'/home/zlyoon/ros2_racing/log_pid_{timestamp}.csv',
            'w',
            newline=''
        )
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow([
            't',
            'x', 'y', 'yaw', 'v_meas',
            'e_lat', 'e_y', 'e_th',  # PID는 lateral error 중심
            'v_cmd', 'steer_norm',
            'throttle', 'brake'
        ])


    # --------------------------------------------------------------
    # 경로 heading 계산
    # --------------------------------------------------------------
    def compute_heading(self, x, y):
        dx = np.gradient(x)
        dy = np.gradient(y)
        yaw = np.arctan2(dy, dx)
        return yaw

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

    # --------------------------------------------------------------
    # 시뮬레이터 상태 콜백
    # --------------------------------------------------------------
    def ego_callback(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) < 5:
            self.get_logger().warn('[PID] ego_vehicle data length < 5')
            return

        x = float(data[0])
        y = float(data[1])
        yaw = float(data[2])
        v_meas = float(data[3])
        # steering_meas = float(data[4])  # 필요시 사용

        # 시간 간격 dt 계산 (노드 clock 기준)
        now = self.get_clock().now()
        if self.prev_time is None:
            dt = 0.02  # 초기에는 대략 50Hz 가정
        else:
            dt = (now - self.prev_time).nanoseconds * 1e-9
            if dt <= 0.0:
                dt = 1e-3
        self.prev_time = now

        # 1) 경로에서 가장 가까운 점 찾기
        idx = self.find_nearest_index(x, y)
        xr = self.path_x[idx]
        yr = self.path_y[idx]
        thr = self.path_yaw[idx]

        # 2) 경로 좌표계 오차 계산
        dx_w = x - xr
        dy_w = y - yr

        cos_th = math.cos(thr)
        sin_th = math.sin(thr)

        e_x =  cos_th * dx_w + sin_th * dy_w       # 진행 방향 오차
        e_y = -sin_th * dx_w + cos_th * dy_w       # 횡방향 오차
        e_th = self.normalize_angle(yaw - thr)     # heading 오차

        # --- 2-1) lateral combined error ---
        e_lat = e_y + self.K_heading * e_th

        # 3) 종방향 PID (속도 제어)
        v_cmd, throttle, brake = self.longitudinal_pid(v_meas, dt)

        # 4) 횡방향 PID (조향 제어)
        steer_norm = self.lateral_pid(e_lat, dt, v_cmd)

        # 5) 제어 메시지 구성 (Vector3Stamped)
        ctrl_msg = Vector3Stamped()
        ctrl_msg.header.stamp = now.to_msg()
        ctrl_msg.header.frame_id = TEAM_NAME
        ctrl_msg.vector.x = float(throttle)
        ctrl_msg.vector.y = float(steer_norm)
        ctrl_msg.vector.z = float(brake)
        self.ctrl_pub.publish(ctrl_msg)

        # 6) 디버깅용 publish
        self.publish_debug_odom(x, y, yaw, v_meas)
        self.publish_debug_cmd(v_cmd, steer_norm, v_meas)

        # === 로그 저장 ===
        t_sec = now.nanoseconds * 1e-9

        self.log_writer.writerow([
            t_sec,
            x, y, yaw, v_meas,
            e_lat, e_y, e_th,
            v_cmd, steer_norm,
            throttle, brake
        ])

    def destroy_node(self):
        try:
            self.log_file.close()
        except Exception:
            pass
        return super().destroy_node()


    # --------------------------------------------------------------
    # 종방향 PID (속도)
    # --------------------------------------------------------------
    def longitudinal_pid(self, v_meas, dt):
        max_speed = 15.5  # [m/s] 56 km/h

        e_v = self.v_ref - v_meas

        # 적분 & 미분
        self.int_e_v += e_v * dt
        de_v = (e_v - self.prev_e_v) / dt
        self.prev_e_v = e_v

        u_v = self.Kp_v * e_v + self.Ki_v * self.int_e_v + self.Kd_v * de_v

        # u_v > 0 → throttle, u_v < 0 → brake
        if u_v >= 0.0:
            throttle = u_v
            brake = 0.0
        else:
            throttle = 0.0
            brake = -u_v

        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))

        # v_cmd는 v_ref를 그대로 쓰되, 상한 제한
        v_cmd = max(0.0, min(self.v_ref, max_speed))

        return v_cmd, throttle, brake

    # --------------------------------------------------------------
    # 횡방향 PID (조향)
    # --------------------------------------------------------------
    def lateral_pid(self, e_lat, dt, v_cmd):
        # PID for lateral error → steering angle
        self.int_e_lat += e_lat * dt
        de_lat = (e_lat - self.prev_e_lat) / dt
        self.prev_e_lat = e_lat

        u_lat = self.Kp_lat * e_lat + self.Ki_lat * self.int_e_lat + self.Kd_lat * de_lat

        # PID 출력(라디안) → 조향각
        max_steer_deg = 20.0
        max_steer_rad = math.radians(max_steer_deg)

        steer_angle = u_lat
        steer_angle = max(-max_steer_rad, min(max_steer_rad, steer_angle))

        # 정규화 [-1, 1]
        steer_norm = steer_angle / max_steer_rad
        steer_norm = max(-1.0, min(1.0, steer_norm))

        return steer_norm

    # --------------------------------------------------------------
    # 디버그용 publish
    # --------------------------------------------------------------
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

    def publish_debug_cmd(self, v_cmd, steer_norm, v_meas):
        # 디버그용으로 Twist에 "대략적인" v, w를 넣어줌
        wheelbase = 1.023  # [m]
        max_steer_deg = 20.0
        max_steer_rad = math.radians(max_steer_deg)

        steer_angle = steer_norm * max_steer_rad
        # v_meas 기준 yaw rate 근사
        w_cmd = math.tan(steer_angle) * v_meas / wheelbase if abs(wheelbase) > 1e-3 else 0.0

        cmd = Twist()
        cmd.linear.x = float(v_cmd)
        cmd.angular.z = float(w_cmd)
        self.debug_cmd_pub.publish(cmd)

    # --------------------------------------------------------------
    # 기타 유틸 함수
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

    def yaw_to_quaternion(self, yaw):
        q = Quaternion()
        q.w = math.cos(yaw / 2.0)
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        return q


def main(args=None):
    rclpy.init(args=args)
    node = PIDControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
