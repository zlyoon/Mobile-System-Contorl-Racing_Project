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


class PurePursuitControllerNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_controller')

        # === 파라미터 ===
        self.declare_parameter('path_csv', 'optimized_path_eps015.csv')

        # 목표 속도 [m/s] (56 km/h ≈ 15.5 m/s)
        self.declare_parameter('v_ref', 8.0)   # 대략 28.8 km/h

        # Pure Pursuit 룩어헤드 거리 [m]
        self.declare_parameter('Ld', 3.0)

        # 종방향 PID (속도 제어용)
        self.declare_parameter('Kp_v', 0.4)
        self.declare_parameter('Ki_v', 0.1)
        self.declare_parameter('Kd_v', 0.05)

        # 시뮬레이터 topic
        self.declare_parameter('ego_topic', '/mobile_system_control/ego_vehicle')
        self.declare_parameter('ctrl_topic', '/mobile_system_control/control_msg')

        # 디버그 토픽
        self.declare_parameter('debug_odom_topic', '/debug_odom_pp')
        self.declare_parameter('debug_cmd_vel_topic', '/debug_cmd_vel_pp')

        # 파라미터 읽기
        path_csv = self.get_parameter('path_csv').get_parameter_value().string_value
        self.v_ref = self.get_parameter('v_ref').get_parameter_value().double_value
        self.Ld = self.get_parameter('Ld').get_parameter_value().double_value

        self.Kp_v = self.get_parameter('Kp_v').get_parameter_value().double_value
        self.Ki_v = self.get_parameter('Ki_v').get_parameter_value().double_value
        self.Kd_v = self.get_parameter('Kd_v').get_parameter_value().double_value

        ego_topic = self.get_parameter('ego_topic').get_parameter_value().string_value
        ctrl_topic = self.get_parameter('ctrl_topic').get_parameter_value().string_value
        debug_odom_topic = self.get_parameter('debug_odom_topic').get_parameter_value().string_value
        debug_cmd_vel_topic = self.get_parameter('debug_cmd_vel_topic').get_parameter_value().string_value

        # === 경로 로드 ===
        self.get_logger().info(f'[PP] Loading global path from: {path_csv}')
        data = np.loadtxt(path_csv, delimiter=',')
        self.path_x = data[:, 0]
        self.path_y = data[:, 1]
        self.num_points = len(self.path_x)

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
        self.path_pub = self.create_publisher(Path, '/global_path_pp', 1)
        self.global_path_msg = self.build_path_msg()
        self.path_timer = self.create_timer(0.5, self.path_timer_callback)

        # 종방향 PID 내부 상태
        self.prev_time = None
        self.int_e_v = 0.0
        self.prev_e_v = 0.0

        self.get_logger().info('[PP] PurePursuitControllerNode initialized.')

        # === 로그 파일 열기 ===
        timestamp = int(time.time())
        self.log_file = open(
            f'/home/zlyoon/ros2_racing/log_pp_{timestamp}.csv',
            'w',
            newline=''
        )
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow([
            't',
            'x', 'y', 'yaw', 'v_meas',
            'x_local', 'y_local',  # Pure Pursuit 핵심 오차 (local frame)
            'kappa', 'steer_angle', 'steer_norm',
            'v_cmd', 'throttle', 'brake'
        ])


    # --------------------------------------------------------------
    # Path 관련
    # --------------------------------------------------------------
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
            self.get_logger().warn('[PP] ego_vehicle data length < 5')
            return

        x = float(data[0])
        y = float(data[1])
        yaw = float(data[2])
        v_meas = float(data[3])
        # steering_meas = float(data[4])  # 필요하면 사용

        now = self.get_clock().now()
        if self.prev_time is None:
            dt = 0.02
        else:
            dt = (now - self.prev_time).nanoseconds * 1e-9
            if dt <= 0.0:
                dt = 1e-3
        self.prev_time = now

        # 1) Pure Pursuit: look-ahead target 찾기
        target_x, target_y = self.find_lookahead_point(x, y)

        # 2) vehicle 좌표계로 변환
        # world → vehicle frame (x_forward, y_left)
        dx = target_x - x
        dy = target_y - y

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        x_local =  cos_yaw * dx + sin_yaw * dy
        y_local = -sin_yaw * dx + cos_yaw * dy

        # look-ahead 거리
        Ld = max(self.Ld, math.sqrt(x_local**2 + y_local**2) + 1e-6)

        # Pure Pursuit 곡률 kappa = 2 * y_local / Ld^2
        kappa = 2.0 * y_local / (Ld * Ld)

        # steering angle = atan(L * kappa)
        wheelbase = 1.023  # [m] 차량 축거
        steer_angle = math.atan(wheelbase * kappa)

        # 스티어 제한 (±20 deg)
        max_steer_deg = 20.0
        max_steer_rad = math.radians(max_steer_deg)
        steer_angle = max(-max_steer_rad, min(max_steer_rad, steer_angle))

        # 정규화 [-1, 1]
        steer_norm = steer_angle / max_steer_rad
        steer_norm = max(-1.0, min(1.0, steer_norm))

        # 3) 종방향 PID로 throttle / brake 계산
        v_cmd, throttle, brake = self.longitudinal_pid(v_meas, dt)

        # 4) 제어 메시지 publish (Vector3Stamped)
        ctrl_msg = Vector3Stamped()
        ctrl_msg.header.stamp = now.to_msg()
        ctrl_msg.header.frame_id = TEAM_NAME
        ctrl_msg.vector.x = float(throttle)
        ctrl_msg.vector.y = float(steer_norm)
        ctrl_msg.vector.z = float(brake)
        self.ctrl_pub.publish(ctrl_msg)

        # 5) 디버그용 publish
        self.publish_debug_odom(x, y, yaw, v_meas)
        self.publish_debug_cmd(v_cmd, steer_angle, v_meas, wheelbase)

        # === 로그 저장 ===
        t_sec = now.nanoseconds * 1e-9

        self.log_writer.writerow([
            t_sec,
            x, y, yaw, v_meas,
            x_local, y_local,
            kappa, steer_angle, steer_norm,
            v_cmd, throttle, brake
        ])

    def destroy_node(self):
        try:
            self.log_file.close()
        except Exception:
            pass
        return super().destroy_node()


    # --------------------------------------------------------------
    # Look-ahead target 찾기
    # --------------------------------------------------------------
    def find_lookahead_point(self, x, y):
        """
        현재 위치 (x, y)에서 경로 점들까지 거리 중
        Ld에 가장 가까운 점을 look-ahead target으로 선택
        """
        dx = self.path_x - x
        dy = self.path_y - y
        dist = np.sqrt(dx*dx + dy*dy)

        # Ld보다 큰 점들 중에서 가장 가까운 것 선택
        candidates = np.where(dist > self.Ld)[0]
        if len(candidates) == 0:
            # 모든 점이 너무 가까우면 그냥 가장 먼 점 사용
            idx = int(np.argmax(dist))
        else:
            idx = candidates[int(np.argmin(dist[candidates]))]

        return float(self.path_x[idx]), float(self.path_y[idx])

    # --------------------------------------------------------------
    # 종방향 PID (속도 제어)
    # --------------------------------------------------------------
    def longitudinal_pid(self, v_meas, dt):
        max_speed = 15.5  # [m/s] 56 km/h

        e_v = self.v_ref - v_meas

        self.int_e_v += e_v * dt
        de_v = (e_v - self.prev_e_v) / dt
        self.prev_e_v = e_v

        u_v = self.Kp_v * e_v + self.Ki_v * self.int_e_v + self.Kd_v * de_v

        if u_v >= 0.0:
            throttle = u_v
            brake = 0.0
        else:
            throttle = 0.0
            brake = -u_v

        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))

        v_cmd = max(0.0, min(self.v_ref, max_speed))

        return v_cmd, throttle, brake

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

    def publish_debug_cmd(self, v_cmd, steer_angle, v_meas, wheelbase):
        # 조향각으로부터 대략적인 yaw rate 근사
        if abs(wheelbase) > 1e-3:
            w_cmd = math.tan(steer_angle) * v_meas / wheelbase
        else:
            w_cmd = 0.0

        cmd = Twist()
        cmd.linear.x = float(v_cmd)
        cmd.angular.z = float(w_cmd)
        self.debug_cmd_pub.publish(cmd)

    # --------------------------------------------------------------
    # 기타 유틸
    # --------------------------------------------------------------
    def yaw_to_quaternion(self, yaw):
        q = Quaternion()
        q.w = math.cos(yaw / 2.0)
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        return q


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
