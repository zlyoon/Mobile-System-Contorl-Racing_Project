#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import math
import numpy as np


class TestOdomPublisher(Node):
    def __init__(self):
        super().__init__('test_odom_publisher')

        self.declare_parameter('path_csv', 'optimized_path_eps015.csv')
        path_csv = self.get_parameter('path_csv').get_parameter_value().string_value

        self.get_logger().info(f'Loading path for fake odom from: {path_csv}')
        data = np.loadtxt(path_csv, delimiter=',')
        self.path_x = data[:, 0]
        self.path_y = data[:, 1]
        self.num_points = len(self.path_x)

        self.idx = 0  # 현재 인덱스

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.timer = self.create_timer(0.02, self.timer_callback)  # 50 Hz

    def timer_callback(self):
        x = float(self.path_x[self.idx])
        y = float(self.path_y[self.idx])

        next_idx = (self.idx + 1) % self.num_points
        dx = self.path_x[next_idx] - x
        dy = self.path_y[next_idx] - y
        yaw = math.atan2(dy, dx)

        q = self.yaw_to_quaternion(yaw)

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link'

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = q

        self.odom_pub.publish(msg)

        self.idx = next_idx

    def yaw_to_quaternion(self, yaw):
        q = Quaternion()
        q.w = math.cos(yaw / 2.0)
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        return q


def main(args=None):
    rclpy.init(args=args)
    node = TestOdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
