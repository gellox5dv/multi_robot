#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


def yaw_from_quat(q):
    # yaw from quaternion (x,y,z,w)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class OdomState(Node):
    def __init__(self):
        super().__init__("odom_state")

        self.r1: Optional[Pose2D] = None
        self.r2: Optional[Pose2D] = None

        self.create_subscription(Odometry, "/r1/odom", self.cb_r1, 10)
        self.create_subscription(Odometry, "/r2/odom", self.cb_r2, 10)

        self.timer = self.create_timer(0.2, self.tick)  # 5 Hz

    def cb_r1(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.r1 = Pose2D(p.x, p.y, yaw_from_quat(q))

    def cb_r2(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.r2 = Pose2D(p.x, p.y, yaw_from_quat(q))

    def tick(self):
        if self.r1 is None or self.r2 is None:
            self.get_logger().info("Waiting for /r1/odom and /r2/odom...")
            return

        dx = self.r1.x - self.r2.x
        dy = self.r1.y - self.r2.y
        dist = math.hypot(dx, dy)

        self.get_logger().info(
            f"r1: ({self.r1.x:.2f}, {self.r1.y:.2f}, yaw={self.r1.yaw:.2f}) | "
            f"r2: ({self.r2.x:.2f}, {self.r2.y:.2f}, yaw={self.r2.yaw:.2f}) | "
            f"dist={dist:.2f} m"
        )


def main():
    rclpy.init()
    node = OdomState()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
