#!/usr/bin/env python3
import math
import random
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


def yaw_from_quat(q) -> float:
    # yaw from quaternion (x,y,z,w)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class NashController(Node):
    """
    Two-player, two-action repeated game:
      Actions: PUSH (move) or YIELD (stop)
    State: distance between robots from /r1/odom and /r2/odom
    Output: publish Twist to /r1/cmd_vel and /r2/cmd_vel
    """

    def __init__(self):
        super().__init__("nash_controller")

        # --- Subscribers (state) ---
        self.r1_odom: Optional[Odometry] = None
        self.r2_odom: Optional[Odometry] = None
        self.create_subscription(Odometry, "/r1/odom", self.cb_r1, 10)
        self.create_subscription(Odometry, "/r2/odom", self.cb_r2, 10)

        # --- Publishers (actions) ---
        self.pub_r1 = self.create_publisher(Twist, "/r1/cmd_vel", 10)
        self.pub_r2 = self.create_publisher(Twist, "/r2/cmd_vel", 10)

        # --- Game parameters you can tune ---
        self.tick_hz = 5.0
        self.close_dist = 1.0     # meters: "conflict zone"
        self.v_push = 0.2         # m/s forward when pushing
        self.v_yield = 0.0        # stop when yielding

        # Nash: we’ll compute a simple pure-strategy equilibrium for a 2x2 game
        # If two equilibria exist (coordination game), we choose deterministically using a tie-breaker.

        self.timer = self.create_timer(1.0 / self.tick_hz, self.tick)

        self.get_logger().info("NashController started. Waiting for /r1/odom and /r2/odom...")

    def cb_r1(self, msg: Odometry):
        self.r1_odom = msg

    def cb_r2(self, msg: Odometry):
        self.r2_odom = msg

    def get_xy_yaw(self, odom: Odometry) -> Tuple[float, float, float]:
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        return p.x, p.y, yaw_from_quat(q)

    def payoff_matrices(self, dist: float):
        """
        Return (A, B) payoff matrices for actions [PUSH, YIELD].
        A[a1][a2] = payoff for robot1, B[a1][a2] for robot2
        where a1 is r1 action index, a2 is r2 action index.

        We switch between:
          FAR: both pushing is good (2,2)
          CLOSE: both pushing is bad (-2,-2), one yields/one pushes is best
        """
        PUSH, YIELD = 0, 1

        if dist > self.close_dist:
            # Far: both can go
            A = [
                [2, 1],   # r1 PUSH vs (r2 PUSH, r2 YIELD)
                [0, 1],   # r1 YIELD vs (r2 PUSH, r2 YIELD)
            ]
            B = [
                [2, 0],
                [1, 1],
            ]
        else:
            # Close: congestion/collision risk if both push
            A = [
                [-2, 2],  # r1 PUSH vs (PUSH, YIELD)
                [0, 1],   # r1 YIELD vs (PUSH, YIELD)
            ]
            B = [
                [-2, 0],
                [2, 1],
            ]

        return A, B

    def best_responses(self, A, B):
        """
        Compute best responses for each player in a 2x2 game.
        Returns:
          BR1[a2] = set of best actions for player1 given player2 action a2
          BR2[a1] = set of best actions for player2 given player1 action a1
        """
        BR1 = []
        for a2 in [0, 1]:
            col = [A[a1][a2] for a1 in [0, 1]]
            m = max(col)
            BR1.append({a1 for a1, val in enumerate(col) if val == m})

        BR2 = []
        for a1 in [0, 1]:
            row = [B[a1][a2] for a2 in [0, 1]]
            m = max(row)
            BR2.append({a2 for a2, val in enumerate(row) if val == m})

        return BR1, BR2

    def pure_nash_equilibria(self, A, B):
        """
        Return list of pure Nash equilibria (a1, a2) in actions {0,1}.
        """
        BR1, BR2 = self.best_responses(A, B)
        eq = []
        for a1 in [0, 1]:
            for a2 in [0, 1]:
                if a1 in BR1[a2] and a2 in BR2[a1]:
                    eq.append((a1, a2))
        return eq

    def choose_equilibrium(self, eq_list, x1, y1, x2, y2):
        """
        Tie-break if multiple equilibria exist.
        Simple deterministic rule: robot closer to origin gets priority to PUSH.
        (You can replace with goal-distance or right-of-way later.)
        """
        # If only one equilibrium, take it
        if len(eq_list) == 1:
            return eq_list[0]

        # If multiple, pick who "wins" based on distance to (0,0)
        d1 = math.hypot(x1, y1)
        d2 = math.hypot(x2, y2)
        r1_priority = d1 <= d2

        # prefer (PUSH,YIELD) if r1 priority else (YIELD,PUSH)
        # only select if that equilibrium exists
        prefer = (0, 1) if r1_priority else (1, 0)
        if prefer in eq_list:
            return prefer

        # fallback: just pick the first
        return eq_list[0]

    def publish_action(self, pub, action: int):
        PUSH, YIELD = 0, 1
        msg = Twist()
        msg.linear.x = self.v_push if action == PUSH else self.v_yield
        msg.angular.z = 0.0
        pub.publish(msg)

    def tick(self):
        if self.r1_odom is None or self.r2_odom is None:
            return

        x1, y1, yaw1 = self.get_xy_yaw(self.r1_odom)
        x2, y2, yaw2 = self.get_xy_yaw(self.r2_odom)

        dist = math.hypot(x1 - x2, y1 - y2)

        A, B = self.payoff_matrices(dist)
        eq = self.pure_nash_equilibria(A, B)

        if not eq:
            # This shouldn't happen in our matrices, but safe fallback: both yield
            a1, a2 = (1, 1)
        else:
            a1, a2 = self.choose_equilibrium(eq, x1, y1, x2, y2)

        # publish commands
        self.publish_action(self.pub_r1, a1)
        self.publish_action(self.pub_r2, a2)

        # log
        name = {0: "PUSH", 1: "YIELD"}
        self.get_logger().info(
            f"dist={dist:.2f} close<{self.close_dist:.2f} => "
            f"eq={eq} chosen=({name[a1]},{name[a2]}) | "
            f"r1=({x1:.2f},{y1:.2f}) r2=({x2:.2f},{y2:.2f})"
        )


def main():
    rclpy.init()
    node = NashController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
