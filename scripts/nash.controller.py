#!/usr/bin/env python3
import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


def yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def wrap_to_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))

class NashController(Node):
    """
    Two-player repeated 2x2 game each tick.
    Actions: PUSH (move forward) or YIELD (stop).
    Payoffs depend on:
      - progress toward each robot's goal
      - collision risk penalty when close
      - small time penalty for yielding too much
    """

    PUSH, YIELD = 0, 1

    def __init__(self):
        super().__init__("nash_controller")

        self.r1_odom: Optional[Odometry] = None
        self.r2_odom: Optional[Odometry] = None
        self.create_subscription(Odometry, "/r1/odom", self.cb_r1, 10)
        self.create_subscription(Odometry, "/r2/odom", self.cb_r2, 10)

        self.pub_r1 = self.create_publisher(Twist, "/r1/cmd_vel", 10)
        self.pub_r2 = self.create_publisher(Twist, "/r2/cmd_vel", 10)

        # steering controller (simple P controller)
        self.k_yaw = 1.5                # steering gain
        self.max_w = 1.2                # max angular speed
        self.turn_in_place_yaw = 0.9    # if heading error 

        # ---- Tune these ----
        self.tick_hz = 5.0
        self.v_push  = 0.5
        self.v_yield = 0.0

        # Goals (meters). Change to match your world.
        self.r1_goal = (-2.0, -5.0)
        self.r2_goal = (-5.0, 0.0)

        # Safety distances
        self.safe_dist = 0.9     # start caring
        self.collision_dist = 0.7  # heavy penalty

        # Payoff weights
        self.w_progress  = 2.0     # reward for moving toward goal
        self.w_yield_pen = 0.2    # small penalty for yielding (time loss)
        self.w_collision = 6.0    # penalty scale if close
        self.w_both_push_extra = 2.0  # extra penalty if both push when close
        self.w_r1_yield_bias = 2.5    # extra penalty for r1 pushing when too close

        self.timer = self.create_timer(1.0 / self.tick_hz, self.tick)
        self.get_logger().info("Goal-based NashController started.")

    def cb_r1(self, msg: Odometry):
        self.r1_odom = msg

    def cb_r2(self, msg: Odometry):
        self.r2_odom = msg

    def get_xy_yaw(self, odom: Odometry) -> Tuple[float, float, float]:
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        return p.x, p.y, yaw_from_quat(q)

    def dist_to_goal(self, x: float, y: float, goal: Tuple[float, float]) -> float:
        return math.hypot(goal[0] - x, goal[1] - y)

    def collision_penalty(self, dist: float) -> float:
        """
        Smooth-ish penalty: 0 when far, rises as robots get close.
        """
        if dist >= self.safe_dist:
            return 0.0
        # normalize 0..1 where 1 is collision_dist or closer
        if dist <= self.collision_dist:
            t = 1.0
        else:
            t = (self.safe_dist - dist) / (self.safe_dist - self.collision_dist)
        # quadratic grows faster near collision
        return self.w_collision * (t * t)

    def payoff_matrices(self, x1, y1, x2, y2, dist: float):
        """
        2x2 payoff matrices for actions [PUSH, YIELD].
        Payoff idea:
          - moving (PUSH) earns progress reward
          - yielding costs small time penalty
          - collision risk penalizes both, and extra penalizes BOTH PUSH when close
        """
        # progress reward: how much closer you'd get this tick if you PUSH
        # simple approximation: progress ≈ v_push * dt toward goal direction
        dt = 1.0 / self.tick_hz

        d1_now = self.dist_to_goal(x1, y1, self.r1_goal)
        d2_now = self.dist_to_goal(x2, y2, self.r2_goal)

        # max possible progress per tick (upper bound)
        prog1 = min(self.v_push * dt, d1_now)
        prog2 = min(self.v_push * dt, d2_now)

        base_collision = self.collision_penalty(dist)

        # Helper to compute payoff for a given action pair
        def payoff_for(a1, a2):
            # progress reward if PUSH
            r1 = self.w_progress * (prog1 if a1 == self.PUSH else 0.0)
            r2 = self.w_progress * (prog2 if a2 == self.PUSH else 0.0)

            # time/yield penalty
            r1 -= self.w_yield_pen if a1 == self.YIELD else 0.0
            r2 -= self.w_yield_pen if a2 == self.YIELD else 0.0

            # collision penalty applies to both when close
            r1 -= base_collision
            r2 -= base_collision

            # extra penalty if BOTH push when close (more dangerous)
            if a1 == self.PUSH and a2 == self.PUSH and dist < self.safe_dist:
                r1 -= self.w_both_push_extra
                r2 -= self.w_both_push_extra

            # Preference: when very close, discourage r1 from pushing
            if dist < self.collision_dist and a1 == self.PUSH:
                r1 -= self.w_r1_yield_bias

            return r1, r2

        A = [[0.0, 0.0], [0.0, 0.0]]
        B = [[0.0, 0.0], [0.0, 0.0]]
        for a1 in [self.PUSH, self.YIELD]:
            for a2 in [self.PUSH, self.YIELD]:
                p1, p2 = payoff_for(a1, a2)
                A[a1][a2] = p1
                B[a1][a2] = p2

        return A, B

    def best_responses(self, A, B):
        BR1 = []
        for a2 in [0, 1]:
            col = [A[a1][a2] for a1 in [0, 1]]
            m = max(col)
            BR1.append({a1 for a1, val in enumerate(col) if abs(val - m) < 1e-9})

        BR2 = []
        for a1 in [0, 1]:
            row = [B[a1][a2] for a2 in [0, 1]]
            m = max(row)
            BR2.append({a2 for a2, val in enumerate(row) if abs(val - m) < 1e-9})

        return BR1, BR2

    def pure_nash_equilibria(self, A, B):
        BR1, BR2 = self.best_responses(A, B)
        eq = []
        for a1 in [0, 1]:
            for a2 in [0, 1]:
                if a1 in BR1[a2] and a2 in BR2[a1]:
                    eq.append((a1, a2))
        return eq

    def choose_equilibrium(self, eq_list, d1_goal, d2_goal):
        """
        If multiple equilibria exist, give priority to the robot
        that is farther from its goal (needs more progress).
        """
        if len(eq_list) == 1:
            return eq_list[0]

        r1_priority = d1_goal >= d2_goal  # farther gets priority to PUSH
        prefer = (self.PUSH, self.YIELD) if r1_priority else (self.YIELD, self.PUSH)
        if prefer in eq_list:
            return prefer
        return eq_list[0]

    def publish_action(self, pub, action: int, x: float, y: float, yaw: float, goal):
        """
        If action is PUSH: drive toward goal with simple steering controller.
        If action is yield: stop.
        """
        msg = Twist()

        if action == self.YIELD:
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            pub.publish(msg)
            return

        # Desired goal
        gx, gy = goal
        desired = math.atan2(gy - y, gx - x)
        err = wrap_to_pi(desired - yaw)

        # Angular control (P controlle)
        w = clamp(self.k_yaw * err, -self.max_w, self.max_w)

        # Reduce forward speed if facing away
        v = self.v_push
        if abs(err) > self.turn_in_place_yaw:
            v = 0.1 * self.v_push #mostly rotate
            
        msg.linear.x = v
        msg.angular.z = w
        pub.publish(msg)

    def tick(self):
        if self.r1_odom is None or self.r2_odom is None:
            return

        x1, y1, yaw1 = self.get_xy_yaw(self.r1_odom)
        x2, y2, yaw2= self.get_xy_yaw(self.r2_odom)
        dist = math.hypot(x1 - x2, y1 - y2)

        d1_goal = self.dist_to_goal(x1, y1, self.r1_goal)
        d2_goal = self.dist_to_goal(x2, y2, self.r2_goal)

        # Stop if reached goal (simple)
        r1_done = d1_goal < 0.25
        r2_done = d2_goal < 0.25

        if r1_done and r2_done:
            self.publish_action(self.pub_r1, self.YIELD, x1, y1, yaw1, self.r1_goal)
            self.publish_action(self.pub_r2, self.YIELD, x2, y2, yaw2, self.r2_goal)
            self.get_logger().info("Both goals reached. Stopping.")
            return

        A, B = self.payoff_matrices(x1, y1, x2, y2, dist)
        eq = self.pure_nash_equilibria(A, B)

        if not eq:
            a1, a2 = (self.YIELD, self.YIELD)
        else:
            a1, a2 = self.choose_equilibrium(eq, d1_goal, d2_goal)

        # If a robot reached goal, force it to yield
        if r1_done:
            a1 = self.YIELD
        if r2_done:
            a2 = self.YIELD

        self.publish_action(self.pub_r1, a1, x1, y1, yaw1, self.r1_goal)
        self.publish_action(self.pub_r2, a2, x2, y2, yaw2, self.r2_goal)

        name = {0: "PUSH", 1: "YIELD"}
        self.get_logger().info(
            f"dist={dist:.2f}  d_goal(r1)={d1_goal:.2f} d_goal(r2)={d2_goal:.2f} | "
            f"eq={eq} chosen=({name[a1]},{name[a2]}) | "
            f"A={[[round(x,2) for x in row] for row in A]} B={[[round(x,2) for x in row] for row in B]}"
        )
        
def main():
    rclpy.init()
    node = NashController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
