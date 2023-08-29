#!/usr/bin/env python3

"""
A collision avoidance node that detects potential collision based on onboard
    multirange sensor measurements, provides nearest obstacle point coordinates,
    and provide collision avoidance maneuvers.

    2023 = Shyam Rauniyar (IASRL)
"""
import rclpy
from rclpy.node import Node

from crazyflie_interfaces.srv import CA
from crazyflie_interfaces.msg import CollDetect
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from functools import partial
import numpy as np
import math

class DetectAndAvoid(Node):
    def __init__(self):
        super().__init__('detect_avoid',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,)
        # self.declare_parameter('ca_threshold1', 0.2)
        # self.declare_parameter('ca_threshold2', 0.4)
        # self.declare_parameter('avoidance_vel', 0.5)
        # self.declare_parameter('robot_prefix', '/cf2')

        self.ca_threshold1  = self.get_parameter('ca_threshold1').value
        self.ca_threshold2  = self.get_parameter('ca_threshold2').value
        self.avoidance_vel  = self.get_parameter('avoidance_vel').value
        robot_prefix  = self.get_parameter('robot_prefix').value
        
        # Publishers
        self.publisher_collision = self.create_publisher(
            CollDetect, 
            robot_prefix + '/coll_detect', 
            10)

        # Subscriptions
        self.pose_sub = self.create_subscription(
            PoseStamped,
            robot_prefix + '/pose',
            self.pose_callback,
            10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            robot_prefix + '/scan',
            self.scan_callback,
            10)

        # Clients
        
        
        # Services
        self.create_service(
            CA,
            robot_prefix + "/ca", 
            partial(self.smooth_ca))

        # Initializations
        self.nearest = [float("inf")] * 3
        self.nearest_dist = float("inf")

        self.get_logger().info(f"Detect and Avoid set for {robot_prefix}"+
                               f" with ca_th1 = {self.ca_threshold1} m, ca_th2 = {self.ca_threshold2} m")

    def pose_callback(self, msg:PoseStamped):
        self.msg_pose = msg
        cf_position = self.msg_pose.pose.position
        self.cf_inertial = [cf_position.x, cf_position.y, cf_position.z]

    def scan_callback(self, msg:LaserScan):
        self.msg_scan = msg
        ranges = list(msg.ranges)
        coll_check_ranges = [i <= self.ca_threshold2 for i in ranges] # Collision check via ranges
        self.nearest_range = min(ranges)
        self.nearest_range_index = ranges.index(self.nearest_range)
        self.current_nearest = self.range_inertial(self.nearest_range, self.nearest_range_index)
        self.nearest, self.nearest_dist = self.compare_nearest(self.current_nearest, self.nearest)
        coll_check = any(coll_check_ranges)
        # coll_check = self.nearest_dist <= self.ca_threshold2 # Collision check via nearest sensed obs pt.

        msg_collision = CollDetect()
        msg_collision.nearest = self.current_nearest
        if coll_check:
            msg_collision.collision = True
        else:
            msg_collision.collision = False
        self.publisher_collision.publish(msg_collision)

    def repel_ca(self, request, response):
        old_hover = request.old_hover
        new_hover = old_hover
        vel_avoid = self.avoidance_vel
        new_hover.vx = 0.0
        new_hover.vy = 0.0
        new_hover.yaw_rate = 0.0

        if self.nearest_range_index == 0: # back
            new_hover.vx = vel_avoid
        if self.nearest_range_index == 1: # right
            new_hover.vy = vel_avoid
        if self.nearest_range_index == 2: # front
            new_hover.vx = -vel_avoid
        if self.nearest_range_index == 3: # left
            new_hover.vy = -vel_avoid

        response.new_hover = new_hover

        self.get_logger().info("Repel CA engaged.")
        return response

    def repel_ca2(self, request, response):
        old_hover = request.old_hover
        new_hover = old_hover
        vel_avoid = self.avoidance_vel
        new_hover.vx = 0.0
        new_hover.vy = 0.0
        new_hover.yaw_rate = 0.0

        x = self.nearest[0] # x axis, -x is back, +x is front
        y = self.nearest[1] # y axis, -y is right, +y is left
        new_hover.vx = - vel_avoid * float(np.sign(x))
        new_hover.vy = - vel_avoid * float(np.sign(y))

        response.new_hover = new_hover

        self.get_logger().info("Repel CA engaged.")
        return response

    def smooth_ca(self, request, response):
        old_hover = request.old_hover
        new_hover = old_hover
        vel_avoid = self.avoidance_vel
        m1, c1 = self.vel_eq(vel_avoid, 0.0, self.ca_threshold2, self.ca_threshold1)
        m2, c2 = self.vel_eq(0.0, vel_avoid, self.ca_threshold2, self.ca_threshold1)

        axial_speed = self.smooth_vel_change(self.nearest_range, m1, c1)
        lateral_speed = self.smooth_vel_change(self.nearest_range, m2, c2)
        if self.nearest_range_index == 0: # back
            new_hover.vx = -axial_speed
            new_hover.vy = lateral_speed
        if self.nearest_range_index == 1: # right
            new_hover.vx = lateral_speed
            new_hover.vy = -axial_speed
        if self.nearest_range_index == 2: # front
            new_hover.vx = axial_speed
            new_hover.vy = lateral_speed
        if self.nearest_range_index == 3: # left
            new_hover.vx = lateral_speed
            new_hover.vy = axial_speed

        response.new_hover = new_hover

        self.get_logger().info("Smooth CA engaged.")
        return response

    def range_inertial(self, rangex, index):
        if index == 0: # back
            range_relative = [0, 0, -rangex]
        elif index == 1: # right
            range_relative = [-rangex, 0, 0]
        elif index == 2: # front
            range_relative = [0, 0, rangex]
        elif index == 3: # left
            range_relative = [rangex, 0, 0]                                   

        range_inertial = [range_relative[i] + self.cf_inertial[i] for i in range(len(self.cf_inertial))]

        return range_inertial
    
    def compare_nearest(self, pos1, pos2):
        pos1_rel = np.array([pos1[i] - self.cf_inertial[i] for i in range(len(self.cf_inertial))])
        pos2_rel = np.array([pos2[i] - self.cf_inertial[i] for i in range(len(self.cf_inertial))])
        pos1_reldist = np.linalg.norm(pos1_rel)
        pos2_reldist = np.linalg.norm(pos2_rel)

        if pos1_reldist < pos2_reldist:
            return pos1, float(pos1_reldist)
        else:
            return pos2, float(pos2_reldist)
        
    def smooth_vel_change(self, dist, m, c):
        v = m * dist + c

        return v
    
    def vel_eq(self, v1, v2, x1, x2):
        m = (v2 - v1) / (x2 - x1)
        c = v1 - m * x1

        return m, c

def main(args=None):
    rclpy.init(args=args)

    detect_avoid = DetectAndAvoid()

    rclpy.spin(detect_avoid)

    detect_avoid.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
