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
from crazyflie_interfaces.msg import CollDetect, WayPoint, Position, Hover, InFlight
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
        # self.declare_parameter('ca_threshold1', 0.1)
        # self.declare_parameter('ca_threshold2', 0.2)
        # self.declare_parameter('avoidance_vel', 0.2)
        # self.declare_parameter('robot_prefix', '/cf2')

        self.ca_threshold1  = self.get_parameter('ca_threshold1').value
        self.ca_threshold2  = self.get_parameter('ca_threshold2').value
        self.avoidance_vel  = self.get_parameter('avoidance_vel').value
        self.ca_on  = self.get_parameter('ca_on').value
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
        self.waypoint_sub = self.create_subscription(
            WayPoint,
            robot_prefix + '/waypoint',
            self.waypoint_callback,
            10)
        self.inflight_sub = self.create_subscription(
            InFlight,
            robot_prefix + '/inflight',
            self.inflight_callback,
            10)

        # Clients
        
        
        # Services
        self.ca_choice = self.apf_ca
        self.create_service(
            CA,
            robot_prefix + "/ca", 
            partial(self.ca_choice))

        # Initializations
        inf = float("inf")
        self.nearest_inertial = [inf] * 3
        self.nearest_relative = [inf] * 3
        self.nearest_dist = inf
        self.m1, self.c1 = self.vel_eq(self.avoidance_vel, 0.0, self.ca_threshold2, self.ca_threshold1) # Axial Slow down
        self.m2, self.c2 = self.vel_eq(0.0, self.avoidance_vel, self.ca_threshold2, self.ca_threshold1) # Lateral Speed up
        self.coll_check = False

        self.get_logger().info(f"Detect and Avoid set for {robot_prefix}"+
                               f" with ca_th1 = {self.ca_threshold1} m, ca_th2 = {self.ca_threshold2} m")

    def pose_callback(self, msg:PoseStamped):
        self.msg_pose = msg
        self.cf_position = msg.pose.position
        self.cf_inertial = [self.cf_position.x, self.cf_position.y, self.cf_position.z]

    def waypoint_callback(self, msg:WayPoint):
        self.msg_wp = msg
        self.wp = msg.pose
        self.wp_inertial = [self.wp.x, self.wp.y, self.wp.z]
        self.wp_relative_array = self.inertial2relative(self.wp_inertial)
        self.wp_dist = self.dist2pos(self.wp_relative_array)

    def inflight_callback(self, msg:InFlight):
        self.msg_inflight = msg

    def scan_callback(self, msg:LaserScan):
        self.msg_scan = msg
        ranges = list(msg.ranges)
        
        if self.msg_inflight.inflight and self.ca_on:
            nearest_range = min(ranges)
            self.nearest_range = max(nearest_range, self.msg_scan.range_min) # Avoid zero division error
            self.nearest_range_index = ranges.index(nearest_range)
            self.nearest_range_relative = self.range_relative(self.nearest_range, self.nearest_range_index)
            self.nearest_range_inertial = self.relative2inertial(self.nearest_range_relative)
                
            self.nearest_relative = self.inertial2relative(self.nearest_inertial)
            self.nearest_dist = self.dist2pos(self.nearest_relative) # Consider previous nearest
            self.nearest_dist = self.nearest_range # Only consider sensed nearest
            if self.nearest_range <= self.nearest_dist:
                # self.get_logger().warn(f"Nearest range: [{self.nearest_range}, {self.nearest_range_index}].")
                self.nearest_relative = self.nearest_range_relative
                self.nearest_inertial = self.nearest_range_inertial
                self.nearest_dist = self.nearest_range

            self.coll_check = self.nearest_dist <= self.ca_threshold2 # Collision check via nearest obs pt.

            msg_collision = CollDetect()
            msg_collision.nearest = self.nearest_relative
            if self.coll_check:
                self.get_logger().warn(f"Nearest collision at {self.nearest_dist}.")
                self.get_logger().warn(f"Obs inertially at: " +
                                       f"obs_x = {self.nearest_inertial[0]}, " +
                                       f"obs_y = {self.nearest_inertial[1]}, ")
                self.get_logger().info(f"CF position at: "+
                                       f"cf_x = {self.cf_inertial[0]}, " + 
                                       f"cf_y = {self.cf_inertial[1]}, " +
                                       f"cf_z = {self.cf_inertial[2]}")
                msg_collision.collision = True
            else:
                msg_collision.collision = False
            self.publisher_collision.publish(msg_collision)

    def repel_ca(self, request, response):
        old_hover = request.old_hover
        new_hover = old_hover
        new_hover = self.set_hover_z(new_hover)
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
        new_hover = self.set_hover_z(new_hover)
        new_hover.vx = 0.0
        new_hover.vy = 0.0
        new_hover.yaw_rate = 0.0

        x = self.nearest_relative[0] # x axis, - is back, + is front
        y = self.nearest_relative[1] # y axis, - is right, + is left
        new_hover.vx = - self.avoidance_vel * float(np.sign(x))
        new_hover.vy = - self.avoidance_vel * float(np.sign(y))

        response.new_hover = new_hover

        self.get_logger().info("Repel CA2 engaged.")
        return response

    def smooth_ca(self, request, response): # Best performance till now with ca1 = 0.2 and ca2 = 0.4
        old_hover = request.old_hover
        new_hover = old_hover
        new_hover = self.set_hover_z(new_hover)

        axial_speed = self.smooth_vel_change(self.nearest_dist, self.m1, self.c1)
        lateral_speed = self.smooth_vel_change(self.nearest_dist, self.m2, self.c2)
        if self.nearest_range_index == 0: # back
            new_hover.vx = -axial_speed
            new_hover.vy = -lateral_speed
        if self.nearest_range_index == 1: # right
            new_hover.vx = lateral_speed
            new_hover.vy = -axial_speed
        if self.nearest_range_index == 2: # front
            new_hover.vx = axial_speed
            new_hover.vy = lateral_speed
        if self.nearest_range_index == 3: # left
            new_hover.vx = -lateral_speed
            new_hover.vy = axial_speed

        response.new_hover = new_hover

        self.get_logger().info("Smooth CA engaged.")
        return response

    def smooth_ca2(self, request, response):
        old_hover = request.old_hover
        new_hover = old_hover
        new_hover = self.set_hover_z(new_hover)

        axial_speed = self.smooth_vel_change(self.nearest_dist, self.m1, self.c1)
        lateral_speed = self.smooth_vel_change(self.nearest_dist, self.m2, self.c2)
        x = self.nearest_relative[0] # x axis, - is back, + is front
        y = self.nearest_relative[1] # y axis, - is right, + is left
        theta = math.atan2(y, x)

        # Method 4 - makes circles around the obstacle, performs well.
        new_hover.vx = self.avoidance_vel * math.cos(theta - math.pi/2)
        new_hover.vy = self.avoidance_vel * math.sin(theta - math.pi/2)

        response.new_hover = new_hover

        self.get_logger().info("Smooth CA2 engaged.")
        return response
    
    def apf_ca(self, request, response): # best settings in comments, worked well when nearest not stored
        old_hover = request.old_hover
        new_hover = old_hover
        new_hover = self.set_hover_z(new_hover)
        x = self.nearest_relative[0] # x axis, - is back, + is front
        y = self.nearest_relative[1] # y axis, - is right, + is left
        rho_h = math.atan2(abs(y), x)

        ka = 0.3 # 0.3
        kr = 0.01 # 0.01
        ng = 2
        n = 2
        do = self.ca_threshold2 # with ca1 = 0.2 and ca2 = 0.4
        dr = max(min(self.nearest_dist, do), self.ca_threshold1)
        da = self.wp_dist
        unit_direction_nearest = np.array(self.nearest_relative) / dr
        unit_direction_wp = np.array(self.wp_relative_array) / da
        
        gamma = math.pi / 4
        alpha = 1

        R_1 = np.array([[np.cos(gamma),-np.sin(gamma), 0],
                        [np.sin(gamma), np.cos(gamma), 0],
                        [0            , 0            , 1]])
        
        R_2 = np.array([[ np.cos(gamma), 0, np.sin(gamma)],
                        [ 0            , 1,             0],
                        [-np.sin(gamma), 0, np.cos(gamma)]])

        if rho_h >= 0:
            R_h = R_1
            R_v = np.transpose(R_2)
        else:
            R_h = np.transpose(R_1)
            R_v = R_2

        p_h = np.matmul(np.array(self.nearest_relative), R_h)
        p_v = np.matmul(np.array(self.nearest_relative), R_v)
        p = np.array([alpha * p_h[0], 
                      alpha * p_h[1], 
                      (1-alpha) * p_v[2]])
        unit_p = p / max(min(self.dist2pos(p), self.nearest_dist), -self.nearest_dist)
        
        fa = ka * da * unit_direction_wp
        fr_o = -(1/2) * kr * (da**ng / dr**2) * n * (((1/dr) - (1/do))**(n-1)) * unit_p
        fr_g = -(1/2) * kr * ng * da**(ng-1)  * (((1/dr) - (1/do))**n) * unit_direction_wp
        fr = fr_o + fr_g
        fc = fa + fr

        new_hover.vx = max(min(fc[0], self.avoidance_vel), -self.avoidance_vel)
        new_hover.vy = max(min(fc[1], self.avoidance_vel), -self.avoidance_vel)

        response.new_hover = new_hover

        self.get_logger().info(f"fc_x = {fc[0]}, fc_y = {fc[1]}")
        self.get_logger().info("APF CA engaged.")
        return response
    
    def set_hover_z(self, msg_hover: Hover):
        try:
            msg_hover.z_distance = self.msg_wp.pose.z
        except Exception as e:
            msg_hover.z_distance = self.cf_position.z

        return msg_hover

    def range_relative(self, rangex, index):
        if index == 0: # back
            range_relative = [-rangex, 0.0, 0.0]
        elif index == 1: # right
            range_relative = [0.0, -rangex, 0.0]
        elif index == 2: # front
            range_relative = [rangex, 0.0, 0.0]
        elif index == 3: # left
            range_relative = [0.0, rangex, 0.0]

        return range_relative
    
    def compare_nearest(self, pos1, pos2): # Overwrites the nearest obs in memory
        pos1_rel_array = self.inertial2relative(pos1)
        pos2_rel_array = self.inertial2relative(pos2)
        pos1_reldist = self.dist2pos(pos1_rel_array)
        pos2_reldist = self.dist2pos(pos2_rel_array)

        if pos1_reldist < pos2_reldist:
            return pos1, pos1_rel_array, pos1_reldist
        else:
            return pos2, pos2_rel_array, pos1_reldist

    def dist2pos(self, pos):
        pos_reldist = np.linalg.norm(pos)

        return float(pos_reldist)
    
    def smooth_vel_change(self, dist, m, c):
        v = m * dist + c

        return v
    
    def vel_eq(self, v1, v2, x1, x2):
        m = (v2 - v1) / (x2 - x1)
        c = v1 - m * x1

        return m, c
    
    def inertial2relative(self, pos): # assuming yaw is at zero
        pos_rel_array = np.array([pos[i] - self.cf_inertial[i] for i in range(len(self.cf_inertial))])
        pos_rel = Position()
        pos_rel.x = pos_rel_array[0]
        pos_rel.y = pos_rel_array[1]
        pos_rel.z = pos_rel_array[2]
        pos_rel.yaw = math.atan2(pos_rel.y, pos_rel.z)

        return list(pos_rel_array)

    def relative2inertial(self, pos_rel): # assuming yaw is at zero
        pos_inertial_array = np.array([pos_rel[i] + self.cf_inertial[i] for i in range(len(self.cf_inertial))])
        pos_inertial = Position()
        pos_inertial.x = pos_inertial_array[0]
        pos_inertial.y = pos_inertial_array[1]
        pos_inertial.z = pos_inertial_array[2]
        pos_inertial.yaw = math.atan2(pos_inertial.y, pos_inertial.z)

        return list(pos_inertial_array)

def main(args=None):
    rclpy.init(args=args)

    detect_avoid = DetectAndAvoid()

    rclpy.spin(detect_avoid)

    detect_avoid.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
