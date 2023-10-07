#!/usr/bin/env python3

"""
A Twist message handler that get incoming twist messages from 
    external packages and handles proper takeoff, landing and
    hover commands of connected crazyflie in the crazyflie_server
    node

    2022 - K. N. McGuire (Bitcraze AB)
    2023 = Shyam Rauniyar (IASRL)
"""
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from crazyflie_interfaces.srv import Takeoff, Land, GoTo, NotifySetpointsStop, CA, InitiateWayPoint, ChangeWayPoint
from crazyflie_interfaces.msg import Hover, CollDetect, WayPoint, InFlight
from geometry_msgs.msg import PoseStamped
from functools import partial
import time
import numpy as np
import random as rd

class GoalCommander(Node):
    def __init__(self):
        super().__init__('goal_commander',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,)
        # self.declare_parameter('hover_height', 0.5)
        # self.declare_parameter('ca_on', True)
        # self.declare_parameter('incoming_twist_topic', '/cmd_vel')
        # self.declare_parameter('robot_prefix', '/cf2')

        self.hover_height  = self.get_parameter('hover_height').value
        self.ca_on  = self.get_parameter('ca_on').value
        incoming_twist_topic  = self.get_parameter('incoming_twist_topic').value
        robot_prefix  = self.get_parameter('robot_prefix').value
        
        # Publishers
        self.publisher_hover = self.create_publisher(
            Hover, 
            robot_prefix + '/cmd_hover', 
            10)
        self.publisher_inflight = self.create_publisher(
            InFlight, 
            robot_prefix + '/inflight', 
            10)

        # Subscriptions
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            incoming_twist_topic,
            self.cmd_vel_callback,
            10)
        self.pose_sub = self.create_subscription(
            PoseStamped,
            robot_prefix + '/pose',
            self.pose_callback,
            10)
        self.coll_detect_sub = self.create_subscription(
            CollDetect,
            robot_prefix + '/coll_detect',
            self.coll_detect_callback,
            10)
        self.waypoints_sub = self.create_subscription(
            WayPoint,
            robot_prefix + '/waypoint',
            self.waypoint_callback,
            10)

        # Clients
        self.takeoff_client = self.create_client(
            Takeoff, 
            robot_prefix + '/takeoff')
        self.land_client = self.create_client(
            Land, 
            robot_prefix + '/land')
        self.goto_client = self.create_client(
            GoTo, 
            robot_prefix + '/go_to')
        self.notifysps_client = self.create_client(
            NotifySetpointsStop, 
            robot_prefix + '/notify_setpoints_stop')
        self.ca_client = self.create_client(
            CA, 
            robot_prefix + '/ca')
        self.init_wp_client = self.create_client(
            InitiateWayPoint, 
            robot_prefix + '/init_wp')
        self.change_wp_client = self.create_client(
            ChangeWayPoint, 
            robot_prefix + '/change_wp')

        # Services


        # Initializations
        self.msg_hover = Hover()
        self.inflight = False
        self.cf_has_taken_off = False
        self.command_hover = False
        self.command_takeoff = False
        self.command_goto = False
        self.command_land = False
        self.in_goto_mode = False
        self.VELOCITY = 0.18 # m/s
        self.TAKEOFFVELOCITY = 0.5 # m/s
        self.LANDVELOCITY = 0.02 # m/s
        self.land_z = 0.05   # m
        self.z_limit = 2.0   # m

        self.msg_collision = CollDetect()
        self.msg_collision.collision = False

        timer_period = 0.1 # Same as for /scan
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Service Call Check
        while not self.takeoff_client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for Takeoff Service...")
        while not self.land_client.wait_for_service():
            self.get_logger().warn("Waiting for Land Service...")
        while not self.goto_client.wait_for_service():
            self.get_logger().warn("Waiting for GoTo Service...")
        while not self.notifysps_client.wait_for_service():
            self.get_logger().warn("Waiting for NotifySetpointsStop Service...")
        while not self.init_wp_client.wait_for_service():
            self.get_logger().warn("Waiting for Initiate Waypoint Service...")

        self.get_logger().info(f"Goal Commander set for {robot_prefix}")

    def pose_callback(self, msg:PoseStamped):
        self.msg_pose = msg
        self.z_position = msg.pose.position.z
        msg_inflight = InFlight()
        msg_inflight.inflight = self.inflight
        self.publisher_inflight.publish(msg_inflight)

    def coll_detect_callback(self, msg:CollDetect):
        self.msg_collision = msg

    def waypoint_callback(self, msg:WayPoint):
        self.msg_waypoint = msg

    def cmd_vel_callback(self, msg:Twist):
        msg_is_zero = msg.linear.x == 0.0 and msg.linear.y == 0.0 and msg.angular.z == 0.0 and msg.linear.z == 0.0 # key pressed: k or non-assigned keys
          
        self.command_takeoff = msg.linear.z > 0.0 and not self.cf_has_taken_off # key pressed: t
        self.command_hover = not msg_is_zero and self.cf_has_taken_off # key pressed: any assigned-keys except k  
        self.command_goup = self.command_hover and msg.linear.z > 0.0 # key pressed: t
        self.command_godown = self.command_hover and msg.linear.z < 0.0 # key pressed: b
        self.command_land =  self.command_godown and self.z_position <= self.land_z # key pressed: b       
        self.command_stopandhover = msg_is_zero and self.cf_has_taken_off # key pressed: k or non-assigned keys

        self.msg_hover.vx = msg.linear.x
        self.msg_hover.vy = msg.linear.y
        self.msg_hover.yaw_rate = msg.angular.z
        
        if self.in_goto_mode:
            self.msg_hover.z_distance = self.z_position

    def timer_callback(self):
        if self.command_takeoff and not self.cf_has_taken_off:
            req = Takeoff.Request()
            req.height = self.hover_height
            takeoff_duration = self.hover_height / self.TAKEOFFVELOCITY
            req.duration = rclpy.duration.Duration(seconds=takeoff_duration).to_msg()
            self.get_logger().info("Requesting Takeoff...")
            self.takeoff_client.call_async(req)
            self.command_takeoff = False
            self.cf_has_taken_off = True
            self.command_stopandhover = False
            self.msg_hover.z_distance = self.hover_height
            self.inflight = True
            time.sleep(2.0)

            if self.msg_waypoint.num > 0:
                req = InitiateWayPoint.Request()
                req.wp_id = 0
                self.init_wp_client.call_async(req)
                self.command_goto = True
                self.first_waypoint = True
            else:
                self.command_goto = False
                self.first_waypoint = False

        if self.cf_has_taken_off:
            # random waypoint check
            # self.msg_waypoint.wp_reached = rd.choice([True] + [False] * 20)
            # if self.msg_waypoint.wp_reached:
            #     self.msg_waypoint.goal_reached = rd.choice([True] + [False] * 2)
            #     if self.msg_waypoint.goal_reached:
            #         self.get_logger().info("Goal Reached.")

            if self.msg_collision.collision and self.ca_on:
                self.in_goto_mode = False
                req = CA.Request()
                req.old_hover = self.msg_hover
                self.get_logger().info("Requesting Collision Avoidance...")
                future = self.ca_client.call_async(req)
                future.add_done_callback(partial(self.ca_client_callback))
                self.command_goto = False
                self.command_stopandhover = True
                self.msg_collision.collision = False

            elif self.command_hover:
                self.in_goto_mode = False
                if self.command_goup:
                    self.msg_hover.z_distance += 0.1
                elif self.command_godown:
                    self.msg_hover.z_distance -= 0.05
                self.msg_hover.z_distance = min(max(self.msg_hover.z_distance, self.land_z), self.z_limit)
                self.get_logger().info("Publishing Hover...")
                self.publisher_hover.publish(self.msg_hover)
                self.command_hover = False
                self.command_goto = False
                self.command_stopandhover = True

            elif self.command_stopandhover:
                self.in_goto_mode = False
                self.msg_hover = self.set_zero_hover(self.msg_hover)
                self.get_logger().info("Publishing Zero Hover...")
                self.publisher_hover.publish(self.msg_hover)
                if self.msg_waypoint.num > 0:
                    req = NotifySetpointsStop.Request()
                    req.remain_valid_millisecs = 100
                    self.get_logger().info("Requesting Notify Setpoints Stop...")
                    self.notifysps_client.call_async(req)
                    self.command_goto = True
            
            if self.command_goto or (self.msg_waypoint.wp_reached and not self.msg_waypoint.goal_reached):
                if self.msg_waypoint.wp_reached:
                    req = ChangeWayPoint.Request()
                    req.wp_id = self.msg_waypoint.wp_id
                    self.change_wp_client.call_async(req)
                    self.command_goto = True
                    self.first_waypoint = False
                else:
                    req = GoTo.Request()
                    req.goal.x = self.msg_waypoint.pose.x
                    req.goal.y = self.msg_waypoint.pose.y
                    req.goal.z = self.msg_waypoint.pose.z
                    req.yaw = self.msg_waypoint.pose.yaw
                    goto_duration = self.msg_waypoint.reldist / self.VELOCITY
                    req.duration = rclpy.duration.Duration(seconds=goto_duration).to_msg()
                    self.get_logger().info("Requesting GoTo...")
                    self.goto_client.call_async(req)
                    self.command_goto = False
                    self.command_hover = False
                    self.command_stopandhover = False
                    self.in_goto_mode = True
            
            if self.command_land or (self.msg_waypoint.goal_reached and not self.first_waypoint):
                self.in_goto_mode = False
                land_height = self.land_z
                land_duration = land_height / self.LANDVELOCITY
                req = Land.Request()
                req.height = land_height
                req.duration = rclpy.duration.Duration(seconds=land_duration).to_msg()
                self.get_logger().info("Requesting Land...")
                self.land_client.call_async(req)
                self.cf_has_taken_off = False
                self.inflight = False

    def set_zero_hover(self, msg:Hover):
        msg.vx = 0.0
        msg.vy = 0.0
        msg.yaw_rate = 0.0

        return msg

    def ca_client_callback(self, future):
        try:
            response = future.result()
            self.msg_hover = response.new_hover
            # self.msg_hover.z_distance = self.z_position
            if not self.command_land:
                self.get_logger().info("Publishing CA Hover...")
                self.publisher_hover.publish(self.msg_hover)

        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))
        
        return self.msg_hover

def main(args=None):
    rclpy.init(args=args)
    
    goal_commander = GoalCommander()

    rclpy.spin(goal_commander)

    goal_commander.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
