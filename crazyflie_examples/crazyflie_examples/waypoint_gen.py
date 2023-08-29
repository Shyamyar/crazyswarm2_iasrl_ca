#!/usr/bin/env python3

"""
A waypoint generating node that also calculates relative position of waypoint
    w.r.t crazyflie and provides relative distance information as well. Publishes
    message with these details and if waypoint is reached, and goal is reached.

    2023 = Shyam Rauniyar (IASRL)
"""
import rclpy
from rclpy.node import Node

from crazyflie_interfaces.srv import InitiateWayPoint, ChangeWayPoint
from crazyflie_interfaces.msg import WayPoint
from geometry_msgs.msg import PoseStamped
from functools import partial
import tf_transformations
import numpy as np

class WaypointGenerator(Node):
    def __init__(self):
        super().__init__('waypoint_gen',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,)
        self.declare_parameter('robot_prefix', '/cf2')

        robot_prefix  = self.get_parameter('robot_prefix').value

        # Publishers
        self.publisher_waypoint = self.create_publisher(
            WayPoint, 
            robot_prefix + '/waypoint', 
            10)

        # Subscriptions
        self.pose_sub = self.create_subscription(
            PoseStamped,
            robot_prefix + '/pose',
            self.pose_callback,
            10)

        # Clients


        # Services
        self.create_service(
            InitiateWayPoint,
            robot_prefix + "/init_wp", 
            partial(self.initiate_waypoint_callback))
        self.create_service(
            ChangeWayPoint,
            robot_prefix + "/change_wp", 
            partial(self.change_waypoint_callback))
        

        # Initializations
        self.waypoints = []
        self.waypoints = [[0.5,-0.4, 0.7, 0.1],
                          [0.0, 0.0, 0.3, 0.0]]       
        self.num_wp = len(self.waypoints)
        self.waypoint_id = 0
        self.wp_reached = False
        self.goal_reached = False
        self.reldist = float("inf")
        self.reldist_th = 0.15

        # Service Call Check


        self.get_logger().info(f"Waypoint Generator set for {robot_prefix}")

    def pose_callback(self, msg:PoseStamped):
        self.msg_pose = msg
        pos_all = msg.pose.position
        cf_x = pos_all.x
        cf_y = pos_all.y
        cf_z = pos_all.z
        q_all = msg.pose.orientation
        cf_q = [q_all.w, q_all.x, q_all.y, q_all.z]
        [cf_roll, cf_pitch, cf_yaw] = tf_transformations.euler_from_quaternion(cf_q)

        self.msg_waypoint = WayPoint()
        if self.num_wp > 0:
            wp_x = self.waypoints[self.waypoint_id][0]
            wp_y = self.waypoints[self.waypoint_id][1]
            wp_z = self.waypoints[self.waypoint_id][2]
            wp_yaw = self.waypoints[self.waypoint_id][3]

            relative_pose = np.array([wp_x - cf_x,
                                    wp_y - cf_y,
                                    wp_z - cf_z,
                                    wp_yaw - cf_yaw])
            
            self.reldist = float(np.linalg.norm(relative_pose))

            self.msg_waypoint.num = self.num_wp
            self.msg_waypoint.wp_id = self.waypoint_id
            self.msg_waypoint.pose.x = wp_x
            self.msg_waypoint.pose.y = wp_y
            self.msg_waypoint.pose.z = wp_z
            self.msg_waypoint.pose.yaw = wp_yaw
            self.msg_waypoint.relpose.x = relative_pose[0]
            self.msg_waypoint.relpose.y = relative_pose[1]
            self.msg_waypoint.relpose.z = relative_pose[2]
            self.msg_waypoint.relpose.yaw = relative_pose[3]
            self.msg_waypoint.reldist = self.reldist

        self.wp_reached = self.reldist <= self.reldist_th
        self.publish_waypoint_msg()

    def change_waypoint_callback(self, request, response):
        self.get_logger().info("Waypoint Reached. Requesting next waypoint...")
        self.waypoint_id = self.waypoint_id + 1
        self.goal_reached = self.waypoint_id == self.num_wp and self.wp_reached
        self.waypoint_id = min(self.waypoint_id, self.num_wp - 1)

        if self.goal_reached:
            self.get_logger().info("Goal Reached.")

        self.publish_waypoint_msg()
        return response

    def initiate_waypoint_callback(self, request, response):
        self.get_logger().info("Initiating Waypoints with waypoint " + 
                               f"{request.wp_id}")
        self.waypoint_id = request.wp_id
        self.wp_reached = False
        self.goal_reached = False

        self.publish_waypoint_msg()
        return response
    
    def publish_waypoint_msg(self):
        self.msg_waypoint.wp_reached = self.wp_reached
        self.msg_waypoint.goal_reached = self.goal_reached
        self.publisher_waypoint.publish(self.msg_waypoint)

    
def main(args=None):
    rclpy.init(args=args)

    waypoint_gen = WaypointGenerator()

    rclpy.spin(waypoint_gen)

    waypoint_gen.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
