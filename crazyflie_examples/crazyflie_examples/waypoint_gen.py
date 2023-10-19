#!/usr/bin/env python3

"""
A waypoint generating node that also calculates relative position of waypoint
    w.r.t crazyflie and provides relative distance information as well. Publishes
    message with these details and if waypoint is reached, and goal is reached.

    2023 = Shyam Rauniyar (IASRL)
"""
import rclpy
from rclpy.node import Node

from crazyflie_interfaces.srv import ChangeWayPoint
from crazyflie_interfaces.msg import WayPoint
from geometry_msgs.msg import PoseStamped
from functools import partial
import tf_transformations
import numpy as np

class WaypointGenerator(Node,):
    def __init__(self):
        super().__init__('waypoint_gen',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,)
        # self.declare_parameter('robot_prefix', '/cf2')
        # self.declare_parameter('waypoints', False)

        # Turn ROS parameters into a dictionary
        self._ros_parameters = self._param_to_dict(self._parameters)
        robot_prefix = self._ros_parameters["robot_prefix"]
        waypoints = self._ros_parameters["waypoints"]

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
        # self.create_service(
        #     ChangeWayPoint,
        #     robot_prefix + "/init_wp", 
        #     partial(self.initiate_waypoint_callback))
        self.create_service(
            ChangeWayPoint,
            robot_prefix + "/change_wp", 
            partial(self.change_waypoint_callback))
        

        # Initializations
        self.waypoints = list()
        if waypoints:
            for key in waypoints:
                wp = waypoints[key]
                self.waypoints.append(wp)

        self.num_wp = len(self.waypoints)
        self.wp_initiated = False
        self.reldist_th = 0.15

        # Service Call Check


        self.get_logger().info(f"Waypoint Generator set for {robot_prefix} with {self.num_wp} waypoints")

    def pose_callback(self, msg:PoseStamped):
        self.msg_pose = msg
        pos_all = msg.pose.position
        q_all = msg.pose.orientation
        cf_q = [q_all.w, q_all.x, q_all.y, q_all.z]
        cf_eul = tf_transformations.euler_from_quaternion(cf_q)
        self.cf_x = pos_all.x
        self.cf_y = pos_all.y
        self.cf_z = pos_all.z
        self.cf_yaw = cf_eul[2]
        self.publish_waypoint_msg()

    def change_waypoint_callback(self, request, response):
        self.waypoint_id = request.wp_id
        self.get_logger().info(f"Requesting waypoint {self.waypoint_id}...")
        if self.num_wp > 0:
            self.wp_initiated = True
            if self.waypoints[self.waypoint_id][4] == 1:
                self.wp_x = self.cf_x + self.waypoints[self.waypoint_id][0]
                self.wp_y = self.cf_y + self.waypoints[self.waypoint_id][1]
                self.wp_z = self.cf_z + self.waypoints[self.waypoint_id][2]
                self.wp_yaw = self.cf_yaw + self.waypoints[self.waypoint_id][3]
            else:
                self.wp_x = self.waypoints[self.waypoint_id][0]
                self.wp_y = self.waypoints[self.waypoint_id][1]
                self.wp_z = self.waypoints[self.waypoint_id][2]
                self.wp_yaw = self.waypoints[self.waypoint_id][3]
        else:
            self.get_logger().info("Waypoints unavailable...")

        return response

    # def initiate_waypoint_callback(self, request, response):
    #     self.get_logger().info("Initiating Waypoints with waypoint " + 
    #                            f"{request.wp_id}")
    #     self.waypoint_id = request.wp_id
    #     self.wp_reached = False
    #     self.goal_reached = False

    #     self.publish_waypoint_msg()
    #     return response
    
    def publish_waypoint_msg(self):
        self.msg_waypoint = WayPoint()
        if self.wp_initiated:
            self.relative_pose = np.array([self.wp_x - self.cf_x,
                                            self.wp_y - self.cf_y,
                                            self.wp_z - self.cf_z,
                                            self.wp_yaw - self.cf_yaw])
            self.reldist = float(np.linalg.norm(self.relative_pose))
            self.wp_reached = self.reldist <= self.reldist_th    
            self.goal_reached = self.waypoint_id == self.num_wp - 1 and self.wp_reached
            self.waypoint_id = max(min(self.waypoint_id, self.num_wp - 1),0)

            if self.goal_reached:
                self.get_logger().info("Goal Reached.")
            elif self.wp_reached:
                self.get_logger().info("Waypoint Reached.")

            self.msg_waypoint.num = self.num_wp
            self.msg_waypoint.wp_id = self.waypoint_id
            self.msg_waypoint.pose.x = self.wp_x
            self.msg_waypoint.pose.y = self.wp_y
            self.msg_waypoint.pose.z = self.wp_z
            self.msg_waypoint.pose.yaw = self.wp_yaw
            self.msg_waypoint.relpose.x = self.relative_pose[0]
            self.msg_waypoint.relpose.y = self.relative_pose[1]
            self.msg_waypoint.relpose.z = self.relative_pose[2]
            self.msg_waypoint.relpose.yaw = self.relative_pose[3]
            self.msg_waypoint.reldist = self.reldist
            self.msg_waypoint.wp_reached = self.wp_reached
            self.msg_waypoint.goal_reached = self.goal_reached
            
        self.publisher_waypoint.publish(self.msg_waypoint)

    def _param_to_dict(self, param_ros):
        """
        Turn ROS 2 parameters from the node into a dict
        """
        tree = {}
        for item in param_ros:
            t = tree
            for part in item.split('.'):
                if part == item.split('.')[-1]:
                    t = t.setdefault(part, param_ros[item].value)
                else:
                    t = t.setdefault(part, {})
        return tree

    
def main(args=None):
    rclpy.init(args=args)

    waypoint_gen = WaypointGenerator()

    rclpy.spin(waypoint_gen)

    waypoint_gen.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
