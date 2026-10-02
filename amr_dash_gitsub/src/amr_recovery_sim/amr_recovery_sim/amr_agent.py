import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
import math

class AMRAgent(Node):
    def __init__(self):
        super().__init__('amr_agent')
        self.declare_parameter('amr_name', 'amr1')
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)
        
        self.amr_name = self.get_parameter('amr_name').value
        
        self.x = self.get_parameter('start_x').value
        self.y = self.get_parameter('start_y').value
        self.theta = 0.0
        
        self.v = 0.0
        self.w = 0.0
        
        self.pose_pub = self.create_publisher(PoseStamped, f'/{self.amr_name}/local_pose', 10)
        self.heartbeat_pub = self.create_publisher(String, f'/{self.amr_name}/local_heartbeat', 10)
        
        self.create_subscription(PoseStamped, f'/{self.amr_name}/goal_pose', self.goal_cb, 10)
        self.create_subscription(Twist, f'/{self.amr_name}/cmd_vel', self.cmd_cb, 10)
        self.create_subscription(String, '/network/lora_status', self.lora_cb, 10)
        
        self.goal = None
        self.lora_active_with = []
        self.lora_goal_pub = self.create_publisher(PoseStamped, '/lora/amr1/goal_pose', 10)
        
        self.timer = self.create_timer(0.1, self.update_kinematics)
        self.heartbeat_timer = self.create_timer(1.0, self.send_heartbeat)
        
        self.get_logger().info(f"AMR Agent {self.amr_name} started at ({self.x}, {self.y})")

    def goal_cb(self, msg):
        self.goal = msg.pose.position
        self.get_logger().info(f"{self.amr_name} received goal: ({self.goal.x}, {self.goal.y})")

    def cmd_cb(self, msg):
        self.v = msg.linear.x
        self.w = msg.angular.z

    def lora_cb(self, msg):
        links = msg.data.split(',')
        self.lora_active_with = [link for link in links if self.amr_name in link]
        if self.lora_active_with:
            self.get_logger().info(f"{self.amr_name} has LoRa connection: {self.lora_active_with}")

    def update_kinematics(self):
        # Simple navigation to goal if set
        if self.goal:
            dx = self.goal.x - self.x
            dy = self.goal.y - self.y
            dist = math.hypot(dx, dy)
            if dist > 0.5:
                self.v = min(0.5, dist)
                target_theta = math.atan2(dy, dx)
                # simple proportional heading control
                err = target_theta - self.theta
                # wrap to pi
                err = (err + math.pi) % (2 * math.pi) - math.pi
                self.w = 0.5 * err
            else:
                self.v = 0.0
                self.w = 0.0
                self.goal = None
                self.get_logger().info(f"{self.amr_name} reached goal.")
                
                # Check if we need to rescue someone via LoRa
                if 'amr1' in ''.join(self.lora_active_with) and self.amr_name != 'amr1':
                    self.get_logger().warn(f"{self.amr_name} found AMR-1! Sending LoRa recovery waypoint...")
                    rec_goal = PoseStamped()
                    rec_goal.header.frame_id = 'map'
                    rec_goal.pose.position.x = 0.0 # Back to base/Wi-Fi zone
                    rec_goal.pose.position.y = 0.0
                    self.lora_goal_pub.publish(rec_goal)
                
        # Update pose
        dt = 0.1
        self.theta += self.w * dt
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = self.x
        msg.pose.position.y = self.y
        msg.pose.orientation.z = math.sin(self.theta / 2.0)
        msg.pose.orientation.w = math.cos(self.theta / 2.0)
        self.pose_pub.publish(msg)

    def send_heartbeat(self):
        msg = String()
        msg.data = f"{self.amr_name} alive"
        self.heartbeat_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AMRAgent()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
