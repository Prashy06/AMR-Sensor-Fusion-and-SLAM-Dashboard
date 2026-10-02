import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String

class HITLBridge(Node):
    def __init__(self):
        super().__init__('hitl_bridge')
        
        # Real physical robot subscriptions
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        
        # Virtual local topics for the simulator to pick up
        self.pose_pub = self.create_publisher(PoseStamped, '/amr1/local_pose', 10)
        self.heartbeat_pub = self.create_publisher(String, '/amr1/local_heartbeat', 10)
        
        # LoRa recovery bridge
        self.lora_status = []
        self.create_subscription(String, '/network/lora_status', self.lora_cb, 10)
        
        # Commands received via LoRa from the rescue AMR (AMR-2)
        self.create_subscription(PoseStamped, '/lora/amr1/goal_pose', self.lora_goal_cb, 10)
        
        # Publisher to physical Nav2
        self.physical_goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        self.create_timer(1.0, self.send_heartbeat)
        
        self.get_logger().info("HITL Bridge active. Translating physical /odom to virtual AMR-1.")

    def odom_cb(self, msg):
        # Convert physical Odometry to virtual PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.header.frame_id = 'map'
        pose_msg.pose = msg.pose.pose
        self.pose_pub.publish(pose_msg)

    def send_heartbeat(self):
        msg = String()
        msg.data = "amr1 alive (physical)"
        self.heartbeat_pub.publish(msg)

    def lora_cb(self, msg):
        self.lora_status = msg.data.split(',')
        if any('amr1' in link for link in self.lora_status):
            pass # We have a LoRa link!

    def lora_goal_cb(self, msg):
        # If we have a LoRa connection to AMR-2, accept the goal and forward to physical Nav2
        if 'amr1-amr2' in self.lora_status or 'amr2-amr1' in self.lora_status:
            self.get_logger().warn(f"Physical AMR-1 received RECOVERY GOAL via LoRa! Forwarding to Nav2: X={msg.pose.position.x}")
            self.physical_goal_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = HITLBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
