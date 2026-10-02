import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import math

class NetworkSimulator(Node):
    def __init__(self):
        super().__init__('network_sim')
        self.declare_parameter('wifi_boundary_x', 10.0)
        self.declare_parameter('lora_range_m', 5.0)
        
        self.wifi_boundary = self.get_parameter('wifi_boundary_x').value
        self.lora_range = self.get_parameter('lora_range_m').value
        
        self.amrs = ['amr1', 'amr2']
        self.poses = {}
        
        # Setup bridges
        self.pubs_pose = {}
        self.pubs_heartbeat = {}
        
        for amr in self.amrs:
            self.poses[amr] = None
            # Subscribe to local (raw) topics
            self.create_subscription(PoseStamped, f'/{amr}/local_pose', lambda msg, a=amr: self.pose_cb(msg, a), 10)
            self.create_subscription(String, f'/{amr}/local_heartbeat', lambda msg, a=amr: self.heartbeat_cb(msg, a), 10)
            
            # Publishers to fleet manager (bridged)
            self.pubs_pose[amr] = self.create_publisher(PoseStamped, f'/fleet/{amr}/pose', 10)
            self.pubs_heartbeat[amr] = self.create_publisher(String, f'/fleet/{amr}/heartbeat', 10)
            
        # LoRa bridge publisher
        self.lora_pub = self.create_publisher(String, '/network/lora_status', 10)
        self.create_timer(1.0, self.check_lora)
        
        self.get_logger().info(f"Network Simulator started. Wi-Fi boundary X < {self.wifi_boundary}m")

    def pose_cb(self, msg, amr):
        self.poses[amr] = msg
        # If in Wi-Fi, bridge the pose
        if msg.pose.position.x < self.wifi_boundary:
            self.pubs_pose[amr].publish(msg)

    def heartbeat_cb(self, msg, amr):
        pose = self.poses.get(amr)
        if pose and pose.pose.position.x < self.wifi_boundary:
            self.pubs_heartbeat[amr].publish(msg)

    def check_lora(self):
        # Broadcast which pairs are in LoRa range
        if self.poses['amr1'] is None:
            return
            
        p1 = self.poses['amr1'].pose.position
        
        active_links = []
        for amr in ['amr2']:
            if self.poses.get(amr) is None:
                continue
            p2 = self.poses[amr].pose.position
            dist = math.hypot(p1.x - p2.x, p1.y - p2.y)
            if dist <= self.lora_range:
                active_links.append(f"amr1-{amr}")
                
        if active_links:
            msg = String()
            msg.data = ",".join(active_links)
            self.lora_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = NetworkSimulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
