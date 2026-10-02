import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import time
import math

class FleetManager(Node):
    def __init__(self):
        super().__init__('fleet_manager')
        
        self.amrs = ['amr1', 'amr2']
        self.last_seen = {}
        self.last_pose = {}
        self.status = {}
        
        for amr in self.amrs:
            self.last_seen[amr] = time.time()
            self.last_pose[amr] = None
            self.status[amr] = 'ONLINE'
            
            self.create_subscription(PoseStamped, f'/fleet/{amr}/pose', lambda msg, a=amr: self.pose_cb(msg, a), 10)
            self.create_subscription(String, f'/fleet/{amr}/heartbeat', lambda msg, a=amr: self.heartbeat_cb(msg, a), 10)
            
        self.cmd_pubs = {amr: self.create_publisher(PoseStamped, f'/{amr}/goal_pose', 10) for amr in self.amrs}
            
        self.create_timer(1.0, self.monitor_fleet)
        self.get_logger().info("Fleet Manager started.")

    def pose_cb(self, msg, amr):
        self.last_pose[amr] = msg
        self.last_seen[amr] = time.time()
        if self.status[amr] == 'RECOVERING':
            self.status[amr] = 'ONLINE'
            self.get_logger().info(f"{amr} is back ONLINE!")

    def heartbeat_cb(self, msg, amr):
        self.last_seen[amr] = time.time()
        if self.status[amr] in ['OFFLINE', 'RECOVERING']:
            self.status[amr] = 'ONLINE'
            self.get_logger().info(f"{amr} reconnected to Wi-Fi. State: ONLINE")

    def monitor_fleet(self):
        now = time.time()
        for amr in self.amrs:
            if self.last_pose[amr] is not None and self.status[amr] == 'ONLINE' and (now - self.last_seen[amr]) > 3.0:
                self.status[amr] = 'OFFLINE'
                pose = self.last_pose[amr]
                x = pose.pose.position.x if pose else "UNKNOWN"
                y = pose.pose.position.y if pose else "UNKNOWN"
                self.get_logger().warn(f"{amr} went OFFLINE! Last known pos: ({x}, {y})")
                
                # Trigger recovery
                self.dispatch_recovery(amr)
                
    def dispatch_recovery(self, lost_amr):
        lost_pose = self.last_pose[lost_amr]
        if not lost_pose:
            self.get_logger().error(f"Cannot recover {lost_amr}, no known position.")
            return
            
        target_x = lost_pose.pose.position.x
        target_y = lost_pose.pose.position.y
        
        # Find nearest available AMR
        best_amr = None
        best_dist = float('inf')
        
        for amr in self.amrs:
            if amr != lost_amr and self.status[amr] == 'ONLINE' and self.last_pose[amr]:
                p = self.last_pose[amr].pose.position
                dist = math.hypot(target_x - p.x, target_y - p.y)
                if dist < best_dist:
                    best_dist = dist
                    best_amr = amr
                    
        if best_amr:
            self.get_logger().info(f"Dispatching {best_amr} to recover {lost_amr} at ({target_x:.2f}, {target_y:.2f})")
            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.pose = lost_pose.pose
            self.cmd_pubs[best_amr].publish(goal)
            self.status[lost_amr] = 'RECOVERING'
        else:
            self.get_logger().error("No AMRs available for recovery!")

def main(args=None):
    rclpy.init(args=args)
    node = FleetManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
