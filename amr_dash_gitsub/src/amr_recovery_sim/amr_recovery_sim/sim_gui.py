import sys
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QFrame, QGridLayout
)
from PyQt5.QtCore import QThread, QTimer, Qt
from PyQt5.QtGui import QFont, QColor
import pyqtgraph as pg

class ROS2Worker(QThread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.node = None
        
        self.poses = {'amr1': (0,0,0), 'amr2': (5,0,0), 'amr3': (0,0,0)}
        self.lora_links = []
        
        self.map_data = None
        self.map_info = None
        self.map_updated = False
        
        self.scan_points = None

    def run(self):
        rclpy.init()
        self.node = rclpy.create_node('sim_gui_node')
        
        self.node.create_subscription(PoseStamped, '/amr1/local_pose', lambda msg: self.pose_cb('amr1', msg), 10)
        self.node.create_subscription(PoseStamped, '/amr2/local_pose', lambda msg: self.pose_cb('amr2', msg), 10)
        self.node.create_subscription(String, '/network/lora_status', self.lora_cb, 10)
        
        # Subscribe to real LiDAR physical map and raw scan
        self.node.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.node.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        
        self.cmd_pub = self.node.create_publisher(PoseStamped, '/amr1/goal_pose', 10)
        
        while self.running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.05)
            
        self.node.destroy_node()
        rclpy.shutdown()
        
    def pose_cb(self, amr, msg):
        q = msg.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.poses[amr] = (msg.pose.position.x, msg.pose.position.y, yaw)
        
    def lora_cb(self, msg):
        self.lora_links = msg.data.split(',')
        
    def scan_cb(self, msg):
        if 'amr1' not in self.poses:
            return
        rx, ry, ryaw = self.poses['amr1']
        
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        
        # Filter valid ranges
        valid = (ranges >= msg.range_min) & (ranges <= msg.range_max)
        r = ranges[valid]
        a = angles[valid]
        
        # Transform to map frame
        global_a = a + ryaw
        pts_x = rx + r * np.cos(global_a)
        pts_y = ry + r * np.sin(global_a)
        
        self.scan_points = np.column_stack((pts_x, pts_y))
        
    def map_cb(self, msg):
        # Convert map 1D array to 2D numpy array
        w = msg.info.width
        h = msg.info.height
        data = np.array(msg.data, dtype=np.int8).reshape((h, w))
        
        # We only want to map ROS coordinates to PyQTGraph image coordinates.
        # Unknown (-1) -> 127, Free (0) -> 255, Occupied (100) -> 0
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[data == -1] = [50, 50, 60] # Unknown background dark
        img[data == 0] = [240, 240, 240] # Free space light
        img[data >= 1] = [30, 30, 30] # Occupied space very dark
        
        # Pyqtgraph expects image as (width, height)
        self.map_data = np.rot90(img, k=-1)
        self.map_data = np.flipud(self.map_data)
        
        self.map_info = msg.info
        self.map_updated = True
        
    def send_goal(self, amr, x, y):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        self.cmd_pub.publish(msg)

    def stop(self):
        self.running = False


class SimGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AMR Recovery Simulation")
        self.resize(1000, 600)
        self.setStyleSheet("background-color: #1a1b1e; color: #c9d1d9;")
        
        self.ros_worker = ROS2Worker()
        self.ros_worker.start()
        
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(50)
        
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Left Panel - Map
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#0d1117')
        self.plot_widget.setXRange(-2, 20)
        self.plot_widget.setYRange(-10, 10)
        
        # Add Map ImageItem
        self.map_img = pg.ImageItem()
        self.plot_widget.addItem(self.map_img)
        
        layout.addWidget(self.plot_widget, 70)
        
        # Add Wi-Fi Zone boundary (X=10)
        wifi_zone = pg.LinearRegionItem([0, 10], orientation=pg.LinearRegionItem.Vertical, movable=False)
        wifi_zone.setBrush(pg.mkBrush(0, 255, 0, 20)) # faint green
        self.plot_widget.addItem(wifi_zone)
        
        self.scatter = pg.ScatterPlotItem(size=15, pen=pg.mkPen(None))
        self.plot_widget.addItem(self.scatter)
        
        self.laser_scatter = pg.ScatterPlotItem(size=3, pen=pg.mkPen(None), brush=pg.mkBrush(255, 50, 50, 150))
        self.plot_widget.addItem(self.laser_scatter)
        
        self.lora_line = pg.PlotDataItem(pen=pg.mkPen('c', width=3, style=Qt.DashLine))
        self.plot_widget.addItem(self.lora_line)
        
        # Add boundary text
        text = pg.TextItem("Wi-Fi Zone", color=(100, 255, 100), anchor=(0, 1))
        text.setPos(0, 9)
        self.plot_widget.addItem(text)
        text2 = pg.TextItem("Dead Zone", color=(255, 100, 100), anchor=(0, 1))
        text2.setPos(10.5, 9)
        self.plot_widget.addItem(text2)
        
        # Right Panel - Controls
        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet("background-color: #21262d; border-radius: 8px;")
        ctrl_layout = QVBoxLayout(ctrl_frame)
        
        title = QLabel("SIMULATION CONTROLS")
        title.setFont(QFont("Monospace", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        ctrl_layout.addWidget(title)
        
        # Buttons
        btn1 = QPushButton("Drive AMR-1 to Dead Zone (X=12)")
        btn1.setStyleSheet("background-color: #b32d2e; color: white; padding: 10px; border-radius: 4px;")
        btn1.clicked.connect(lambda: self.ros_worker.send_goal('amr1', 12.0, 0.0))
        ctrl_layout.addWidget(btn1)
        
        btn2 = QPushButton("Return AMR-1 to Base (X=0)")
        btn2.setStyleSheet("background-color: #238636; color: white; padding: 10px; border-radius: 4px;")
        btn2.clicked.connect(lambda: self.ros_worker.send_goal('amr1', 0.0, 0.0))
        ctrl_layout.addWidget(btn2)
        
        ctrl_layout.addStretch()
        
        self.status_lbl = QLabel("Awaiting data...")
        self.status_lbl.setFont(QFont("Monospace", 10))
        ctrl_layout.addWidget(self.status_lbl)
        
        layout.addWidget(ctrl_frame, 30)
        
    def update_gui(self):
        # Update map if new
        if self.ros_worker.map_updated and self.ros_worker.map_data is not None:
            self.map_img.setImage(self.ros_worker.map_data)
            
            # Position map correctly
            info = self.ros_worker.map_info
            res = info.resolution
            ox = info.origin.position.x
            oy = info.origin.position.y
            
            # Scale and translate image
            # Note: pyqtgraph transforms apply to the image rect
            tr = pg.QtGui.QTransform()
            tr.translate(ox, oy)
            tr.scale(res, res)
            self.map_img.setTransform(tr)
            
            self.ros_worker.map_updated = False

        # Update Laser Scan
        if self.ros_worker.scan_points is not None:
            sp = self.ros_worker.scan_points
            self.laser_scatter.setData(x=sp[:,0], y=sp[:,1])

        pts = []
        brushes = []
        
        x1, y1, _ = self.ros_worker.poses.get('amr1', (0,0,0))
        x2, y2, _ = self.ros_worker.poses.get('amr2', (5,0,0))
        
        # AMR1
        pts.append({'pos': (x1, y1), 'data': 1})
        brushes.append(pg.mkBrush(255, 50, 50) if x1 >= 10.0 else pg.mkBrush(50, 255, 50))
        
        # AMR2
        pts.append({'pos': (x2, y2), 'data': 2})
        brushes.append(pg.mkBrush(50, 100, 255))
        
        self.scatter.setData(pts, brush=brushes)
        
        # Draw LoRa link if active
        if 'amr1-amr2' in self.ros_worker.lora_links:
            self.lora_line.setData([x1, x2], [y1, y2])
        else:
            self.lora_line.setData([], [])
            
        status_text = f"AMR-1 Position: ({x1:.1f}, {y1:.1f})\n"
        status_text += f"AMR-2 Position: ({x2:.1f}, {y2:.1f})\n"
        status_text += f"LoRa Links: {', '.join(self.ros_worker.lora_links)}"
        self.status_lbl.setText(status_text)
        
    def closeEvent(self, event):
        self.ros_worker.stop()
        self.ros_worker.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    gui = SimGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
