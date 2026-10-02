#!/usr/bin/env python3
import os
import sys
import time
import json
import subprocess
import threading
import math
import cv2
import numpy as np

# Force single OpenMP thread to prevent collision with Qt
os.environ["OMP_NUM_THREADS"] = "1"

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QHBoxLayout, QWidget, QVBoxLayout,
    QGridLayout, QFrame, QSizePolicy, QSlider, QSpinBox, QPushButton, QSpacerItem
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan

from trt_yolo import TensorRTYOLO
from fusion_mapper import SpatialFusionMapper
from opencr_bridge import OpenCRBridge

# Resolve config path relative to this file — works from any clone location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(_SCRIPT_DIR, "amr_config.json")

# --------------------------------------------------------------
# Depth Anything V3 topics -- confirm with `ros2 topic list` if
# these ever change on the device
# --------------------------------------------------------------
DEPTH_TOPIC = "/depth_anything_v3/output/depth_image"
POINTCLOUD_TOPIC = "/depth_anything_v3/output/point_cloud"

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from sensor_msgs.msg import Image as RosImage, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge

def _load_depth_colormap():
    for name, kwargs in [
        ("turbo", {}),
        ("turbo", {"source": "matplotlib"}),
        ("viridis", {}),
        ("CET-L17", {}),  # pyqtgraph built-in, always available
    ]:
        try:
            return pg.colormap.get(name, **kwargs)
        except Exception:
            continue
    return None

DEPTH_CMAP = _load_depth_colormap()



# ==============================================================================
# Background LiDAR & Spatial Fusion Worker (Decoupled from GUI)
# ==============================================================================
class LidarFusionWorker(QThread):
    def __init__(self, mapper):
        super().__init__()
        self.mapper = mapper
        self.running = True
        self.node = None
        self.driver_process = None
        
        self._lock = threading.Lock()
        self.latest_fusion_res = {
            'front_3d_points': [],
            'peripheral_2d_points': [],
            'detected_3d_objects': [],
            'front_offset_deg': mapper.get_front_offset_deg(),
            'safety': {
                'front': {'dist': 9.9, 'safe': True, 'warn': False},
                'back':  {'dist': 9.9, 'safe': True, 'warn': False},
                'left':  {'dist': 9.9, 'safe': True, 'warn': False},
                'right': {'dist': 9.9, 'safe': True, 'warn': False},
                'recommended_cmd': 'FORWARD'
            }
        }
        self.current_detections = []

    def set_camera_detections(self, dets):
        with self._lock:
            self.current_detections = list(dets)

    def get_latest_fusion(self):
        with self._lock:
            return self.latest_fusion_res

    def set_front_offset_deg(self, deg):
        self.mapper.set_front_offset_deg(deg)

    def ensure_ld08_running(self):
        try:
            pgrep = subprocess.run(["pgrep", "-f", "ld08_driver"], stdout=subprocess.PIPE, text=True)
            if not pgrep.stdout.strip():
                print("[LidarFusionWorker] Auto-launching ld08_driver in background...", flush=True)
                cmd = f"source /opt/ros/humble/setup.bash && [ -f {_SCRIPT_DIR}/install/setup.bash ] && source {_SCRIPT_DIR}/install/setup.bash ; ros2 run ld08_driver ld08_driver"
                self.driver_process = subprocess.Popen(
                    cmd, shell=True, executable="/bin/bash",
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                time.sleep(1.5)
        except Exception as e:
            print(f"[LidarFusionWorker] Driver check warning: {e}", flush=True)

    def run(self):
        self.ensure_ld08_running()
        
        if not rclpy.ok():
            rclpy.init()
            
        self.node = rclpy.create_node('dashboard_fusion_worker')
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2
        )
        
        def scan_cb(msg):
            v_ang, v_rng, x, z = self.mapper.process_lidar_scan(
                msg.ranges, msg.angle_min, msg.angle_increment
            )
            with self._lock:
                dets = self.current_detections
                res = self.mapper.fuse(v_ang, v_rng, x, z, dets, img_w=640, img_h=360)
                self.latest_fusion_res = res

        self.node.create_subscription(LaserScan, 'scan', scan_cb, qos)
        print("[LidarFusionWorker] ROS 2 /scan subscription active.", flush=True)
        
        
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self.node)
        while self.running and rclpy.ok():
            try:
                executor.spin_once(timeout_sec=0.03)
            except Exception:
                break

    def stop(self):
        self.running = False
        if self.node:
            self.node.destroy_node()
        if self.driver_process:
            self.driver_process.terminate()


# ==============================================================================
# Background Depth Anything V3 Worker -- depth map + point cloud
# (Decoupled from GUI, same pattern as LidarFusionWorker)
# ==============================================================================
class DepthCloudWorker(QThread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.node = None
        self.cv_bridge = CvBridge()

        self._lock = threading.Lock()
        self.latest_depth_pixmap = None
        self.latest_depth_min = 0.0
        self.latest_depth_max = 0.0
        self.latest_points = np.zeros((0, 3), dtype=np.float32)
        self.latest_colors = None

    def get_latest(self):
        with self._lock:
            return (self.latest_depth_pixmap, self.latest_depth_min,
                    self.latest_depth_max, self.latest_points, self.latest_colors)

    def _on_depth(self, msg):
        depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        valid = depth[depth > 0]
        if valid.size == 0:
            d_min, d_max = 0.0, 0.0
        else:
            d_min, d_max = float(valid.min()), float(valid.max())

        # Normalize to the colormap's own 0.0-1.0 domain BEFORE mapping --
        # normalizing to 0-255 first clips every non-zero pixel to the
        # colormap's top color (a solid block, no gradient).
        norm = np.zeros_like(depth, dtype=np.float32)
        if d_max > d_min:
            norm = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0).astype(np.float32)

        if DEPTH_CMAP is not None:
            colored = DEPTH_CMAP.map(norm, mode="byte")  # (H, W, 4) uint8 RGBA
            h, w = norm.shape
            rgb = colored.reshape(h, w, 4)[:, :, :3].copy()
        else:
            gray = (norm * 255.0).astype(np.uint8)
            rgb = np.stack([gray] * 3, axis=-1)

        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())

        with self._lock:
            self.latest_depth_pixmap = pix
            self.latest_depth_min = d_min
            self.latest_depth_max = d_max

    def _on_points(self, msg):
        pts = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        if pts.size == 0:
            return
        if pts.shape[0] > 40000:
            idx = np.random.choice(pts.shape[0], 40000, replace=False)
            pts = pts[idx]

        gl_pts = np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]]).astype(np.float32)
        
        # Make points dark for light theme
        colors = np.full((gl_pts.shape[0], 4), (0.6, 0.65, 0.7, 1.0), dtype=np.float32)

        with self._lock:
            self.latest_points = gl_pts
            self.latest_colors = colors

    def run(self):
        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node('dashboard_depth_worker')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2
        )

        self.node.create_subscription(RosImage, DEPTH_TOPIC, self._on_depth, qos)
        self.node.create_subscription(PointCloud2, POINTCLOUD_TOPIC, self._on_points, 1)
        print(f"[DepthCloudWorker] Subscribed to {DEPTH_TOPIC} and {POINTCLOUD_TOPIC}", flush=True)

        
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self.node)
        while self.running and rclpy.ok():
            try:
                executor.spin_once(timeout_sec=0.03)
            except Exception:
                break

    def stop(self):
        self.running = False
        if self.node:
            self.node.destroy_node()


# ==============================================================================
# Background Camera, TensorRT & VSLAM Worker (Decoupled from GUI)
# ==============================================================================
class CameraYoloWorker(QThread):
    def __init__(self, camera_id=0, engine_path=None):
        super().__init__()
        self.camera_id = camera_id
        # Resolve engine relative to repo root when not explicitly passed
        self.engine_path = engine_path if engine_path is not None else os.path.join(_SCRIPT_DIR, "yolov8n.engine")
        self.running = True
        
        
        self._lock = threading.Lock()
        self.latest_qpixmap = None
        self.latest_detections = []

        self.fps = 30.0
        self.infer_ms = 4.0
        self.latest_front_3d_points = []
        self.latest_3d_objects = []

    def set_front_3d_data(self, front_pts, objs_3d):
        with self._lock:
            self.latest_front_3d_points = front_pts
            self.latest_3d_objects = objs_3d


    def get_latest_data(self):
        with self._lock:
            return self.latest_qpixmap, self.latest_detections, self.fps, self.infer_ms

    def run(self):
        detector = TensorRTYOLO(self.engine_path, conf_thresh=0.35, nms_thresh=0.45)
        
        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node('dashboard_camera_worker')
        cv_bridge = CvBridge()

        frame_holder = {'frame': None}
        frame_lock = threading.Lock()

        def _on_image(msg):
            try:
                f = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception:
                return
            with frame_lock:
                frame_holder['frame'] = f

        node.create_subscription(RosImage, '/image_raw', _on_image, 5)
        print("[CameraYoloWorker] Subscribed to /image_raw (shared camera feed)", flush=True)

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)

        frame_count = 0
        t_start = time.perf_counter()

        while self.running and rclpy.ok():
            try:
                executor.spin_once(timeout_sec=0.03)
            except Exception:
                break

            with frame_lock:
                frame = frame_holder['frame']
                frame_holder['frame'] = None  # consume so we don't reprocess the same frame

            if frame is None:
                continue
            frame = frame.copy()
            frame_count += 1
            if frame_count % 15 == 0:
                dt = time.perf_counter() - t_start
                if dt > 0:
                    self.fps = 15.0 / dt
                t_start = time.perf_counter()

            # 1. Run CUDA TensorRT Inference
            detections, infer_ms = detector.detect(frame)
            self.infer_ms = infer_ms

            with self._lock:
                front_pts = self.latest_front_3d_points
                objs_3d = self.latest_3d_objects

            # Draw HUD
            cv2.putText(
                frame, f"TRT: {infer_ms:.1f}ms | {self.fps:.1f} FPS",
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
            )

            # Draw Detected Object Bounding Boxes
            for det in detections:
                x1, y1, x2, y2 = det['box']
                cname = det['class_name']
                score = det['score']
                
                z_tag = ""
                for obj in objs_3d:
                    bx = obj['box_2d']
                    if abs(bx[0] - x1) < 30:
                        z_tag = f" | {obj['z3d']:.2f}m"
                        break

                label = f"{cname.upper()} {int(score*100)}%{z_tag}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
                
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(0, y1)), (15, 20, 30), -1)
                cv2.putText(frame, label, (x1 + 3, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Convert to QPixmap directly in worker thread
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)

            with self._lock:
                self.latest_qpixmap = pix
                self.latest_detections = detections

        node.destroy_node()

    def stop(self):
        self.running = False


# ==============================================================================
# Background OpenCR UART Telemetry Worker
# ==============================================================================
class UARTWorker(QThread):
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self.running = True
        self._lock = threading.Lock()
        self.latest_safety = None

    def update_safety(self, safety):
        with self._lock:
            self.latest_safety = safety

    def run(self):
        while self.running:
            with self._lock:
                s = self.latest_safety
            if s:
                self.bridge.send_safety_command(s)
            time.sleep(0.1)

    def stop(self):
        self.running = False


# ==============================================================================
# Main Dashboard Application with Live AMR Front Heading Calibration & VSLAM
# ==============================================================================
class DashboardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jetson Orin Nano - VSLAM Sub-3D Sensor Fusion & AMR Dashboard")
        self.resize(1420, 880)
        self.setStyleSheet("background-color: #1a1b1e; color: #c9d1d9;")

        # Load Saved AMR Configuration
        self.front_offset_deg = 0.0
        self.load_amr_config()

        self.fusion_mapper = SpatialFusionMapper(
            fov_deg=68.0, safe_dist_m=0.55, warn_dist_m=1.00,
            front_offset_deg=self.front_offset_deg
        )
        self.opencr_bridge = OpenCRBridge(port='/dev/ttyACM0', baudrate=115200)

        self.init_ui()

        # Start Background Workers
        self.lidar_worker = LidarFusionWorker(self.fusion_mapper)
        self.camera_worker = CameraYoloWorker()
        self.uart_worker = UARTWorker(self.opencr_bridge)
        self.depth_worker = DepthCloudWorker()

        self.lidar_worker.start()
        self.camera_worker.start()
        self.uart_worker.start()
        self.depth_worker.start()

        # 30 FPS UI Refresh Timer (33ms)
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.tick_gui)
        self.ui_timer.start(33)

    def load_amr_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    cfg = json.load(f)
                    self.front_offset_deg = float(cfg.get('front_offset_deg', 0.0))
                    print(f"[DashboardApp] Loaded AMR front offset: {self.front_offset_deg}° from {CONFIG_FILE}")
            except Exception as e:
                print(f"[DashboardApp] Warning loading config: {e}")

    def save_amr_config(self):
        try:
            cfg = {'front_offset_deg': self.front_offset_deg}
            with open(CONFIG_FILE, 'w') as f:
                json.dump(cfg, f, indent=2)
            self.save_btn.setText("✓ SAVED!")
            QTimer.singleShot(1500, lambda: self.save_btn.setText("💾 SAVE CONFIG"))
            print(f"[DashboardApp] Saved AMR front offset {self.front_offset_deg}° to {CONFIG_FILE}")
        except Exception as e:
            print(f"[DashboardApp] Error saving config: {e}")

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 10, 14, 10)
        root_layout.setSpacing(8)

        # -------------------------------------------------------------
        # 1. TOP HEADER & TELEMETRY BADGES
        # -------------------------------------------------------------
        header = QFrame()
        header.setStyleSheet("background-color: #21262d; border: 1px solid #30363d; border-radius: 6px; padding: 6px 14px;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(0, 0, 0, 0)

        title_lbl = QLabel("JETSON ORIN NANO | DEPTH ANYTHING V3 & SENSOR FUSION")
        title_lbl.setFont(QFont("Monospace", 11, QFont.Bold))

        self.cuda_badge = QLabel("CUDA: TensorRT FP16 (ACTIVE)")
        self.cuda_badge.setStyleSheet("color: #3fb950; font-family: Monospace; font-weight: bold;")

        self.lidar_badge = QLabel("LiDAR: LD_08 (/dev/ttyUSB0)")
        self.lidar_badge.setStyleSheet("color: #58a6ff; font-family: Monospace;")

        self.opencr_badge = QLabel("OpenCR: /dev/ttyACM0 (CONNECTED)")
        self.opencr_badge.setStyleSheet("color: #3fb950; font-family: Monospace; font-weight: bold;")

        h_layout.addWidget(title_lbl)
        h_layout.addStretch()
        h_layout.addWidget(self.cuda_badge)
        h_layout.addSpacing(16)
        h_layout.addWidget(self.lidar_badge)
        h_layout.addSpacing(16)
        h_layout.addWidget(self.opencr_badge)
        root_layout.addWidget(header)

        # -------------------------------------------------------------
        # 2. AMR FRONT HEADING CALIBRATION & VSLAM CONTROL BAR
        # -------------------------------------------------------------
        calib_frame = QFrame()
        calib_layout = QHBoxLayout(calib_frame)
        calib_layout.setContentsMargins(4, 2, 4, 2)
        calib_layout.setSpacing(10)

        # Heading Icon & Title
        calib_title = QLabel("🧭 AMR FRONT HEADING:")
        calib_title.setFont(QFont("Monospace", 9, QFont.Bold))
        calib_layout.addWidget(calib_title)

        # Heading Slider (-180 to +180 deg)
        self.heading_slider = QSlider(Qt.Horizontal)
        self.heading_slider.setRange(-180, 180)
        self.heading_slider.setValue(int(self.front_offset_deg))
        self.heading_slider.setFixedWidth(200)
        self.heading_slider.valueChanged.connect(self.on_heading_slider_changed)
        calib_layout.addWidget(self.heading_slider)

        # Heading Spinbox for fine-tuning
        self.heading_spin = QSpinBox()
        self.heading_spin.setRange(-180, 180)
        self.heading_spin.setValue(int(self.front_offset_deg))
        self.heading_spin.setSuffix("°")
        self.heading_spin.setFont(QFont("Monospace", 9, QFont.Bold))
        self.heading_spin.valueChanged.connect(self.on_heading_spin_changed)
        calib_layout.addWidget(self.heading_spin)

        # Preset Quick Buttons
        for name, deg in [("0° FWD", 0), ("+90° R", 90), ("180° REAR", 180), ("-90° L", -90)]:
            btn = QPushButton(name)
            btn.setFont(QFont("Monospace", 8))
            btn.clicked.connect(lambda checked, d=deg: self.set_preset_heading(d))
            calib_layout.addWidget(btn)

        # Save Button
        self.save_btn = QPushButton("💾 SAVE CONFIG")
        self.save_btn.setFont(QFont("Monospace", 8, QFont.Bold))
        self.save_btn.clicked.connect(self.save_amr_config)
        calib_layout.addWidget(self.save_btn)

        calib_layout.addSpacing(15)
        calib_layout.addWidget(QFrame(frameShape=QFrame.VLine))



        calib_layout.addStretch()
        root_layout.addWidget(calib_frame)

        # -------------------------------------------------------------
        # 3. MAIN CENTER SPLIT: Camera Video + Sub-3D VSLAM Map
        # -------------------------------------------------------------
        main_split = QHBoxLayout()
        main_split.setSpacing(12)

        # Left: Camera Video
        cam_frame = QFrame()
        cam_layout = QVBoxLayout(cam_frame)
        cam_layout.setContentsMargins(8, 8, 8, 8)

        cam_title = QLabel("CAMERA POV - YOLOv8 CLASSIFICATION + VSLAM FEATURES")
        cam_title.setFont(QFont("Monospace", 9, QFont.Bold))
        cam_layout.addWidget(cam_title)

        self.cam_display = QLabel("Initializing Video Feed...")
        self.cam_display.setAlignment(Qt.AlignCenter)
        self.cam_display.setMinimumSize(560, 360)
        self.cam_display.setStyleSheet("background-color: #0d1117; border: 1px solid #30363d; border-radius: 4px;")
        self.cam_display.setScaledContents(True)
        self.cam_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cam_layout.addWidget(self.cam_display)

        main_split.addWidget(cam_frame, 44)

        # Right: LiDAR Radar + Live 3D Point Cloud
        map_frame = QFrame()
        map_layout = QVBoxLayout(map_frame)
        map_layout.setContentsMargins(8, 8, 8, 8)

        map_title = QLabel("LIDAR MAPPING & DEPTH ANYTHING V3 POINT CLOUD")
        map_title.setFont(QFont("Monospace", 9, QFont.Bold))
        map_layout.addWidget(map_title)
        
        split_layout = QHBoxLayout()
        split_layout.setSpacing(8)

        # Left half: Depth Map Output (Small Preview)
        depth_sub_frame = QFrame()
        depth_sub_frame.setFixedWidth(260)
        depth_layout = QVBoxLayout(depth_sub_frame)
        depth_layout.setContentsMargins(0, 0, 0, 0)
        self.depth_display = QLabel("Waiting for depth...")
        self.depth_display.setAlignment(Qt.AlignCenter)
        self.depth_display.setStyleSheet("background-color: #0d1117; border-radius: 4px; color: #8b949e;")
        self.depth_display.setScaledContents(True)
        self.depth_display.setFixedHeight(160)
        depth_layout.addWidget(self.depth_display)
        depth_layout.addStretch() # push preview to top
        split_layout.addWidget(depth_sub_frame)

        # Right half: live 3D point cloud
        cloud_sub_frame = QFrame()
        cloud_sub_layout = QVBoxLayout(cloud_sub_frame)
        cloud_sub_layout.setContentsMargins(0, 0, 0, 0)
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.gl_view.setCameraPosition(distance=8, elevation=20, azimuth=-90)
        self.gl_view.setBackgroundColor(pg.mkColor("#0d1117"))
        grid = gl.GLGridItem()
        grid.setSize(x=20, y=20)
        grid.setSpacing(x=1, y=1)
        grid.setColor((200, 200, 200, 80)) # Dark grid
        self.gl_view.addItem(grid)
        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=2.5, pxMode=True)
        self.gl_view.addItem(self.scatter)
        self.lidar_scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=4.0, color=(0.81, 0.13, 0.18, 1.0), pxMode=True)
        self.gl_view.addItem(self.lidar_scatter)
        
        # Add lidar rays
        self.lidar_rays = gl.GLLinePlotItem(mode='lines')
        self.gl_view.addItem(self.lidar_rays)
        
        cloud_sub_layout.addWidget(self.gl_view)
        
        self.cloud_stats_label = QLabel("points: 0")
        self.cloud_stats_label.setFont(QFont("Monospace", 8))
        self.cloud_stats_label.setStyleSheet("color: #8b949e;")
        cloud_sub_layout.addWidget(self.cloud_stats_label)
        split_layout.addWidget(cloud_sub_frame, 1) # Take all remaining horizontal space

        map_layout.addLayout(split_layout)
        self._cloud_autofit_done = False

        main_split.addWidget(map_frame, 56)
        root_layout.addLayout(main_split, 65)

        # -------------------------------------------------------------
        # 4. BOTTOM HUD PANEL: Directional Safety & OpenCR UART Telemetry
        # -------------------------------------------------------------
        bottom_frame = QFrame()
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setSpacing(14)

        # 4 Directional Cards
        cards_layout = QGridLayout()
        cards_layout.setHorizontalSpacing(10)
        cards_layout.setVerticalSpacing(6)

        self.card_f = self.create_dir_card("▲ FRONT")
        self.card_b = self.create_dir_card("▼ BACK")
        self.card_l = self.create_dir_card("◀ LEFT")
        self.card_r = self.create_dir_card("▶ RIGHT")

        cards_layout.addWidget(self.card_f, 0, 1)
        cards_layout.addWidget(self.card_l, 1, 0)
        cards_layout.addWidget(self.card_b, 1, 1)
        cards_layout.addWidget(self.card_r, 1, 2)

        bottom_layout.addLayout(cards_layout, 38)

        # Recommended Action Banner
        rec_frame = QFrame()
        rec_layout = QVBoxLayout(rec_frame)
        rec_title = QLabel("AUTONOMOUS SAFETY ARBITER")
        rec_title.setFont(QFont("Monospace", 9, QFont.Bold))
        self.rec_label = QLabel("RECOMMENDED: FORWARD")
        self.rec_label.setFont(QFont("Monospace", 13, QFont.Bold))
        self.rec_label.setStyleSheet("color: #3fb950;")
        self.rec_label.setAlignment(Qt.AlignCenter)
        rec_layout.addWidget(rec_title)
        rec_layout.addWidget(self.rec_label)
        bottom_layout.addWidget(rec_frame, 26)

        # OpenCR UART Monitor
        uart_frame = QFrame()
        uart_layout = QVBoxLayout(uart_frame)
        uart_title = QLabel("OPENCR 1.0 UART MONITOR (/dev/ttyACM0)")
        uart_title.setFont(QFont("Monospace", 9, QFont.Bold))
        self.uart_tx_label = QLabel("Tx: Initializing...")
        self.uart_tx_label.setFont(QFont("Monospace", 8))
        self.uart_tx_label.setStyleSheet("color: #58a6ff;")
        self.uart_tx_label.setWordWrap(True)
        uart_layout.addWidget(uart_title)
        uart_layout.addWidget(self.uart_tx_label)
        bottom_layout.addWidget(uart_frame, 36)

        root_layout.addWidget(bottom_frame, 25)

    def create_dir_card(self, title):
        card = QFrame()
        card.setMinimumWidth(105)
        card.current_state = "SAFE"
        card.setStyleSheet("background-color: #0d1117; border: 1.5px solid #238636; border-radius: 6px; padding: 3px;")
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(4, 3, 4, 3)
        vbox.setSpacing(2)

        lbl_title = QLabel(title)
        lbl_title.setFont(QFont("Monospace", 9, QFont.Bold))
        lbl_title.setAlignment(Qt.AlignCenter)

        lbl_stat = QLabel("SAFE (>5m)")
        lbl_stat.setFont(QFont("Monospace", 8, QFont.Bold))
        lbl_stat.setAlignment(Qt.AlignCenter)
        lbl_stat.setStyleSheet("color: #3fb950;")

        vbox.addWidget(lbl_title)
        vbox.addWidget(lbl_stat)
        card.lbl_stat = lbl_stat
        return card

    def update_dir_card_fast(self, card, info):
        dist = info.get('dist', 9.9)
        safe = info.get('safe', True)
        warn = info.get('warn', False)

        dist_str = f"{dist:.2f}m" if dist < 9.0 else ">5m"
        state = "BLOCKED" if not safe else ("WARN" if warn else "SAFE")
        
        if card.current_state != state:
            card.current_state = state
            if state == "BLOCKED":
                card.setStyleSheet("background-color: #2e0f14; border: 1.5px solid #f85149; border-radius: 6px;")
                card.lbl_stat.setStyleSheet("color: #f85149; font-weight: bold;")
            elif state == "WARN":
                card.setStyleSheet("background-color: #2b1d06; border: 1.5px solid #d29922; border-radius: 6px;")
                card.lbl_stat.setStyleSheet("color: #d29922; font-weight: bold;")
            else:
                card.setStyleSheet("background-color: #0d1117; border: 1.5px solid #238636; border-radius: 6px;")
                card.lbl_stat.setStyleSheet("color: #3fb950; font-weight: bold;")

        card.lbl_stat.setText(f"{state} ({dist_str})")

    def on_heading_slider_changed(self, val):
        self.front_offset_deg = float(val)
        self.heading_spin.blockSignals(True)
        self.heading_spin.setValue(val)
        self.heading_spin.blockSignals(False)
        self.lidar_worker.set_front_offset_deg(self.front_offset_deg)

    def on_heading_spin_changed(self, val):
        self.front_offset_deg = float(val)
        self.heading_slider.blockSignals(True)
        self.heading_slider.setValue(val)
        self.heading_slider.blockSignals(False)
        self.lidar_worker.set_front_offset_deg(self.front_offset_deg)

    def set_preset_heading(self, deg):
        self.front_offset_deg = float(deg)
        self.heading_slider.setValue(deg)
        self.heading_spin.setValue(deg)
        self.lidar_worker.set_front_offset_deg(self.front_offset_deg)



    def tick_gui(self):
        # 1. Update Camera Display
        pix, dets, fps, infer_ms = self.camera_worker.get_latest_data()
        if pix is not None and not pix.isNull():
            self.cam_display.setPixmap(pix)
            self.lidar_worker.set_camera_detections(dets)

        # 2. Update Fusion Map (Safety)
        fusion_data = self.lidar_worker.get_latest_fusion()

        # 2b. Update Depth Anything V3 point cloud
        depth_pix, d_min, d_max, points, colors = self.depth_worker.get_latest()

        if hasattr(self, 'depth_display') and depth_pix is not None and not depth_pix.isNull():
            self.depth_display.setPixmap(depth_pix)

        if points.shape[0] > 0:
            # Set colors to dark grey for light theme
            colors = np.full((points.shape[0], 4), (0.7, 0.75, 0.8, 1.0), dtype=np.float32)
            self.scatter.setData(pos=points, color=colors, size=2.5)
            self.cloud_stats_label.setText(f"points: {points.shape[0]}")

        # LiDAR points (X=right, Y=forward, Z=up)
        l_pts = []
        for pt in fusion_data.get('front_3d_points', []):
            l_pts.append([pt['x'], pt['z'], pt['y']])
        for pt in fusion_data.get('peripheral_2d_points', []):
            l_pts.append([pt['x'], pt['z'], 0.0])
        
        if l_pts:
            l_pts_arr = np.array(l_pts, dtype=np.float32)
            self.lidar_scatter.setData(pos=l_pts_arr)
            
            # Create lines from origin (camera optical center) to the points
            rays = np.zeros((len(l_pts) * 2, 3), dtype=np.float32)
            # Origin is (0,0,0) so even indices are left as 0
            rays[1::2] = l_pts_arr
            # Create a fading color gradient for the rays
            ray_colors = np.zeros((len(l_pts) * 2, 4), dtype=np.float32)
            ray_colors[0::2] = [1.0, 0.2, 0.2, 0.05] # Faded at origin
            ray_colors[1::2] = [1.0, 0.1, 0.1, 0.8]  # Bright at the hit point
            self.lidar_rays.setData(pos=rays, color=ray_colors)
        else:
            self.lidar_scatter.setData(pos=np.zeros((1,3)))
            self.lidar_rays.setData(pos=np.zeros((2,3)))
            if not self._cloud_autofit_done and points.shape[0] > 100:
                center = points.mean(axis=0)
                spread = float(np.percentile(np.linalg.norm(points - center, axis=1), 90))
                spread = max(spread, 1.0)
                self.gl_view.setCameraPosition(distance=spread * 2.5)
                self.gl_view.opts['center'] = pg.Vector(*center)
                self._cloud_autofit_done = True

        # Pass front 3D points to camera worker for VSLAM depth back-projection
        self.camera_worker.set_front_3d_data(
            fusion_data.get('front_3d_points', []),
            fusion_data.get('detected_3d_objects', [])
        )

        # 3. Update Safety Status & OpenCR
        safety = fusion_data['safety']
        self.update_dir_card_fast(self.card_f, safety['front'])
        self.update_dir_card_fast(self.card_b, safety['back'])
        self.update_dir_card_fast(self.card_l, safety['left'])
        self.update_dir_card_fast(self.card_r, safety['right'])

        rec = safety['recommended_cmd']
        self.rec_label.setText(f"RECOMMENDED: {rec}")
        self.uart_worker.update_safety(safety)

        # UART Monitor readout
        self.uart_tx_label.setText(f"Tx: {self.opencr_bridge.last_tx_msg}")
        if self.opencr_bridge.is_connected:
            self.opencr_badge.setText("OpenCR: /dev/ttyACM0 (CONNECTED)")
            self.opencr_badge.setStyleSheet("color: #3fb950; font-family: Monospace; font-weight: bold;")
        else:
            self.opencr_badge.setText("OpenCR: /dev/ttyACM0 (OFFLINE)")
            self.opencr_badge.setStyleSheet("color: #dc2626; font-family: Monospace; font-weight: bold;")

    def closeEvent(self, event):
        self.ui_timer.stop()
        if hasattr(self, 'camera_worker'):
            self.camera_worker.stop()
        if hasattr(self, 'lidar_worker'):
            self.lidar_worker.stop()
        if hasattr(self, 'uart_worker'):
            self.uart_worker.stop()
        if hasattr(self, 'depth_worker'):
            self.depth_worker.stop()
        if hasattr(self, 'opencr_bridge'):
            self.opencr_bridge.close()
            
        if hasattr(self, 'camera_worker'):
            self.camera_worker.wait(500)
        if hasattr(self, 'lidar_worker'):
            self.lidar_worker.wait(500)
        if hasattr(self, 'uart_worker'):
            self.uart_worker.wait(500)
        if hasattr(self, 'depth_worker'):
            self.depth_worker.wait(500)
        
        try:
            rclpy.shutdown()
        except Exception:
            pass
        event.accept()


if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)
    window = DashboardApp()
    window.show()
    sys.exit(app.exec_())
