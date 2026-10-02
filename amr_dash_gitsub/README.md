# 🤖 AMR Sub-3D Sensor Fusion Dashboard
### Jetson Orin Nano · ROS 2 Humble · TensorRT YOLOv8 · LiDAR LD-08 · Depth Anything V3

[![ROS 2 Humble](https://img.shields.io/badge/ROS%202-Humble-blue?logo=ros)](https://docs.ros.org/en/humble/)
[![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin%20Nano-green?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-orin-nano)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![TensorRT](https://img.shields.io/badge/TensorRT-FP16-orange?logo=nvidia)](https://developer.nvidia.com/tensorrt)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Software Prerequisites](#software-prerequisites)
- [Repository Structure](#repository-structure)
- [Installation & Build](#installation--build)
- [Configuration](#configuration)
- [Running the Dashboard](#running-the-dashboard)
- [ROS 2 Topics](#ros-2-topics)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **AMR Sub-3D Sensor Fusion Dashboard** is a real-time autonomous safety monitoring GUI designed to run on a **NVIDIA Jetson Orin Nano**. It fuses data from:

- 🟢 **LD-08 2D LiDAR** — 360° obstacle mapping via `/scan` topic
- 🔵 **Depth Anything V3** — Monocular pseudo-depth estimation (no stereo camera needed)
- 🟡 **YOLOv8n TensorRT FP16** — Real-time object detection at the edge
- 🔴 **OpenCR 1.0 UART Bridge** — Safety command relay to the robot motor controller

The dashboard provides a unified Sub-3D spatial fusion view (LiDAR 2D + depth-lifted 3D), live camera feed with detection overlays, depth heatmap, and real-time safety zone alerting (SAFE / WARN / STOP).

> **Note:** This project is designed specifically for NVIDIA Jetson hardware (JetPack 6.x). TensorRT and CUDA are not available on standard x86 machines without an NVIDIA GPU and the CUDA toolkit.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Jetson Orin Nano (Host)                       │
│                                                                  │
│  ┌──────────────┐   /scan    ┌───────────────────────────────┐  │
│  │  LD-08 LiDAR │──────────▶│                               │  │
│  │  /dev/ttyUSB0│           │     dashboard.py (PyQt5)      │  │
│  └──────────────┘           │                               │  │
│                             │  ┌─────────────────────────┐  │  │
│  ┌──────────────┐ /image_raw│  │  Sub-3D Fusion Map View  │  │  │
│  │  USB Camera  │──────────▶│  │  (sub3d_map_widget.py)   │  │  │
│  │  /dev/video0 │           │  └─────────────────────────┘  │  │
│  └──────────────┘           │                               │  │
│                             │  ┌─────────────────────────┐  │  │
│  ┌──────────────┐  /depth.. │  │  TensorRT YOLOv8n FP16  │  │  │
│  │DepthAnythingV3────────▶  │  │  (trt_yolo.py)          │  │  │
│  │  ROS 2 Node  │  /pc2     │  └─────────────────────────┘  │  │
│  └──────────────┘           │                               │  │
│                             │  ┌─────────────────────────┐  │  │
│  ┌──────────────┐  UART     │  │  SpatialFusionMapper     │  │  │
│  │  OpenCR 1.0  │◀─────────│  │  (fusion_mapper.py)      │  │  │
│  │ /dev/ttyACM0 │           │  └─────────────────────────┘  │  │
│  └──────────────┘           └───────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

| Component | Model | Interface | Notes |
|-----------|-------|-----------|-------|
| **Compute** | NVIDIA Jetson Orin Nano | — | JetPack 6.x required |
| **2D LiDAR** | LDRobot LD-08 (LDS-02) | `/dev/ttyUSB0` | 115200 baud |
| **Motor Controller** | OpenCR 1.0 | `/dev/ttyACM0` | 115200 baud |
| **RGB Camera** | USB UVC Camera | `/dev/video0` | 640×360 @ V4L2 |
| **Display** | HDMI / DisplayPort | `$DISPLAY` | Required for PyQt5 GUI |

> **Minimum Jetson RAM:** 8 GB. The TensorRT engine is compiled specifically for the Orin Nano's GPU architecture and **cannot** be transferred to other devices.

---

## Software Prerequisites

### 1. Operating System & JetPack

**Required:** NVIDIA JetPack 6.x on Jetson Orin Nano.

Verify your JetPack version:
```bash
cat /etc/nv_tegra_release
# or
dpkg -l | grep nvidia-jetpack
```

### 2. ROS 2 Humble (Full Desktop)

```bash
# Add ROS 2 apt repo (if not already done)
sudo apt install -y software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list

sudo apt update && sudo apt install -y ros-humble-desktop
```

### 3. ROS 2 Package Dependencies

```bash
sudo apt install -y \
  ros-humble-v4l2-camera \
  ros-humble-cv-bridge \
  ros-humble-sensor-msgs \
  python3-sensor-msgs-py \
  python3-colcon-common-extensions
```

### 4. Python Dependencies

All Python dependencies are listed in [`requirements.txt`](requirements.txt).

```bash
pip3 install -r requirements.txt
```

> **Important:** `tensorrt` and `rclpy` are **NOT** in `requirements.txt` — they ship with JetPack and ROS 2 respectively and must not be pip-installed.

Verify they are available:
```bash
python3 -c "import tensorrt; print('TensorRT:', tensorrt.__version__)"
source /opt/ros/humble/setup.bash && python3 -c "import rclpy; print('rclpy OK')"
```

### 5. Depth Anything V3 ROS 2 Node

The launcher requires the `depth_anything_v3` ROS 2 package in a separate workspace:

```bash
# Create the depth workspace
mkdir -p ~/depth_ws/src && cd ~/depth_ws/src

# Clone the Depth Anything V3 ROS 2 wrapper
git clone https://github.com/saurabh1002/depth_anything_ros.git depth_anything_v3

# Build
cd ~/depth_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

> If your `depth_ws` is at a different path, export `DEPTH_WS=/your/path` before running the launcher.

### 6. Serial Port Permissions (One-Time Setup)

```bash
sudo usermod -aG dialout $USER
# Log out and back in, or apply immediately:
newgrp dialout
```

---

## Repository Structure

```
ros2_ws/
├── run_sub3d_dashboard.sh       # ⭐ Main launcher — run this to start everything
├── dashboard.py                 # PyQt5 GUI — main application window
├── sub3d_map_widget.py          # Custom Sub-3D spatial fusion map widget
├── fusion_mapper.py             # LiDAR + depth spatial fusion logic
├── trt_yolo.py                  # TensorRT YOLOv8n FP16 inference engine
├── opencr_bridge.py             # OpenCR UART serial bridge
├── amr_config.json              # Runtime config (LiDAR front offset calibration)
├── requirements.txt             # Python pip dependencies
├── yolov8n.engine               # ⚠️ NOT in repo — must be generated on-device (see below)
├── src/
│   ├── ld08_driver/             # LiDAR ROS 2 C++ driver package
│   │   ├── CMakeLists.txt
│   │   ├── package.xml
│   │   ├── include/             # C++ headers
│   │   ├── launch/ld08.launch.py
│   │   └── src/                 # C++ source files
│   └── amr_recovery_sim/        # AMR fleet recovery simulation package
│       ├── package.xml
│       ├── setup.py
│       └── amr_recovery_sim/
│           ├── amr_agent.py
│           ├── fleet_manager.py
│           ├── hitl_bridge.py
│           └── network_sim.py
└── build/                       # colcon build output (git-ignored)
```

---

## Installation & Build

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-org>/ros2amrdash.git
cd ros2amrdash/ros2_ws
```

### Step 2 — Install Python Dependencies

```bash
pip3 install -r requirements.txt
```

### Step 3 — Build the ROS 2 Workspace

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### Step 4 — Generate the TensorRT Engine (On-Device Only)

The `yolov8n.engine` file is **not included** in the repository because TensorRT engines are compiled for a specific GPU architecture and are not portable. Generate it on the Jetson:

```bash
# Install ultralytics (YOLOv8 export tool)
pip3 install ultralytics

# Download YOLOv8n weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Export to TensorRT FP16 engine (runs on Jetson GPU — takes ~5–10 min)
yolo export model=yolov8n.pt format=engine device=0 half=True imgsz=640

# Place the engine in the repo root
cp yolov8n.engine /path/to/ros2amrdash/ros2_ws/
```

### Step 5 — Setup Depth Anything V3 Workspace

Follow the instructions in [Software Prerequisites §5](#5-depth-anything-v3-ros-2-node).

---

## Configuration

Edit [`amr_config.json`](amr_config.json) to calibrate the LiDAR front-facing direction:

```json
{
  "front_offset_deg": 180.0
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `front_offset_deg` | float | `180.0` | Rotation offset (degrees) from LiDAR hardware 0° mark to the AMR's forward direction. Adjust if the scan map appears rotated. |

> This file is auto-saved by the dashboard GUI when you adjust the offset slider at runtime.

---

## Running the Dashboard

### ⚡ Quick Start (Recommended)

```bash
# From the repo root — works from any clone path
chmod +x run_sub3d_dashboard.sh
./run_sub3d_dashboard.sh
```

This single script:
1. Prints the resolved workspace path and display info
2. Detects and grants permissions to `/dev/ttyUSB0` (LiDAR) and `/dev/ttyACM0` (OpenCR)
3. Verifies the TensorRT `.engine` file exists in the repo directory
4. Sources ROS 2 Humble and workspace overlays automatically
5. Launches background ROS 2 nodes (`v4l2_camera`, `ld08_driver`, `depth_anything_v3`)
6. Starts the PyQt5 dashboard GUI
7. Tears down all background nodes cleanly on `Ctrl+C`

#### Custom depth_ws path

If your Depth Anything V3 workspace is not at `~/depth_ws`:

```bash
DEPTH_WS=/path/to/your/depth_ws ./run_sub3d_dashboard.sh
```

### 🔧 Manual Step-by-Step

If you need to run nodes individually for debugging:

```bash
# Source environments (all terminals)
source /opt/ros/humble/setup.bash
source ~/depth_ws/install/setup.bash        # adjust path if different
source /path/to/ros2amrdash/ros2_ws/install/setup.bash
export OMP_NUM_THREADS=1

# Terminal 1 — USB Camera
ros2 run v4l2_camera v4l2_camera_node \
  --ros-args -p video_device:=/dev/video0 -p image_size:="[640,360]"

# Terminal 2 — LiDAR Driver
ros2 run ld08_driver ld08_driver

# Terminal 3 — Depth Anything V3
ros2 launch depth_anything_v3 depth_anything_v3.launch.py \
  input_image_topic:=/image_raw \
  input_camera_info_topic:=/camera_info

# Terminal 4 — Dashboard GUI
cd /path/to/ros2amrdash/ros2_ws
python3 dashboard.py
```

### SSH / Remote Display

If running over SSH with the Jetson's local screen:

```bash
ssh <your-user>@<jetson-ip>
export DISPLAY=:0
export XAUTHORITY=$HOME/.Xauthority
./run_sub3d_dashboard.sh
```

---

## ROS 2 Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/scan` | `sensor_msgs/LaserScan` | Subscribe | 360° LiDAR scan from LD-08 |
| `/image_raw` | `sensor_msgs/Image` | Subscribe | RGB camera frames (640×360) |
| `/camera_info` | `sensor_msgs/CameraInfo` | Subscribe | Camera calibration info |
| `/depth_anything_v3/output/depth_image` | `sensor_msgs/Image` | Subscribe | Monocular depth map |
| `/depth_anything_v3/output/point_cloud` | `sensor_msgs/PointCloud2` | Subscribe | Lifted 3D point cloud |

---

## Troubleshooting

### `ERROR: yolov8n.engine missing`

The TensorRT engine was not found in the repo directory. Generate it on-device — see [Installation Step 4](#step-4--generate-the-tensorrt-engine-on-device-only).

### `WARNING: /dev/ttyUSB0 not found`

```bash
ls /dev/ttyUSB*          # verify device appears
dmesg | tail -20         # look for USB serial attach events
sudo chmod 666 /dev/ttyUSB0
```

### `WARNING: /dev/ttyACM0 not found`

The OpenCR controller is not connected or not powered. The dashboard will still run but UART commands won't be sent.

### `ModuleNotFoundError: No module named 'tensorrt'`

TensorRT is not installed. On JetPack 6, it ships pre-installed. Try:
```bash
python3 -c "import tensorrt"
# If this fails, reinstall TensorRT via JetPack SDK Manager
```

### `ModuleNotFoundError: No module named 'sensor_msgs_py'`

```bash
sudo apt install python3-sensor-msgs-py
```

### `ModuleNotFoundError: No module named 'OpenGL'`

```bash
pip3 install PyOpenGL PyOpenGL-accelerate
```

### PyQt5 / Display errors (`cannot connect to X server`)

```bash
echo $DISPLAY          # should output :0 or :1
xhost +local:          # allow local connections (dev/debug only)
```

### LiDAR scan appears rotated

Adjust `front_offset_deg` in `amr_config.json`. A value of `180.0` means the LiDAR's hardware 0° mark faces backward relative to the AMR's forward direction.

### `depth_anything_v3` launch fails

Verify the depth workspace is built and sourced:
```bash
source ~/depth_ws/install/setup.bash
ros2 pkg list | grep depth_anything
```
If not found, see [Software Prerequisites §5](#5-depth-anything-v3-ros-2-node).

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

The `ld08_driver` ROS 2 package is originally from [ROBOTIS-GIT/ld08_driver](https://github.com/ROBOTIS-GIT/ld08_driver), also under Apache 2.0.

---

<div align="center">
  Built for <strong>Smart India Hackathon (SIH)</strong> &nbsp;·&nbsp; Jetson Orin Nano &nbsp;·&nbsp; ROS 2 Humble
</div>
