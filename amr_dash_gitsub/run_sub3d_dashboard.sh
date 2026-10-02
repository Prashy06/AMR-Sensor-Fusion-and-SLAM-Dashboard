#!/bin/bash
# ==============================================================================
# Jetson Orin Nano - Sub-3D Sensor Fusion & Autonomous Safety Dashboard Launcher
# ==============================================================================
#
# Usage:   ./run_sub3d_dashboard.sh
# Works from ANY clone location — no hardcoded paths.
# ==============================================================================

set -e

# Resolve the directory this script lives in (portable — works from any path)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ANSI Color Codes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${CYAN}${BOLD}===================================================================${NC}"
echo -e "${CYAN}${BOLD}     JETSON ORIN NANO - SENSOR FUSION & SUB-3D GUI LAUNCHER       ${NC}"
echo -e "${CYAN}${BOLD}===================================================================${NC}"
echo -e "  -> Workspace: ${GREEN}${SCRIPT_DIR}${NC}"
echo -e "  -> Using DISPLAY:     ${GREEN}${DISPLAY}${NC}"
echo -e "  -> Using XAUTHORITY:  ${GREEN}${XAUTHORITY}${NC}"

# 2. Check Connected Hardware
echo -e "\n${BOLD}[2/5] Verifying Connected Hardware...${NC}"

# LiDAR Check
if [ -e "/dev/ttyUSB0" ]; then
    echo -e "  -> LiDAR:    ${GREEN}/dev/ttyUSB0 detected (LD_08)${NC}"
    chmod 666 /dev/ttyUSB0 2>/dev/null || true
else
    echo -e "  -> LiDAR:    ${YELLOW}WARNING: /dev/ttyUSB0 not found${NC}"
fi

# OpenCR Check
if [ -e "/dev/ttyACM0" ]; then
    echo -e "  -> OpenCR:   ${GREEN}/dev/ttyACM0 detected (OpenCR 1.0)${NC}"
    chmod 666 /dev/ttyACM0 2>/dev/null || true
else
    echo -e "  -> OpenCR:   ${YELLOW}WARNING: /dev/ttyACM0 not found${NC}"
fi

# 3. Verify CUDA TensorRT Engine
echo -e "\n${BOLD}[3/5] Checking CUDA TensorRT Model...${NC}"
ENGINE_PATH="${SCRIPT_DIR}/yolov8n.engine"
if [ -f "$ENGINE_PATH" ]; then
    echo -e "  -> Model:    ${GREEN}TensorRT FP16 engine verified (${ENGINE_PATH})${NC}"
else
    echo -e "  -> Model:    ${RED}ERROR: ${ENGINE_PATH} missing!${NC}"
    echo -e "  -> ${YELLOW}Tip: Generate it on-device with:${NC}"
    echo -e "     pip3 install ultralytics"
    echo -e "     yolo export model=yolov8n.pt format=engine device=0 half=True imgsz=640"
    echo -e "     cp yolov8n.engine ${SCRIPT_DIR}/"
    exit 1
fi

# 4. Source ROS 2 and Workspace Environment
echo -e "\n${BOLD}[4/5] Sourcing ROS 2 Humble Environment...${NC}"
source /opt/ros/humble/setup.bash
echo -e "  -> ${GREEN}ROS 2 Humble sourced${NC}"

# Source this workspace's overlay (if already built with colcon)
if [ -f "${SCRIPT_DIR}/install/setup.bash" ]; then
    source "${SCRIPT_DIR}/install/setup.bash"
    echo -e "  -> ${GREEN}Workspace overlay sourced (${SCRIPT_DIR}/install/setup.bash)${NC}"
else
    echo -e "  -> ${YELLOW}WARNING: Workspace not built yet. Run: colcon build --symlink-install${NC}"
fi

# Source Depth Anything V3 workspace overlay if it exists
# Default: ~/depth_ws — override by setting DEPTH_WS env var before calling this script
DEPTH_WS="${DEPTH_WS:-${HOME}/depth_ws}"
if [ -f "${DEPTH_WS}/install/setup.bash" ]; then
    source "${DEPTH_WS}/install/setup.bash"
    echo -e "  -> ${GREEN}depth_ws overlay sourced (${DEPTH_WS}/install/setup.bash)${NC}"
else
    echo -e "  -> ${YELLOW}WARNING: depth_ws not found at ${DEPTH_WS}. depth_anything_v3 node may not launch.${NC}"
    echo -e "     Set DEPTH_WS=/your/path before running this script, or see README for setup."
fi

# Prevent OpenMP and Qt collisions
export OMP_NUM_THREADS=1

# Clean up any lingering background processes
killall -9 ld08_driver v4l2_camera_node depth_anything_v3_main 2>/dev/null || true

# 5. Trap Exit Signals for Clean Shutdown
cleanup() {
    trap - SIGINT SIGTERM EXIT
    echo -e "\n${YELLOW}[!] Shutting down GUI and ROS 2 background nodes...${NC}"
    killall -9 ld08_driver v4l2_camera_node depth_anything_v3_main 2>/dev/null || true
    echo -e "${GREEN}[✓] Clean shutdown complete.${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 6. Launch background ROS nodes
echo -e "\n${BOLD}Launch Background Nodes...${NC}"
ros2 run v4l2_camera v4l2_camera_node --ros-args -p video_device:=/dev/video0 -p image_size:="[640,360]" > /dev/null 2>&1 &
ros2 run ld08_driver ld08_driver > /dev/null 2>&1 &
ros2 launch depth_anything_v3 depth_anything_v3.launch.py input_image_topic:=/image_raw input_camera_info_topic:=/camera_info > /dev/null 2>&1 &

# 7. Launch Sub-3D Dashboard GUI
echo -e "\n${BOLD}[5/5] Launching Sensor Fusion Dashboard on display ${DISPLAY}...${NC}"
echo -e "${CYAN}Press Ctrl+C in this terminal to terminate the application and all drivers.${NC}\n"

cd "${SCRIPT_DIR}"
python3 "${SCRIPT_DIR}/dashboard.py"
