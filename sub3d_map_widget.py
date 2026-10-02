import math
import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF, QLinearGradient, QPainterPath
from PyQt5.QtCore import Qt, QPointF, QRectF

class Sub3DMapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 420)
        self.front_3d_points = []
        self.peripheral_2d_points = []
        self.detected_3d_objects = []
        self.front_offset_deg = 0.0
        self.fov_deg = 68.0
        self.max_range = 5.0 # meters
        
        # VSLAM Map Data
        self.vslam_tracking = False
        self.vslam_pose = (0.0, 0.0, 0.0, 0.0) # (x, y, z, yaw)
        self.vslam_trajectory = []
        self.vslam_keyframes = []
        self.vslam_map_points = []

    def update_map(self, fusion_data, vslam_data=None):
        self.front_3d_points = fusion_data.get('front_3d_points', [])
        self.peripheral_2d_points = fusion_data.get('peripheral_2d_points', [])
        self.detected_3d_objects = fusion_data.get('detected_3d_objects', [])
        self.front_offset_deg = fusion_data.get('front_offset_deg', 0.0)
        
        if vslam_data:
            self.vslam_tracking = vslam_data.get('tracking', False)
            self.vslam_pose = vslam_data.get('pose', (0.0, 0.0, 0.0, 0.0))
            self.vslam_trajectory = vslam_data.get('trajectory', [])
            self.vslam_keyframes = vslam_data.get('keyframes', [])
            self.vslam_map_points = vslam_data.get('map_points', [])
            
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        w = self.width()
        h = self.height()

        # Deep Graphite Cyberpunk Background
        painter.fillRect(0, 0, w, h, QColor(13, 17, 23))

        # Split width: 56% Front 3D Frustum + VSLAM, 44% Peripheral 2D Radar
        split_x = int(w * 0.56)

        # 1. Paint Front 3D VSLAM Sub-3D View
        self.paint_front_vslam_view(painter, 0, 0, split_x, h)

        # 2. Vertical Divider Line
        painter.setPen(QPen(QColor(208, 215, 222), 2))
        painter.drawLine(split_x, 15, split_x, h - 15)

        # 3. Paint Peripheral 2D Radar Map & AMR Heading
        self.paint_peripheral_2d_view(painter, split_x, 0, w - split_x, h)

    def paint_front_vslam_view(self, painter, vx, vy, vw, vh):
        # 1. Title & VSLAM Status Badges
        painter.setFont(QFont("Monospace", 9, QFont.Bold))
        painter.setPen(QColor(88, 166, 255))
        painter.drawText(vx + 16, vy + 24, "VSLAM SUB-3D SPATIAL MAP (CAMERA POV)")

        # VSLAM Status Indicator
        status_col = QColor(26, 127, 55) if self.vslam_tracking else QColor(210, 153, 34)
        status_str = "VSLAM: TRACKING (RGB-D FUSED)" if self.vslam_tracking else "VSLAM: INITIALIZING..."
        painter.setFont(QFont("Monospace", 8, QFont.Bold))
        painter.setPen(status_col)
        painter.drawText(vx + 16, vy + 40, status_str)

        # Frustum Origin at bottom center
        cx = vx + vw / 2.0
        cy = vy + vh * 0.88

        # 3D isometric perspective projection
        scale_depth = (vh * 0.65) / self.max_range
        scale_lat = (vw * 0.45) / (self.max_range * math.tan(math.radians(self.fov_deg / 2.0)))

        def project_3d(x_m, y_m, z_m):
            z_clamped = max(0.15, min(z_m, self.max_range + 0.8))
            persp = 1.0 / (1.0 + 0.28 * (z_clamped / self.max_range))
            sx = cx + (x_m * scale_lat * persp)
            sy = cy - (z_clamped * scale_depth) + (y_m * scale_depth * 0.8 * persp)
            return sx, sy

        half_fov = math.radians(self.fov_deg / 2.0)

        # 2. Draw Ground Frustum Shading & Metric Grid
        far_z = self.max_range
        rx_l = -far_z * math.tan(half_fov)
        rx_r = far_z * math.tan(half_fov)

        sx0, sy0 = project_3d(0, 0, 0.05)
        sxl, syl = project_3d(rx_l, 0, far_z)
        sxr, syr = project_3d(rx_r, 0, far_z)

        frustum_poly = QPolygonF([QPointF(sx0, sy0), QPointF(sxl, syl), QPointF(sxr, syr)])
        grad = QLinearGradient(sx0, sy0, (sxl + sxr) / 2, (syl + syr) / 2)
        grad.setColorAt(0.0, QColor(0, 140, 255, 40))
        grad.setColorAt(1.0, QColor(0, 70, 180, 10))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(0, 140, 255, 95), 1.5, Qt.DashLine))
        painter.drawPolygon(frustum_poly)

        # Depth Grid Arcs (1m, 2m, 3m, 4m, 5m)
        painter.setFont(QFont("Monospace", 7))
        for r_m in [1.0, 2.0, 3.0, 4.0, 5.0]:
            if r_m > self.max_range:
                continue
            arc_pts = []
            x_bound = r_m * math.tan(half_fov)
            for xi in np.linspace(-x_bound, x_bound, 13):
                sx, sy = project_3d(xi, 0.0, r_m)
                arc_pts.append(QPointF(sx, sy))
            
            painter.setPen(QPen(QColor(40, 65, 95), 1, Qt.DotLine))
            for i in range(len(arc_pts) - 1):
                painter.drawLine(arc_pts[i], arc_pts[i+1])
                
            if len(arc_pts) > 0:
                mid_pt = arc_pts[len(arc_pts)//2]
                painter.setPen(QColor(88, 166, 255, 180))
                painter.drawText(int(mid_pt.x() + 4), int(mid_pt.y() - 2), f"{r_m:.0f}m")

        # 3. Draw VSLAM Camera Trajectory Ribbon
        if len(self.vslam_trajectory) > 1:
            painter.setPen(QPen(QColor(0, 255, 200, 180), 2.0))
            traj_pts = []
            for (tx, ty, tz, tyaw) in self.vslam_trajectory:
                # Relative to current camera pose
                rel_x = tx - self.vslam_pose[0]
                rel_z = tz - self.vslam_pose[2]
                if abs(rel_z) <= self.max_range and abs(rel_x) <= self.max_range:
                    sx, sy = project_3d(rel_x, 0.0, rel_z)
                    traj_pts.append(QPointF(sx, sy))
            for i in range(len(traj_pts) - 1):
                painter.drawLine(traj_pts[i], traj_pts[i+1])

        # 4. Draw VSLAM 3D Accumulated Visual Landmark Points
        # Points in self.vslam_map_points
        for pt in self.vslam_map_points:
            # Transform relative to current camera position
            rel_x = pt['x'] - self.vslam_pose[0]
            rel_y = pt['y']
            rel_z = pt['z'] - self.vslam_pose[2]
            
            if 0.2 <= rel_z <= self.max_range and abs(rel_x) <= rel_z * math.tan(half_fov) + 0.3:
                sx, sy = project_3d(rel_x, rel_y, rel_z)
                r, g, b = pt.get('r', 0), pt.get('g', 220), pt.get('b', 255)
                
                # Draw glowing VSLAM landmark diamond/dot
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(r, g, b, 230)))
                painter.drawEllipse(QPointF(sx, sy), 3.0, 3.0)
                # Outer glow
                painter.setBrush(QBrush(QColor(r, g, b, 60)))
                painter.drawEllipse(QPointF(sx, sy), 6.0, 6.0)

        # 5. Draw Front 3D LiDAR Obstacle Pillars
        closest_front_dist = 9.9
        for pt in self.front_3d_points:
            px = pt['x']
            pz = pt['z']
            pr = pt['range']
            if pr < closest_front_dist:
                closest_front_dist = pr

            if pz > self.max_range:
                continue

            bx, by = project_3d(px, 0.0, pz)
            tx, ty = project_3d(px, -0.45, pz) # 0.45m obstacle height

            if pr < 0.55:
                bar_col = QColor(255, 70, 70, 240)
            elif pr < 1.0:
                bar_col = QColor(255, 180, 40, 220)
            else:
                bar_col = QColor(0, 225, 255, 200)

            painter.setPen(QPen(bar_col, 2.5))
            painter.drawLine(QPointF(bx, by), QPointF(tx, ty))
            painter.setBrush(QBrush(bar_col))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(tx, ty), 3.5, 3.5)

        # 6. Draw 3D Model Classification Bounding Cuboids
        for obj in self.detected_3d_objects:
            cname = obj['class_name'].upper()
            score = int(obj['score'] * 100)
            x3d = obj['x3d']
            y3d = obj['y3d']
            z3d = obj['z3d']
            w3d = max(0.35, obj['width'])
            h3d = max(0.40, obj['height'])
            d3d = max(0.35, obj['depth'])

            if z3d > self.max_range:
                continue

            hw, hh, hd = w3d / 2.0, h3d / 2.0, d3d / 2.0
            corners_3d = [
                (x3d - hw, y3d - hh, z3d - hd),
                (x3d + hw, y3d - hh, z3d - hd),
                (x3d + hw, y3d + hh, z3d - hd),
                (x3d - hw, y3d + hh, z3d - hd),
                (x3d - hw, y3d - hh, z3d + hd),
                (x3d + hw, y3d - hh, z3d + hd),
                (x3d + hw, y3d + hh, z3d + hd),
                (x3d - hw, y3d + hh, z3d + hd),
            ]
            scr_pts = [QPointF(*project_3d(cx3, cy3, cz3)) for (cx3, cy3, cz3) in corners_3d]

            box_col = QColor(26, 127, 55) if z3d >= 1.0 else (QColor(210, 153, 34) if z3d >= 0.55 else QColor(248, 81, 73))
            painter.setPen(QPen(box_col, 2.0))
            painter.setBrush(QColor(box_col.red(), box_col.green(), box_col.blue(), 40))

            front_face = QPolygonF([scr_pts[0], scr_pts[1], scr_pts[2], scr_pts[3]])
            painter.drawPolygon(front_face)

            painter.drawLine(scr_pts[0], scr_pts[4])
            painter.drawLine(scr_pts[1], scr_pts[5])
            painter.drawLine(scr_pts[2], scr_pts[6])
            painter.drawLine(scr_pts[3], scr_pts[7])

            back_face = QPolygonF([scr_pts[4], scr_pts[5], scr_pts[6], scr_pts[7]])
            painter.drawPolygon(back_face)

            # 3D Tag
            top_pt = scr_pts[0]
            tag1 = f"3D: {cname} {score}%"
            tag2 = f"X:{x3d:+.2f}m | Z:{z3d:.2f}m"

            painter.setFont(QFont("Monospace", 8, QFont.Bold))
            tw = max(painter.fontMetrics().horizontalAdvance(tag1), painter.fontMetrics().horizontalAdvance(tag2)) + 14
            th = 34
            bx = top_pt.x() - tw / 2.0
            by = top_pt.y() - th - 8

            painter.setBrush(QBrush(QColor(22, 27, 34, 230)))
            painter.setPen(QPen(box_col, 1.5))
            painter.drawRoundedRect(QRectF(bx, by, tw, th), 4, 4)

            painter.setPen(box_col)
            painter.drawText(int(bx + 7), int(by + 14), tag1)
            painter.setPen(QColor(201, 209, 217))
            painter.drawText(int(bx + 7), int(by + 28), tag2)

        # 7. Camera Base Marker & VSLAM Pose Readout
        painter.setBrush(QBrush(QColor(88, 166, 255)))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(sx0, sy0), 7, 7)
        painter.setFont(QFont("Monospace", 8, QFont.Bold))
        painter.setPen(QColor(88, 166, 255))
        painter.drawText(int(sx0 - 28), int(sy0 + 18), "CAMERA")

        # Bottom HUD Summary
        painter.setFont(QFont("Monospace", 8))
        painter.setPen(QColor(87, 96, 106))
        lm_count = len(self.vslam_map_points)
        c_dist = f"{closest_front_dist:.2f}m" if closest_front_dist < 9.0 else "CLEAR"
        painter.drawText(vx + 16, vy + 56, f"Landmarks: {lm_count} | Closest: {c_dist} | Front Pts: {len(self.front_3d_points)}")

    def paint_peripheral_2d_view(self, painter, vx, vy, vw, vh):
        # 1. Header with AMR Heading Offset
        painter.setFont(QFont("Monospace", 9, QFont.Bold))
        painter.setPen(QColor("#24292f")) # Dark text for light theme
        painter.drawText(vx + 16, vy + 24, "LiDAR 2D RADAR MAPPING")

        painter.setFont(QFont("Monospace", 8, QFont.Bold))
        painter.setPen(QColor("#0969da")) # Blue for light theme
        painter.drawText(vx + 16, vy + 40, f"AMR Front Calibrated: {self.front_offset_deg:+.1f}°")

        # Radar Center
        rcx = vx + vw / 2.0
        rcy = vy + vh * 0.54
        radar_radius = min(vw, vh) * 0.38
        scale = radar_radius / self.max_range

        # Concentric Metric Range Rings
        painter.setFont(QFont("Monospace", 7))
        for r_m in [0.55, 1.0, 2.0, 3.0, 4.0, 5.0]:
            rad_px = r_m * scale
            if r_m == 0.55:
                painter.setPen(QPen(QColor(248, 81, 73, 140), 1.5, Qt.DashLine))
            else:
                painter.setPen(QPen(QColor(208, 215, 222), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(rcx, rcy), rad_px, rad_px)
            painter.setPen(QColor(87, 96, 106))
            painter.drawText(int(rcx + 4), int(rcy - rad_px + 10), f"{r_m:.1f}m")

        # Radar Crosshairs
        painter.setPen(QPen(QColor(175, 184, 193), 1.5))
        painter.drawLine(int(rcx - radar_radius), int(rcy), int(rcx + radar_radius), int(rcy))
        painter.drawLine(int(rcx), int(rcy - radar_radius), int(rcx), int(rcy + radar_radius))

        # Camera FOV Frustum Wedge on Radar (indicates where the 3D view is looking)
        half_fov = math.radians(self.fov_deg / 2.0)
        x_left = rcx - radar_radius * math.sin(half_fov)
        y_left = rcy - radar_radius * math.cos(half_fov)
        x_right = rcx + radar_radius * math.sin(half_fov)
        y_right = rcy - radar_radius * math.cos(half_fov)

        fov_wedge = QPolygonF([QPointF(rcx, rcy), QPointF(x_left, y_left), QPointF(x_right, y_right)])
        painter.setBrush(QBrush(QColor(9, 105, 218, 40))) # Stronger blue highlight for FOV
        painter.setPen(QPen(QColor(9, 105, 218, 180), 1.5, Qt.DashLine))
        painter.drawPolygon(fov_wedge)
        
        # Add a text label inside the FOV
        painter.setPen(QColor("#0969da"))
        painter.setFont(QFont("Monospace", 8, QFont.Bold))
        painter.drawText(int(rcx - 20), int(rcy - radar_radius/2), "FRONT FOV")

        # Plot Peripheral LiDAR Scan Points
        for pt in self.peripheral_2d_points:
            ang = pt['angle']
            rng = pt['range']
            if rng > self.max_range:
                continue

            px = rcx + rng * scale * math.sin(ang)
            py = rcy - rng * scale * math.cos(ang)

            if rng < 0.55:
                pt_col = QColor(207, 34, 46, 255) # Strong red
                dot_size = 5.0
            elif rng < 1.0:
                pt_col = QColor(154, 103, 0, 255) # Strong amber
                dot_size = 4.0
            else:
                pt_col = QColor(36, 41, 47, 200) # Dark industrial grey for points
                dot_size = 3.0

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(pt_col))
            painter.drawEllipse(QPointF(px, py), dot_size, dot_size)

        # Draw Glowing AMR Front Direction Arrow
        arrow_len = radar_radius * 0.45
        ax = rcx
        ay = rcy - arrow_len
        painter.setPen(QPen(QColor(26, 127, 55), 2.5))
        painter.drawLine(int(rcx), int(rcy), int(ax), int(ay))
        
        # Arrowhead
        arrow_head = QPolygonF([
            QPointF(ax, ay),
            QPointF(ax - 6, ay + 12),
            QPointF(ax + 6, ay + 12)
        ])
        painter.setBrush(QBrush(QColor(26, 127, 55)))
        painter.drawPolygon(arrow_head)

        # Sector Labels
        painter.setFont(QFont("Monospace", 8, QFont.Bold))
        painter.setPen(QColor(26, 127, 55))
        painter.drawText(int(rcx - 22), int(rcy - radar_radius - 8), "▲ FRONT")
        painter.setPen(QColor(210, 153, 34))
        painter.drawText(int(rcx - 18), int(rcy + radar_radius + 18), "▼ REAR")
        painter.drawText(int(rcx - radar_radius - 40), int(rcy + 4), "◀ LEFT")
        painter.drawText(int(rcx + radar_radius + 8), int(rcy + 4), "▶ RIGHT")

        # Central AMR Body Icon
        painter.setBrush(QBrush(QColor(26, 127, 55)))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(rcx, rcy), 6, 6)

        # Peripheral Point Count
        painter.setFont(QFont("Monospace", 8))
        painter.setPen(QColor(87, 96, 106))
        painter.drawText(vx + 16, vy + 56, f"Peripheral Points: {len(self.peripheral_2d_points)}")
