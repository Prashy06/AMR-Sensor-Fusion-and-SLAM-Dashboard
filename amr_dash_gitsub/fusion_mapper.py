import numpy as np
import math

class SpatialFusionMapper:
    def __init__(self, fov_deg=68.0, safe_dist_m=0.55, warn_dist_m=1.00, front_offset_deg=0.0):
        self.fov_deg = fov_deg
        self.fov_rad = math.radians(fov_deg)
        self.half_fov = self.fov_rad / 2.0
        self.safe_dist_m = safe_dist_m
        self.warn_dist_m = warn_dist_m
        self.front_offset_deg = float(front_offset_deg)
        
        # Virtual camera focal length approximation (normalized for 640px width)
        self.fx = 640.0 / (2.0 * math.tan(self.half_fov))
        self.fy = self.fx

    def set_front_offset_deg(self, offset_deg):
        """Set the AMR front heading offset angle relative to LiDAR 0 mark (in degrees)"""
        self.front_offset_deg = float(offset_deg)

    def get_front_offset_deg(self):
        return self.front_offset_deg

    def process_lidar_scan(self, ranges, angle_min, angle_increment, range_min=0.05, range_max=12.0):
        """
        Transforms raw 1D ranges array into 2D polar/Cartesian points.
        Applies front_offset_deg so:
          theta = 0 rad is STRAIGHT FORWARD of the AMR (+Z in camera/body frame)
          positive theta is to the LEFT (+X)
          negative theta is to the RIGHT (-X)
        """
        num_points = len(ranges)
        raw_angles = angle_min + np.arange(num_points) * angle_increment
        
        # Apply AMR Front Heading Offset
        angles = raw_angles - math.radians(self.front_offset_deg)
        
        # Normalize angles to [-pi, pi] where 0 is straight ahead
        angles = (angles + np.pi) % (2.0 * np.pi) - np.pi
        
        ranges = np.array(ranges, dtype=np.float32)
        valid = (ranges >= range_min) & (ranges <= range_max) & np.isfinite(ranges)
        
        v_angles = angles[valid]
        v_ranges = ranges[valid]
        
        # Coordinate frame:
        # Z = forward (along camera optical axis / AMR forward)
        # X = lateral (right: +X, left: -X in camera frame)
        z = v_ranges * np.cos(v_angles)
        x = v_ranges * np.sin(v_angles)
        
        return v_angles, v_ranges, x, z

    def fuse(self, v_angles, v_ranges, x_pts, z_pts, detections, img_w=640, img_h=360):
        front_3d_points = []
        peripheral_2d_points = []
        
        # 1. Partition scan points into Front Camera FOV vs Peripheral
        if len(v_angles) > 0:
            in_front_fov = (np.abs(v_angles) <= self.half_fov) & (z_pts > 0.05)
            
            # Front points (3D)
            if np.any(in_front_fov):
                f_x = x_pts[in_front_fov]
                f_z = z_pts[in_front_fov]
                f_r = v_ranges[in_front_fov]
                f_ang = v_angles[in_front_fov]
                
                for i in range(len(f_x)):
                    front_3d_points.append({
                        'x': float(f_x[i]),
                        'y': 0.0,
                        'z': float(f_z[i]),
                        'range': float(f_r[i]),
                        'angle': float(f_ang[i])
                    })
            
            # Peripheral points (2D)
            in_peripheral = ~in_front_fov
            if np.any(in_peripheral):
                p_x = x_pts[in_peripheral]
                p_z = z_pts[in_peripheral]
                p_r = v_ranges[in_peripheral]
                p_ang = v_angles[in_peripheral]
                for i in range(len(p_x)):
                    peripheral_2d_points.append({
                        'x': float(p_x[i]),
                        'z': float(p_z[i]),
                        'range': float(p_r[i]),
                        'angle': float(p_ang[i])
                    })
        
        # 2. 3D Model Classification: Project 2D bounding boxes into 3D metric cuboids
        detected_3d_objects = []
        matched_front_indices = set()

        for det in detections:
            box = det['box'] # [x1, y1, x2, y2]
            x1, y1, x2, y2 = box
            u_center = (x1 + x2) / 2.0
            v_center = (y1 + y2) / 2.0
            box_w = x2 - x1
            box_h = y2 - y1
            
            # Angular span of bounding box
            angle_obj = math.atan2((u_center - img_w / 2.0), self.fx)
            angle_span = math.atan2(box_w / 2.0, self.fx)
            
            # Find matching LiDAR points
            obj_depth = 1.5
            if len(front_3d_points) > 0:
                obj_angles = np.array([pt['angle'] for pt in front_3d_points])
                obj_ranges = np.array([pt['range'] for pt in front_3d_points])
                
                angular_dist = np.abs(obj_angles - angle_obj)
                matched = np.where(angular_dist <= max(angle_span, 0.08))[0]
                
                if len(matched) > 0:
                    matched_ranges = obj_ranges[matched]
                    matched_ranges = np.sort(matched_ranges)
                    sample_size = max(1, int(len(matched_ranges) * 0.5))
                    obj_depth = float(np.median(matched_ranges[:sample_size]))
                    for idx in matched:
                        matched_front_indices.add(idx)
                else:
                    obj_depth = max(0.5, float(self.fy * 0.5 / max(box_h, 10.0)))
                    
            X_3d = float(obj_depth * math.sin(angle_obj))
            Z_3d = float(obj_depth * math.cos(angle_obj))
            Y_center = float(((v_center - img_h / 2.0) / self.fy) * obj_depth)
            
            metric_w = max(0.35, float((box_w / self.fx) * obj_depth))
            metric_h = max(0.40, float((box_h / self.fy) * obj_depth))
            metric_d = max(0.35, metric_w * 0.7)
            
            detected_3d_objects.append({
                'class_name': det['class_name'],
                'score': det['score'],
                'box_2d': box,
                'x3d': X_3d,
                'y3d': Y_center,
                'z3d': Z_3d,
                'width': metric_w,
                'height': metric_h,
                'depth': metric_d,
                'is_yolo': True
            })

        # If no YOLO objects detected, identify closest front obstacle cluster
        if len(detected_3d_objects) == 0 and len(front_3d_points) > 0:
            unmatched_front = [front_3d_points[i] for i in range(len(front_3d_points)) if i not in matched_front_indices]
            if len(unmatched_front) > 0:
                closest_pt = min(unmatched_front, key=lambda p: p['z'])
                if closest_pt['z'] < 3.5:
                    detected_3d_objects.append({
                        'class_name': 'obstacle',
                        'score': 0.88,
                        'box_2d': [240, 100, 400, 260],
                        'x3d': closest_pt['x'],
                        'y3d': -0.15,
                        'z3d': closest_pt['z'],
                        'width': 0.5,
                        'height': 0.6,
                        'depth': 0.5,
                        'is_yolo': False
                    })
                    
        # 3. Directional Safety Arbiter
        if len(v_angles) > 0:
            front_mask = np.abs(v_angles) <= math.radians(35.0)
            back_mask = np.abs(v_angles) >= math.radians(145.0)
            left_mask = (v_angles > math.radians(35.0)) & (v_angles < math.radians(145.0))
            right_mask = (v_angles < -math.radians(35.0)) & (v_angles > -math.radians(145.0))
            
            front_dists = v_ranges[front_mask] if np.any(front_mask) else np.array([9.9])
            back_dists = v_ranges[back_mask] if np.any(back_mask) else np.array([9.9])
            left_dists = v_ranges[left_mask] if np.any(left_mask) else np.array([9.9])
            right_dists = v_ranges[right_mask] if np.any(right_mask) else np.array([9.9])
            
            min_f = float(np.min(front_dists)) if len(front_dists) > 0 else 9.9
            min_b = float(np.min(back_dists)) if len(back_dists) > 0 else 9.9
            min_l = float(np.min(left_dists)) if len(left_dists) > 0 else 9.9
            min_r = float(np.min(right_dists)) if len(right_dists) > 0 else 9.9
        else:
            min_f, min_b, min_l, min_r = 9.9, 9.9, 9.9, 9.9

        for obj in detected_3d_objects:
            if obj['z3d'] < min_f:
                min_f = obj['z3d']

        safe_f = min_f >= self.safe_dist_m
        safe_b = min_b >= self.safe_dist_m
        safe_l = min_l >= self.safe_dist_m
        safe_r = min_r >= self.safe_dist_m

        if safe_f:
            rec_cmd = "FORWARD"
        elif safe_r and (min_r >= min_l):
            rec_cmd = "TURN_RIGHT"
        elif safe_l:
            rec_cmd = "TURN_LEFT"
        elif safe_b:
            rec_cmd = "REVERSE"
        else:
            rec_cmd = "HALT"
            
        safety_status = {
            'front': {'dist': min_f, 'safe': safe_f, 'warn': min_f < self.warn_dist_m},
            'back':  {'dist': min_b, 'safe': safe_b, 'warn': min_b < self.warn_dist_m},
            'left':  {'dist': min_l, 'safe': safe_l, 'warn': min_l < self.warn_dist_m},
            'right': {'dist': min_r, 'safe': safe_r, 'warn': min_r < self.warn_dist_m},
            'recommended_cmd': rec_cmd
        }
        
        return {
            'front_3d_points': front_3d_points,
            'peripheral_2d_points': peripheral_2d_points,
            'detected_3d_objects': detected_3d_objects,
            'safety': safety_status,
            'front_offset_deg': self.front_offset_deg
        }
