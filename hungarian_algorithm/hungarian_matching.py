import numpy as np
from scipy.optimize import linear_sum_assignment    
from geometric_fusion.transforms import global_to_sensor, project_to_image

LIDAR_GATE_PX = 60    # LiDAR is dense and precise
RADAR_GATE_PX = 100   # Radar is sparse → looser gate


class CrossSensorFusion:
    
    def fuse(self, camera_event, lidar_points_global, radar_points_global):
    
        K     = np.array(camera_event['K'])
        calib = camera_event['calib']
        ego   = camera_event['ego']
        dets  = camera_event['objects']

        if K.size == 0:
            return []                       # camera has no calibration data

        # Project lidar and radar points into this camera image
        lidar_proj = self._project_points(lidar_points_global, calib, ego, K)
        radar_proj = self._project_points(radar_points_global, calib, ego, K)

        # Associate detections with projected points
        lidar_matches = self._assign(dets, lidar_proj, LIDAR_GATE_PX)
        radar_matches = self._assign(dets, radar_proj, RADAR_GATE_PX)

        # Build output
        fused = []
        for i, det in enumerate(dets):
            obj = {
                'id'          : f"{det['class']}_{i}",
                'class'       : det['class'],
                'conf'        : det['conf'],
                'pixel_center': det['px'],
                # LiDAR match (position)
                'lidar_x'    : None,
                'lidar_y'    : None,
                'lidar_score': None,
                'lidar_px_dist': None,
                # Radar match — Cartesian (for Linear KF)
                'radar_x'    : None,
                'radar_y'    : None,
                'radar_vx'   : None,
                'radar_vy'   : None,
                # Radar match — native polar/Doppler (for EKF)
                'radar_rho'   : None,
                'radar_phi'   : None,
                'radar_rhodot': None,
                'radar_px_dist': None,
            }
            if i in lidar_matches:
                lp, dist = lidar_matches[i]
                obj['lidar_x']     = lp['x']
                obj['lidar_y']     = lp['y']
                obj['lidar_score'] = lp.get('score')
                obj['lidar_px_dist'] = dist

            if i in radar_matches:
                rp, dist = radar_matches[i]
                obj['radar_x']    = rp['x']
                obj['radar_y']    = rp['y']
                obj['radar_vx']   = rp.get('vx')
                obj['radar_vy']   = rp.get('vy')
                obj['radar_rho']    = rp.get('rho')
                obj['radar_phi']    = rp.get('phi')
                obj['radar_rhodot'] = rp.get('rhodot')
                obj['radar_px_dist'] = dist

            fused.append(obj)

        return fused
    # point_global full lidar points , calib camera calibration data , ego , k extrinsic and intrinsic
    def _project_points(self,point_global, calib , ego , K):
        proj = []
        for i in point_global:
            if 'x' in i and 'y' in i:
                x, y = i['x'], i['y']
            elif 'rho' in i and 'phi' in i:
                # polar-only radar point (--ekf --polar mode) -> derive Cartesian for projection
                x = i['rho'] * np.cos(i['phi'])
                y = i['rho'] * np.sin(i['phi'])
            else:
                continue  # can't locate this point, skip it

            p_global = np.array([x, y, 0.0]) #neglect the z and make it 1d vecctor
            p_cam = global_to_sensor(p_global , calib , ego)
            uv = project_to_image(p_cam , K)
            if uv is not None:
                point = i if ('x' in i and 'y' in i) else {**i, 'x': x, 'y': y}
                proj.append((uv, point))

        return proj
    
    def _assign(self, dets, projected, gate):
      
        if not dets or not projected:
            return {}

        nd = len(dets)
        np_ = len(projected)

        # Build cost matrix
        C = np.full((nd, np_), np.inf)
        for i, det in enumerate(dets):
            center = np.array(det['px'], dtype=float)
            for j, (uv, _) in enumerate(projected):
                dist = np.linalg.norm(center - uv)
                if dist < gate:
                    C[i, j] = dist

        matches = {}
        if np.all(np.isinf(C)):
            return matches                  # nothing within gate
        C_finite = np.where(np.isinf(C), gate * 10, C)
        row_ind, col_ind = linear_sum_assignment(C_finite)

        for ri, ci in zip(row_ind, col_ind):
            if C[ri, ci] < gate:            # only accept gated matches
                _, point = projected[ci]
                matches[ri] = (point, float(C[ri, ci]))

        return matches