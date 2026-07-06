import os
import json
import numpy as np
import bisect
from collections import defaultdict

from geometric_fusion.transforms import *

class TimelineBuilder:


    def __init__(self, calib_dir, detections_dir):

        self.calib_dir = calib_dir
        self.det_dir   = detections_dir
        self._load_calibration_tables()

    def _load(self, name):
        path = os.path.join(self.calib_dir, name)
        with open(path) as f:
            return json.load(f)

    def _load_calibration_tables(self):
        sensors       = self._load('/home/omar_ben_emad/3d_object_detection_and_tracking/data/v1.0-mini/Calibration Files/sensor.json')
        calib_sensors = self._load('/home/omar_ben_emad/3d_object_detection_and_tracking/data/v1.0-mini/Calibration Files/calibrated_sensor.json')
        ego_poses     = self._load('/home/omar_ben_emad/3d_object_detection_and_tracking/data/v1.0-mini/Calibration Files/ego_pose.json')
        sample_data   = self._load('/home/omar_ben_emad/3d_object_detection_and_tracking/data/v1.0-mini/Calibration Files/sample_data.json')

        # fast lookup dicts
        self.sensor_by_token = {s['token']: s for s in sensors}
        self.calib_by_token  = {c['token']: c for c in calib_sensors}
        self.ego_by_token    = {e['token']: e for e in ego_poses}

        # group sample_data by channel name, sorted by timestamp
        by_channel = defaultdict(list)
        for r in sample_data:
            calib  = self.calib_by_token.get(r['calibrated_sensor_token'])
            if not calib:
                continue
            sensor = self.sensor_by_token.get(calib['sensor_token'])
            if not sensor:
                continue
            by_channel[sensor['channel']].append(r)

        def _sorted(ch):
            return sorted(by_channel[ch], key=lambda r: r['timestamp'])

        # one sorted list per channel we care about
        self.cam_front_sd  = _sorted('CAM_FRONT')
        self.lidar_top_sd  = _sorted('LIDAR_TOP')
        # radar: only non-keyframe sweeps (those are what the radar script processed)
        self.radar_front_sd = sorted(
            [r for r in by_channel['RADAR_FRONT'] if not r['is_key_frame']],
            key=lambda r: r['timestamp']
        )

        # timestamp lists for bisect-based nearest-neighbour search
        self.cam_ts   = [r['timestamp'] for r in self.cam_front_sd]
        self.lidar_ts = [r['timestamp'] for r in self.lidar_top_sd]
        self.radar_ts = [r['timestamp'] for r in self.radar_front_sd]

        # filename→sample_data for lidar (needed to look up calib by sweep filename)
        self.lidar_fname_to_sd = {
            os.path.basename(r['filename']): r
            for r in self.lidar_top_sd
        }

    def _nearest_idx(self, sorted_ts, ts):
        """Return the index in sorted_ts whose value is closest to ts."""
        i = bisect.bisect_left(sorted_ts, ts)
        cands = [j for j in (i - 1, i) if 0 <= j < len(sorted_ts)]
        return min(cands, key=lambda j: abs(sorted_ts[j] - ts))

    def _get_calib_ego(self, sd_record):
        calib = self.calib_by_token[sd_record['calibrated_sensor_token']]
        ego   = self.ego_by_token[sd_record['ego_pose_token']]
        return calib, ego

    # ──────────────────────────────────────────────
    # Public: build the unified timeline
    # ──────────────────────────────────────────────
    def build(self):

        events = []
        events += self._build_lidar_events()
        events += self._build_radar_events()
        events += self._build_camera_events()
        events.sort(key=lambda e: e['timestamp'])
        return events

    def _build_lidar_events(self):
        path = ("/home/omar_ben_emad/3d_object_detection_and_tracking/Json_files_output/lidar_detections_xy.json")
        with open(path) as f:
            raw = json.load(f)

        events = []
        for fname, dets in raw.items():
            sd = self.lidar_fname_to_sd.get(fname)
            if sd is None:
                continue                    # sweep not in our calibration table, skip
            calib, ego = self._get_calib_ego(sd)

            objects = []
            for d in dets:
                # Transform from lidar sensor frame → global map frame
                # z=0 because the 3D detector output is projected to the XY plane
                import numpy as np
                p_sensor = np.array([d['x'], d['y'], 0.0])
                p_global = sensor_to_global(p_sensor, calib, ego)
                objects.append({
                    'class': d['class'],
                    'score': d['score'],
                    'x'    : float(p_global[0]),
                    'y'    : float(p_global[1]),
                })

            events.append({
                'timestamp': sd['timestamp'],
                'type'     : 'lidar',
                'source'   : fname,
                'objects'  : objects,
                'sd_token' : sd['token'],
            })
        return events

    def _build_radar_events(self):
        path = ("/home/omar_ben_emad/3d_object_detection_and_tracking/Json_files_output/radar_all_frames.json")
        with open(path) as f:
            raw = json.load(f)['frames']

        events = []
        for i, frame in enumerate(raw):
            if i >= len(self.radar_front_sd):
                break
            sd = self.radar_front_sd[i]
            calib, ego = self._get_calib_ego(sd)

            objects = []
            for p in frame['radar_points']:
                p_sensor = np.array([p['x'], p['y'], 0.0])
                p_global = sensor_to_global(p_sensor, calib, ego)
                v_global = vel_sensor_to_global(p['vx'], p['vy'], calib, ego)
                objects.append({
                    'x' : float(p_global[0]),
                    'y' : float(p_global[1]),
                    'vx': float(v_global[0]),
                    'vy': float(v_global[1]),
                })

            events.append({
                'timestamp': sd['timestamp'],
                'type'     : 'radar',
                'source'   : f'radar_frame_{i}',
                'objects'  : objects,
                'sd_token' : sd['token'],
            })
        return events

    def _build_camera_events(self):
        path = ("/home/omar_ben_emad/3d_object_detection_and_tracking/Json_files_output/camera_detections_xy.json")
        with open(path) as f:
            raw = json.load(f)

        events = []
        for dets, img_path in raw:
            # parse timestamp from filename:  ..._CAM_FRONT__<TIMESTAMP>.jpg
            fname_base = os.path.basename(img_path)
            ts = int(fname_base.split('__')[-1].split('.')[0])

            # find matching sample_data record for this camera frame
            idx = self._nearest_idx(self.cam_ts, ts)
            sd  = self.cam_front_sd[idx]
            calib, ego = self._get_calib_ego(sd)

            # store raw K so fusion.py can project 3D→pixel
            K = calib['camera_intrinsic']   # 3×3 list of lists

            objects = []
            for d in dets:
                objects.append({
                    'class': d['class'],
                    'conf' : d['conf'],
                    'px'   : d['center'],   # [u, v] pixel coordinates
                })

            events.append({
                'timestamp'  : ts,
                'type'       : 'camera',
                'source'     : img_path,
                'objects'    : objects,
                'sd_token'   : sd['token'],
                'K'          : K,
                'calib'      : calib,
                'ego'        : ego,
            })
        return events
    
if __name__ == "__main__":

    calib_dir = "/home/omar_ben_emad/3d_object_detection_and_tracking/data/v1.0-mini/Calibration Files"
    detections_dir = "/home/omar_ben_emad/3d_object_detection_and_tracking/Json_files_output"
    builder = TimelineBuilder(calib_dir, detections_dir)
    timeline = builder.build()

    output_path = os.path.join(detections_dir, "data_with_timestamps.json")
    with open(output_path, "w") as f:
        json.dump(timeline, f, indent=2)

    print(f"Timeline built successfully. Total events: {len(timeline)}")
    print(f"Saved to: {output_path}")
