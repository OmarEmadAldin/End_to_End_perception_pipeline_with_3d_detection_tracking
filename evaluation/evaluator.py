import json
import os
import bisect
import numpy as np

from evaluation.metrics import MOTMetrics


class Evaluator:
    def __init__(self, calib_dir, categories=None, threshold_m=2.0):
        self.calib_dir  = calib_dir
        self.categories = set(categories) if categories else None
        self.threshold  = threshold_m
        self._load_ground_truth()

    def _load(self, name):
        with open(os.path.join(self.calib_dir, name)) as f:
            return json.load(f)

    def _load_ground_truth(self):
        samples     = self._load('/home/omar_ben_emad/3d_object_detection_and_tracking/data/v1.0-mini/Calibration Files/sample.json')
        annotations = self._load('/home/omar_ben_emad/3d_object_detection_and_tracking/data/v1.0-mini/Calibration Files/sample_annotation.json')
        instances   = self._load('/home/omar_ben_emad/3d_object_detection_and_tracking/data/v1.0-mini/Calibration Files/instance.json')
        categories  = self._load('/home/omar_ben_emad/3d_object_detection_and_tracking/data/v1.0-mini/Calibration Files/category.json')

        cat_by_token  = {c['token']: c['name'] for c in categories}
        inst_by_token = {i['token']: i for i in instances}

        # Map sample_token → list of GT objects
        self.gt_by_sample = {}
        for ann in annotations:
            st = ann['sample_token']
            inst = inst_by_token.get(ann['instance_token'], {})
            cat_name = cat_by_token.get(inst.get('category_token', ''), 'unknown')

            # Optional category filter
            if self.categories:
                # nuScenes uses 'vehicle.car' etc.; we compare the suffix
                short = cat_name.split('.')[-1]
                if short not in self.categories and cat_name not in self.categories:
                    continue

            obj = {
                'id'   : ann['instance_token'],   # stable across frames
                'x'    : ann['translation'][0],
                'y'    : ann['translation'][1],
                'class': cat_name,
            }
            self.gt_by_sample.setdefault(st, []).append(obj)

        # Map sample timestamp → sample_token (for time-based lookup)
        self.sample_ts_to_token = {s['timestamp']: s['token'] for s in samples}
        self.sample_tokens_sorted = sorted(
            samples, key=lambda s: s['timestamp']
        )
        self.sample_ts_list = [s['timestamp'] for s in self.sample_tokens_sorted]

    def evaluate(self, tracks):
        track_histories = self._build_track_histories(tracks)

        metrics = MOTMetrics(threshold_m=self.threshold)

        for sample in self.sample_tokens_sorted:
            ts    = sample['timestamp']
            token = sample['token']

            gt_objs = self.gt_by_sample.get(token, [])

            # Find tracker predictions at this timestamp
            pred_objs = []
            for tid, hist_ts, hist_xy in track_histories:
                xy = self._interpolate(hist_ts, hist_xy, ts)
                if xy is not None:
                    pred_objs.append({'id': tid, 'x': xy[0], 'y': xy[1]})

            metrics.add_frame(gt_objs, pred_objs)

        results = metrics.compute()
        return results

    def _build_track_histories(self, tracks):
 
        out = []
        for t in tracks:
            metric_snaps = [
                h for h in t['history']
                if h['sensor'] in ('lidar', 'radar', 'birth(lidar)')
            ]
            if len(metric_snaps) < 2:
                continue
            metric_snaps.sort(key=lambda h: h['timestamp'])
            ts_list = [h['timestamp'] for h in metric_snaps]
            xy_list = np.array([[h['x'], h['y']] for h in metric_snaps])
            out.append((t['id'], ts_list, xy_list))
        return out

    def _interpolate(self, hist_ts, hist_xy, query_ts,
                     max_extrap_us=500_000):   # 0.5 s extrapolation limit

        if not hist_ts:
            return None
        t_min, t_max = hist_ts[0], hist_ts[-1]
        if query_ts < t_min - max_extrap_us:
            return None
        if query_ts > t_max + max_extrap_us:
            return None

        # Find surrounding indices
        idx = bisect.bisect_left(hist_ts, query_ts)
        if idx == 0:
            return hist_xy[0]
        if idx >= len(hist_ts):
            return hist_xy[-1]

        t0, t1 = hist_ts[idx - 1], hist_ts[idx]
        if t1 == t0:
            return hist_xy[idx]

        alpha = (query_ts - t0) / (t1 - t0)
        return hist_xy[idx - 1] * (1 - alpha) + hist_xy[idx] * alpha

    def print_results(self, results):
        for k, v in results.items():
            print(f"  {k:<12}: {v}")
        print("="*45 + "\n")
