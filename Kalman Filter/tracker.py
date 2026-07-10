import numpy as np
from scipy.optimize import linear_sum_assignment

from kalman_filter.linear_kf import LinearKalmanFilter
from kalman_filter.ekf import ExtendedKalmanFilter
from geometric_fusion.transforms import global_to_sensor, project_to_image
from hungarian_algorithm.hungarian_matching import CrossSensorFusion

LIDAR_GATE_M   = 3.0    # metres — max dist for lidar→track match
RADAR_GATE_M   = 5.0    # metres — max dist for radar→track match
CAMERA_GATE_PX = 80.0   # pixels — max dist for camera→track match
MAX_MISS_SEC   = 1.0    # seconds — delete track if no update this long
SCENE_BREAK_SEC = 5.0   # seconds — gap this large means a new scene
MIN_LIDAR_SCORE = 0.3   # minimum detection confidence to spawn a track


class Track:
    
    _next_id = 0

    def __init__(self, px, py, vx, vy, timestamp_us, cls='unknown', use_ekf=False):
        self.id  = f"{cls}_{Track._next_id}"; Track._next_id += 1
        self.cls = cls

        if use_ekf:
            self.kf = ExtendedKalmanFilter(px, py, vx, vy)
        else:
            self.kf = LinearKalmanFilter(px, py, vx, vy)

        self.last_ts   = timestamp_us   # microseconds
        self.birth_ts  = timestamp_us
        self.n_updates = 0
        self.history   = []             # list of snapshot dicts

    def predict(self, timestamp_us):
        """Advance the filter to this timestamp."""
        dt = (timestamp_us - self.last_ts) / 1e6   # µs → s
        if dt > 0:
            self.kf.predict(dt)
            self.last_ts = timestamp_us

    def update_lidar(self, px, py, timestamp_us, score=None):
        self.kf.update_lidar(px, py)
        self._record('lidar', timestamp_us, score=score)

    def update_radar(self, px, py, vx, vy, timestamp_us,
                    use_polar=False, rho=None, phi=None, rhodot=None):

        if use_polar and isinstance(self.kf, ExtendedKalmanFilter):
            # EXPECT polar directly (no conversion fallback)
            if rho is None or phi is None or rhodot is None:
                raise ValueError("Polar radar update requires rho, phi, rhodot")

            self.kf.update_radar_polar(rho, phi, rhodot)

        else:
            # Linear KF OR non-polar mode
            if px is None or py is None:
                raise ValueError("Linear radar update requires x, y")

            self.kf.update_radar(px, py, vx, vy)

        self._record('radar', timestamp_us, vx=vx, vy=vy)
    def confirm_camera(self, cls, conf, px_uv, timestamp_us):
        """Camera provides class label and confidence — no metric update."""
        self.cls = cls
        self._record('camera', timestamp_us, conf=conf, px=px_uv)

    def _record(self, sensor, timestamp_us, **extras):
        self.n_updates += 1
        snap = {
            'timestamp': timestamp_us,
            'sensor'   : sensor,
            'x'        : float(self.kf.state[0]),
            'y'        : float(self.kf.state[1]),
            'vx'       : float(self.kf.state[2]),
            'vy'       : float(self.kf.state[3]),
        }
        snap.update(extras)
        self.history.append(snap)

    @property
    def pos(self):
        return self.kf.position    # [px, py]

    def to_dict(self):
        return {
            'id'          : self.id,
            'class'       : self.cls,
            'n_updates'   : self.n_updates,
            'birth_ts'    : self.birth_ts,
            'final_state' : {
                'x' : float(self.kf.state[0]),
                'y' : float(self.kf.state[1]),
                'vx': float(self.kf.state[2]),
                'vy': float(self.kf.state[3]),
            },
            'history': self.history,
        }


class MultiObjectTracker:
    

    def __init__(self, use_ekf=False, use_polar=False):
        self.use_ekf   = use_ekf
        self.use_polar = use_polar
        self.tracks    = []
        self.dead      = []          # completed tracks (for evaluation)
        self._last_ts  = None
        self._fuser    = CrossSensorFusion()   # used inside _process_camera

    def run(self, events):
      
        for ev in events:
            ts = ev['timestamp']

            # Scene boundary detection: large time gap → reset tracker
            if self._last_ts is not None:
                gap_sec = (ts - self._last_ts) / 1e6
                if gap_sec > SCENE_BREAK_SEC:
                    self._flush_all_tracks()

            self._last_ts = ts

            # Predict every active track forward to this event's time
            for t in self.tracks:
                t.predict(ts)

            # Process by sensor type
            if ev['type'] == 'lidar':
                self._process_lidar(ev)
            elif ev['type'] == 'radar':
                self._process_radar(ev)
            elif ev['type'] == 'camera':
                self._process_camera(ev)

            # Prune stale tracks
            alive, dead = [], []
            for t in self.tracks:
                if (ts - t.last_ts) / 1e6 < MAX_MISS_SEC:
                    alive.append(t)
                else:
                    dead.append(t)
            self.tracks = alive
            self.dead.extend(dead)

        self._flush_all_tracks()

        return [t.to_dict() for t in self.dead]

    def _process_lidar(self, ev):
        ts   = ev['timestamp']
        objs = ev['objects']

        pairs, unmatched_o = self._associate_metric(objs, LIDAR_GATE_M)

        for ti, oi in pairs:
            o = objs[oi]
            self.tracks[ti].update_lidar(o['x'], o['y'], ts, score=o.get('score'))

        for oi in unmatched_o:
            o = objs[oi]
            if o.get('score', 1.0) >= MIN_LIDAR_SCORE:
                tr = Track(o['x'], o['y'], 0.0, 0.0, ts,
                           cls=o.get('class', 'unknown'),
                           use_ekf=self.use_ekf)
                tr.update_lidar(o['x'], o['y'], ts, score=o.get('score'))
                self.tracks.append(tr)

    def _process_radar(self, ev):
        ts   = ev['timestamp']
        objs = ev['objects']

        assoc_objs = []
        for o in objs:
            if 'x' in o and 'y' in o:
                assoc_objs.append(o)
            else:
                # derive temporary Cartesian for matching only
                x = o['rho'] * np.cos(o['phi'])
                y = o['rho'] * np.sin(o['phi'])
                assoc_objs.append({**o, 'x': x, 'y': y})

        # -------- STEP 2: associate --------
        pairs, _ = self._associate_metric(assoc_objs, RADAR_GATE_M)

        # -------- STEP 3: update tracks --------
        for ti, oi in pairs:
            o = objs[oi]  # original (NOT assoc one)

            if self.use_polar:
                self.tracks[ti].update_radar(
                    None, None, None, None, ts,
                    use_polar=True,
                    rho=o['rho'],
                    phi=o['phi'],
                    rhodot=o['rhodot']
                )
            else:
                self.tracks[ti].update_radar(
                    o['x'], o['y'], o['vx'], o['vy'], ts
                )

    def _process_camera(self, ev):
        
        ts    = ev['timestamp']
        objs  = ev['objects']
        calib = ev.get('calib')
        ego   = ev.get('ego')
        K     = ev.get('K')

        if not (calib and ego and K):
            return
        K = np.array(K)
        if K.size == 0:
            return

        nearby_lidar = ev.get('nearby_lidar', [])
        nearby_radar = ev.get('nearby_radar', [])

        fused = self._fuser.fuse(ev, nearby_lidar, nearby_radar)
        lidar_det_idxs = [i for i, f in enumerate(fused) if f['lidar_x'] is not None]

        lidar_objs_metric = [
            {'x': fused[i]['lidar_x'], 'y': fused[i]['lidar_y'],
             'class': fused[i]['class'], 'score': fused[i].get('lidar_score', 1.0),
             'fused_idx': i}
            for i in lidar_det_idxs
        ]

        # Hungarian in metric space (same logic as _process_lidar)
        if lidar_objs_metric:
            pairs_l, unmatched_l = self._associate_metric(lidar_objs_metric, LIDAR_GATE_M)

            for ti, oi in pairs_l:
                o = lidar_objs_metric[oi]
                fi = o['fused_idx']
                self.tracks[ti].update_lidar(o['x'], o['y'], ts, score=o['score'])
                self.tracks[ti].cls = fused[fi]['class']   # update class from camera

                f = fused[fi]
                if f['radar_vx'] is not None:
                    self.tracks[ti].update_radar(
                        f['radar_x'], f['radar_y'],
                        f['radar_vx'], f['radar_vy'],
                        ts, use_polar=self.use_polar,
                        rho=f.get('radar_rho'), phi=f.get('radar_phi'),
                        rhodot=f.get('radar_rhodot'),
                    )

            # Spawn new tracks from unmatched lidar-confirmed detections
            for oi in unmatched_l:
                o = lidar_objs_metric[oi]
                fi = o['fused_idx']
                if o.get('score', 1.0) >= MIN_LIDAR_SCORE:
                    tr = Track(o['x'], o['y'], 0.0, 0.0, ts,
                               cls=o.get('class', 'unknown'),
                               use_ekf=self.use_ekf)
                    tr.update_lidar(o['x'], o['y'], ts, score=o['score'])
                    # ── STEP 3b: radar update on newly spawned track ──
                    f = fused[fi]
                    if f['radar_vx'] is not None:
                        tr.update_radar(
                            f['radar_x'], f['radar_y'],
                            f['radar_vx'], f['radar_vy'],
                            ts, use_polar=self.use_polar,
                            rho=f.get('radar_rho'), phi=f.get('radar_phi'),
                            rhodot=f.get('radar_rhodot'),
                        )
                    self.tracks.append(tr)


        nd = len(objs)
        nt = len(self.tracks)
        if nd == 0 or nt == 0:
            return

        track_uvs = []
        for t in self.tracks:
            p_global = np.array([t.pos[0], t.pos[1], 0.0])
            p_cam    = global_to_sensor(p_global, calib, ego)
            uv       = project_to_image(p_cam, K)
            track_uvs.append(uv)

        C = np.full((nt, nd), np.inf)
        for ti, uv in enumerate(track_uvs):
            if uv is None:
                continue
            for di, det in enumerate(objs):
                center = np.array(det['px'], dtype=float)
                dist   = np.linalg.norm(uv - center)
                if dist < CAMERA_GATE_PX:
                    C[ti, di] = dist

        if not np.all(np.isinf(C)):
            C_finite = np.where(np.isinf(C), CAMERA_GATE_PX * 10, C)
            row_ind, col_ind = linear_sum_assignment(C_finite)
            for ri, ci in zip(row_ind, col_ind):
                if C[ri, ci] < CAMERA_GATE_PX:
                    det = objs[ci]
                    self.tracks[ri].confirm_camera(
                        det['class'], det['conf'], det['px'], ts
                    )

    def _associate_metric(self, objs, gate_m):

        if not self.tracks or not objs:
            return [], set(range(len(objs)))

        nt = len(self.tracks)
        nd = len(objs)
        C  = np.full((nt, nd), np.inf)

        for ti, t in enumerate(self.tracks):
            for oi, o in enumerate(objs):
                dist = np.hypot(t.pos[0] - o['x'], t.pos[1] - o['y'])
                if dist < gate_m:
                    C[ti, oi] = dist

        pairs = []
        if not np.all(np.isinf(C)):
            C_finite        = np.where(np.isinf(C), gate_m * 10, C)
            row_ind, col_ind = linear_sum_assignment(C_finite)
            for ri, ci in zip(row_ind, col_ind):
                if C[ri, ci] < gate_m:
                    pairs.append((ri, ci))

        matched_o   = {ci for _, ci in pairs}
        unmatched_o = set(range(nd)) - matched_o
        return pairs, unmatched_o

    def _flush_all_tracks(self):
        self.dead.extend(self.tracks)
        self.tracks = []