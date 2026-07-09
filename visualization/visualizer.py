import os
import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import matplotlib
matplotlib.use('Agg')
from geometric_fusion.transforms import global_to_sensor, project_to_image
from collections import Counter
from datetime import datetime

_COLOURS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
]

MAX_TIME_GAP = 0.5

def _colour(track_id):
    idx = int(track_id.split('_')[-1]) % len(_COLOURS)
    return _COLOURS[idx]

def _colour_bgr(track_id):
    colors = [
        (0,0,255),(0,255,0),(255,0,0),
        (0,255,255),(255,0,255),(255,255,0),
        (128,128,255)
    ]
    return colors[int(track_id.split('_')[-1]) % len(colors)]


def _nearest_snap(snaps, ts):
    ts_list = [s['timestamp'] for s in snaps]
    idx = bisect.bisect_left(ts_list, ts)
    cands = [j for j in (idx-1, idx) if 0 <= j < len(snaps)]
    return snaps[min(cands, key=lambda j: abs(ts_list[j] - ts))]


def camera_overlay(image_path, tracks, camera_event,
                   output_path=None, min_updates=2):


    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    K     = np.array(camera_event['K'])
    calib = camera_event['calib']
    ego   = camera_event['ego']
    ts    = camera_event['timestamp']

    if K.size == 0:
        return

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(img)
    ax.axis('off')

    for t in tracks:
        if t['n_updates'] < min_updates:
            continue

        # Find the track's state at this camera timestamp
        snaps = sorted(
            [h for h in t['history'] if h['sensor'] in ('lidar', 'radar', 'birth(lidar)')],
            key=lambda h: h['timestamp']
        )
        if not snaps:
            continue
        snap = _nearest_snap(snaps, ts)

        p_global = np.array([snap['x'], snap['y'], 0.0])
        p_cam    = global_to_sensor(p_global, calib, ego)
        uv       = project_to_image(p_cam, K)
        if uv is None:
            continue

        u, v = float(uv[0]), float(uv[1])
        h, w  = img.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            continue     # projected off-screen

        col = _colour(t['id'])

        # Circle at projected position
        circ = plt.Circle((u, v), radius=8, color=col, fill=False, linewidth=1.5)
        ax.add_patch(circ)

        # Label
        ax.text(u + 10, v - 10, f"{t['id']}", fontsize=7, color=col,
                bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.5, ec='none'))

        # Velocity arrow (scaled to pixels; ~5 px per m/s)
        vx, vy = snap.get('vx', 0), snap.get('vy', 0)
        speed = np.hypot(vx, vy)
        if speed > 0.5:
            # Project velocity tip as well
            v_tip_global = np.array([snap['x'] + vx, snap['y'] + vy, 0.0])
            v_tip_cam    = global_to_sensor(v_tip_global, calib, ego)
            uv_tip       = project_to_image(v_tip_cam, K)
            if uv_tip is not None:
                ax.annotate('', xy=(float(uv_tip[0]), float(uv_tip[1])),
                            xytext=(u, v),
                            arrowprops=dict(arrowstyle='->', color=col, lw=1.5))

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches='tight')
        print(f"Saved camera overlay → {output_path}")
    else:
        plt.show()
    plt.close(fig)
    
def camera_overlay_video(events, tracks, output_path, fps=10, min_updates=3):

    cam_events = [e for e in events if e['type'] == 'camera']
    cam_events.sort(key=lambda e: e['timestamp'])

    if not cam_events:
        print("No camera events")
        return

    # --- preprocess tracks ---
    track_snaps = {}
    for t in tracks:
        if t['n_updates'] < min_updates:
            continue

        snaps = sorted(
            [h for h in t['history']
             if h['sensor'] in ('lidar','radar','birth(lidar)')],
            key=lambda h: h['timestamp']
        )
        if snaps:
            track_snaps[t['id']] = snaps

    writer = None

    for cam_ev in cam_events:

        img_path = cam_ev.get('source')
        if not img_path or not os.path.isfile(img_path):
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        K = np.array(cam_ev['K'])
        if K.size == 0:
            continue

        calib = cam_ev['calib']
        ego   = cam_ev['ego']
        ts    = cam_ev['timestamp']
        h, w  = frame.shape[:2]

        detections = cam_ev.get("detections", [])

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])

            track_id = det.get("track_id", None)
            col = (0,255,0) if track_id is None else _colour_bgr(track_id)
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)

            if track_id:
                cv2.putText(frame, track_id,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, col, 1, cv2.LINE_AA)

        for track_id, snaps in track_snaps.items():

            snap = _nearest_snap(snaps, ts)

            if abs(snap['timestamp'] - ts) > MAX_TIME_GAP:
                continue

            # --- project ---
            p = np.array([snap['x'], snap['y'], 0.0])
            p_cam = global_to_sensor(p, calib, ego)
            uv = project_to_image(p_cam, K)

            if uv is None:
                continue

            u, v = int(uv[0]), int(uv[1])
            if not (0 <= u < w and 0 <= v < h):
                continue

            col = _colour_bgr(track_id)
            cv2.circle(frame, (u, v), 4, (255,255,255), -1)
            cv2.circle(frame, (u, v), 6, col, 2)

            # --- velocity ---
            vx, vy = snap.get('vx', 0), snap.get('vy', 0)

            if np.hypot(vx, vy) > 0.5:
                tip = np.array([snap['x']+vx, snap['y']+vy, 0.0])
                tip_cam = global_to_sensor(tip, calib, ego)
                uv_tip = project_to_image(tip_cam, K)

                if uv_tip is not None:
                    cv2.arrowedLine(frame,
                                    (u, v),
                                    (int(uv_tip[0]), int(uv_tip[1])),
                                    col, 2)

            label = (
                f"{track_id} | "
                f"x={snap['x']:.1f} y={snap['y']:.1f} "
                f"vx={vx:.1f} vy={vy:.1f}"
            )

            (tw, th), _ = cv2.getTextSize(label,
                                         cv2.FONT_HERSHEY_SIMPLEX,0.45, 1)

            text_x = u + 10
            text_y = v - 10

            # background box
            cv2.rectangle(frame,(text_x, text_y - th - 4),(text_x + tw, text_y),(0,0,0),-1)
            # text
            cv2.putText(frame,label,(text_x, text_y - 2),cv2.FONT_HERSHEY_SIMPLEX,0.45,col,1,cv2.LINE_AA)                 
                        
        cv2.putText(frame,
                    f"t={ts:.2f}s",(10, 25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        writer.write(frame)

    if writer:
        writer.release()
        print(f"Saved video → {output_path}")

def plot_track_stats(tracks, output_path=None):

    counts = Counter(t['class'] for t in tracks if t['n_updates'] >= 3)
    if not counts:
        print("No tracks with >= 3 updates to plot.")
        return

    classes = list(counts.keys())
    values  = [counts[c] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classes, values,
                  color=[_COLOURS[i % len(_COLOURS)] for i in range(len(classes))],
                  edgecolor='black', linewidth=0.5)
    ax.bar_label(bars)
    ax.set_xlabel('Object class')
    ax.set_ylabel('Number of tracks (≥3 updates)')
    ax.set_title('Tracks per class')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=120)
        print(f"Saved stats plot → {output_path}")
    else:
        plt.show()
    plt.close(fig)



def plot_kf(history, output_dir=None, prefix="kf_plot", show=False):
    pred = np.array(history["predicted"])
    meas = np.array(history["measurement"])
    err  = np.array(history["residual"])

    if len(pred) == 0:
        print("No KF data to plot")
        return

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure()
    plt.title("Prediction vs Measurement")
    plt.plot(pred[:, 0], label="Predicted")
    plt.plot(meas[:, 0], label="Measured")
    plt.legend()

    if output_dir:
        path1 = os.path.join(output_dir, f"{prefix}_pred_vs_meas_{timestamp}.png")
        plt.savefig(path1)
        print(f"Saved: {path1}")

    if show:
        plt.show()
    else:
        plt.close()
    plt.figure()
    plt.title("Residual Error")
    plt.plot(err[:, 0], label="Error")
    plt.legend()

    if output_dir:
        path2 = os.path.join(output_dir, f"{prefix}_residual_{timestamp}.png")
        plt.savefig(path2)
        print(f"Saved: {path2}")

    if show:
        plt.show()
    else:
        plt.close()
