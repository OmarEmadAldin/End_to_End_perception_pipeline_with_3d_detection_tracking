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

_COLOURS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
]

def _colour(track_id):
    idx = int(track_id.split('_')[-1]) % len(_COLOURS)
    return _COLOURS[idx]


def bird_eye_view(tracks, output_path=None, title='Track trajectories (global frame)',
                  min_updates=3):

    fig, ax = plt.subplots(figsize=(14, 14))
    legend_handles = []
    for t in tracks:
        if t['n_updates'] < min_updates:
            continue

        # Extract metric snapshots (lidar/radar only — have real x,y)
        snaps = [h for h in t['history'] if h['sensor'] in ('lidar', 'radar', 'birth(lidar)')]
        if len(snaps) < 2:
            continue
        snaps.sort(key=lambda h: h['timestamp'])

        xs = [h['x'] for h in snaps]
        ys = [h['y'] for h in snaps]
        col = _colour(t['id'])

        # Trajectory line
        ax.plot(xs, ys, '-', color=col, linewidth=0.8, alpha=0.7)

        # Latest position dot
        ax.scatter(xs[-1], ys[-1], color=col, s=30, zorder=5)

        # Velocity arrow at last position
        last = snaps[-1]
        if abs(last.get('vx', 0)) + abs(last.get('vy', 0)) > 0.2:
            ax.annotate('',
                xy=(last['x'] + last['vx'], last['y'] + last['vy']),
                xytext=(last['x'], last['y']),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.5)
            )

        # Track ID label
        ax.text(xs[-1] + 0.3, ys[-1] + 0.3, t['id'],
                fontsize=5, color=col, zorder=6)

        legend_handles.append(
            mpatches.Patch(color=col, label=f"{t['id']} ({t['class']})")
        )

    ax.set_xlabel('X (global frame, metres)', fontsize=11)
    ax.set_ylabel('Y (global frame, metres)', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_aspect('equal')
    ax.grid(True, linewidth=0.3, alpha=0.5)

    if legend_handles:
        ax.legend(handles=legend_handles[:30], fontsize=5,
                  loc='upper right', ncol=2)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved bird's-eye view → {output_path}")
    else:
        plt.show()
    plt.close(fig)


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


def _nearest_snap(snaps, ts):
    """Return the snap whose timestamp is closest to ts."""
    ts_list = [s['timestamp'] for s in snaps]
    idx = bisect.bisect_left(ts_list, ts)
    cands = [j for j in (idx - 1, idx) if 0 <= j < len(snaps)]
    return snaps[min(cands, key=lambda j: abs(ts_list[j] - ts))]

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
