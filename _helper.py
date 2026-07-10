import os
import argparse
import json

CALIB_DIR   = '/home/omar_ben_emad/3d_object_detection_and_tracking/data/v1.0-mini/Calibration Files'
DETECT_DIR  = '/home/omar_ben_emad/3d_object_detection_and_tracking/Json_files_output'
OUTPUT_DIR  = '/home/omar_ben_emad/3d_object_detection_and_tracking/output/Final_tracking'

TIMELINE_LINEAR = os.path.join(DETECT_DIR, 'data_with_timestamps.json')
TIMELINE_EKF    = os.path.join(DETECT_DIR, 'data_with_timestamps_for_EKF.json')
TRACKS_OUT   = os.path.join(OUTPUT_DIR, 'tracks_output.json')
EVAL_OUT     = os.path.join(OUTPUT_DIR, 'evaluation_results.json')


def convert_history_for_plot(track_history):
    predicted = []
    measurement = []
    residual = []

    last_meas = None

    for h in track_history:
        # state = predicted
        pred = [h['x'], h['y']]
        predicted.append(pred)

        # measurement depends on sensor
        if h['sensor'] in ['lidar', 'radar']:
            meas = [h['x'], h['y']]
            last_meas = meas
        else:
            meas = last_meas if last_meas is not None else pred

        measurement.append(meas)

        # residual = pred - meas
        res = [pred[0] - meas[0], pred[1] - meas[1]]
        residual.append(res)

    return {
        "predicted": predicted,
        "measurement": measurement,
        "residual": residual
    }


def parse_args():
    p = argparse.ArgumentParser(description='Multi-sensor KF tracker for nuScenes')
    p.add_argument('--ekf',      action='store_true',
                   help='Use Extended Kalman Filter instead of Linear KF')
    p.add_argument('--polar',    action='store_true',
                   help='Use polar radar measurements in EKF (requires --ekf)')
    p.add_argument('--no-eval',  action='store_true',
                   help='Skip evaluation against ground truth')
    p.add_argument('--no-viz',   action='store_true',
                   help='Skip visualization outputs')
    return p.parse_args()


def load_events(args):
    
    path = TIMELINE_EKF if args.ekf else TIMELINE_LINEAR
    with open(path) as f:
        events = json.load(f)

    print(f"Total events: {len(events)}")
    radar_objs = next((e['objects'] for e in events
                        if e['type'] == 'radar' and e['objects']), [])
    if radar_objs:
        sample = radar_objs[0]
        missing_xy = 'x' not in sample or 'y' not in sample
        missing_polar = args.ekf and args.polar and (
            'rho' not in sample or 'phi' not in sample or 'rhodot' not in sample)
        
        if missing_xy and not (args.ekf and args.polar):
            raise ValueError(
                f"{path} radar objects are missing 'x'/'y' -- required for "
                "association and camera fusion in ANY filter mode. "
                "This looks like a polar-only timeline file."
            )
        if missing_polar:
            raise ValueError(
                f"{path} radar objects are missing 'rho'/'phi'/'rhodot' -- "
                "required for --ekf --polar. This looks like a Cartesian-"
                "only timeline file."
            )

    return events