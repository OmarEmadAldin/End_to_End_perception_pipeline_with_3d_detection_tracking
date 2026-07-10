import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kalman_filter.tracker     import MultiObjectTracker
from evaluation.evaluator      import Evaluator
from visualization.visualizer  import *
from _helper import *



def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    events = load_events(args)
    print(f"\nRunning {'EKF' if args.ekf else 'Linear KF'} tracker...")
    tracker = MultiObjectTracker(
        use_ekf=args.ekf,
        use_polar=(args.polar and args.ekf)
    )

    tracks = tracker.run(events)
    print(f"Total tracks: {len(tracks)}")
    tracks_path = os.path.join(OUTPUT_DIR, 'tracks_output.json')
    with open(tracks_path, 'w') as f:
        json.dump(tracks, f, indent=2)

    print(f"Saved {tracks_path}")

    # for track in tracks:
    #     history = track.get("history", [])

    #     if len(history) == 0:
    #         continue
    #     kf_history = convert_history_for_plot(history)
    #     plot_kf(
    #         kf_history,
    #         output_dir=OUTPUT_DIR,
    #         prefix=track["id"],
    #         show=False
    #     )

    if not args.no_eval:
        print("\nEvaluation")
        try:
            evaluator = Evaluator(calib_dir=CALIB_DIR,categories=None,threshold_m=2.0,)
            results = evaluator.evaluate(tracks)
            evaluator.print_results(results)

            eval_path = os.path.join(OUTPUT_DIR, 'evaluation_results.json')
            with open(eval_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved → {eval_path}")

        except FileNotFoundError as e:
            print(f"Ground truth missing: {e}")
    else:
        print("Evaluation skipped")

    if not args.no_viz:

        print("\nGenerating camera tracking video...")
        video_path = os.path.join(OUTPUT_DIR, 'tracking_video.mp4')

        camera_overlay_video(
            events,
            tracks,
            output_path=video_path,
            min_updates=3,
            fps=10
        )

        print(f"Saved video → {video_path}")


        
    else:
        print("Visualization skipped")
    print("\nAll outputs saved to:", OUTPUT_DIR)

if __name__ == '__main__':
    main()