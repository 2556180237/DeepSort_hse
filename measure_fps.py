import os
import time
import numpy as np
import deep_sort_app

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
DETECTIONS_DIR = os.path.join(BASE_DIR, "resources", "detections")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def measure_fps(seq_name):
    seq_dir = os.path.join(VIDEOS_DIR, seq_name)
    det_file = os.path.join(DETECTIONS_DIR, f"{seq_name}.npy")
    output_file = os.path.join(OUTPUT_DIR, f"{seq_name}_fps.txt")

    seq_info = deep_sort_app.gather_sequence_info(seq_dir, det_file)
    num_frames = seq_info["max_frame_idx"] - seq_info["min_frame_idx"] + 1

    start = time.time()
    deep_sort_app.run(
        seq_dir, det_file, output_file,
        min_confidence=0.3, nms_max_overlap=1.0,
        min_detection_height=0, max_cosine_distance=0.2,
        nn_budget=100, display=False
    )
    elapsed = time.time() - start

    fps = num_frames / elapsed if elapsed > 0 else 0
    return num_frames, elapsed, fps


def main():
    sequences = ["TUD-Campus", "TUD-Stadtmitte", "KITTI-17",
                 "PETS09-S2L1", "MOT16-09", "MOT16-11"]

    print("=" * 65)
    print(f"{'Video':<16} {'Frames':>8} {'Time(s)':>10} {'FPS':>10}")
    print("=" * 65)

    all_fps = []
    for seq in sequences:
        frames, elapsed, fps = measure_fps(seq)
        all_fps.append(fps)
        print(f"{seq:<16} {frames:>8d} {elapsed:>10.2f} {fps:>10.2f}")

    print("-" * 65)
    print(f"{'AVERAGE':<16} {'':>8} {'':>10} {np.mean(all_fps):>10.2f}")
    print("=" * 65)

    # Save
    results_file = os.path.join(OUTPUT_DIR, "baseline_fps.txt")
    with open(results_file, 'w') as f:
        f.write(f"{'Video':<16} {'Frames':>8} {'Time(s)':>10} {'FPS':>10}\n")
        f.write("=" * 65 + "\n")
        for seq in sequences:
            frames, elapsed, fps = measure_fps(seq)
            f.write(f"{seq:<16} {frames:>8d} {elapsed:>10.2f} {fps:>10.2f}\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'AVERAGE':<16} {'':>8} {'':>10} {np.mean(all_fps):>10.2f}\n")
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
