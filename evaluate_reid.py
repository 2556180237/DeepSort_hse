"""Evaluate REID models independently using ground-truth bounding boxes.

Instead of running the full DeepSORT pipeline, this script:
1. Loads ground-truth bounding boxes from MOT Challenge
2. Extracts REID features for each GT box
3. Associates detections across frames using cosine distance matching
4. Computes HOTA, MOTA, IDF1 metrics

This isolates the REID model's contribution from the detector's quality.

Usage:
    python evaluate_reid.py --reid osnet_x1_0
    python evaluate_reid.py --reid osnet_x1_0 --sequences TUD-Campus
    python evaluate_reid.py --list_reid
"""
import argparse
import os
import time

import cv2
import numpy as np

from reid import create_reid, list_reid_models
from evaluate_hota import compute_hota_per_frame, compute_motmetrics, load_mot_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def load_gt_boxes(gt_file):
    """Load ground-truth boxes grouped by frame.

    Returns dict: frame_idx -> list of (track_id, x, y, w, h)
    """
    if not os.path.exists(gt_file):
        return {}
    data = np.loadtxt(gt_file, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    gt_by_frame = {}
    for row in data:
        frame = int(row[0])
        track_id = int(row[1])
        bbox = row[2:6]
        if frame not in gt_by_frame:
            gt_by_frame[frame] = []
        gt_by_frame[frame].append((track_id, bbox))
    return gt_by_frame


def run_reid_tracking(sequence_dir, reid_model, max_cosine_distance=0.2,
                      nn_budget=100):
    """Run tracking using GT boxes + REID features (no detector, no SORT).

    Uses simple nearest-neighbor association with cosine distance.
    """
    from deep_sort import nn_matching
    from deep_sort.detection import Detection
    from deep_sort.tracker import Tracker
    from application_util import preprocessing

    gt_file = os.path.join(sequence_dir, "gt", "gt.txt")
    gt_by_frame = load_gt_boxes(gt_file)

    img_dir = os.path.join(sequence_dir, "img1")
    image_files = sorted(
        [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(os.path.splitext(x)[0]))

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    total_frames = len(image_files)
    start_time = time.time()

    for i, img_file in enumerate(image_files):
        frame_idx = int(os.path.splitext(img_file)[0])
        print(f"\rFrame {frame_idx:05d} ({i+1}/{total_frames})", end="", flush=True)

        image = cv2.imread(os.path.join(img_dir, img_file), cv2.IMREAD_COLOR)
        if image is None:
            continue

        gt_entries = gt_by_frame.get(frame_idx, [])
        if len(gt_entries) == 0:
            tracker.predict()
            tracker.update([])
        else:
            boxes = [entry[1] for entry in gt_entries]
            features = reid_model.extract_features(image, boxes)
            detections = []
            for (_, bbox), feat in zip(gt_entries, features):
                detections.append(Detection(bbox, 1.0, feat))

            boxes_arr = np.array([d.tlwh for d in detections])
            if len(boxes_arr) > 0:
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(
                    boxes_arr, 1.0, scores)
                detections = [detections[i] for i in indices]

            tracker.predict()
            tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    elapsed = time.time() - start_time
    fps = total_frames / elapsed if elapsed > 0 else 0
    print(f"\nDone: {total_frames} frames in {elapsed:.2f}s ({fps:.2f} FPS)")

    return results, fps


def evaluate_reid_model(reid_name, sequences, device="cpu", max_cosine_distance=0.2):
    """Evaluate a REID model on all sequences using GT boxes."""
    print(f"\n{'='*60}")
    print(f"REID model: {reid_name}")
    print(f"{'='*60}")

    reid_model = create_reid(reid_name, device=device)
    print(f"Loading {reid_model}...")
    reid_model.load_model()
    print(f"Loaded. Feature dim: {reid_model.feature_dim}")

    all_results = {}
    all_fps = []

    for seq in sequences:
        seq_dir = os.path.join(VIDEOS_DIR, seq)
        if not os.path.isdir(seq_dir):
            print(f"SKIP: {seq_dir} not found")
            continue

        print(f"\n--- {seq} ---")
        results, fps = run_reid_tracking(seq_dir, reid_model, max_cosine_distance)
        all_fps.append(fps)

        # Save tracking results
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_file = os.path.join(OUTPUT_DIR, f"{seq}_gt_{reid_name}.txt")
        with open(out_file, 'w') as f:
            for row in results:
                f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n' % (
                    row[0], row[1], row[2], row[3], row[4], row[5]))

        # Compute HOTA
        gt_file = os.path.join(seq_dir, "gt", "gt.txt")
        gt_data = load_mot_file(gt_file)
        pred_data = np.array(results) if len(results) > 0 else np.zeros((0, 6))
        if pred_data.ndim == 1:
            pred_data = pred_data.reshape(1, -1)

        hota_res = compute_hota_per_frame(gt_data, pred_data)
        mot_res = compute_motmetrics(gt_data, pred_data)

        if hota_res is None:
            hota_res = {'HOTA': 0, 'DetJ': 0, 'AssA': 0, 'TP': 0, 'FP': 0, 'FN': 0,
                        'Precision': 0, 'Recall': 0}
        if mot_res is None:
            mot_res = {'MOTA': 0, 'IDF1': 0}

        all_results[seq] = {
            "hota": hota_res['HOTA'], "detj": hota_res['DetJ'], "assa": hota_res['AssA'],
            "mota": mot_res['MOTA'], "idf1": mot_res['IDF1'],
            "tp": hota_res['TP'], "fp": hota_res['FP'], "fn": hota_res['FN'],
            "precision": hota_res['Precision'], "recall": hota_res['Recall'],
            "fps": fps
        }

        print(f"  HOTA={hota_res['HOTA']:.2f} DetJ={hota_res['DetJ']:.2f} AssA={hota_res['AssA']:.2f} "
              f"MOTA={mot_res['MOTA']:.2f} IDF1={mot_res['IDF1']:.2f} FPS={fps:.2f}")

    # Average
    if all_results:
        avg_hota = np.mean([r["hota"] for r in all_results.values()])
        avg_detj = np.mean([r["detj"] for r in all_results.values()])
        avg_assa = np.mean([r["assa"] for r in all_results.values()])
        avg_mota = np.mean([r["mota"] for r in all_results.values()])
        avg_idf1 = np.mean([r["idf1"] for r in all_results.values()])
        avg_fps = np.mean(all_fps)

        all_results["AVERAGE"] = {
            "hota": avg_hota, "detj": avg_detj, "assa": avg_assa,
            "mota": avg_mota, "idf1": avg_idf1, "fps": avg_fps
        }

        print(f"\n{'='*60}")
        print(f"AVERAGE: HOTA={avg_hota:.2f} DetJ={avg_detj:.2f} AssA={avg_assa:.2f} "
              f"MOTA={avg_mota:.2f} IDF1={avg_idf1:.2f} FPS={avg_fps:.2f}")
        print(f"{'='*60}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate REID models with GT boxes")
    parser.add_argument("--reid", default=None, help="REID model name")
    parser.add_argument("--sequences", nargs='+',
                        default=["TUD-Campus", "TUD-Stadtmitte", "KITTI-17",
                                 "PETS09-S2L1", "MOT16-09", "MOT16-11"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_cosine_distance", type=float, default=0.2)
    parser.add_argument("--list_reid", action="store_true")
    args = parser.parse_args()

    if args.list_reid:
        print("Available REID models:", list_reid_models())
        exit(0)

    if args.reid is None:
        print("Specify --reid. Use --list_reid to see options.")
        exit(1)

    all_results = evaluate_reid_model(args.reid, args.sequences,
                                      device=args.device,
                                      max_cosine_distance=args.max_cosine_distance)

    # Save summary
    results_file = os.path.join(OUTPUT_DIR, f"reid_evaluation_{args.reid}.txt")
    with open(results_file, 'w') as f:
        f.write(f"REID Model: {args.reid}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Max cosine distance: {args.max_cosine_distance}\n\n")
        f.write(f"{'Video':<16} {'HOTA':>8} {'DetJ':>8} {'AssA':>8} "
                f"{'MOTA':>8} {'IDF1':>8} {'FPS':>8}\n")
        f.write(f"{'-'*60}\n")
        for seq in args.sequences + ["AVERAGE"]:
            if seq in all_results:
                r = all_results[seq]
                f.write(f"{seq:<16} {r['hota']:>8.2f} {r['detj']:>8.2f} "
                        f"{r['assa']:>8.2f} {r['mota']:>8.2f} "
                        f"{r['idf1']:>8.2f} {r['fps']:>8.2f}\n")
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
