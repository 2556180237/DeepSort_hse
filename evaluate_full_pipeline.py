"""Evaluate the full DeepSORT pipeline (detector + REID + tracker) with HOTA metrics.

Runs YOLO11s + osnet_x0_25 (or other detector/REID combos) on all test videos,
computes HOTA/MOTA/IDF1, and supports parameter tuning.

Usage:
    # Default parameters
    python evaluate_full_pipeline.py --detector yolo11s --reid osnet_x0_25

    # Custom parameters
    python evaluate_full_pipeline.py --detector yolo11s --reid osnet_x0_25 \
        --conf_threshold 0.4 --max_cosine_distance 0.3 --max_age 30 --n_init 3

    # Only specific sequences
    python evaluate_full_pipeline.py --sequences TUD-Campus TUD-Stadtmitte
"""
import argparse
import os
import time
import json

import cv2
import numpy as np

from application_util import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from detectors import create_detector
from reid import create_reid
from evaluate_hota import compute_hota_per_frame, compute_motmetrics, load_mot_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def gather_sequence_info(sequence_dir):
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))
    }
    return {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "min_frame_idx": min(image_filenames.keys()) if image_filenames else 0,
        "max_frame_idx": max(image_filenames.keys()) if image_filenames else 0,
    }


def run_pipeline(sequence_dir, detector, reid_model, params):
    """Run full pipeline on one sequence and return tracking results + FPS."""
    seq_info = gather_sequence_info(sequence_dir)
    seq_name = seq_info["sequence_name"]

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", params["max_cosine_distance"], params["nn_budget"])
    tracker = Tracker(metric, max_iou_distance=params["max_iou_distance"],
                      max_age=params["max_age"], n_init=params["n_init"])
    results = []

    frame_indices = sorted(seq_info["image_filenames"].keys())
    total_frames = len(frame_indices)
    start_time = time.time()

    for i, frame_idx in enumerate(frame_indices):
        print(f"\r  {seq_name} frame {frame_idx:05d} ({i+1}/{total_frames})",
              end="", flush=True)

        image = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
        if image is None:
            continue

        det_results = detector.detect(image)
        boxes = [dr.bbox for dr in det_results]

        if reid_model is not None and len(boxes) > 0:
            features = reid_model.extract_features(image, boxes)
        else:
            features = [np.random.rand(128).astype(np.float32) for _ in boxes]

        detections = []
        for dr, feat in zip(det_results, features):
            detections.append(Detection(dr.bbox, dr.confidence, feat))

        boxes_arr = np.array([d.tlwh for d in detections])
        if len(boxes_arr) > 0:
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes_arr, params["nms_max_overlap"], scores)
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
    print(f"\n  Done: {total_frames} frames in {elapsed:.2f}s ({fps:.2f} FPS)")

    return results, fps


def evaluate(detector_name, reid_name, sequences, device, params):
    """Run full pipeline on all sequences and compute metrics."""
    print(f"\n{'='*70}")
    print(f"Detector: {detector_name} | REID: {reid_name} | Device: {device}")
    print(f"Params: {json.dumps(params)}")
    print(f"{'='*70}")

    detector = create_detector(detector_name, device=device,
                               conf_threshold=params["conf_threshold"])
    print(f"Loading detector: {detector}")
    detector.load_model()

    reid_model = None
    if reid_name is not None:
        reid_model = create_reid(reid_name, device=device)
        print(f"Loading REID: {reid_model}")
        reid_model.load_model()

    all_results = {}
    all_fps = []

    for seq in sequences:
        seq_dir = os.path.join(VIDEOS_DIR, seq)
        if not os.path.isdir(seq_dir):
            print(f"SKIP: {seq_dir} not found")
            continue

        print(f"\n--- {seq} ---")
        results, fps = run_pipeline(seq_dir, detector, reid_model, params)
        all_fps.append(fps)

        # Save tracking results
        param_tag = f"{detector_name}_{reid_name}" if reid_name else detector_name
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_file = os.path.join(OUTPUT_DIR, f"{seq}_{param_tag}.txt")
        with open(out_file, 'w') as f:
            for row in results:
                f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n' % (
                    row[0], row[1], row[2], row[3], row[4], row[5]))

        # Compute metrics
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
            "hota": hota_res['HOTA'], "detj": hota_res['DetJ'],
            "assa": hota_res['AssA'], "mota": mot_res['MOTA'],
            "idf1": mot_res['IDF1'], "tp": hota_res['TP'],
            "fp": hota_res['FP'], "fn": hota_res['FN'],
            "precision": hota_res['Precision'], "recall": hota_res['Recall'],
            "fps": fps
        }

        print(f"  HOTA={hota_res['HOTA']:.2f} DetJ={hota_res['DetJ']:.2f} "
              f"AssA={hota_res['AssA']:.2f} MOTA={mot_res['MOTA']:.2f} "
              f"IDF1={mot_res['IDF1']:.2f} FPS={fps:.2f}")

    if all_results:
        avg = {k: np.mean([r[k] for r in all_results.values()])
               for k in ["hota", "detj", "assa", "mota", "idf1", "fps"]}
        all_results["AVERAGE"] = avg
        print(f"\n{'='*70}")
        print(f"AVERAGE: HOTA={avg['hota']:.2f} DetJ={avg['detj']:.2f} "
              f"AssA={avg['assa']:.2f} MOTA={avg['mota']:.2f} "
              f"IDF1={avg['idf1']:.2f} FPS={avg['fps']:.2f}")
        print(f"{'='*70}")

    return all_results, params


def save_results(all_results, params, detector_name, reid_name, tag="default"):
    param_tag = f"{detector_name}_{reid_name}" if reid_name else detector_name
    results_file = os.path.join(OUTPUT_DIR, f"pipeline_{param_tag}_{tag}.txt")
    with open(results_file, 'w') as f:
        f.write(f"Detector: {detector_name}\n")
        f.write(f"REID: {reid_name}\n")
        f.write(f"Params: {json.dumps(params)}\n\n")
        f.write(f"{'Video':<16} {'HOTA':>8} {'DetJ':>8} {'AssA':>8} "
                f"{'MOTA':>8} {'IDF1':>8} {'FPS':>8}\n")
        f.write(f"{'-'*64}\n")
        for seq in list(all_results.keys()):
            r = all_results[seq]
            f.write(f"{seq:<16} {r['hota']:>8.2f} {r['detj']:>8.2f} "
                    f"{r['assa']:>8.2f} {r['mota']:>8.2f} "
                    f"{r['idf1']:>8.2f} {r['fps']:>8.2f}\n")
    print(f"Results saved to {results_file}")
    return results_file


def main():
    parser = argparse.ArgumentParser(description="Evaluate full DeepSORT pipeline")
    parser.add_argument("--detector", default="yolo11s")
    parser.add_argument("--reid", default="osnet_x0_25")
    parser.add_argument("--sequences", nargs='+',
                        default=["TUD-Campus", "TUD-Stadtmitte", "KITTI-17",
                                 "PETS09-S2L1", "MOT16-09", "MOT16-11"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf_threshold", type=float, default=0.3)
    parser.add_argument("--nms_max_overlap", type=float, default=1.0)
    parser.add_argument("--max_cosine_distance", type=float, default=0.2)
    parser.add_argument("--nn_budget", type=int, default=100)
    parser.add_argument("--max_age", type=int, default=30)
    parser.add_argument("--n_init", type=int, default=3)
    parser.add_argument("--max_iou_distance", type=float, default=0.7)
    parser.add_argument("--tag", default="default", help="Tag for output file")
    args = parser.parse_args()

    params = {
        "conf_threshold": args.conf_threshold,
        "nms_max_overlap": args.nms_max_overlap,
        "max_cosine_distance": args.max_cosine_distance,
        "nn_budget": args.nn_budget,
        "max_age": args.max_age,
        "n_init": args.n_init,
        "max_iou_distance": args.max_iou_distance,
    }

    all_results, params = evaluate(
        args.detector, args.reid, args.sequences, args.device, params)
    save_results(all_results, params, args.detector, args.reid, args.tag)


if __name__ == "__main__":
    main()
