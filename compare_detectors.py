"""Compare detectors on MOT Challenge ground-truth bounding boxes.

Runs each detector on all test videos, matches detections to ground-truth
using IoU, and computes Precision / Recall / F1 per detector and per video.

Usage:
    python compare_detectors.py
    python compare_detectors.py --detectors yolo11s yolov8n
    python compare_detectors.py --sequences TUD-Campus
"""
import argparse
import os
import time

import cv2
import numpy as np

from detectors import create_detector, list_detectors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def load_gt(gt_file):
    if not os.path.exists(gt_file):
        return {}
    data = np.loadtxt(gt_file, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    gt_by_frame = {}
    for row in data:
        frame = int(row[0])
        if frame not in gt_by_frame:
            gt_by_frame[frame] = []
        gt_by_frame[frame].append(row[2:6])
    return gt_by_frame


def compute_iou_matrix(boxes_a, boxes_b):
    """Compute IoU between two sets of boxes in (x, y, w, h) format."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    a = np.array(boxes_a, dtype=np.float64)
    b = np.array(boxes_b, dtype=np.float64)

    a_x2 = a[:, 0] + a[:, 2]
    a_y2 = a[:, 1] + a[:, 3]
    b_x2 = b[:, 0] + b[:, 2]
    b_y2 = b[:, 1] + b[:, 3]

    inter_x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    inter_y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    inter_x2 = np.minimum(a_x2[:, None], b_x2[None, :])
    inter_y2 = np.minimum(a_y2[:, None], b_y2[None, :])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = (b[:, 2] * b[:, 3])[None, :]

    iou = inter_area / (area_a + area_b - inter_area + 1e-10)
    return iou


def match_detections_to_gt(det_boxes, gt_boxes, iou_threshold=0.5):
    """Match detections to ground-truth using greedy IoU matching.

    Returns (tp, fp, fn).
    """
    if len(gt_boxes) == 0:
        return 0, len(det_boxes), 0
    if len(det_boxes) == 0:
        return 0, 0, len(gt_boxes)

    iou_matrix = compute_iou_matrix(det_boxes, gt_boxes)
    matched_gt = set()
    matched_det = set()

    # Greedy matching: sort by IoU descending
    flat_indices = np.argsort(-iou_matrix, axis=None)
    for flat_idx in flat_indices:
        d_idx = flat_idx // iou_matrix.shape[1]
        g_idx = flat_idx % iou_matrix.shape[1]
        if iou_matrix[d_idx, g_idx] < iou_threshold:
            break
        if d_idx in matched_det or g_idx in matched_gt:
            continue
        matched_det.add(d_idx)
        matched_gt.add(g_idx)

    tp = len(matched_det)
    fp = len(det_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def evaluate_detector(detector_name, sequences, device="cpu", conf_threshold=0.3,
                      iou_threshold=0.5):
    """Run detector on sequences and compute detection metrics."""
    detector = create_detector(detector_name, device=device,
                               conf_threshold=conf_threshold)
    print(f"  Loading {detector_name}...")
    detector.load_model()

    results = {}
    total_tp = total_fp = total_fn = 0
    total_time = 0
    total_frames = 0

    for seq in sequences:
        seq_dir = os.path.join(VIDEOS_DIR, seq)
        img_dir = os.path.join(seq_dir, "img1")
        gt_file = os.path.join(seq_dir, "gt", "gt.txt")

        if not os.path.isdir(img_dir):
            print(f"    SKIP {seq}: no img1")
            continue

        gt_by_frame = load_gt(gt_file)

        image_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))],
            key=lambda x: int(os.path.splitext(x)[0]))

        tp = fp = fn = 0
        seq_time = 0

        for img_file in image_files:
            frame_idx = int(os.path.splitext(img_file)[0])
            image = cv2.imread(os.path.join(img_dir, img_file), cv2.IMREAD_COLOR)
            if image is None:
                continue

            t0 = time.time()
            detections = detector.detect(image)
            seq_time += time.time() - t0

            det_boxes = [d.bbox for d in detections]
            gt_boxes = gt_by_frame.get(frame_idx, [])

            f_tp, f_fp, f_fn = match_detections_to_gt(det_boxes, gt_boxes, iou_threshold)
            tp += f_tp
            fp += f_fp
            fn += f_fn

        n_frames = len(image_files)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fps = n_frames / seq_time if seq_time > 0 else 0

        results[seq] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "fps": fps, "frames": n_frames
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_time += seq_time
        total_frames += n_frames

        print(f"    {seq}: P={precision:.4f} R={recall:.4f} F1={f1:.4f} FPS={fps:.2f} "
              f"(TP={tp} FP={fp} FN={fn})")

    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) \
        if (total_precision + total_recall) > 0 else 0
    total_fps = total_frames / total_time if total_time > 0 else 0

    results["AVERAGE"] = {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": total_precision, "recall": total_recall, "f1": total_f1,
        "fps": total_fps, "frames": total_frames
    }

    print(f"    AVERAGE: P={total_precision:.4f} R={total_recall:.4f} "
          f"F1={total_f1:.4f} FPS={total_fps:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare detectors on GT bbox")
    parser.add_argument("--detectors", nargs='+', default=None,
                        help="Detector names to evaluate (default: all)")
    parser.add_argument("--sequences", nargs='+',
                        default=["TUD-Campus", "TUD-Stadtmitte", "KITTI-17",
                                 "PETS09-S2L1", "MOT16-09", "MOT16-11"],
                        help="Sequence names to evaluate")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf_threshold", type=float, default=0.3)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    args = parser.parse_args()

    detector_names = args.detectors or list_detectors()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}
    for det_name in detector_names:
        print(f"\n{'='*60}")
        print(f"Detector: {det_name}")
        print(f"{'='*60}")
        results = evaluate_detector(det_name, args.sequences,
                                    device=args.device,
                                    conf_threshold=args.conf_threshold,
                                    iou_threshold=args.iou_threshold)
        all_results[det_name] = results

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Detector':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPS':>10} "
          f"{'TP':>6} {'FP':>6} {'FN':>6}")
    print(f"{'='*80}")
    for det_name in detector_names:
        r = all_results[det_name]["AVERAGE"]
        print(f"{det_name:<14} {r['precision']:>10.4f} {r['recall']:>10.4f} "
              f"{r['f1']:>10.4f} {r['fps']:>10.2f} "
              f"{r['tp']:>6d} {r['fp']:>6d} {r['fn']:>6d}")
    print(f"{'='*80}")

    # Save to file
    results_file = os.path.join(OUTPUT_DIR, "detector_comparison.txt")
    with open(results_file, 'w') as f:
        for det_name in detector_names:
            f.write(f"\n{'='*60}\n")
            f.write(f"Detector: {det_name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"{'Video':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} "
                    f"{'FPS':>10} {'TP':>6} {'FP':>6} {'FN':>6}\n")
            f.write(f"{'-'*60}\n")
            for seq in args.sequences + ["AVERAGE"]:
                if seq in all_results[det_name]:
                    r = all_results[det_name][seq]
                    f.write(f"{seq:<16} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                            f"{r['f1']:>10.4f} {r['fps']:>10.2f} "
                            f"{r['tp']:>6d} {r['fp']:>6d} {r['fn']:>6d}\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"{'Detector':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} "
                f"{'FPS':>10} {'TP':>6} {'FP':>6} {'FN':>6}\n")
        f.write(f"{'='*80}\n")
        for det_name in detector_names:
            r = all_results[det_name]["AVERAGE"]
            f.write(f"{det_name:<14} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                    f"{r['f1']:>10.4f} {r['fps']:>10.2f} "
                    f"{r['tp']:>6d} {r['fp']:>6d} {r['fn']:>6d}\n")
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
