import os
import numpy as np
import motmetrics as mm
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def load_mot_file(filepath):
    if not os.path.exists(filepath):
        return None
    data = np.loadtxt(filepath, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def compute_hota_per_frame(gt_data, pred_data, iou_threshold=0.5):
    """Compute HOTA metrics averaged over all frames.

    HOTA = sqrt(J * A) where:
      J (detection) = TP / (TP + FP + FN)
      A (association) = TPA / (TPA + FPA + FNA)
    """
    if gt_data is None or pred_data is None:
        return None

    gt_frames = np.unique(gt_data[:, 0].astype(int))
    pred_frames = np.unique(pred_data[:, 0].astype(int))
    all_frames = sorted(set(gt_frames) | set(pred_frames))

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # For association: track global assignment
    # HOTA association is computed via global alignment
    # We use the simpler per-frame approach with accumulated counts

    # For proper HOTA, we need to track per-pair (gt_id, pred_id) match counts
    from collections import defaultdict
    match_counts = defaultdict(int)  # (gt_id, pred_id) -> num matched frames
    gt_total = defaultdict(int)      # gt_id -> total frames visible
    pred_total = defaultdict(int)    # pred_id -> total frames predicted

    for frame in all_frames:
        gt_mask = gt_data[:, 0].astype(int) == frame
        pred_mask = pred_data[:, 0].astype(int) == frame

        gt_boxes = gt_data[gt_mask]
        pred_boxes = pred_data[pred_mask]

        gt_ids = gt_boxes[:, 1].astype(int)
        pred_ids = pred_boxes[:, 1].astype(int)

        gt_bboxes = gt_boxes[:, 2:6]  # x, y, w, h
        pred_bboxes = pred_boxes[:, 2:6]

        for gid in gt_ids:
            gt_total[gid] += 1
        for pid in pred_ids:
            pred_total[pid] += 1

        if len(gt_bboxes) == 0:
            total_fp += len(pred_bboxes)
            continue
        if len(pred_bboxes) == 0:
            total_fn += len(gt_bboxes)
            continue

        # Compute IoU matrix
        iou_matrix = compute_iou_matrix(gt_bboxes, pred_bboxes)

        # Hungarian matching
        from scipy.optimize import linear_sum_assignment
        cost = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_pairs = []
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                matched_pairs.append((gt_ids[r], pred_ids[c]))
                total_tp += 1
            else:
                total_fn += 1
                total_fp += 1

        unmatched_gt = len(gt_bboxes) - len(matched_pairs)
        unmatched_pred = len(pred_bboxes) - len(matched_pairs)
        total_fn += unmatched_gt
        total_fp += unmatched_pred

        for gid, pid in matched_pairs:
            match_counts[(gid, pid)] += 1

    # Detection Jaccard
    if total_tp + total_fp + total_fn == 0:
        J = 0
    else:
        J = total_tp / (total_tp + total_fp + total_fn)

    # Association Jaccard (HOTA-style: average per-match association)
    # For each matched pair, A_local = c(g,p) / (|gt_id| + |pred_id| - c(g,p))
    # HOTA_A = (1/|TP|) * sum over matched pairs of A_local
    # Then overall A = HOTA_A
    if total_tp == 0:
        A = 0
    else:
        a_sum = 0
        for (gid, pid), c in match_counts.items():
            if c > 0:
                a_local = c / (gt_total[gid] + pred_total[pid] - c)
                a_sum += a_local
        A = a_sum / total_tp

    HOTA = np.sqrt(J * A)

    # Additional metrics
    if total_tp + total_fp == 0:
        precision = 0
    else:
        precision = total_tp / (total_tp + total_fp)
    if total_tp + total_fn == 0:
        recall = 0
    else:
        recall = total_tp / (total_tp + total_fn)

    return {
        'HOTA': HOTA * 100,
        'DetJ': J * 100,
        'AssA': A * 100,
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'Precision': precision * 100,
        'Recall': recall * 100,
    }


def compute_iou_matrix(boxes_a, boxes_b):
    """Compute IoU matrix between two sets of boxes in (x, y, w, h) format."""
    # Convert to (x1, y1, x2, y2)
    a = boxes_a.copy().astype(float)
    b = boxes_b.copy().astype(float)
    a[:, 2] += a[:, 0]  # x2 = x + w
    a[:, 3] += a[:, 1]  # y2 = y + h
    b[:, 2] += b[:, 0]
    b[:, 3] += b[:, 1]

    # Intersection
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter = inter_w * inter_h

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / (union + 1e-10)
    return iou


def compute_motmetrics(gt_data, pred_data):
    """Compute MOTA, IDF1 using motmetrics library."""
    if gt_data is None or pred_data is None:
        return None

    gt_frames = np.unique(gt_data[:, 0].astype(int))
    pred_frames = np.unique(pred_data[:, 0].astype(int))
    all_frames = sorted(set(gt_frames) | set(pred_frames))

    acc = mm.MOTAccumulator(auto_id=True)

    for frame in all_frames:
        gt_mask = gt_data[:, 0].astype(int) == frame
        pred_mask = pred_data[:, 0].astype(int) == frame

        gt_boxes = gt_data[gt_mask]
        pred_boxes = pred_data[pred_mask]

        gt_ids = gt_boxes[:, 1].astype(int) if len(gt_boxes) > 0 else []
        pred_ids = pred_boxes[:, 1].astype(int) if len(pred_boxes) > 0 else []

        if len(gt_boxes) > 0:
            gt_bboxes = gt_boxes[:, 2:6]
        else:
            gt_bboxes = np.zeros((0, 4))

        if len(pred_boxes) > 0:
            pred_bboxes = pred_boxes[:, 2:6]
        else:
            pred_bboxes = np.zeros((0, 4))

        # motmetrics expects (x, y, w, h) format
        acc.update(
            gt_ids,
            pred_ids,
            mm.distances.iou_matrix(gt_bboxes, pred_bboxes, max_iou=0.5)
        )

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'idf1', 'precision', 'recall'], name='overall')
    return {
        'MOTA': summary['mota']['overall'] * 100,
        'IDF1': summary['idf1']['overall'] * 100,
        'Precision_mm': summary['precision']['overall'] * 100,
        'Recall_mm': summary['recall']['overall'] * 100,
    }


def main():
    sequences = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')])

    print("=" * 90)
    print(f"{'Video':<16} {'HOTA':>7} {'DetJ':>7} {'AssA':>7} {'MOTA':>7} {'IDF1':>7} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Rec':>7}")
    print("=" * 90)

    results = []
    for seq_file in sequences:
        seq_name = seq_file.replace('.txt', '')
        gt_file = os.path.join(VIDEOS_DIR, seq_name, "gt", "gt.txt")
        pred_file = os.path.join(OUTPUT_DIR, seq_file)

        gt_data = load_mot_file(gt_file)
        pred_data = load_mot_file(pred_file)

        hota_metrics = compute_hota_per_frame(gt_data, pred_data)
        mot_metrics = compute_motmetrics(gt_data, pred_data)

        if hota_metrics is None or mot_metrics is None:
            print(f"{seq_name:<16} -- no data --")
            continue

        row = {
            'Video': seq_name,
            **hota_metrics,
            **mot_metrics,
        }
        results.append(row)

        print(f"{seq_name:<16} {hota_metrics['HOTA']:>7.2f} {hota_metrics['DetJ']:>7.2f} {hota_metrics['AssA']:>7.2f} "
              f"{mot_metrics['MOTA']:>7.2f} {mot_metrics['IDF1']:>7.2f} {hota_metrics['TP']:>6d} {hota_metrics['FP']:>6d} "
              f"{hota_metrics['FN']:>6d} {hota_metrics['Precision']:>7.2f} {hota_metrics['Recall']:>7.2f}")

    # Average HOTA
    if results:
        avg_hota = np.mean([r['HOTA'] for r in results])
        avg_mota = np.mean([r['MOTA'] for r in results])
        avg_idf1 = np.mean([r['IDF1'] for r in results])
        print("-" * 90)
        print(f"{'AVERAGE':<16} {avg_hota:>7.2f} {'':>7} {'':>7} {avg_mota:>7.2f} {avg_idf1:>7.2f}")
        print("=" * 90)
        print(f"\nAverage HOTA: {avg_hota:.2f}")
        print(f"Average MOTA: {avg_mota:.2f}")
        print(f"Average IDF1: {avg_idf1:.2f}")

        # Save results to file
        results_file = os.path.join(BASE_DIR, "output", "baseline_metrics.txt")
        with open(results_file, 'w') as f:
            f.write(f"{'Video':<16} {'HOTA':>7} {'DetJ':>7} {'AssA':>7} {'MOTA':>7} {'IDF1':>7} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Rec':>7}\n")
            f.write("=" * 90 + "\n")
            for r in results:
                f.write(f"{r['Video']:<16} {r['HOTA']:>7.2f} {r['DetJ']:>7.2f} {r['AssA']:>7.2f} "
                        f"{r['MOTA']:>7.2f} {r['IDF1']:>7.2f} {r['TP']:>6d} {r['FP']:>6d} "
                        f"{r['FN']:>6d} {r['Precision']:>7.2f} {r['Recall']:>7.2f}\n")
            f.write("-" * 90 + "\n")
            f.write(f"{'AVERAGE':<16} {avg_hota:>7.2f} {'':>7} {'':>7} {avg_mota:>7.2f} {avg_idf1:>7.2f}\n")
        print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
