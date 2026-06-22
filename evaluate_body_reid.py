"""Evaluate autonomous body REID using clustering metrics.

Extracts body crops with ground-truth track IDs, computes REID descriptors,
builds identity clusters, and evaluates with:
- Fowlkes-Mallows score
- Silhouette Coefficient
- Calinski-Harabasz Index

Also runs the full IdentityManager pipeline on tracking output to evaluate
the complete body REID system.

Usage:
    # Evaluate with OSNet x0_25
    python evaluate_body_reid.py --reid osnet_x0_25

    # Evaluate with ResNet50
    python evaluate_body_reid.py --reid resnet50_reid

    # Specific sequences
    python evaluate_body_reid.py --reid osnet_x0_25 --sequences TUD-Campus
"""
import argparse
import os
import time
import json

import cv2
import numpy as np
from sklearn.metrics import fowlkes_mallows_score, silhouette_score
from sklearn.metrics import calinski_harabasz_score

from reid import create_reid
from evaluate_hota import load_mot_file
from body_reid import IdentityManager

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


def extract_gt_crops_and_descriptors(sequence_dir, reid_model, max_frames=None):
    """Extract body crops using GT bounding boxes and compute REID descriptors.

    Returns
    -------
    descriptors : np.ndarray (N, D)
        REID feature vectors for all crops.
    gt_track_ids : np.ndarray (N,)
        Ground-truth track IDs for each crop.
    frame_ids : np.ndarray (N,)
        Frame ID for each crop.
    """
    seq_info = gather_sequence_info(sequence_dir)
    seq_name = seq_info["sequence_name"]

    gt_file = os.path.join(sequence_dir, "gt", "gt.txt")
    if not os.path.exists(gt_file):
        print(f"  No GT file for {seq_name}")
        return None, None, None

    gt_data = load_mot_file(gt_file)
    if gt_data is None or len(gt_data) == 0:
        return None, None, None

    # Group GT by frame: frame -> list of (x, y, w, h, track_id)
    gt_by_frame = {}
    for row in gt_data:
        f = int(row[0])
        tid = int(row[1])
        x, y, w, h = row[2], row[3], row[4], row[5]
        if f not in gt_by_frame:
            gt_by_frame[f] = []
        gt_by_frame[f].append((x, y, w, h, tid))

    frame_indices = sorted(seq_info["image_filenames"].keys())
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]

    descriptors = []
    gt_track_ids = []
    frame_ids = []

    for i, frame_idx in enumerate(frame_indices):
        if frame_idx not in gt_by_frame:
            continue

        image = cv2.imread(seq_info["image_filenames"][frame_idx],
                           cv2.IMREAD_COLOR)
        if image is None:
            continue

        frame_dets = gt_by_frame[frame_idx]
        boxes = []
        track_ids = []
        for det in frame_dets:
            x, y, w, h, track_id = det[0], det[1], det[2], det[3], det[4]
            if w < 10 or h < 10:
                continue
            boxes.append([x, y, w, h])
            track_ids.append(track_id)

        if not boxes:
            continue

        features = reid_model.extract_features(image, boxes)
        for feat, tid in zip(features, track_ids):
            descriptors.append(feat)
            gt_track_ids.append(tid)
            frame_ids.append(frame_idx)

        if (i + 1) % 50 == 0:
            print(f"\r  {seq_name} extracting crops: frame {frame_idx} "
                  f"({i+1}/{len(frame_indices)})", end="", flush=True)

    print(f"\n  {seq_name}: extracted {len(descriptors)} crops, "
          f"{len(set(gt_track_ids))} unique GT track IDs")

    return np.array(descriptors), np.array(gt_track_ids), np.array(frame_ids)


def evaluate_clustering(descriptors, gt_track_ids, reid_name, seq_name):
    """Evaluate REID clustering quality against GT track IDs.

    Uses the REID descriptors to cluster crops, then compares clusters
    against GT track IDs using standard clustering metrics.
    """
    if len(descriptors) < 2 or len(set(gt_track_ids)) < 2:
        print(f"  {seq_name}: insufficient data for clustering metrics")
        return None

    # Normalize descriptors
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    norm_descriptors = descriptors / norms

    # Cluster using cosine distance threshold
    # Assign each descriptor to a cluster via greedy matching
    cluster_labels = _threshold_cluster(norm_descriptors, threshold=0.3)

    n_clusters = len(set(cluster_labels))
    n_gt = len(set(gt_track_ids))

    # Compute metrics
    fmi = fowlkes_mallows_score(gt_track_ids, cluster_labels)

    # Silhouette requires at least 2 clusters and <= n_samples-1 clusters
    if n_clusters >= 2 and n_clusters < len(descriptors):
        sil = silhouette_score(norm_descriptors, cluster_labels, metric="cosine")
    else:
        sil = -1.0

    # Calinski-Harabasz
    if n_clusters >= 2 and n_clusters < len(descriptors):
        ch = calinski_harabasz_score(norm_descriptors, cluster_labels)
    else:
        ch = 0.0

    # Purity
    purity = _compute_purity(gt_track_ids, cluster_labels)

    print(f"  {seq_name}: GT IDs={n_gt}, Clusters={n_clusters}, "
          f"FMI={fmi:.4f}, Silhouette={sil:.4f}, CH={ch:.2f}, "
          f"Purity={purity:.4f}")

    return {
        "fmi": fmi, "silhouette": sil, "calinski_harabasz": ch,
        "purity": purity, "n_gt_ids": n_gt, "n_clusters": n_clusters,
        "n_samples": len(descriptors),
    }


def _threshold_cluster(descriptors, threshold=0.3):
    """Greedy threshold-based clustering using cosine distance.

    Assigns each descriptor to the first existing cluster whose centroid
    is within `threshold` cosine distance. Otherwise creates a new cluster.
    """
    n = len(descriptors)
    labels = np.zeros(n, dtype=int)
    centroids = [descriptors[0].copy()]
    labels[0] = 0
    n_clusters = 1

    for i in range(1, n):
        best_dist = 1.0
        best_cluster = -1
        for c in range(n_clusters):
            dist = 1.0 - np.dot(descriptors[i], centroids[c])
            if dist < best_dist:
                best_dist = dist
                best_cluster = c

        if best_dist <= threshold and best_cluster >= 0:
            labels[i] = best_cluster
            n_in_cluster = np.sum(labels[:i+1] == best_cluster)
            centroids[best_cluster] = (
                centroids[best_cluster] * (n_in_cluster - 1) / n_in_cluster
                + descriptors[i] / n_in_cluster
            )
        else:
            labels[i] = n_clusters
            centroids.append(descriptors[i].copy())
            n_clusters += 1

    return labels


def _compute_purity(gt_labels, pred_labels):
    """Compute clustering purity."""
    n = len(gt_labels)
    contingency = {}
    for gt, pred in zip(gt_labels, pred_labels):
        key = (gt, pred)
        contingency[key] = contingency.get(key, 0) + 1

    pred_to_gt = {}
    for (gt, pred), count in contingency.items():
        if pred not in pred_to_gt or count > pred_to_gt[pred][1]:
            pred_to_gt[pred] = (gt, count)

    correct = sum(count for _, count in pred_to_gt.values())
    return correct / n


def evaluate_identity_manager(sequence_dir, reid_model, params, max_frames=None):
    """Run the full IdentityManager pipeline on a sequence using GT boxes.

    This simulates the body REID system: for each frame, extract REID
    descriptors from GT boxes, run identity management, and compare
    assigned identities against GT track IDs.
    """
    seq_info = gather_sequence_info(sequence_dir)
    seq_name = seq_info["sequence_name"]

    gt_file = os.path.join(sequence_dir, "gt", "gt.txt")
    if not os.path.exists(gt_file):
        return None

    gt_data = load_mot_file(gt_file)
    if gt_data is None or len(gt_data) == 0:
        return None

    # Group GT by frame
    gt_by_frame = {}
    for row in gt_data:
        f = int(row[0])
        tid = int(row[1])
        x, y, w, h = row[2], row[3], row[4], row[5]
        if f not in gt_by_frame:
            gt_by_frame[f] = []
        gt_by_frame[f].append((x, y, w, h, tid))

    manager = IdentityManager(
        distance_threshold=params["distance_threshold"],
        temporal_window=params["temporal_window"],
        min_votes=params["min_votes"],
        use_centroid=params["use_centroid"],
        max_descriptors=params["max_descriptors"],
        conflict_strategy=params["conflict_strategy"])

    frame_indices = sorted(seq_info["image_filenames"].keys())
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]

    # Collect all (gt_track_id, assigned_identity_id) pairs
    all_gt_ids = []
    all_assigned_ids = []

    for i, frame_idx in enumerate(frame_indices):
        if frame_idx not in gt_by_frame:
            continue

        image = cv2.imread(seq_info["image_filenames"][frame_idx],
                           cv2.IMREAD_COLOR)
        if image is None:
            continue

        frame_dets = gt_by_frame[frame_idx]
        boxes = []
        track_ids = []
        for det in frame_dets:
            x, y, w, h, track_id = det[0], det[1], det[2], det[3], det[4]
            if w < 10 or h < 10:
                continue
            boxes.append([x, y, w, h])
            track_ids.append(track_id)

        if not boxes:
            continue

        features = reid_model.extract_features(image, boxes)

        # Build track_descriptors dict
        track_descriptors = {}
        for tid, feat in zip(track_ids, features):
            track_descriptors[tid] = feat

        # Run identity manager
        resolved = manager.update(frame_idx, track_descriptors)

        for tid in track_ids:
            all_gt_ids.append(tid)
            all_assigned_ids.append(resolved.get(tid, -1))

        if (i + 1) % 50 == 0:
            stats = manager.get_statistics()
            print(f"\r  {seq_name} identity manager: frame {frame_idx} "
                  f"({i+1}/{len(frame_indices)}), identities={stats['num_identities']}",
                  end="", flush=True)

    print(f"\n  {seq_name}: {len(all_gt_ids)} observations, "
          f"{manager.get_statistics()['num_identities']} identities created")

    # Evaluate: compare assigned identities to GT track IDs
    all_gt_ids = np.array(all_gt_ids)
    all_assigned_ids = np.array([-1 if x is None else x for x in all_assigned_ids])

    if len(all_gt_ids) < 2 or len(set(all_gt_ids)) < 2:
        return None

    # Filter out unassigned (-1) for clustering metrics
    valid_mask = all_assigned_ids >= 0
    if valid_mask.sum() < 2 or len(set(all_gt_ids[valid_mask])) < 2:
        return None

    # Fowlkes-Mallows (only on valid assignments)
    fmi = fowlkes_mallows_score(all_gt_ids[valid_mask], all_assigned_ids[valid_mask])

    # Purity
    purity = _compute_purity(all_gt_ids, all_assigned_ids)

    # Inverse purity (colocation)
    inverse_purity = _compute_purity(all_assigned_ids, all_gt_ids)

    stats = manager.get_statistics()
    n_gt = len(set(all_gt_ids))
    n_assigned = len(set(all_assigned_ids))

    print(f"  {seq_name}: GT IDs={n_gt}, Assigned IDs={n_assigned}, "
          f"FMI={fmi:.4f}, Purity={purity:.4f}, InvPurity={inverse_purity:.4f}")

    return {
        "fmi": fmi, "purity": purity, "inverse_purity": inverse_purity,
        "n_gt_ids": n_gt, "n_assigned_ids": n_assigned,
        "n_observations": len(all_gt_ids),
        "num_identities": stats["num_identities"],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate body REID")
    parser.add_argument("--reid", default="osnet_x0_25")
    parser.add_argument("--sequences", nargs='+',
                        default=["TUD-Campus", "TUD-Stadtmitte", "KITTI-17",
                                 "PETS09-S2L1", "MOT16-09", "MOT16-11"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--distance_threshold", type=float, default=0.3)
    parser.add_argument("--temporal_window", type=int, default=30)
    parser.add_argument("--min_votes", type=int, default=1)
    parser.add_argument("--use_centroid", action="store_true", default=True)
    parser.add_argument("--max_descriptors", type=int, default=50)
    parser.add_argument("--conflict_strategy", default="reset",
                        choices=["reset", "keep_best"])
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit frames per sequence (for speed)")
    args = parser.parse_args()

    params = {
        "distance_threshold": args.distance_threshold,
        "temporal_window": args.temporal_window,
        "min_votes": args.min_votes,
        "use_centroid": args.use_centroid,
        "max_descriptors": args.max_descriptors,
        "conflict_strategy": args.conflict_strategy,
    }

    print(f"\n{'='*70}")
    print(f"Body REID Evaluation")
    print(f"REID: {args.reid} | Device: {args.device}")
    print(f"Params: {json.dumps(params)}")
    print(f"{'='*70}")

    reid_model = create_reid(args.reid, device=args.device)
    print(f"Loading REID: {reid_model}")
    reid_model.load_model()

    all_clustering = {}
    all_manager = {}

    for seq in args.sequences:
        seq_dir = os.path.join(VIDEOS_DIR, seq)
        if not os.path.isdir(seq_dir):
            print(f"SKIP: {seq_dir} not found")
            continue

        print(f"\n--- {seq} ---")

        # Part 1: Clustering evaluation (offline)
        print("  [1] Clustering evaluation (offline):")
        descriptors, gt_track_ids, frame_ids = extract_gt_crops_and_descriptors(
            seq_dir, reid_model, max_frames=args.max_frames)
        if descriptors is not None:
            clustering_res = evaluate_clustering(
                descriptors, gt_track_ids, args.reid, seq)
            all_clustering[seq] = clustering_res

        # Part 2: IdentityManager evaluation (online pipeline)
        print("  [2] IdentityManager evaluation (online):")
        manager_res = evaluate_identity_manager(
            seq_dir, reid_model, params, max_frames=args.max_frames)
        all_manager[seq] = manager_res

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Clustering Metrics (offline)")
    print(f"{'='*70}")
    print(f"{'Video':<16} {'FMI':>8} {'Silhouette':>12} {'CH':>10} "
          f"{'Purity':>8} {'GT IDs':>8} {'Clusters':>10}")
    print(f"{'-'*72}")
    for seq, res in all_clustering.items():
        if res:
            print(f"{seq:<16} {res['fmi']:>8.4f} {res['silhouette']:>12.4f} "
                  f"{res['calinski_harabasz']:>10.2f} {res['purity']:>8.4f} "
                  f"{res['n_gt_ids']:>8} {res['n_clusters']:>10}")
    if all_clustering:
        avg_fmi = np.mean([r['fmi'] for r in all_clustering.values() if r])
        avg_sil = np.mean([r['silhouette'] for r in all_clustering.values() if r])
        avg_pur = np.mean([r['purity'] for r in all_clustering.values() if r])
        print(f"{'AVERAGE':<16} {avg_fmi:>8.4f} {avg_sil:>12.4f} "
              f"{'':>10} {avg_pur:>8.4f}")

    print(f"\n{'='*70}")
    print("SUMMARY: IdentityManager Metrics (online)")
    print(f"{'='*70}")
    print(f"{'Video':<16} {'FMI':>8} {'Purity':>8} {'InvPurity':>10} "
          f"{'GT IDs':>8} {'Assigned':>10} {'Identities':>12}")
    print(f"{'-'*72}")
    for seq, res in all_manager.items():
        if res:
            print(f"{seq:<16} {res['fmi']:>8.4f} {res['purity']:>8.4f} "
                  f"{res['inverse_purity']:>10.4f} {res['n_gt_ids']:>8} "
                  f"{res['n_assigned_ids']:>10} {res['num_identities']:>12}")
    if all_manager:
        avg_fmi_m = np.mean([r['fmi'] for r in all_manager.values() if r])
        avg_pur_m = np.mean([r['purity'] for r in all_manager.values() if r])
        avg_inv_m = np.mean([r['inverse_purity'] for r in all_manager.values() if r])
        print(f"{'AVERAGE':<16} {avg_fmi_m:>8.4f} {avg_pur_m:>8.4f} "
              f"{avg_inv_m:>10.4f}")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_file = os.path.join(
        OUTPUT_DIR, f"body_reid_{args.reid}.txt")
    with open(results_file, 'w') as f:
        f.write(f"Body REID Evaluation\n")
        f.write(f"REID: {args.reid}\n")
        f.write(f"Params: {json.dumps(params)}\n\n")

        f.write("=== Clustering Metrics (offline) ===\n")
        f.write(f"{'Video':<16} {'FMI':>8} {'Silhouette':>12} {'CH':>10} "
                f"{'Purity':>8} {'GT IDs':>8} {'Clusters':>10}\n")
        for seq, res in all_clustering.items():
            if res:
                f.write(f"{seq:<16} {res['fmi']:>8.4f} {res['silhouette']:>12.4f} "
                        f"{res['calinski_harabasz']:>10.2f} {res['purity']:>8.4f} "
                        f"{res['n_gt_ids']:>8} {res['n_clusters']:>10}\n")

        f.write(f"\n=== IdentityManager Metrics (online) ===\n")
        f.write(f"{'Video':<16} {'FMI':>8} {'Purity':>8} {'InvPurity':>10} "
                f"{'GT IDs':>8} {'Assigned':>10} {'Identities':>12}\n")
        for seq, res in all_manager.items():
            if res:
                f.write(f"{seq:<16} {res['fmi']:>8.4f} {res['purity']:>8.4f} "
                        f"{res['inverse_purity']:>10.4f} {res['n_gt_ids']:>8} "
                        f"{res['n_assigned_ids']:>10} {res['num_identities']:>12}\n")

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
