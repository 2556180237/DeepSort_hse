"""Full pipeline with autonomous body REID.

Runs YOLO11s (or YOLO11s-seg) + OSNet x0_25 + DeepSORT tracker + IdentityManager.
Each detection gets a track ID (from DeepSORT) and an identity ID (from body REID).

Usage:
    python run_body_reid.py --sequence_dir videos/TUD-Campus \
        --detector yolo11s --reid osnet_x0_25 \
        --output_file output/TUD-Campus_body_reid.txt

    # With segmentation detector
    python run_body_reid.py --sequence_dir videos/TUD-Campus \
        --detector yolo11s-seg --reid osnet_x0_25

    # Run on all videos
    python run_body_reid.py --all --detector yolo11s --reid osnet_x0_25
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
from detectors import create_detector, list_detectors
from reid import create_reid, list_reid_models
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


def run(sequence_dir, detector_name, reid_name, output_file, device="cpu",
        conf_threshold=0.4, nms_max_overlap=1.0, max_cosine_distance=0.2,
        nn_budget=100, max_age=30, n_init=3, max_iou_distance=0.7,
        body_reid_threshold=0.3, temporal_window=30, min_votes=1,
        conflict_strategy="reset"):
    """Run full pipeline with body REID."""
    seq_info = gather_sequence_info(sequence_dir)
    seq_name = seq_info["sequence_name"]

    print(f"Sequence: {seq_name}")
    print(f"Detector: {detector_name} | REID: {reid_name}")
    print(f"Body REID: threshold={body_reid_threshold}, window={temporal_window}")

    detector = create_detector(detector_name, device=device,
                               conf_threshold=conf_threshold)
    detector.load_model()

    reid_model = create_reid(reid_name, device=device)
    reid_model.load_model()

    identity_manager = IdentityManager(
        distance_threshold=body_reid_threshold,
        temporal_window=temporal_window,
        min_votes=min_votes,
        use_centroid=True,
        max_descriptors=50,
        conflict_strategy=conflict_strategy)

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_iou_distance=max_iou_distance,
                      max_age=max_age, n_init=n_init)

    results = []
    frame_indices = sorted(seq_info["image_filenames"].keys())
    total_frames = len(frame_indices)
    start_time = time.time()

    for i, frame_idx in enumerate(frame_indices):
        print(f"\r{seq_name} frame {frame_idx:05d} ({i+1}/{total_frames})",
              end="", flush=True)

        image = cv2.imread(seq_info["image_filenames"][frame_idx],
                           cv2.IMREAD_COLOR)
        if image is None:
            continue

        det_results = detector.detect(image)
        boxes = [dr.bbox for dr in det_results]

        if len(boxes) > 0:
            features = reid_model.extract_features(image, boxes)
        else:
            features = []

        detections = []
        for dr, feat in zip(det_results, features):
            detections.append(Detection(dr.bbox, dr.confidence, feat))

        boxes_arr = np.array([d.tlwh for d in detections])
        if len(boxes_arr) > 0:
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes_arr, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        # Collect confirmed track descriptors for body REID
        track_descriptors = {}
        track_bboxes = {}
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # Find the matching detection's feature
            bbox = track.to_tlwh()
            track_bboxes[track.track_id] = bbox

            # Use the last feature from the track
            if track.features:
                track_descriptors[track.track_id] = track.features[-1]

        # Run identity manager
        if track_descriptors:
            resolved = identity_manager.update(frame_idx, track_descriptors)
        else:
            resolved = {}

        # Write results: frame, track_id, identity_id, bbox
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            identity_id = resolved.get(track.track_id, -1)
            results.append([
                frame_idx, track.track_id, identity_id,
                bbox[0], bbox[1], bbox[2], bbox[3]])

    elapsed = time.time() - start_time
    fps = total_frames / elapsed if elapsed > 0 else 0
    print(f"\nDone: {total_frames} frames in {elapsed:.2f}s ({fps:.2f} FPS)")

    stats = identity_manager.get_statistics()
    print(f"Identities created: {stats['num_identities']}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for row in results:
            f.write('%d,%d,%d,%.2f,%.2f,%.2f,%.2f\n' % (
                row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
    print(f"Results saved to {output_file}")

    return fps, stats


def run_all(detector_name, reid_name, **kwargs):
    sequences = ["TUD-Campus", "TUD-Stadtmitte", "KITTI-17",
                 "PETS09-S2L1", "MOT16-09", "MOT16-11"]
    for seq in sequences:
        seq_dir = os.path.join(VIDEOS_DIR, seq)
        if not os.path.isdir(seq_dir):
            print(f"SKIP: {seq_dir} not found")
            continue
        output_file = os.path.join(
            OUTPUT_DIR, f"{seq}_{detector_name}_{reid_name}_body_reid.txt")
        run(seq_dir, detector_name, reid_name, output_file, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="DeepSORT + Body REID")
    parser.add_argument("--sequence_dir", default=None)
    parser.add_argument("--detector", default="yolo11s")
    parser.add_argument("--reid", default="osnet_x0_25")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf_threshold", type=float, default=0.4)
    parser.add_argument("--max_cosine_distance", type=float, default=0.2)
    parser.add_argument("--nn_budget", type=int, default=100)
    parser.add_argument("--max_age", type=int, default=30)
    parser.add_argument("--n_init", type=int, default=3)
    parser.add_argument("--max_iou_distance", type=float, default=0.7)
    parser.add_argument("--body_reid_threshold", type=float, default=0.3)
    parser.add_argument("--temporal_window", type=int, default=30)
    parser.add_argument("--min_votes", type=int, default=1)
    parser.add_argument("--conflict_strategy", default="reset",
                        choices=["reset", "keep_best"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list_detectors", action="store_true")
    parser.add_argument("--list_reid", action="store_true")
    args = parser.parse_args()

    if args.list_detectors:
        print("Available detectors:", list_detectors())
        exit(0)
    if args.list_reid:
        print("Available REID models:", list_reid_models())
        exit(0)

    if args.all:
        run_all(args.detector, args.reid, device=args.device,
                conf_threshold=args.conf_threshold,
                max_cosine_distance=args.max_cosine_distance,
                nn_budget=args.nn_budget, max_age=args.max_age,
                n_init=args.n_init, max_iou_distance=args.max_iou_distance,
                body_reid_threshold=args.body_reid_threshold,
                temporal_window=args.temporal_window,
                min_votes=args.min_votes,
                conflict_strategy=args.conflict_strategy)
    elif args.sequence_dir:
        suffix = f"{args.detector}_{args.reid}_body_reid"
        output_file = args.output_file or os.path.join(
            OUTPUT_DIR, f"{os.path.basename(args.sequence_dir)}_{suffix}.txt")
        run(args.sequence_dir, args.detector, args.reid, output_file,
            device=args.device, conf_threshold=args.conf_threshold,
            max_cosine_distance=args.max_cosine_distance,
            nn_budget=args.nn_budget, max_age=args.max_age,
            n_init=args.n_init, max_iou_distance=args.max_iou_distance,
            body_reid_threshold=args.body_reid_threshold,
            temporal_window=args.temporal_window,
            min_votes=args.min_votes,
            conflict_strategy=args.conflict_strategy)
    else:
        print("Specify --sequence_dir or --all.")


if __name__ == "__main__":
    main()
