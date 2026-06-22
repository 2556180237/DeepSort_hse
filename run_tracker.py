"""DeepSORT tracking pipeline with swappable detectors.

Usage:
    # Run with YOLO11s detector
    python run_tracker.py --sequence_dir videos/TUD-Campus --detector yolo11s --output_file output/TUD-Campus_yolo11s.txt

    # Run with YOLOv8n detector
    python run_tracker.py --sequence_dir videos/TUD-Campus --detector yolov8n --output_file output/TUD-Campus_yolov8n.txt

    # Run with RT-DETR detector
    python run_tracker.py --sequence_dir videos/TUD-Campus --detector rtdetr-r50 --output_file output/TUD-Campus_rtdetr.txt

    # Run on all videos
    python run_tracker.py --all --detector yolo11s

    # List available detectors
    python run_tracker.py --list_detectors
"""
import argparse
import os
import time

import cv2
import numpy as np

from application_util import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from detectors import create_detector, list_detectors
from reid import create_reid, list_reid_models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def gather_sequence_info(sequence_dir):
    """Gather sequence information from MOTChallenge directory."""
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))}

    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())), cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        image_size = None
        min_frame_idx = 0
        max_frame_idx = 0

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    update_ms = None
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)
        update_ms = 1000 / int(info_dict["frameRate"])

    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "update_ms": update_ms
    }
    return seq_info


def run(sequence_dir, detector_name, output_file, device="cpu",
        conf_threshold=0.3, nms_max_overlap=1.0, max_cosine_distance=0.2,
        nn_budget=100, reid_name=None):
    """Run DeepSORT with a live detector on a sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to MOTChallenge sequence directory.
    detector_name : str
        Registered detector name (e.g. "yolo11s", "yolov8n", "rtdetr-r50").
    output_file : str
        Path to tracking output file.
    device : str
        "cpu" or "cuda".
    conf_threshold : float
        Detection confidence threshold.
    nms_max_overlap : float
        NMS threshold.
    max_cosine_distance : float
        Cosine distance gating threshold.
    nn_budget : int or None
        Appearance descriptor gallery size.
    reid_name : str or None
        REID model name. If None, uses random features (baseline behavior).
    """
    seq_info = gather_sequence_info(sequence_dir)
    seq_name = seq_info["sequence_name"]

    print(f"Sequence: {seq_name}")
    print(f"Frames: {seq_info['min_frame_idx']}..{seq_info['max_frame_idx']}")
    print(f"Detector: {detector_name} (device={device})")
    print(f"REID: {reid_name or 'random_features'}")

    detector = create_detector(detector_name, device=device,
                               conf_threshold=conf_threshold)
    print(f"Loading detector: {detector}")
    detector.load_model()
    print("Detector loaded.")

    reid_model = None
    if reid_name is not None:
        reid_model = create_reid(reid_name, device=device)
        print(f"Loading REID: {reid_model}")
        reid_model.load_model()
        print("REID loaded.")

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    np.random.seed(42)
    frame_indices = sorted(seq_info["image_filenames"].keys())
    total_frames = len(frame_indices)
    start_time = time.time()

    for i, frame_idx in enumerate(frame_indices):
        print(f"\rProcessing frame {frame_idx:05d}/{seq_info['max_frame_idx']} "
              f"({i+1}/{total_frames})", end="", flush=True)

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

        boxes = np.array([d.tlwh for d in detections])
        if len(boxes) > 0:
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes, nms_max_overlap, scores)
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

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for row in results:
            f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n' % (
                row[0], row[1], row[2], row[3], row[4], row[5]))
    print(f"Results saved to {output_file}")

    return fps


def run_all(detector_name, device="cpu", conf_threshold=0.3, reid_name=None, **kwargs):
    """Run tracker on all 6 test videos."""
    sequences = ["TUD-Campus", "TUD-Stadtmitte", "KITTI-17",
                 "PETS09-S2L1", "MOT16-09", "MOT16-11"]

    all_fps = []
    for seq in sequences:
        seq_dir = os.path.join(VIDEOS_DIR, seq)
        if not os.path.isdir(seq_dir):
            print(f"SKIP: {seq_dir} not found")
            continue
        suffix = f"{detector_name}_{reid_name}" if reid_name else detector_name
        output_file = os.path.join(OUTPUT_DIR, f"{seq}_{suffix}.txt")
        fps = run(seq_dir, detector_name, output_file, device=device,
                  conf_threshold=conf_threshold, reid_name=reid_name, **kwargs)
        all_fps.append(fps)

    print(f"\n{'='*60}")
    print(f"Average FPS: {np.mean(all_fps):.2f}")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSORT with swappable detectors")
    parser.add_argument("--sequence_dir", help="Path to MOTChallenge sequence directory",
                        default=None)
    parser.add_argument("--detector", help="Detector name", default="yolo11s")
    parser.add_argument("--output_file", help="Output file path", default=None)
    parser.add_argument("--device", help="cpu or cuda", default="cpu")
    parser.add_argument("--conf_threshold", type=float, default=0.3)
    parser.add_argument("--nms_max_overlap", type=float, default=1.0)
    parser.add_argument("--max_cosine_distance", type=float, default=0.2)
    parser.add_argument("--nn_budget", type=int, default=100)
    parser.add_argument("--reid", help="REID model name", default=None)
    parser.add_argument("--all", action="store_true", help="Run on all videos")
    parser.add_argument("--list_detectors", action="store_true", help="List available detectors")
    parser.add_argument("--list_reid", action="store_true", help="List available REID models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_detectors:
        print("Available detectors:", list_detectors())
        exit(0)

    if args.list_reid:
        print("Available REID models:", list_reid_models())
        exit(0)

    if args.all:
        run_all(args.detector, device=args.device,
                conf_threshold=args.conf_threshold,
                nms_max_overlap=args.nms_max_overlap,
                max_cosine_distance=args.max_cosine_distance,
                nn_budget=args.nn_budget,
                reid_name=args.reid)
    elif args.sequence_dir:
        suffix = f"{args.detector}_{args.reid}" if args.reid else args.detector
        output_file = args.output_file or os.path.join(
            OUTPUT_DIR, f"{os.path.basename(args.sequence_dir)}_{suffix}.txt")
        run(args.sequence_dir, args.detector, output_file, device=args.device,
            conf_threshold=args.conf_threshold,
            nms_max_overlap=args.nms_max_overlap,
            max_cosine_distance=args.max_cosine_distance,
            nn_budget=args.nn_budget,
            reid_name=args.reid)
    else:
        print("Specify --sequence_dir or --all. Use --list_detectors to see options.")
