"""Generate overlay videos for baseline and best implementation.

Creates side-by-side comparison videos:
  Left:  Baseline (Original DeepSORT)
  Right: YOLO11s + OSNet x0_25 (Best)

Each track is drawn with a colored bounding box and track ID.

Usage:
    python generate_overlays.py
    python generate_overlays.py --sequences TUD-Campus TUD-Stadtmitte
"""
import os
import argparse
import configparser
import numpy as np
import cv2
import colorsys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OVERLAYS_DIR = os.path.join(BASE_DIR, "overlays")


def create_unique_color(tag, hue_step=0.41):
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return int(255 * r), int(255 * g), int(255 * b)


def load_mot_file(filepath):
    if not os.path.exists(filepath):
        return None
    data = np.loadtxt(filepath, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def read_seqinfo(seq_dir):
    info_file = os.path.join(seq_dir, "seqinfo.ini")
    config = configparser.ConfigParser()
    if os.path.exists(info_file):
        config.read(info_file)
        if "Sequence" in config:
            return dict(config["Sequence"])
    return {}


def draw_tracks(image, results, frame_idx):
    """Draw bounding boxes with track IDs for a given frame."""
    if results is None or len(results) == 0:
        return
    mask = results[:, 0].astype(int) == frame_idx
    frame_tracks = results[mask]

    for row in frame_tracks:
        track_id = int(row[1])
        x, y, w, h = row[2:6]
        color = create_unique_color(track_id)
        pt1 = (int(x), int(y))
        pt2 = (int(x + w), int(y + h))
        cv2.rectangle(image, pt1, pt2, color, 2)

        label = str(track_id)
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_pt = (pt1[0], max(pt1[1] - 5, text_size[1] + 5))
        cv2.rectangle(image, pt1,
                      (pt1[0] + text_size[0] + 8, pt1[1] - text_size[1] - 8),
                      color, -1)
        cv2.putText(image, label, (pt1[0] + 2, label_pt[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def generate_comparison(seq_name, baseline_file, best_file):
    """Generate a side-by-side comparison video."""
    seq_dir = os.path.join(VIDEOS_DIR, seq_name)
    img_dir = os.path.join(seq_dir, "img1")

    if not os.path.isdir(img_dir):
        print(f"  SKIP: no img1 dir for {seq_name}")
        return False

    info = read_seqinfo(seq_dir)
    frame_rate = int(info.get("frameRate", "25"))

    image_files = sorted(
        [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(os.path.splitext(x)[0]))
    if not image_files:
        print(f"  SKIP: no images for {seq_name}")
        return False

    # Read first frame to get actual dimensions
    first_img = cv2.imread(os.path.join(img_dir, image_files[0]))
    im_height, im_width = first_img.shape[:2]

    baseline_results = load_mot_file(baseline_file)
    best_results = load_mot_file(best_file)

    out_dir = os.path.join(OVERLAYS_DIR, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{seq_name}_comparison.mp4")

    gap = 4
    label_h = 30
    combined_w = im_width * 2 + gap
    combined_h = im_height + label_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, frame_rate,
                             (combined_w, combined_h))

    total = len(image_files)
    print(f"  Generating {seq_name}: {total} frames -> {out_path}")

    for i, img_file in enumerate(image_files):
        frame_idx = int(os.path.splitext(img_file)[0])
        img_path = os.path.join(img_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        left = image.copy()
        right = image.copy()

        draw_tracks(left, baseline_results, frame_idx)
        draw_tracks(right, best_results, frame_idx)

        # Labels
        cv2.rectangle(left, (0, im_height),
                      (im_width, im_height + label_h), (0, 0, 0), -1)
        cv2.putText(left, "Baseline (Original DeepSORT)", (5, im_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

        cv2.rectangle(right, (0, im_height),
                      (im_width, im_height + label_h), (0, 0, 0), -1)
        cv2.putText(right, "YOLO11s + OSNet x0_25 (Best)",
                    (5, im_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

        combined = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
        combined[:im_height, :im_width] = left
        combined[:im_height, im_width + gap:] = right
        combined[:, im_width:im_width + gap] = (0, 255, 255)

        writer.write(combined)

        if (i + 1) % 100 == 0 or i == 0:
            print(f"\r  {seq_name}: frame {frame_idx} ({i+1}/{total})",
                  end="", flush=True)

    writer.release()
    print(f"\n  DONE: {out_path}")
    return True


def generate_single(seq_name, result_file, label, sub_dir):
    """Generate a single overlay video."""
    seq_dir = os.path.join(VIDEOS_DIR, seq_name)
    img_dir = os.path.join(seq_dir, "img1")

    if not os.path.isdir(img_dir):
        return False

    info = read_seqinfo(seq_dir)
    frame_rate = int(info.get("frameRate", "25"))

    image_files = sorted(
        [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(os.path.splitext(x)[0]))
    if not image_files:
        return False

    # Read first frame to get actual dimensions
    first_img = cv2.imread(os.path.join(img_dir, image_files[0]))
    im_height, im_width = first_img.shape[:2]

    results = load_mot_file(result_file)

    out_dir = os.path.join(OVERLAYS_DIR, sub_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{seq_name}.mp4")

    label_h = 30
    out_h = im_height + label_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, frame_rate,
                             (im_width, out_h))

    total = len(image_files)
    print(f"  Generating {seq_name}: {total} frames -> {out_path}")

    for i, img_file in enumerate(image_files):
        frame_idx = int(os.path.splitext(img_file)[0])
        image = cv2.imread(os.path.join(img_dir, img_file))
        if image is None:
            continue

        draw_tracks(image, results, frame_idx)

        # Pad to out_h: copy image to top, draw label in bottom bar
        frame = np.zeros((out_h, im_width, 3), dtype=np.uint8)
        frame[:im_height, :] = image
        cv2.rectangle(frame, (0, im_height), (im_width, out_h), (0, 0, 0), -1)
        cv2.putText(frame, f"{seq_name} — {label}", (5, im_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

        writer.write(frame)

        if (i + 1) % 100 == 0 or i == 0:
            print(f"\r  {seq_name}: frame {frame_idx} ({i+1}/{total})",
                  end="", flush=True)

    writer.release()
    print(f"\n  DONE: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate overlay videos")
    parser.add_argument("--sequences", nargs='+',
                        default=["TUD-Campus", "TUD-Stadtmitte", "KITTI-17",
                                 "PETS09-S2L1", "MOT16-09", "MOT16-11"])
    parser.add_argument("--mode", choices=["comparison", "separate", "both"],
                        default="comparison",
                        help="comparison: side-by-side; separate: individual; both: all")
    args = parser.parse_args()

    best_suffix = "yolo11s_osnet_x0_25"
    sequences = args.sequences

    print(f"\n{'=' * 60}")
    print(f"Generating overlays for {len(sequences)} sequences")
    print(f"Mode: {args.mode}")
    print(f"Output: {OVERLAYS_DIR}/")
    print(f"{'=' * 60}")

    success = 0
    for seq in sequences:
        print(f"\n[{seq}]")
        baseline_file = os.path.join(OUTPUT_DIR, f"{seq}.txt")
        best_file = os.path.join(OUTPUT_DIR, f"{seq}_{best_suffix}.txt")

        if not os.path.exists(baseline_file):
            print(f"  SKIP: baseline file not found: {baseline_file}")
            continue
        if not os.path.exists(best_file):
            print(f"  SKIP: best file not found: {best_file}")
            continue

        ok = False
        if args.mode in ("comparison", "both"):
            ok = generate_comparison(seq, baseline_file, best_file) or ok
        if args.mode in ("separate", "both"):
            generate_single(seq, baseline_file,
                            "Baseline (Original DeepSORT)", "baseline")
            generate_single(seq, best_file,
                            "YOLO11s + OSNet x0_25 (Best)", "best")
            ok = True

        if ok:
            success += 1

    print(f"\n{'=' * 60}")
    print(f"Done: {success}/{len(sequences)} sequences processed")
    print(f"Overlays in: {OVERLAYS_DIR}/")


if __name__ == "__main__":
    main()
