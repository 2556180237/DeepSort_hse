import os
import numpy as np
import cv2
import colorsys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OVERLAYS_DIR = os.path.join(BASE_DIR, "overlays", "baseline")


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
    info = {}
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('['):
                    k, v = line.split('=', 1)
                    info[k.strip()] = v.strip()
    return info


def generate_overlay(seq_name):
    seq_dir = os.path.join(VIDEOS_DIR, seq_name)
    result_file = os.path.join(OUTPUT_DIR, f"{seq_name}.txt")
    img_dir = os.path.join(seq_dir, "img1")

    if not os.path.isdir(img_dir):
        print(f"  SKIP: no img1 dir for {seq_name}")
        return False
    if not os.path.exists(result_file):
        print(f"  SKIP: no result file for {seq_name}")
        return False

    info = read_seqinfo(seq_dir)
    frame_rate = int(info.get("frameRate", 25))
    im_width = int(info.get("imWidth", 640))
    im_height = int(info.get("imHeight", 480))

    image_files = sorted(
        [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(os.path.splitext(x)[0]))
    if not image_files:
        print(f"  SKIP: no images for {seq_name}")
        return False

    results = load_mot_file(result_file)

    os.makedirs(OVERLAYS_DIR, exist_ok=True)
    out_path = os.path.join(OVERLAYS_DIR, f"{seq_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, frame_rate, (im_width, im_height))

    print(f"  Generating {seq_name}: {len(image_files)} frames -> {out_path}")

    for img_file in image_files:
        frame_idx = int(os.path.splitext(img_file)[0])
        img_path = os.path.join(img_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        if results is not None and len(results) > 0:
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
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                label_pt = (pt1[0], max(pt1[1] - 5, text_size[1] + 5))
                cv2.rectangle(image, pt1, (pt1[0] + text_size[0] + 8, pt1[1] - text_size[1] - 8), color, -1)
                cv2.putText(image, label, (pt1[0] + 2, label_pt[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(image)

    writer.release()
    print(f"  DONE: {out_path}")
    return True


def main():
    sequences = ["TUD-Campus", "TUD-Stadtmitte", "KITTI-17",
                 "PETS09-S2L1", "MOT16-09", "MOT16-11"]

    print("Generating baseline overlays...")
    print(f"Output dir: {OVERLAYS_DIR}")
    print("=" * 60)

    success = 0
    for seq in sequences:
        print(f"\n[{seq}]")
        if generate_overlay(seq):
            success += 1

    print(f"\n{'=' * 60}")
    print(f"Done: {success}/{len(sequences)} overlays generated in {OVERLAYS_DIR}")


if __name__ == "__main__":
    main()
