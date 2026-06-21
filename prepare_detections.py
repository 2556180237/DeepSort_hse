import os
import numpy as np

videos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "detections")
os.makedirs(output_dir, exist_ok=True)

feature_dim = 128
np.random.seed(42)

for seq_name in os.listdir(videos_dir):
    seq_dir = os.path.join(videos_dir, seq_name)
    det_file = os.path.join(seq_dir, "det", "det.txt")
    if not os.path.isfile(det_file):
        continue
    print(f"Processing {seq_name}...")
    detections = np.loadtxt(det_file, delimiter=',')
    if detections.ndim == 1:
        detections = detections.reshape(1, -1)
    n = detections.shape[0]
    features = np.random.rand(n, feature_dim).astype(np.float32)
    detections_out = np.hstack([detections, features])
    out_path = os.path.join(output_dir, f"{seq_name}.npy")
    np.save(out_path, detections_out)
    print(f"  Saved {n} detections to {out_path}")

print("Done!")
