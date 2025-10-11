import json
import cv2
import numpy as np
from tqdm import tqdm

INPUT_VIDEO_PATH = '../data/footage/DJI_20251006180546_0001_D.MP4'
OUTPUT_VIDEO_PATH = '../outputs/georeferenced_video.mp4'
GCP_FILE = '../data/gcp.txt'
RESOLUTION = 0.1

with open(GCP_FILE, 'r') as f:
    gcps = json.load(f)

print(f"Loaded {len(gcps)} ground control points.")

src_pts = np.array([p['pixel'] for p in gcps], dtype=np.float32)
lv95_pts = np.array([p['lv95'] for p in gcps], dtype=np.float32)

min_e = np.min(lv95_pts[:, 0])
max_e = np.max(lv95_pts[:, 0])
min_n = np.min(lv95_pts[:, 1])
max_n = np.max(lv95_pts[:, 1])

output_width = int(np.ceil(max_e - min_e) / RESOLUTION)
output_height = int(np.ceil(max_n - min_n) / RESOLUTION)

print(f"Output map will be {output_width}x{output_height} pixels.")

dst_pts = np.array([
    [(p[0] - min_e) / RESOLUTION, (max_n - p[1]) / RESOLUTION]
    for p in lv95_pts
], dtype=np.float32)

H_master, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (output_width, output_height))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(nfeatures=2000)
prev_kps, prev_descs = orb.detectAndCompute(prev_gray, None)

H_motion = np.identity(3)

frame_idx = 1  # Already read the first frame

with tqdm(total=total_frames, desc="Processing frames") as pbar:
    while ret:
        H_final = H_master @ np.linalg.inv(H_motion)

        warped_frame = cv2.warpPerspective(prev_frame, H_final, (output_width, output_height))
        out.write(warped_frame)

        ret, current_frame = cap.read()
        if not ret:
            break

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_kps, current_descs = orb.detectAndCompute(current_gray, None)

        if prev_descs is not None and current_descs is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(prev_descs, current_descs)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:100]

            if len(good_matches) > 10:
                src_match_pts = np.float32([prev_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_match_pts = np.float32([current_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                H_inc, _ = cv2.findHomography(src_match_pts, dst_match_pts, cv2.RANSAC, 5.0)

                if H_inc is not None:
                    H_motion = H_motion @ H_inc

        prev_frame = current_frame
        prev_kps, prev_descs = current_kps, current_descs

        frame_idx += 1
        pbar.update(1)

print("Finished processing. Cleaning up.")
cap.release()
out.release()
cv2.destroyAllWindows()
