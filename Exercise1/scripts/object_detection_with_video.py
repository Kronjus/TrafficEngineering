import csv
import json

import cv2
from tqdm import tqdm
from ultralytics import solutions

# --- Configuration ---
MODEL_PATH = '../data/model.pt'
VIDEO_PATH = '../data/footage/DJI_20251006180546_0001_D.MP4'
ROI_PATH = '../data/roi_coordinates.json'
OUTPUT_VIDEO_PATH = '../outputs/region_counting.avi'
OUTPUT_CSV_PATH = '../outputs/counts_per_frame.csv'
TARGET_FPS = 15

# --- Load ROI Coordinates ---
try:
    with open(ROI_PATH, 'r') as f:
        load = json.load(f)
    roi_points = [tuple(x) for x in load]
except FileNotFoundError:
    print(f"Error: ROI file not found at {ROI_PATH}")
    exit()

# --- Initialize Video Capture ---
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f"Error reading video file at {VIDEO_PATH}"

# --- Get Video Properties ---
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --- Calculate frame skip interval for downsampling ---
if fps > TARGET_FPS:
    frame_skip = round(fps / TARGET_FPS)
else:
    frame_skip = 1
print(f"Original FPS: {fps}. Target FPS: {TARGET_FPS}. Processing 1 of every {frame_skip} frames.")

# --- Initialize Region Counter ---
regioncounter = solutions.RegionCounter(
    show=False,
    region=roi_points,
    model=MODEL_PATH,
    conf=0.67,
    verbose=False
)

# --- Create CSV and write the header ONCE ---
with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Frame', 'Vehicles'])

# --- Process video and append to CSV in a loop ---
with tqdm(total=total_frames, desc="Processing Video Frames") as pbar:
    frame_count = 0
    while cap.isOpened():
        success, im0 = cap.read()

        if not success:
            print("\nVideo frame is empty or processing is complete.")
            break

        if frame_count % frame_skip == 0:
            # Process frame with the counter
            results = regioncounter(im0)

            # Append the count for the current frame to the CSV
            with open(OUTPUT_CSV_PATH, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                vehicle_count = results.region_counts.get('Region#01', 0)
                writer.writerow([frame_count, vehicle_count])

        frame_count += 1
        pbar.update(1)

# --- Cleanup ---
print(f"\nProcessing complete. Output video saved to {OUTPUT_VIDEO_PATH}")
print(f"Frame-by-frame counts saved to {OUTPUT_CSV_PATH}")
cap.release()
cv2.destroyAllWindows()
