"""
Object Counting Script Using YOLO and Region Counter

This script processes a video file to count vehicles within a specified region of interest (ROI) using a YOLO model.
It reads ROI coordinates from a JSON file, processes the video at a target frame rate, and writes the vehicle count
per frame to a CSV file. A progress bar is displayed during processing.

Modules:
    - csv: For writing vehicle counts to a CSV file.
    - json: For loading ROI coordinates from a JSON file.
    - cv2: OpenCV library for video capture and processing.
    - tqdm: For displaying a progress bar.
    - ultralytics.solutions: For the RegionCounter object.

Configuration:
    - MODEL_PATH: Path to the YOLO model file.
    - VIDEO_PATH: Path to the input video file.
    - ROI_PATH: Path to the JSON file containing ROI coordinates.
    - OUTPUT_CSV_PATH: Path to the output CSV file for vehicle counts.
    - TARGET_FPS: Target frames per second for processing.

Usage:
    Ensure the required files (model, video, ROI JSON) are in the specified paths. Run the script to process the video
    and generate a CSV file with vehicle counts per frame.
"""

import csv
import json

import cv2
from tqdm import tqdm
from ultralytics import solutions

# --- Configuration ---
MODEL_PATH = '../data/model.pt'  # Path to the YOLO model file
VIDEO_PATH = '../data/footage/DJI_20251006183040_0002_D.MP4'  # Path to the input video file
ROI_PATH = '../data/roi_coordinates.json'  # Path to the JSON file containing ROI coordinates
OUTPUT_CSV_PATH = '../outputs/counts_per_frame_video_2.csv'  # Path to the output CSV file
TARGET_FPS = 15  # Target frames per second for processing

# --- Load ROI Coordinates ---
try:
    with open(ROI_PATH, 'r') as f:
        load = json.load(f)  # Load ROI coordinates from the JSON file
    roi_points = [tuple(x) for x in load]  # Convert ROI points to a list of tuples
except FileNotFoundError:
    print(f"Error: ROI file not found at {ROI_PATH}")  # Handle missing ROI file
    exit()

# --- Initialize Video Capture ---
cap = cv2.VideoCapture(VIDEO_PATH)  # Open the video file
assert cap.isOpened(), f"Error reading video file at {VIDEO_PATH}"  # Ensure video file is opened

# --- Get Video Properties ---
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                                       cv2.CAP_PROP_FPS))  # Get video width, height, and FPS
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames

# --- Calculate frame skip interval for downsampling ---
if fps > TARGET_FPS:
    frame_skip = round(fps / TARGET_FPS)  # Calculate frame skip interval to match target FPS
else:
    frame_skip = 1  # No skipping if original FPS is less than or equal to target
print(f"Original FPS: {fps}. Target FPS: {TARGET_FPS}. Processing 1 of every {frame_skip} frames.")

# --- Initialize Region Counter ---
regioncounter = solutions.RegionCounter(
    show=False,  # Do not display the video during processing
    region=roi_points,  # Set the region of interest
    model=MODEL_PATH,  # Path to the YOLO model
    conf=0.67,  # Confidence threshold for detections
    verbose=False  # Disable verbose output
)

# --- Create CSV and write the header ONCE ---
with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Frame', 'Vehicles'])  # Write CSV header

# --- Process video and append to CSV in a loop ---
with tqdm(total=total_frames, desc="Processing Video Frames") as pbar:  # Initialize progress bar
    frame_count = 0  # Frame counter
    while cap.isOpened():
        success, im0 = cap.read()  # Read next frame

        if not success:
            print("\nVideo frame is empty or processing is complete.")  # End of video or error
            break

        if frame_count % frame_skip == 0:
            # Process frame with the counter
            results = regioncounter(im0)  # Get region counts for the frame

            # Append the count for the current frame to the CSV
            with open(OUTPUT_CSV_PATH, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                vehicle_count = results.region_counts.get('Region#01', 0)  # Get vehicle count for the region
                writer.writerow([frame_count, vehicle_count])  # Write frame number and count

        frame_count += 1  # Increment frame counter
        pbar.update(1)  # Update progress bar

# --- Cleanup ---
print(f"Frame-by-frame counts saved to {OUTPUT_CSV_PATH}")  # Notify user of completion
cap.release()  # Release video capture object
cv2.destroyAllWindows()  # Close any OpenCV windows
