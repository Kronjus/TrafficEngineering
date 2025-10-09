"""
Object Tracking Script

This script processes a video file to perform object tracking using a YOLO model. It detects objects within a
specified region of interest (ROI) and exports the tracking data to a CSV file. The script uses a target frame
rate to control the frequency of detection.

Modules:
    - json: For loading ROI coordinates from a JSON file.
    - cv2: OpenCV library for video processing.
    - numpy: For handling numerical operations and arrays.
    - supervision: For object tracking and detection utilities.
    - tqdm: For displaying a progress bar during video processing.
    - ultralytics: For loading and using the YOLO model.

Configuration:
    - MODEL_PATH: Path to the YOLO model file.
    - VIDEO_PATH: Path to the input video file.
    - ROI_PATH: Path to the JSON file containing ROI coordinates.
    - OUTPUT_CSV_PATH: Path to the output CSV file for tracking data.
    - TARGET_FPS: Target frames per second for processing.

Usage:
    Ensure the required files (model, video, ROI JSON) are in the specified paths. Run the script to process the video
    and generate a CSV file with tracking data.
"""

import json

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = '../data/model.pt'  # Path to the YOLO model file
VIDEO_PATH = '../data/footage/DJI_20251006180546_0001_D.MP4'  # Path to the input video file
ROI_PATH = '../data/roi_coordinates.json'  # Path to the JSON file containing ROI coordinates
OUTPUT_CSV_PATH = '../outputs/tracks_per_frame.csv'  # Path to the output CSV file
TARGET_FPS = 15  # Target frames per second for processing

# --- Load ROI Coordinates ---
try:
    with open(ROI_PATH, 'r') as f:
        load = json.load(f)  # Load ROI coordinates from the JSON file
    roi_points = np.array([tuple(x) for x in load])  # Convert ROI points to a NumPy array
except FileNotFoundError:
    print(f"Error: ROI file not found at {ROI_PATH}")  # Handle missing ROI file
    exit()

# --- Get Video Info ---
cap = cv2.VideoCapture(VIDEO_PATH)  # Open the video file
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video
original_fps = cap.get(cv2.CAP_PROP_FPS)  # Get the original FPS of the video
frame_interval = int(original_fps // TARGET_FPS)  # Calculate the frame interval for target FPS
frame_idx = 0  # Initialize the frame index

# --- Load Model ---
model = YOLO(MODEL_PATH)  # Load the YOLO model
tracker = sv.ByteTrack()  # Initialize the ByteTrack tracker
smoother = sv.DetectionsSmoother()  # Initialize the detections smoother
csv_sink = sv.CSVSink(OUTPUT_CSV_PATH)  # Initialize the CSV sink for writing tracking data
frames_generator = sv.get_video_frames_generator(VIDEO_PATH)  # Create a generator for video frames

zone = sv.PolygonZone(roi_points)  # Define the ROI as a polygon zone

# --- Process Video Frames ---
with csv_sink as sink:  # Open the CSV sink for writing
    with tqdm(total=total_frames, desc="Processing Video Frames") as pbar:  # Initialize progress bar
        for frame in frames_generator:  # Iterate over video frames
            if frame_idx % frame_interval == 0:  # Process only frames at the specified interval
                results = model(frame, verbose=False)[0]  # Perform object detection on the frame
                detections = sv.Detections.from_ultralytics(results)  # Convert YOLO results to detections
                mask = zone.trigger(detections=detections)  # Check if detections are within the ROI
                detections = detections[mask & np.isin(detections.class_id, [3, 4, 5])]  # Filter detections by class ID
                detections = tracker.update_with_detections(detections)  # Update tracker with filtered detections
                detections = smoother.update_with_detections(detections)  # Smooth the detections
                sink.append(detections, custom_data={'frame': frame_idx})  # Write detections to the CSV file
            frame_idx += 1  # Increment the frame index
            pbar.update(1)  # Update the progress bar

# --- Cleanup ---
cap.release()  # Release the video capture object