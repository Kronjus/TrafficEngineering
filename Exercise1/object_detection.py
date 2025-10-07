import json

import cv2
from ultralytics import solutions

video_path = 'data/footage/DJI_20251006180546_0001_D.MP4'
roi_file = 'roi_coordinates.json'

with open(roi_file, "r") as f:
    loaded = json.load(f)
roi_points_list = [tuple(x) for x in loaded]

cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("data/results/trackzone_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

trackzone = solutions.TrackZone(
    show=False,
    region=roi_points_list,
    model="data/model.pt",
    verbose=True,
    classes=[3, 4, 5],
    device=0,
)
n = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = trackzone(im0)
    video_writer.write(results.plot_im)
    if n % 25 == 0:
        print(results.total_tracks)
    n += 1

cap.release()
video_writer.release()
cv2.destroyAllWindows()
