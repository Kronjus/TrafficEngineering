import json

import numpy as np
import supervision as sv
from ultralytics import YOLO

roi_file = '../data/roi_coordinates.json'

with open(roi_file, "r") as f:
    loaded = json.load(f)
roi_points_list = np.array([tuple(x) for x in loaded])

model = YOLO('../data/model.pt')
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
zone = sv.PolygonZone(roi_points_list)
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()


def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    mask = zone.trigger(detections=detections)
    detections = detections[mask & np.isin(detections.class_id, [3, 4, 5])]
    labels = [
        f"#{tracker_id} {class_name}" for class_name, tracker_id in zip(detections.data['class_name'], detections.tracker_id)
    ]
    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
    return trace_annotator.annotate(annotated_frame, detections=detections)


sv.process_video(source_path='../data/footage/DJI_20251006180546_0001_D.MP4',
                 target_path='../outputs/result_video1.mp4',
                 callback=callback)
