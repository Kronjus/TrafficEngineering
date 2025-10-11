import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm


# --------------
# Data classes
# --------------
@dataclass
class GeorefMeta:
    resolution: float  # meters per pixel
    min_e: float  # LV95 min easting bounding box used for the warp
    max_n: float  # LV95 max northing bounding box used for the warp
    width: int  # pixels
    height: int  # pixels


# --------------
# Utilities
# --------------
def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def save_json(path: str, data: dict, indent: int = 2):
    ensure_parent_dir(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# --------------
# Georeferencing helpers
# --------------
def compute_georef_params_from_gcps(gcp_file: str, resolution_m_per_px: float, ransac_reproj_threshold: float = 5.0):
    gcps = load_json(gcp_file)
    if not isinstance(gcps, list) or len(gcps) < 4:
        raise ValueError("GCP file must contain at least 4 points with 'pixel' and 'lv95'.")

    src_pts = np.array([p["pixel"] for p in gcps], dtype=np.float32)
    lv95_pts = np.array([p["lv95"] for p in gcps], dtype=np.float32)

    min_e = float(np.min(lv95_pts[:, 0]))
    max_e = float(np.max(lv95_pts[:, 0]))
    min_n = float(np.min(lv95_pts[:, 1]))
    max_n = float(np.max(lv95_pts[:, 1]))

    out_w = int(np.ceil((max_e - min_e) / resolution_m_per_px))
    out_h = int(np.ceil((max_n - min_n) / resolution_m_per_px))

    # destination points map LV95 → pixel coords of output canvas
    dst_pts = np.array(
        [[(p[0] - min_e) / resolution_m_per_px, (max_n - p[1]) / resolution_m_per_px] for p in lv95_pts],
        dtype=np.float32,
    )

    H_master, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)
    if H_master is None:
        raise RuntimeError("Failed to compute master homography from GCPs.")

    meta = GeorefMeta(
        resolution=resolution_m_per_px,
        min_e=min_e,
        max_n=max_n,
        width=out_w,
        height=out_h,
    )
    return H_master, meta


def lv95_from_georef_px(meta: GeorefMeta, cx: float, cy: float) -> Tuple[float, float]:
    E = meta.min_e + cx * meta.resolution
    N = meta.max_n - cy * meta.resolution
    return float(E), float(N)


# --------------
# Stage 1 (batch georef) kept for completeness
# --------------
def georeference_video(
        input_video: str,
        gcp_file: str,
        resolution_m_per_px: float,
        output_video: str,
        output_meta_json: Optional[str] = None,
        feature_count: int = 2000,
        ransac_reproj_threshold: float = 5.0,
) -> GeorefMeta:
    H_master, meta = compute_georef_params_from_gcps(gcp_file, resolution_m_per_px, ransac_reproj_threshold)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ensure_parent_dir(output_video)
    out = cv2.VideoWriter(output_video, fourcc, fps, (meta.width, meta.height))

    # Initialize motion model with ORB + BFMatcher
    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        cap.release()
        out.release()
        raise RuntimeError("Could not read first frame from input.")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=feature_count)
    prev_kps, prev_descs = orb.detectAndCompute(prev_gray, None)

    H_motion = np.eye(3, dtype=np.float32)

    with tqdm(total=total_frames if total_frames > 0 else None, desc="Georeferencing") as pbar:
        while True:
            H_final = H_master @ np.linalg.inv(H_motion)
            warped = cv2.warpPerspective(prev_frame, H_final, (meta.width, meta.height))
            out.write(warped)

            ret, cur_frame = cap.read()
            if not ret:
                break

            cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
            cur_kps, cur_descs = orb.detectAndCompute(cur_gray, None)

            if prev_descs is not None and cur_descs is not None and len(prev_descs) > 0 and len(cur_descs) > 0:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(prev_descs, cur_descs)
                if matches:
                    matches = sorted(matches, key=lambda m: m.distance)[:100]
                    if len(matches) >= 10:
                        src_m = np.float32([prev_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                        dst_m = np.float32([cur_kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                        H_inc, _ = cv2.findHomography(src_m, dst_m, cv2.RANSAC, 5.0)
                        if H_inc is not None:
                            H_motion = H_motion @ H_inc

            prev_frame, prev_kps, prev_descs = cur_frame, cur_kps, cur_descs
            if total_frames > 0:
                pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if output_meta_json:
        save_json(output_meta_json, {
            "resolution": meta.resolution,
            "min_e": meta.min_e,
            "max_n": meta.max_n,
            "width": meta.width,
            "height": meta.height,
        })

    return meta


# --------------
# Stage 2: ROI selection (on georeferenced first frame)
# --------------
class _ROISelector:
    def __init__(self, image: np.ndarray, win_name: str = "Select ROI"):
        self.image = image
        self.win = win_name
        self.points: List[Tuple[int, int]] = []
        self.frame_draw = image.copy()
        self.scale = 1.0
        self.ox = 0
        self.oy = 0

    @staticmethod
    def _fit_and_letterbox(img, target_w, target_h):
        h, w = img.shape[:2]
        if target_w <= 0 or target_h <= 0:
            return img.copy(), 1.0, 0, 0
        s = min(target_w / w, target_h / h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x0 = (target_w - nw) // 2
        y0 = (target_h - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas, s, x0, y0

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.frame_draw is not None:
            fx, fy = (x - self.ox), (y - self.oy)
            if fx < 0 or fy < 0 or self.scale <= 0:
                return
            ox, oy = int(round(fx / self.scale)), int(round(fy / self.scale))
            H, W = self.frame_draw.shape[:2]
            if 0 <= ox < W and 0 <= oy < H:
                self.points.append((ox, oy))
                cv2.circle(self.frame_draw, (ox, oy), 5, (0, 0, 255), -1)
                if len(self.points) > 1:
                    cv2.line(self.frame_draw, self.points[-2], self.points[-1], (0, 255, 0), 2)

    def run(self) -> List[Tuple[int, int]]:
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 960, 720)
        cv2.setMouseCallback(self.win, self._mouse)

        print("Instructions:\n"
              " - Click to add polygon points.\n"
              " - Press 's' to save.\n"
              " - Press 'u' to undo last point.\n"
              " - Press 'r' to reset.\n"
              " - Press 'q' to quit without saving.")

        saved_points = None
        while True:
            try:
                _, _, ww, wh = cv2.getWindowImageRect(self.win)
            except Exception:
                ww, wh = 960, 720
            display, self.scale, self.ox, self.oy = self._fit_and_letterbox(self.frame_draw, ww, wh)
            cv2.imshow(self.win, display)

            key = cv2.waitKey(16) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('u'):
                if self.points:
                    self.points.pop()
                    self.frame_draw = self.image.copy()
                    for i, pt in enumerate(self.points):
                        cv2.circle(self.frame_draw, pt, 5, (0, 0, 255), -1)
                        if i > 0:
                            cv2.line(self.frame_draw, self.points[i - 1], self.points[i], (0, 255, 0), 2)
            elif key == ord('r'):
                self.points.clear()
                self.frame_draw = self.image.copy()
            elif key == ord('s'):
                if len(self.points) >= 3:
                    saved_points = list(self.points)
                    break
                else:
                    print("Need at least 3 points.")

        cv2.destroyAllWindows()
        if saved_points is None:
            raise RuntimeError("ROI not saved.")
        return saved_points


def select_and_save_roi_on_frame(frame: np.ndarray, roi_json_out: str) -> List[Tuple[int, int]]:
    selector = _ROISelector(frame)
    pts = selector.run()
    ensure_parent_dir(roi_json_out)
    with open(roi_json_out, "w") as f:
        json.dump(pts, f, indent=2)
    print(f"Saved ROI to: {roi_json_out}")
    return pts


# --------------
# Stage 3a: Streaming georef + tracking per frame
# --------------
def stream_track_with_on_the_fly_georef(
        input_video: str,
        gcp_json: str,
        model_path: str,
        csv_out: str,
        roi_json: Optional[str] = None,
        resolution: float = 0.1,
        target_fps: int = 15,
        feature_count: int = 2000,
        class_ids: Optional[List[int]] = None,
):
    import supervision as sv
    from ultralytics import YOLO

    H_master, meta = compute_georef_params_from_gcps(gcp_json, resolution)

    # Prepare ORB tracking for inter-frame motion
    orb = cv2.ORB_create(nfeatures=feature_count)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(round(original_fps / float(target_fps))))

    # Read first frame and set ROI (georeferenced)
    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame.")

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_kps, prev_descs = orb.detectAndCompute(prev_gray, None)
    H_motion = np.eye(3, dtype=np.float32)
    H_final = H_master @ np.linalg.inv(H_motion)
    first_warped = cv2.warpPerspective(first_frame, H_final, (meta.width, meta.height))

    if roi_json and os.path.exists(roi_json):
        roi_points = np.array([tuple(x) for x in load_json(roi_json)], dtype=np.float32)
    else:
        # Interactive ROI selection on georeferenced first frame
        pts = select_and_save_roi_on_frame(first_warped, roi_json or "./roi.json")
        roi_points = np.array(pts, dtype=np.float32)

    zone = sv.PolygonZone(roi_points)

    # Model + trackers + sinks
    model = YOLO(model_path)
    tracker = sv.ByteTrack()
    smoother = sv.DetectionsSmoother()
    ensure_parent_dir(csv_out)
    sink = sv.CSVSink(csv_out)

    # Process remaining frames (including the first we already warped)
    frames_processed = 0
    frame_idx = 0
    with sink as s, tqdm(total=total_frames if total_frames > 0 else None, desc="Stream tracking") as pbar:
        # handle first warped frame
        if frame_idx % frame_interval == 0:
            results = model(first_warped, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            mask = zone.trigger(detections=detections)
            if class_ids is not None:
                class_mask = np.isin(detections.class_id, class_ids)
                mask = mask & class_mask
            detections = detections[mask]
            detections = tracker.update_with_detections(detections)
            detections = smoother.update_with_detections(detections)

            custom_rows = {"frame": frame_idx}
            if len(detections) > 0:
                centers_lv95 = []
                for xyxy in detections.xyxy:
                    x1, y1, x2, y2 = xyxy
                    cx = 0.5 * (x1 + x2)
                    cy = 0.5 * (y1 + y2)
                    E, N = lv95_from_georef_px(meta, cx, cy)
                    centers_lv95.append((E, N))
                custom_rows["lv95_centers"] = json.dumps(centers_lv95)
            s.append(detections, custom_data=custom_rows)

        frame_idx += 1
        pbar.update(1)

        while True:
            ok, cur_frame = cap.read()
            if not ok:
                break

            cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
            cur_kps, cur_descs = orb.detectAndCompute(cur_gray, None)

            if prev_descs is not None and cur_descs is not None and len(prev_descs) > 0 and len(cur_descs) > 0:
                matches = bf.match(prev_descs, cur_descs)
                if matches:
                    matches = sorted(matches, key=lambda m: m.distance)[:100]
                    if len(matches) >= 10:
                        src_m = np.float32([prev_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                        dst_m = np.float32([cur_kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                        H_inc, _ = cv2.findHomography(src_m, dst_m, cv2.RANSAC, 5.0)
                        if H_inc is not None:
                            H_motion = H_motion @ H_inc

            # Warp current frame with updated motion
            H_final = H_master @ np.linalg.inv(H_motion)
            warped = cv2.warpPerspective(cur_frame, H_final, (meta.width, meta.height))

            if frame_idx % frame_interval == 0:
                results = model(warped, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)

                mask = zone.trigger(detections=detections)
                if class_ids is not None:
                    class_mask = np.isin(detections.class_id, class_ids)
                    mask = mask & class_mask
                detections = detections[mask]

                detections = tracker.update_with_detections(detections)
                detections = smoother.update_with_detections(detections)

                custom_rows = {"frame": frame_idx}
                if len(detections) > 0:
                    centers_lv95 = []
                    for xyxy in detections.xyxy:
                        x1, y1, x2, y2 = xyxy
                        cx = 0.5 * (x1 + x2)
                        cy = 0.5 * (y1 + y2)
                        E, N = lv95_from_georef_px(meta, cx, cy)
                        centers_lv95.append((E, N))
                    custom_rows["lv95_centers"] = json.dumps(centers_lv95)

                s.append(detections, custom_data=custom_rows)

            prev_gray, prev_kps, prev_descs = cur_gray, cur_kps, cur_descs
            frame_idx += 1
            if total_frames > 0:
                pbar.update(1)

    cap.release()


# --------------
# Stage 3b: Batch tracking on already georeferenced video (from previous message)
# --------------
def track_in_roi(
        video_path: str,
        roi_json: str,
        model_path: str,
        csv_out: str,
        target_fps: int = 15,
        class_ids: Optional[List[int]] = None,
        georef_meta_json: Optional[str] = None,
):
    import supervision as sv
    from ultralytics import YOLO

    ensure_parent_dir(csv_out)

    roi_points = np.array([tuple(x) for x in load_json(roi_json)], dtype=np.float32)
    zone = sv.PolygonZone(roi_points)

    meta = None
    if georef_meta_json and os.path.exists(georef_meta_json):
        m = load_json(georef_meta_json)
        meta = GeorefMeta(
            resolution=float(m["resolution"]),
            min_e=float(m["min_e"]),
            max_n=float(m["max_n"]),
            width=int(m["width"]),
            height=int(m["height"]),
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(round(original_fps / float(target_fps))))

    model = YOLO(model_path)
    tracker = sv.ByteTrack()
    smoother = sv.DetectionsSmoother()

    frames = sv.get_video_frames_generator(video_path)
    sink = sv.CSVSink(csv_out)

    frame_idx = 0
    with sink as s, tqdm(total=total_frames if total_frames > 0 else None, desc="Tracking") as pbar:
        for frame in frames:
            if frame_idx % frame_interval == 0:
                results = model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)

                mask = zone.trigger(detections=detections)
                if class_ids is not None:
                    class_mask = np.isin(detections.class_id, class_ids)
                    mask = mask & class_mask
                detections = detections[mask]

                detections = tracker.update_with_detections(detections)
                detections = smoother.update_with_detections(detections)

                custom_rows = {"frame": frame_idx}
                if meta is not None and len(detections) > 0:
                    centers_lv95 = []
                    for xyxy in detections.xyxy:
                        x1, y1, x2, y2 = xyxy
                        cx = 0.5 * (x1 + x2)
                        cy = 0.5 * (y1 + y2)
                        E, N = lv95_from_georef_px(meta, cx, cy)
                        centers_lv95.append((E, N))
                    custom_rows["lv95_centers"] = json.dumps(centers_lv95)

                s.append(detections, custom_data=custom_rows)

            frame_idx += 1
            if total_frames > 0:
                pbar.update(1)

    cap.release()


# --------------
# Orchestrator CLI
# --------------
def main():
    parser = argparse.ArgumentParser(description="Georeference → ROI → Track → CSV pipeline (streaming or batch)")
    parser.add_argument("--input_video", required=True, help="Path to input video (non-georeferenced)")
    parser.add_argument("--gcp_json", required=True, help="Path to GCP json (list of {pixel:[x,y], lv95:[E,N]})")
    parser.add_argument("--model_path", required=True, help="Path to YOLO model file")
    parser.add_argument("--workdir", default="./outputs_pipeline", help="Working directory for outputs")
    parser.add_argument("--resolution", type=float, default=0.1, help="Meters per pixel for georeferencing")
    parser.add_argument("--target_fps", type=int, default=15, help="Target FPS for detection/tracking")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="Optional class IDs filter (e.g. 0 1 2)")

    # Batch mode (optional)
    parser.add_argument("--batch", action="store_true", help="Use batch mode: write a georeferenced video then track")
    parser.add_argument("--georef_video", default=None, help="Path to existing georeferenced video (if batch)")
    parser.add_argument("--roi_json", default=None,
                        help="ROI json path. If missing in streaming, you will be prompted to draw on first warped frame.")

    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    georef_video = args.georef_video or os.path.join(args.workdir, "georeferenced.mp4")
    georef_meta = os.path.join(args.workdir, "georef_meta.json")
    tracks_csv = os.path.join(args.workdir, "tracks.csv")

    if args.batch:
        # Batch: write video then track
        print("[Batch] Georeferencing video…")
        meta = georeference_video(
            input_video=args.input_video,
            gcp_file=args.gcp_json,
            resolution_m_per_px=args.resolution,
            output_video=georef_video,
            output_meta_json=georef_meta,
        )
        print("[Batch] Tracking in ROI…")
        if args.roi_json is None:
            print(
                "ERROR: --roi_json required in batch mode (draw with any video tool), or use streaming to draw interactively.")
            sys.exit(2)
        track_in_roi(
            video_path=georef_video,
            roi_json=args.roi_json,
            model_path=args.model_path,
            csv_out=tracks_csv,
            target_fps=args.target_fps,
            class_ids=args.classes,
            georef_meta_json=georef_meta,
        )
    else:
        # Streaming: georef each frame then track immediately
        print("[Streaming] Georeferencing each frame on-the-fly and tracking…")
        stream_track_with_on_the_fly_georef(
            input_video=args.input_video,
            gcp_json=args.gcp_json,
            model_path=args.model_path,
            csv_out=tracks_csv,
            roi_json=args.roi_json,
            resolution=args.resolution,
            target_fps=args.target_fps,
            class_ids=args.classes,
        )

    print(f"Done! CSV written to: {tracks_csv}")


if __name__ == "__main__":
    main()
