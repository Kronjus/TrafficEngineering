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
    """
    Ensures that the parent directory of the given file path exists.
    If the directory does not exist, it is created.

    Args:
        path (str): The file path for which the parent directory should be ensured.

    Example:
        ensure_parent_dir("/path/to/file.txt")
        # This will create the directory "/path/to" if it does not already exist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def save_json(path: str, data: dict, indent: int = 2):
    """
    Saves a dictionary as a JSON file at the specified path.

    This function ensures that the parent directory of the file exists
    before attempting to save the JSON data. If the directory does not exist,
    it will be created.

    Args:
        path (str): The file path where the JSON data will be saved.
        data (dict): The dictionary to be serialized and saved as JSON.
        indent (int, optional): The number of spaces to use for indentation in the JSON file. Defaults to 2.

    Example:
        save_json("output/data.json", {"key": "value"}, indent=4)
        # This will save the dictionary {"key": "value"} to "output/data.json" with an indentation of 4 spaces.
    """
    ensure_parent_dir(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_json(path: str) -> dict:
    """
    Loads and parses a JSON file from the specified path.

    Args:
        path (str): The file path to the JSON file to be loaded.

    Returns:
        dict: The parsed JSON data as a Python dictionary.

    Example:
        data = load_json("config.json")
        # This will load and return the contents of "config.json" as a dictionary.
    """
    with open(path, "r") as f:
        return json.load(f)


# --------------
# Georeferencing helpers
# --------------
def compute_georef_params_from_gcps(gcp_file: str, resolution_m_per_px: float, ransac_reproj_threshold: float = 5.0):
    """
    Computes georeferencing parameters from Ground Control Points (GCPs).

    This function calculates the transformation matrix (homography) and metadata
    required to map pixel coordinates to georeferenced coordinates based on the
    provided GCPs.

    Args:
        gcp_file (str): Path to the JSON file containing GCPs. The file must contain
                        at least 4 points with 'pixel' and 'lv95' keys.
        resolution_m_per_px (float): Resolution in meters per pixel for the output canvas.
        ransac_reproj_threshold (float, optional): RANSAC reprojection threshold for
                                                   homography computation. Defaults to 5.0.

    Returns:
        Tuple[np.ndarray, GeorefMeta]: A tuple containing:
            - H_master (np.ndarray): The computed homography matrix.
            - meta (GeorefMeta): Metadata describing the georeferenced output canvas.

    Raises:
        ValueError: If the GCP file contains fewer than 4 points or is not a list.
        RuntimeError: If the homography computation fails.

    Example:
        H_master, meta = compute_georef_params_from_gcps("gcps.json", 0.1)
        # Computes the homography and metadata for georeferencing.
    """
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
    """
    Converts pixel coordinates to LV95 georeferenced coordinates.

    This function calculates the LV95 easting (E) and northing (N) coordinates
    based on the provided georeferencing metadata and pixel coordinates.

    Args:
        meta (GeorefMeta): The georeferencing metadata containing resolution,
                           bounding box, and canvas dimensions.
        cx (float): The x-coordinate (column) in pixel space.
        cy (float): The y-coordinate (row) in pixel space.

    Returns:
        Tuple[float, float]: A tuple containing:
            - E (float): The LV95 easting coordinate.
            - N (float): The LV95 northing coordinate.

    Example:
        E, N = lv95_from_georef_px(meta, 100.0, 200.0)
        # Converts pixel coordinates (100.0, 200.0) to LV95 coordinates.
    """
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
    """
    Georeferences a video using Ground Control Points (GCPs) and saves the output.

    This function computes the georeferencing parameters from the provided GCPs,
    processes the input video frame by frame, and writes the georeferenced video
    to the specified output path. It also optionally saves metadata about the
    georeferenced video.

    Args:
        input_video (str): Path to the input video file.
        gcp_file (str): Path to the JSON file containing GCPs. The file must contain
                        at least 4 points with 'pixel' and 'lv95' keys.
        resolution_m_per_px (float): Resolution in meters per pixel for the output canvas.
        output_video (str): Path to save the georeferenced video.
        output_meta_json (Optional[str]): Path to save the georeferencing metadata as JSON.
                                          If None, metadata is not saved.
        feature_count (int, optional): Number of features to detect for motion estimation. Defaults to 2000.
        ransac_reproj_threshold (float, optional): RANSAC reprojection threshold for homography computation. Defaults to 5.0.

    Returns:
        GeorefMeta: Metadata describing the georeferenced video, including resolution,
                    bounding box, and canvas dimensions.

    Raises:
        FileNotFoundError: If the input video cannot be opened.
        RuntimeError: If the first frame of the video cannot be read.

    Example:
        meta = georeference_video(
            input_video="input.mp4",
            gcp_file="gcps.json",
            resolution_m_per_px=0.1,
            output_video="output.mp4",
            output_meta_json="meta.json"
        )
        # Processes "input.mp4" and saves the georeferenced video to "output.mp4".
        # Metadata is saved to "meta.json".
    """
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
    """
    A class for interactively selecting a Region of Interest (ROI) on an image.

    This class provides a graphical interface for users to draw a polygonal ROI
    on an image using mouse clicks. The selected ROI points can be saved, reset,
    or undone during the interaction.
    """

    def __init__(self, image: np.ndarray, win_name: str = "Select ROI"):
        """
        Initializes the ROI selector with the given image and window name.

        Args:
            image (np.ndarray): The image on which the ROI will be selected.
            win_name (str, optional): The name of the OpenCV window. Defaults to "Select ROI".
        """
        self.image = image
        self.win = win_name
        self.points: List[Tuple[int, int]] = []
        self.frame_draw = image.copy()
        self.scale = 1.0
        self.ox = 0
        self.oy = 0

    @staticmethod
    def _fit_and_letterbox(img, target_w, target_h):
        """
        Resizes the image to fit within the target dimensions while maintaining aspect ratio.

        Adds letterboxing (black padding) to ensure the image fits exactly within the target dimensions.

        Args:
            img (np.ndarray): The input image to resize.
            target_w (int): The target width of the output image.
            target_h (int): The target height of the output image.

        Returns:
            Tuple[np.ndarray, float, int, int]: A tuple containing:
                - The resized image with letterboxing.
                - The scaling factor used for resizing.
                - The x-offset of the image within the letterboxed canvas.
                - The y-offset of the image within the letterboxed canvas.
        """
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
        """
        Handles mouse events for selecting ROI points.

        This method is triggered when the user interacts with the OpenCV window using the mouse.

        Args:
            event (int): The type of mouse event (e.g., left button click).
            x (int): The x-coordinate of the mouse event.
            y (int): The y-coordinate of the mouse event.
            flags (int): Any relevant flags passed by OpenCV.
            param (Any): Additional parameters passed by OpenCV.
        """
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
        """
        Starts the interactive ROI selection process.

        Opens an OpenCV window where the user can draw a polygonal ROI by clicking
        to add points. The user can save, undo, reset, or quit the selection process.

        Returns:
            List[Tuple[int, int]]: A list of points representing the selected ROI.

        Raises:
            RuntimeError: If the ROI is not saved before quitting.
        """
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
    """
    Allows the user to interactively select a Region of Interest (ROI) on the given frame
    and saves the selected ROI points to a JSON file.

    This function uses the `_ROISelector` class to provide an interactive interface for
    selecting a polygonal ROI. The selected points are then saved to the specified JSON file.

    Args:
        frame (np.ndarray): The image frame on which the ROI will be selected.
        roi_json_out (str): The file path where the selected ROI points will be saved as JSON.

    Returns:
        List[Tuple[int, int]]: A list of points representing the selected ROI.

    Raises:
        RuntimeError: If the ROI is not saved during the selection process.

    Example:
        roi_points = select_and_save_roi_on_frame(frame, "output/roi.json")
        # Opens an interactive window for ROI selection and saves the points to "output/roi.json".
    """
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
    """
    Streams a video, performs on-the-fly georeferencing, and tracks objects within a Region of Interest (ROI).

    This function processes a video frame by frame, georeferencing each frame using Ground Control Points (GCPs),
    and tracks objects detected within a specified ROI. The results are saved to a CSV file.

    Args:
        input_video (str): Path to the input video file.
        gcp_json (str): Path to the JSON file containing Ground Control Points (GCPs).
        model_path (str): Path to the YOLO model file for object detection.
        csv_out (str): Path to save the tracking results as a CSV file.
        roi_json (Optional[str]): Path to the JSON file containing ROI points. If not provided, the ROI is selected interactively.
        resolution (float, optional): Resolution in meters per pixel for georeferencing. Defaults to 0.1.
        target_fps (int, optional): Target frames per second for processing. Defaults to 15.
        feature_count (int, optional): Number of features to detect for motion estimation. Defaults to 2000.
        class_ids (Optional[List[int]], optional): List of class IDs to filter detections. If None, all classes are included.

    Raises:
        FileNotFoundError: If the input video cannot be opened.
        RuntimeError: If the first frame of the video cannot be read.

    Example:
        stream_track_with_on_the_fly_georef(
            input_video="input.mp4",
            gcp_json="gcps.json",
            model_path="yolo.pt",
            csv_out="output/tracks.csv",
            roi_json="output/roi.json",
            resolution=0.1,
            target_fps=15,
            feature_count=2000,
            class_ids=[0, 1, 2]
        )
        # Processes "input.mp4", georeferences each frame, tracks objects, and saves results to "output/tracks.csv".
    """
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
                E_list, N_list = [], []
                for (x1, y1, x2, y2) in detections.xyxy:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    E = meta.min_e + cx * meta.resolution
                    N = meta.max_n - cy * meta.resolution
                    E_list.append(E)
                    N_list.append(N)
                detections.data["lv95_E"] = np.array(E_list)
                detections.data["lv95_N"] = np.array(N_list)

            s.append(detections, custom_data={"frame": frame_idx})
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

                if len(detections) > 0:
                    E_list, N_list = [], []
                    for (x1, y1, x2, y2) in detections.xyxy:
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        E = meta.min_e + cx * meta.resolution
                        N = meta.max_n - cy * meta.resolution
                        E_list.append(E)
                        N_list.append(N)
                    detections.data["lv95_E"] = np.array(E_list)
                    detections.data["lv95_N"] = np.array(N_list)

                s.append(detections, custom_data={"frame": frame_idx})

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
    """
    Tracks objects within a specified Region of Interest (ROI) in a video.

    This function processes a video frame by frame, detects objects using a YOLO model,
    and tracks them within a predefined ROI. The results are saved to a CSV file. If
    georeferencing metadata is provided, the detections are converted to georeferenced
    coordinates.

    Args:
        video_path (str): Path to the input video file.
        roi_json (str): Path to the JSON file containing ROI points.
        model_path (str): Path to the YOLO model file for object detection.
        csv_out (str): Path to save the tracking results as a CSV file.
        target_fps (int, optional): Target frames per second for processing. Defaults to 15.
        class_ids (Optional[List[int]], optional): List of class IDs to filter detections. If None, all classes are included.
        georef_meta_json (Optional[str], optional): Path to the JSON file containing georeferencing metadata. If None, georeferencing is not applied.

    Raises:
        FileNotFoundError: If the input video cannot be opened.

    Example:
        track_in_roi(
            video_path="video.mp4",
            roi_json="roi.json",
            model_path="yolo.pt",
            csv_out="output/tracks.csv",
            target_fps=15,
            class_ids=[0, 1, 2],
            georef_meta_json="georef_meta.json"
        )
        # Tracks objects in "video.mp4" within the ROI defined in "roi.json",
        # filters detections by class IDs [0, 1, 2], and saves results to "output/tracks.csv".
    """
    import supervision as sv
    from ultralytics import YOLO

    # Ensure the parent directory for the output CSV exists
    ensure_parent_dir(csv_out)

    # Load ROI points from the JSON file
    roi_points = np.array([tuple(x) for x in load_json(roi_json)], dtype=np.float32)
    zone = sv.PolygonZone(roi_points)

    # Load georeferencing metadata if provided
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

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(round(original_fps / float(target_fps))))

    # Initialize the YOLO model, tracker, and smoother
    model = YOLO(model_path)
    tracker = sv.ByteTrack()
    smoother = sv.DetectionsSmoother()

    # Create a generator for video frames and a CSV sink for output
    frames = sv.get_video_frames_generator(video_path)
    sink = sv.CSVSink(csv_out)

    frame_idx = 0
    with sink as s, tqdm(total=total_frames if total_frames > 0 else None, desc="Tracking") as pbar:
        for frame in frames:
            if frame_idx % frame_interval == 0:
                # Perform object detection on the current frame
                results = model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)

                # Filter detections based on the ROI and class IDs
                mask = zone.trigger(detections=detections)
                if class_ids is not None:
                    class_mask = np.isin(detections.class_id, class_ids)
                    mask = mask & class_mask
                detections = detections[mask]

                # Update the tracker and smoother with the filtered detections
                detections = tracker.update_with_detections(detections)
                detections = smoother.update_with_detections(detections)

                # Prepare custom rows for the CSV output
                if meta is not None and len(detections) > 0:
                    E_list, N_list = [], []
                    for (x1, y1, x2, y2) in detections.xyxy:
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        E = meta.min_e + cx * meta.resolution
                        N = meta.max_n - cy * meta.resolution
                        E_list.append(E)
                        N_list.append(N)
                    detections.data["lv95_E"] = np.array(E_list)
                    detections.data["lv95_N"] = np.array(N_list)

                s.append(detections, custom_data={"frame": frame_idx})

            frame_idx += 1
            if total_frames > 0:
                pbar.update(1)

    # Release the video capture object
    cap.release()


# --------------
# Orchestrator CLI
# --------------
def main():
    """
    The main function orchestrates the georeferencing, ROI selection, and object tracking pipeline.

    This function parses command-line arguments to determine the input video, GCP JSON file, YOLO model path,
    and other parameters. It supports two modes of operation:
    - Batch mode: Georeferences the entire video, then tracks objects within a predefined ROI.
    - Streaming mode: Georeferences and tracks objects frame by frame.

    Command-line arguments:
        --input_video (str): Path to the input video (non-georeferenced).
        --gcp_json (str): Path to the GCP JSON file (list of {pixel:[x,y], lv95:[E,N]}).
        --model_path (str): Path to the YOLO model file.
        --workdir (str, optional): Working directory for outputs. Defaults to "./outputs_pipeline".
        --resolution (float, optional): Resolution in meters per pixel for georeferencing. Defaults to 0.1.
        --target_fps (int, optional): Target FPS for detection/tracking. Defaults to 15.
        --classes (List[int], optional): Optional class IDs filter (e.g., 0 1 2). Defaults to None.
        --batch (bool, optional): Use batch mode to write a georeferenced video, then track. Defaults to False.
        --georef_video (str, optional): Path to an existing georeferenced video (if batch mode). Defaults to None.
        --roi_json (str, optional): Path to the ROI JSON file. If missing in streaming mode, the user will be prompted
                                     to draw the ROI on the first warped frame.

    Raises:
        SystemExit: If required arguments are missing or if the ROI JSON file is not provided in batch mode.

    Example:
        python pipeline.py --input_video input.mp4 --gcp_json gcps.json --model_path yolo.pt --batch
    """
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
