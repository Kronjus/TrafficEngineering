import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def load_intrinsics(path):
            """
            Loads camera intrinsics from a JSON file.

            Args:
                path (str): Path to the JSON file containing the camera intrinsics.

            The JSON file should have the following keys:
                - "fx", "fy": Focal lengths of the camera.
                - "cx", "cy": Principal point coordinates.
                - "k1", "k2", "p1", "p2", "k3" (optional): Distortion coefficients.

            Returns:
                tuple:
                    - K (numpy.ndarray): 3x3 camera intrinsic matrix.
                    - k (numpy.ndarray): 1D array of distortion coefficients (length 5).
            """
            with open(path, "r") as f:
                d = json.load(f)
            fx, fy, cx, cy = [float(d.get(k)) for k in ("fx", "fy", "cx", "cy")]
            k = np.array([d.get("k1", 0), d.get("k2", 0), d.get("p1", 0), d.get("p2", 0), d.get("k3", 0)], dtype=np.float32)
            K = np.array(([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]]), dtype=np.float32)
            return K, k


def load_homography(path):
            """
            Loads a homography matrix from a file.

            Args:
                path (str): Path to the file containing the homography matrix.
                    - If the file is a `.npz`, it should contain a key "H" with the matrix.
                    - If the file is a JSON, it should contain a key "H" with a list of 9 values.

            Returns:
                numpy.ndarray: A 3x3 homography matrix of type float32.
            """
            p = Path(path)
            if p.suffix.lower() == ".npz":
                data = np.load(p)
                H = data["H"].astype(np.float32)
            else:
                with open(p, "r") as f:
                    d = json.load(f)
                H = np.array(d["H"], dtype=np.float32).reshape(3, 3)
            return H


def undistort_frame(frame, K, dist, new_K=None, balance=0.0):
            """
            Undistorts a given frame using the provided camera intrinsic matrix and distortion coefficients.

            Args:
                frame (numpy.ndarray): The input image/frame to be undistorted.
                K (numpy.ndarray): The camera intrinsic matrix (3x3).
                dist (numpy.ndarray): The distortion coefficients (1D array).
                new_K (numpy.ndarray, optional): The new camera matrix after undistortion. Defaults to None.
                balance (float, optional): Balance parameter for fisheye undistortion (0.0 to 1.0). Defaults to 0.0.

            Returns:
                numpy.ndarray: The undistorted image/frame.
            """
            h, w = frame.shape[:2]
            if new_K is None:
                new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, dist, (w, h), np.eye(3), balance=balance
                ) if hasattr(cv2, "fisheye") and dist.shape[0] == 4 else K
            map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_16SC2)
            return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


def stabilize_iter(prev_gray, curr_gray, prev_T):
        """
        Estimates a transformation matrix to stabilize the motion between two consecutive frames.

        Args:
            prev_gray (numpy.ndarray): The previous grayscale frame.
            curr_gray (numpy.ndarray): The current grayscale frame.
            prev_T (numpy.ndarray): The previous transformation matrix (3x3).

        Returns:
            numpy.ndarray: The updated transformation matrix (3x3) combining the previous transformation
            and the newly estimated transformation.
        """
        warp_mode = cv2.MOTION_EUCLIDEAN  # Use Euclidean motion model for stabilization.
        warp_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)  # Initialize warp matrix.
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)  # Termination criteria for ECC.

        try:
            # Estimate the transformation using Enhanced Correlation Coefficient (ECC).
            cc, warp_matrix = cv2.findTransformECC(prev_gray, curr_gray, warp_matrix, warp_mode, criteria)
            M = np.vstack([warp_matrix, [0, 0, 1]]).astype(np.float32)  # Convert to 3x3 matrix.
        except cv2.error:
            # Fallback to feature-based method if ECC fails.
            p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=800, qualityLevel=0.01, minDistance=7)
            if p0 is None:
                return np.eye(3, dtype=np.float32)  # Return identity matrix if no features are found.
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)  # Track features.
            good0 = p0[st == 1]  # Filter good points in the previous frame.
            good1 = p1[st == 1]  # Filter good points in the current frame.
            if len(good0) < 10:
                return np.eye(3, dtype=np.float32)  # Return identity matrix if insufficient points.
            M2, _ = cv2.estimateAffinePartial2D(good1, good0, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if M2 is None:
                return np.eye(3, dtype=np.float32)  # Return identity matrix if estimation fails.
            M = np.vstack([M2, [0, 0, 1]]).astype(np.float32)  # Convert to 3x3 matrix.

        return M @ prev_T  # Combine the new transformation with the previous one.


def warp_bev(frame, H, out_size=None):
            """
            Applies a bird's-eye view (BEV) transformation to the input frame using a homography matrix.

            Args:
                frame (numpy.ndarray): The input image/frame to be transformed.
                H (numpy.ndarray): The 3x3 homography matrix for the transformation.
                out_size (tuple, optional): The desired output size as (width, height).
                                            Defaults to the size of the input frame.

            Returns:
                numpy.ndarray: The transformed image/frame with the bird's-eye view applied.
            """
            h, w = frame.shape[:2]
            if out_size is None:
                out_size = (w, h)
            return cv2.warpPerspective(frame, H, out_size, flags=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser(description="Undistort / Stabilize / BEV-warp drone video.")
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--out", required=True, help="Path to output video")
    ap.add_argument("--apply-undistort", action="store_true", help="Apply undistortion using intrinsics JSON")
    ap.add_argument("--intrinsics-json", default="", help="JSON with fx,fy,cx,cy,k1,k2,p1,p2,k3")
    ap.add_argument("--apply-stabilize", action="store_true", help="Apply global stabilization (mild drift)")
    ap.add_argument("--apply-bev", action="store_true", help="Warp to BEV using homography H")
    ap.add_argument("--H", default="", help="Path to homography (npz with H or JSON with 'H': [9 vals])")
    ap.add_argument("--resize-width", type=int, default=0, help="Optional resize width (keeps aspect)")
    ap.add_argument("--codec", default="mp4v", help="FourCC codec (mp4v, avc1, h264 if available)")
    ap.add_argument("--balance", type=float, default=0.0, help="Undistort balance (0..1) if fisheye used")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    K = dist = None
    if args.apply_undistort:
        if not args.intrinsics_json:
            print("ERROR: --apply-undistort requires --intrinsics-json", file=sys.stderr)
            sys.exit(1)
        K, dist = load_intrinsics(args.intrinsics_json)

    H_bev = None
    if args.apply_bev:
        if not args.H:
            print("ERROR: --apply-bev requires --H", file=sys.stderr)
            sys.exit(1)
        H_bev = load_homography(args.H)

    out_w, out_h = W, Hh
    if args.resize_width and args.resize_width > 0:
        out_w = int(args.resize_width)
        out_h = int(round(Hh * (out_w / W)))

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, fourcc, fps, (out_w, out_h))

    prev_T = np.eye(3, dtype=np.float32)
    prev_gray = None

    meta = {
        "source": str(args.video),
        "fps": fps,
        "steps": {
            "undistort": bool(args.apply_undistort),
            "stabilize": bool(args.apply_stabilize),
            "bev": bool(args.apply_bev),
            "resize_width": args.resize_width or None
        }
    }

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.apply_undistort and K is not None:
            frame = undistort_frame(frame, K, dist, balance=args.balance)

        if args.resize_width and args.resize_width > 0:
            frame = cv2.resize(frame(out_w, out_h), interpolation=cv2.INTER_AREA)

        if args.apply_stabilize:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                prev_T = stabilize_iter(prev_gray, gray, prev_T)
                frame = cv2.warpPerspective(frame, prev_T, (out_w, out_h), flags=cv2.INTER_LINEAR)
            prev_gray = gray

        if args.apply_bev:
            frame = warp_bev(frame, H_bev, out_size=(out_w, out_h))

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...", end="\r")

    cap.release()
    writer.release()

    sidecar = Path(args.out).with_suffix(Path(args.out).suffix + ".json")
    with open(sidecar, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: {args.out}")
    print(f"Metadata: {sidecar}")


if __name__ == "__main__":
    main()
