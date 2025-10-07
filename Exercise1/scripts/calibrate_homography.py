import argparse
import os

import cv2
import numpy as np

pts_img = []


def onclick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts_img.append((x, y))
        frame = param.copy()
        for i, (px, py) in enumerate(pts_img):
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{i}", (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Click points (press q when done)", frame)


def main():
    ap = argparse.ArgumentParser(description="Click image points and enter world (X,Y) to compute homography.")
    ap.add_argument("--frame-from-video", required=True, help="Path to video (a representative frame will be used)")
    ap.add_argument("--frame-index", type=int, default=0, help="Frame index to use")
    ap.add_argument("--out", default="configs/homography.npz", help="Output .npz file with H")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.frame_from_video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read frame")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cv2.namedWindow("Click points (press q when done)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click points (press q when done)", onclick, param=frame)
    cv2.imshow("Click points (press q when done)", frame)

    print("[Instructions]")
    print("- Click at least 4 ground points you can also measure in meters.")
    print("- Press 'q' when finished selecting.")
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            break
    cv2.destroyAllWindows()

    if len(pts_img) < 4:
        raise ValueError("Need at least 4 points.")

    img_pts = np.array(pts_img, dtype=np.float32)

    world_pts = []
    print("\nEnter world coordinates (meters) for each point, in the SAME ORDER:")
    for i in range(len(img_pts)):
        s = input(f"World X,Y for point {i} (e.g. 12.3, 7.8): ").strip()
        x_str, y_str = s.replace(" ", "").split(",")
        world_pts.append((float(x_str), float(y_str)))
    world_pts = np.array(world_pts, dtype=np.float32)

    H, mask = cv2.findHomography(img_pts, world_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        raise RuntimeError("findHomography failed")

    # QC: reprojection error
    img_h = np.hstack([img_pts, np.ones((len(img_pts), 1), dtype=np.float32)])
    proj = (H @ img_h.T).T
    proj = proj[:, :2] / proj[:, [2]]
    err = np.linalg.norm(proj - world_pts, axis=1)
    rms = float(np.sqrt(np.mean(err ** 2)))

    np.savez(args.out, H=H, img_pts=img_pts, world_pts=world_pts, rms=rms)
    print(f"\nSaved {args.out}")
    print(f"RMS reprojection error: {rms:.3f} m")

    # Draw projected grid for sanity
    overlay = frame.copy()
    for (u, v), (X, Y) in zip(img_pts, world_pts):
        cv2.circle(overlay, (int(u), int(v)), 6, (0, 0, 255), -1)
        cv2.putText(overlay, f"({X:.1f},{Y:.1f})", (int(u) + 8, int(v) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    qc_path = os.path.splitext(args.out)[0] + "_qc.png"
    cv2.imwrite(qc_path, overlay)
    print(f"QC overlay saved to {qc_path}")


if __name__ == "__main__":
    main()
