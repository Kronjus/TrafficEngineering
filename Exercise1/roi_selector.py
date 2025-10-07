import json

import cv2
import numpy as np

# -------- Globals used by mouse callback --------
roi_points = []
frame_orig = None  # the original first frame (immutable)
frame_draw = None  # a copy we draw points/lines onto (same size as frame_orig)
win_name = "Select ROI"

# Mapping from window -> frame coords for the CURRENTLY SHOWN IMAGE
# These get updated every frame in the render loop
draw_scale = 1.0  # how much the frame was scaled to fit window
offset_x = 0  # left padding in the window canvas
offset_y = 0  # top padding in the window canvas


def fit_and_letterbox(img, target_w, target_h):
    """
    Resize `img` to fit entirely inside (target_w, target_h) with aspect preserved.
    Returns: canvas (H,W,3), scale, offset_x, offset_y
    """
    h, w = img.shape[:2]
    if target_w <= 0 or target_h <= 0:
        # Safety: return the original as-is
        canvas = img.copy()
        return canvas, 1.0, 0, 0

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas, scale, x0, y0


def mouse_callback(event, x, y, flags, param):
    """
    Map window click (x,y) -> original frame coords, ignoring clicks on padding.
    Draw on frame_draw (original resolution) so drawings stay crisp.
    """
    global roi_points, frame_draw, draw_scale, offset_x, offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        # subtract the padding
        fx = (x - offset_x)
        fy = (y - offset_y)

        # If click is outside the drawn image area, ignore
        if fx < 0 or fy < 0:
            return

        # Map back to original-frame coordinates
        if draw_scale <= 0:
            return
        ox = int(round(fx / draw_scale))
        oy = int(round(fy / draw_scale))

        # Bounds check against original frame size
        H, W = frame_draw.shape[:2]
        if not (0 <= ox < W and 0 <= oy < H):
            return

        roi_points.append((ox, oy))
        print(f"Point added (frame coords): ({ox}, {oy}). Total points: {len(roi_points)}")

        # Draw on the original-sized drawing layer
        cv2.circle(frame_draw, (ox, oy), 5, (0, 0, 255), -1)
        if len(roi_points) > 1:
            cv2.line(frame_draw, roi_points[-2], roi_points[-1], (0, 255, 0), 2)


def render_to_window(img):
    """
    Render the original-sized `img` into the current window size with letterbox,
    updating globals (draw_scale, offset_x, offset_y). Returns the canvas to imshow.
    """
    global draw_scale, offset_x, offset_y

    # Get the current drawable area of the window (OpenCV 4.5+)
    try:
        _, _, ww, wh = cv2.getWindowImageRect(win_name)
        if ww <= 0 or wh <= 0:
            ww, wh = 720, 1080  # safety default
    except AttributeError:
        # Older OpenCV: no dynamic window size; just use a fixed one
        ww, wh = 720, 1080

    canvas, s, x0, y0 = fit_and_letterbox(img, ww, wh)
    draw_scale, offset_x, offset_y = s, x0, y0
    return canvas


if __name__ == "__main__":
    video_path = "data/footage/DJI_20251006180546_0001_D.MP4"
    cap = cv2.VideoCapture(video_path)

    ok, frame = cap.read()
    if not ok:
        print("Error: Could not read the first frame.")
        exit(1)

    frame_orig = frame.copy()
    frame_draw = frame.copy()

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # resizable window
    # Start with a portrait-ish size; you can change this to your screen size
    cv2.resizeWindow(win_name, 720, 1080)

    cv2.setMouseCallback(win_name, mouse_callback)

    print("Instructions:")
    print(" - Click to add points for the ROI polygon (coordinates saved in original-frame pixels).")
    print(" - Press 's' to save ROI to 'roi_coordinates.json'.")
    print(" - Press 'r' to reset points.")
    print(" - Press 'u' to undo last point.")
    print(" - Press 'q' to quit.")

    while True:
        # Show the version with drawings (frame_draw) scaled to the current window size
        display = render_to_window(frame_draw)
        cv2.imshow(win_name, display)

        key = cv2.waitKey(15) & 0xFF
        if key == ord("q"):
            break

        elif key == ord("s"):
            if len(roi_points) > 2:
                with open("roi_coordinates.json", "w") as f:
                    json.dump(roi_points, f)
                print(f"ROI coordinates saved to roi_coordinates.json: {roi_points}")
                break
            else:
                print("Error: at least 3 points needed to define a polygon.")

        elif key == ord("r"):
            roi_points = []
            frame_draw = frame_orig.copy()
            print("ROI reset. Start again.")

        elif key == ord("u"):
            if roi_points:
                roi_points.pop()
                # Re-draw everything from scratch on a fresh copy
                frame_draw = frame_orig.copy()
                for i, pt in enumerate(roi_points):
                    cv2.circle(frame_draw, pt, 5, (0, 0, 255), -1)
                    if i > 0:
                        cv2.line(frame_draw, roi_points[i - 1], roi_points[i], (0, 255, 0), 2)
                print("Undid last point.")

    cap.release()
    cv2.destroyAllWindows()
