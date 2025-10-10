import cv2
import json

VIDEO_PATH = '../data/footage/DJI_20251006180546_0001_D.MP4'
points = []

# Globals for coordinate mapping
scale = 1.0
offset_x = 0
offset_y = 0
win_w = None
win_h = None
window_name = 'Select Points'
image = None  # original image (first frame)

def make_display_frame(img, w, h, pts):
    """
    Render a letterboxed frame for the current window size (w,h) while
    preserving aspect ratio. Returns the display frame plus the scale/offset
    used so we can map clicks correctly.
    """
    ih, iw = img.shape[:2]
    # Compute uniform scale to fit image into window while preserving AR
    s = min(w / iw, h / ih)
    s = max(s, 1e-6)  # guard against div-by-zero
    disp_w = max(1, int(iw * s))
    disp_h = max(1, int(ih * s))

    # Resize the image to the display size
    resized = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    # Create letterboxed canvas matching window size
    canvas = (0 * resized).copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    canvas = cv2.resize(canvas, (w, h))  # ensure exact size (all black)

    ox = (w - disp_w) // 2
    oy = (h - disp_h) // 2
    canvas[oy:oy+disp_h, ox:ox+disp_w] = resized

    # Draw any saved points on the displayed (scaled) image
    for p in pts:
        px, py = p["pixel"]
        dx = int(px * s) + ox
        dy = int(py * s) + oy
        cv2.circle(canvas, (dx, dy), 5, (0, 255, 0), -1)

    return canvas, s, ox, oy

def mouse_callback(event, x, y, flags, param):
    global scale, offset_x, offset_y, image, points

    if event == cv2.EVENT_LBUTTONDOWN:
        # Map window click (x,y) → original image coords
        img_x = int(round((x - offset_x) / max(scale, 1e-6)))
        img_y = int(round((y - offset_y) / max(scale, 1e-6)))

        ih, iw = image.shape[:2]
        # Only accept clicks inside the displayed image area
        if 0 <= img_x < iw and 0 <= img_y < ih:
            try:
                print(f"Pixel selected (display): ({x}, {y}) -> (image): ({img_x}, {img_y})")
                easting = float(input("Enter LV95 Easting (E) for this point: "))
                northing = float(input("Enter LV95 Northing (N) for this point: "))

                points.append({
                    "pixel": [img_x, img_y],      # store ORIGINAL image coordinates
                    "lv95": [easting, northing]
                })

                print(f"Point #{len(points)} saved. Select another point or press 'q' to finish.\n")

            except ValueError:
                print("Invalid Input, Please enter numbers only.")
        else:
            print("Click was outside the image area (letterbox). Try again inside the image.")

# Read first frame
cap = cv2.VideoCapture(VIDEO_PATH)
ret, first = cap.read()
cap.release()

if not ret or first is None:
    print("Failed to read the first frame from the video.")
else:
    image = first.copy()
    ih, iw = image.shape[:2]

    # Create a resizable window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    # Start with a nice default size (keeps AR in first render)
    init_w = min(1280, max(640, iw))
    init_h = int(init_w * ih / iw)
    cv2.resizeWindow(window_name, init_w, init_h)

    cv2.setMouseCallback(window_name, mouse_callback)

    print("Click on at least 4 known points in the image.")
    print("After each click, you will be prompted to enter LV95 coordinates in the console.")
    print("Resize the window as you like; the image aspect ratio is preserved.")
    print("Press 'q' when you are finished.")

    # Prime window size
    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
    except Exception:
        # Fallback if getWindowImageRect isn't available on your build
        win_w, win_h = init_w, init_h

    while True:
        # Detect current window size; re-render if changed
        try:
            _, _, cur_w, cur_h = cv2.getWindowImageRect(window_name)
            if (cur_w != win_w) or (cur_h != win_h):
                win_w, win_h = cur_w, cur_h
        except Exception:
            # If function not available, keep last known size
            pass

        # Render letterboxed frame preserving AR + draw existing points
        display_frame, scale, offset_x, offset_y = make_display_frame(image, win_w, win_h, points)

        cv2.imshow(window_name, display_frame)

        # Quit on 'q'
        if cv2.waitKey(16) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Save results
    if len(points) >= 4:
        with open('../data/gcp.txt', 'w') as f:
            json.dump(points, f, indent=4)
        print(f"\nSuccessfully saved {len(points)} ground control points to gcp.txt.")
    else:
        print("\nOperation cancelled. You need at least 4 points to proceed.")
