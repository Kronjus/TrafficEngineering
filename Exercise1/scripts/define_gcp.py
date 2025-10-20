import json

import cv2

VIDEO_PATH = '../data/footage/DJI_20251006180546_0001_D.MP4'
points = []

# Globals for coordinate mapping and interaction
scale = 1.0
offset_x = 0
offset_y = 0
pan_start = None  # For panning with right mouse button
win_w = None
win_h = None
window_name = 'Select Points'
image = None  # original image (first frame)


def make_display_frame(img, w, h, pts, scale, offset_x, offset_y):
    """
    Render a zoomed and panned frame for the current window size (w,h).
    Draws points and returns the display frame.
    """
    ih, iw = img.shape[:2]

    # Compute scaled image size
    disp_w = max(1, int(iw * scale))
    disp_h = max(1, int(ih * scale))

    # Resize the image to the display size
    resized = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    # Create black canvas matching window size
    canvas = (0 * resized).copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    canvas = cv2.resize(canvas, (w, h))  # ensure exact size (all black)

    # Compute top-left corner for placing the image (with pan offset)
    ox = offset_x
    oy = offset_y

    # Compute region of interest in canvas and resized image
    x1 = max(0, -ox)
    y1 = max(0, -oy)
    x2 = min(disp_w, w - ox)
    y2 = min(disp_h, h - oy)

    cx1 = max(0, ox)
    cy1 = max(0, oy)
    cx2 = cx1 + (x2 - x1)
    cy2 = cy1 + (y2 - y1)

    if x2 > x1 and y2 > y1 and cx2 > cx1 and cy2 > cy1:
        canvas[cy1:cy2, cx1:cx2] = resized[y1:y2, x1:x2]

    # Draw any saved points on the displayed (scaled, panned) image
    for p in pts:
        px, py = p["pixel"]
        dx = int(px * scale) + ox
        dy = int(py * scale) + oy
        if 0 <= dx < w and 0 <= dy < h:
            cv2.circle(canvas, (dx, dy), 5, (0, 255, 0), -1)

    return canvas


def mouse_callback(event, x, y, flags, param):
    """
    Handles mouse events for selecting points, zooming, and panning.
    Left click: select point.
    Right button drag: pan.
    Mouse wheel: zoom.
    """
    global scale, offset_x, offset_y, image, points, pan_start, win_w, win_h

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
                    "pixel": [img_x, img_y],  # store ORIGINAL image coordinates
                    "lv95": [easting, northing]
                })

                print(f"Point #{len(points)} saved. Select another point or press 'q' to finish.\n")

            except ValueError:
                print("Invalid Input, Please enter numbers only.")
        else:
            print("Click was outside the image area (letterbox). Try again inside the image.")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Start panning
        pan_start = (x, y)

    elif event == cv2.EVENT_RBUTTONUP:
        # End panning
        pan_start = None

    elif event == cv2.EVENT_MOUSEMOVE:
        # Pan if right button is held
        if pan_start is not None:
            dx = x - pan_start[0]
            dy = y - pan_start[1]
            offset_x += dx
            offset_y += dy
            pan_start = (x, y)

    elif event == cv2.EVENT_MOUSEWHEEL:
        # Zoom in/out, keeping mouse position as center
        if flags > 0:
            zoom_factor = 1.1
        else:
            zoom_factor = 0.9
        old_scale = scale
        scale = max(0.1, min(10.0, scale * zoom_factor))

        # Adjust offset so that the point under the mouse stays under the mouse
        mx, my = x, y
        ox = offset_x
        oy = offset_y
        offset_x = int(mx - (mx - ox) * (scale / old_scale))
        offset_y = int(my - (my - oy) * (scale / old_scale))


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
    print("Use the mouse wheel to zoom, right mouse button to pan.")
    print("Press 'q' when you are finished.")

    # Prime window size
    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
    except Exception:
        win_w, win_h = init_w, init_h

    while True:
        # Detect current window size; re-render if changed
        try:
            _, _, cur_w, cur_h = cv2.getWindowImageRect(window_name)
            if (cur_w != win_w) or (cur_h != win_h):
                win_w, win_h = cur_w, cur_h
        except Exception:
            pass

        # Render zoomed and panned frame + draw existing points
        display_frame = make_display_frame(image, win_w, win_h, points, scale, offset_x, offset_y)

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
