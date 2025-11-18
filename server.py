import cv2
from flask import Flask, Response, request
import threading
import os
import time
import math
import logging

# Quiet Flask access logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

from pymavlink import mavutil

app = Flask(__name__)

# ===================== CONFIG =====================
print(cv2.__version__)
USE_VIDEO = True
#VIDEO_PATH = r"D:\Projects\Thesis\LiveFeed\test_720.mp4"  # or set to 0 for webcam
VIDEO_PATH = 0

# MAVProxy must output to this port (e.g. in MAVProxy: `output add 127.0.0.1:14552`)
# UDP_IN = 'udpin:0.0.0.0:14550'   # keep QGC on 14550

# Camera FOV for bbox->bearing
#
CAMERA_HFOV_DEG = 67.24 # HFOV = 2 × arctan( W / (2 × D) )  D=1m camera distance from the wall W=1.33m
CAMERA_VFOV_DEG = 41.12 # VFOV = 2 × arctan( H / (2 × D) ) D=1m H=0,75M

# How often to print guidance (seconds)
GUIDANCE_PRINT_INTERVAL = 1
# ===================================================

# Shared state
tracker = None
bbox = None
lock = threading.Lock()
current_frame = None

# For throttling console prints
last_guidance_print = 0.0

# ------------- GUIDANCE FROM BBOX -------------

def bbox_center(b):
    """Return (cx, cy) center of bbox in pixels."""
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)

def bbox_to_angles(b, frame_w, frame_h):
    """
    Convert bbox position into yaw/pitch angles relative to frame center.

    Returns (yaw_deg, pitch_deg), where:
      - yaw_deg  > 0: target is to the RIGHT  → turn RIGHT
      - yaw_deg  < 0: target is to the LEFT   → turn LEFT
      - pitch_deg> 0: target is DOWN in image → pitch DOWN
      - pitch_deg< 0: target is UP in image   → pitch UP
    """
    cx, cy = bbox_center(b)

    # Normalize offsets to range [-1, 1]
    dx_norm = (cx - frame_w / 2.0) / (frame_w / 2.0)
    dy_norm = (cy - frame_h / 2.0) / (frame_h / 2.0)

    yaw_deg = dx_norm * (CAMERA_HFOV_DEG / 2.0)
    pitch_deg = dy_norm * (CAMERA_VFOV_DEG / 2.0)

    return yaw_deg, pitch_deg

def format_guidance(yaw_deg, pitch_deg):
    """
    Turn yaw/pitch angles into human-readable text commands.
    """
    # Yaw (left/right)
    if abs(yaw_deg) < 1.0:
        yaw_cmd = "YAW HOLD (centered)"
    elif yaw_deg > 0:
        yaw_cmd = f"YAW RIGHT {abs(yaw_deg):.1f}°"
    else:
        yaw_cmd = f"YAW LEFT {abs(yaw_deg):.1f}°"

    # Pitch (up/down)
    if abs(pitch_deg) < 1.0:
        pitch_cmd = "PITCH HOLD (centered)"
    elif pitch_deg > 0:
        pitch_cmd = f"PITCH DOWN {abs(pitch_deg):.1f}°"
    else:
        pitch_cmd = f"PITCH UP {abs(pitch_deg):.1f}°"

    return yaw_cmd, pitch_cmd

# ---------------- VIDEO THREAD ----------------
def capture_loop():
    global current_frame, tracker, bbox, last_guidance_print
    cap = cv2.VideoCapture(VIDEO_PATH if USE_VIDEO else 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps and fps > 0 else 33
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        os._exit(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            if USE_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            else:
                print("[ERROR] Failed to read frame from camera"); os._exit(0)

        fh, fw = frame.shape[:2]

        # Update tracker
        if tracker is not None and bbox is not None:
            try:
                ok, new_box = tracker.update(frame)
                if ok:
                    x, y, w, h = [int(v) for v in new_box]
                    bbox = (x, y, w, h)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # ---- GUIDANCE PRINTING ----
                    now = time.time()
                    if now - last_guidance_print > GUIDANCE_PRINT_INTERVAL:
                        yaw_deg, pitch_deg = bbox_to_angles(bbox, fw, fh)
                        yaw_cmd, pitch_cmd = format_guidance(yaw_deg, pitch_deg)
                        print(f"[GUIDANCE] {yaw_cmd}, {pitch_cmd}")
                        last_guidance_print = now
                else:
                    print("[INFO] Tracking lost")
                    # Optionally reset tracker & bbox
                    # tracker = None
                    # bbox = None
            except Exception as e:
                print(f"[ERROR] Tracker update failed: {e}")

        with lock:
            current_frame = frame.copy()

        cv2.imshow("Live Feed", frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q') or cv2.getWindowProperty("Live Feed", cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Exiting capture loop…"); break

    cap.release()
    cv2.destroyAllWindows()
    os._exit(0)

# ---------------- FLASK ROUTES ----------------
@app.route('/frame')
def get_frame():
    global current_frame
    with lock:
        if current_frame is None:
            return "No frame available", 503
        _, buffer = cv2.imencode('.jpg', current_frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/bbox', methods=['POST'])
def set_bbox():
    """Init tracker from Android bbox."""
    global bbox, tracker, current_frame, last_guidance_print

    data = request.get_json()
    norm_x = float(data['x'])
    norm_y = float(data['y'])
    norm_w = float(data['width'])
    norm_h = float(data['height'])
    print(f"[DEBUG] Normalized bbox: {norm_x}, {norm_y}, {norm_w}, {norm_h}")

    if current_frame is None:
        return "No frame available", 500

    fh, fw = current_frame.shape[:2]
    x = int(norm_x * fw)
    y = int(norm_y * fh)
    w = int(norm_w * fw)
    h = int(norm_h * fh)

    # clamp
    x = max(0, min(x, fw - 1))
    y = max(0, min(y, fh - 1))
    w = max(1, min(w, fw - x))
    h = max(1, min(h, fh - y))
    bbox = (x, y, w, h)

    try:
        #tracker = cv2.TrackerKCF_create() #Use KCF tracker
        #tracker = cv2.TrackerCSRT_create()
        #tracker = cv2.legacy.TrackerMOSSE_create()
        #tracker = cv2.legacy.TrackerMIL_create()
        #tracker = cv2.legacy.TrackerTLD_create()
        tracker = cv2.legacy.TrackerMedianFlow_create()
        tracker.init(current_frame, bbox)
        last_guidance_print = 0.0  # reset so we print immediately
        print(f"[INFO] Tracker initialized at {bbox}")

        # Optional: print initial guidance right away
        yaw_deg, pitch_deg = bbox_to_angles(bbox, fw, fh)
        yaw_cmd, pitch_cmd = format_guidance(yaw_deg, pitch_deg)
        print(f"[GUIDANCE-INITIAL] {yaw_cmd}, {pitch_cmd}")

        # threading.Thread(target=auto_launch_once, daemon=True).start()
        return "Bounding box received; tracking & guidance started", 200
    except Exception as e:
        print(f"[ERROR] Tracker init error: {e}")
        tracker = None; bbox = None
        return "Tracker init error", 500

# ---------------- MAIN ----------------
if __name__ == '__main__':
    threading.Thread(target=capture_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
