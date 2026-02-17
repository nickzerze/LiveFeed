import cv2
from flask import Flask, Response, request
import threading
import os
import time
#import math
import logging
import numpy as np
#from pymavlink import mavutil


# ===================== OPENCV PERFORMANCE SETTINGS  =====================
# ===================== ONLY IF OPENCV-PYTHON IS INSTALLED  ==============
cv2.setNumThreads(8)        # prevent multi-thread jitter on older CPUs
cv2.setUseOptimized(True)   # enables SIMD & CPU optimizations
# ======================================================================

# Quiet Flask access logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# ===================== CONFIG =====================
# print(cv2.__version__)
#If USE_VIDEO==True -> VIDEO_PATH is used, else if USE_VIDEO==False -> VIDEO_PATH is set to 0
USE_VIDEO = True
VIDEO_PATH = r"D:\Projects\Thesis\LiveFeed\video.mp4"

# MAVProxy must output to this port (e.g. in MAVProxy: `output add 127.0.0.1:14552`)
# UDP_IN = 'udpin:0.0.0.0:14550'   # keep QGC on 14550

# Camera FOV for bbox->bearing
#For resolution 1280x720
CAMERA_HFOV_DEG = 67.24  # HFOV = 2 × arctan( W / (2 × D) )  D=1m camera distance from the wall W=1.33m
CAMERA_VFOV_DEG = 41.12  # VFOV = 2 × arctan( H / (2 × D) ) D=1m H=0,75M

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

# Tracker performance stats
tracker_timing = {
    "name": None,
    "frames": 0,
    "total_time": 0.0
}


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
        yaw_cmd = "YAW 0 deg"
    elif yaw_deg > 0:
        yaw_cmd = f"YAW RIGHT {abs(yaw_deg):.1f} deg"
    else:
        yaw_cmd = f"YAW LEFT {abs(yaw_deg):.1f} deg"

    # Pitch (up/down)
    if abs(pitch_deg) < 1.0:
        pitch_cmd = "PITCH 0 deg"
    elif pitch_deg > 0:
        pitch_cmd = f"PITCH DOWN {abs(pitch_deg):.1f} deg"
    else:
        pitch_cmd = f"PITCH UP {abs(pitch_deg):.1f} deg"

    return yaw_cmd, pitch_cmd

def capture_loop():
    global current_frame, tracker, bbox, last_guidance_print, tracker_timing
    cap = cv2.VideoCapture(VIDEO_PATH if USE_VIDEO else 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps and fps > 0 else 33
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        os._exit(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            if USE_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[ERROR] Failed to read frame from camera")
                os._exit(0)

        fh, fw = frame.shape[:2]

        # === Always draw center crosshair (for reference)
        center = (fw // 2, fh // 2)
        cv2.drawMarker(
            frame, center, (255, 255, 255),
            markerType=cv2.MARKER_CROSS, markerSize=14,
            thickness=1, line_type=cv2.LINE_AA
        )

        # Update tracker
        if tracker is not None and bbox is not None:
            try:
                t0 = time.perf_counter()
                ok, new_box = tracker.update(frame)
                dt = time.perf_counter() - t0

                tracker_timing["frames"] += 1
                tracker_timing["total_time"] += dt
                avg_fps = tracker_timing["frames"] / tracker_timing["total_time"]

                if ok:
                    x, y, w, h = [int(v) for v in new_box]
                    bbox = (x, y, w, h)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # ---- GUIDANCE COMPUTATION ----
                    yaw_deg, pitch_deg = bbox_to_angles(bbox, fw, fh)
                    yaw_cmd, pitch_cmd = format_guidance(yaw_deg, pitch_deg)

                    # ---- GUIDANCE PRINTING (terminal) ----
                    now = time.time()
                    if now - last_guidance_print > GUIDANCE_PRINT_INTERVAL:
                        print(f"[GUIDANCE] {yaw_cmd}, {pitch_cmd}")
                        last_guidance_print = now

                    # ---- GUIDANCE OVERLAY ON FRAME
                    # line from center to bbox center
                    cx, cy = bbox_center(bbox)
                    target_center = (int(cx), int(cy))
                    cv2.line(
                        frame, center, target_center,
                        (255, 0, 0), 2, cv2.LINE_AA
                    )
                    cv2.circle(
                        frame, target_center, 5,
                        (0, 255, 255), -1, cv2.LINE_AA
                    )


                    # text with discrete commands
                    cv2.putText(
                        frame,
                        yaw_cmd,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        #(0, 255, 110),
                        2,
                        cv2.LINE_AA
                    )
                    cv2.putText(
                        frame,
                        pitch_cmd,
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        #(0, 255, 110),
                        2,
                        cv2.LINE_AA
                    )
                    # ================= FPS Overlay =================
                    average_ms = (tracker_timing["total_time"] / tracker_timing["frames"]) * 1000 if tracker_timing[
                                                                                                         "frames"] > 0 else 0

                    text = (
                        f"{tracker_timing['name']}   "
                        f"aver. frame: {average_ms:.1f} ms   "
                        f"FPS: {avg_fps:.2f}"
                    )

                    cv2.putText(
                        frame,
                        text,
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 50),
                        2,
                        cv2.LINE_AA
                    )
                else:
                    # Tracking lost overlay
                    cv2.putText(
                        frame,
                        "TRACKING LOST",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),  # red
                        2,
                        cv2.LINE_AA,
                    )
                    print("[INFO] Tracking lost")
            except Exception as e:
                print(f"[ERROR] Tracker update failed: {e}")

        with lock:
            current_frame = frame.copy()

        cv2.imshow("Live Feed", frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q') or cv2.getWindowProperty("Live Feed", cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Exiting capture loop…")
            break

    cap.release()
    cv2.destroyAllWindows()
    os._exit(0)

# ---------------- TRACKER FACTORY ----------------
def create_tracker(name: str):
    name = name.upper().strip()

    if name == "CSRT":
        return cv2.TrackerCSRT_create()
    if name == "KCF":
        return cv2.TrackerKCF_create()
    if name == "MOSSE":
        return cv2.legacy.TrackerMOSSE_create()
    if name == "MIL":
        return cv2.legacy.TrackerMIL_create()
    if name == "TLD":
        return cv2.legacy.TrackerTLD_create()
    if name == "MEDIAN":
        return cv2.legacy.TrackerMedianFlow_create()

    raise ValueError(f"Unknown tracker type: {name}")


# ---------------- FLASK ROUTES ----------------
@app.route("/ping")
def ping():
    return "OK", 200

@app.route('/frame')
def get_frame():
    global current_frame
    with lock:
        if current_frame is None:
            return "No frame available", 503

        _, buffer = cv2.imencode('.jpg', current_frame)
        response = Response(buffer.tobytes(), mimetype='image/jpeg')
        response.headers['X-Server-Timestamp'] = str(time.time())
        return response



@app.route('/bbox', methods=['POST'])
def set_bbox():
    """Init tracker from Android bbox."""
    global bbox, tracker, current_frame, last_guidance_print, tracker_timing

    data = request.get_json()

    try:
        server_frame_ts = float(data.get("server_frame_ts", 0.0))
        if server_frame_ts > 0.0:
            server_recv_ts = time.time()
            rtt = server_recv_ts - server_frame_ts
            print(f"[LATENCY] RTT={rtt * 1000:.1f} ms")
    except Exception as e:
        print(f"[LATENCY ERROR] {e}")

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

    # --------- NEW: Read tracker name from Android ---------
    tracker_name = data.get("tracker", "CSRT")  # default to CSRT

    try:
        tracker = create_tracker(tracker_name)
    except Exception as e:
        print(f"[ERROR] Unknown tracker: {tracker_name}")
        return f"Unknown tracker: {tracker_name}", 400

    # --------- Initialize tracker ---------
    try:
        tracker.init(current_frame, bbox)

        # RESET TRACKER PERFORMANCE COUNTERS (FPS measurement)
        tracker_timing["name"] = tracker_name
        tracker_timing["frames"] = 0
        tracker_timing["total_time"] = 0.0


        last_guidance_print = 0.0

        print(f"[INFO] Tracker '{tracker_name}' initialized at {bbox}")

        # Initial guidance
        yaw_deg, pitch_deg = bbox_to_angles(bbox, fw, fh)
        yaw_cmd, pitch_cmd = format_guidance(yaw_deg, pitch_deg)
        print(f"[GUIDANCE-INITIAL] {yaw_cmd}, {pitch_cmd}")

        return f"Tracker {tracker_name} started", 200

    except Exception as e:
        print(f"[ERROR] Tracker init error: {e}")
        tracker = None
        bbox = None
        return "Tracker init error", 500


# ---------------- MAIN ----------------
if __name__ == '__main__':
    threading.Thread(target=capture_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=10000, debug=False)
