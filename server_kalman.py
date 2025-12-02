import cv2
from flask import Flask, Response, request
import threading
import os
import time
#import math
import logging
import numpy as np
#from pymavlink import mavutil


class KalmanFilter2D:
    def __init__(self):
        # state: [x, y, vx, vy]
        self.x = np.zeros((4, 1))

        # state transition matrix
        dt = 1
        self.F = np.array([[1, 0, dt, 0],
                           [1, 0, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)

        # process noise
        self.Q = np.eye(4) * 0.01

        # measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)

        # measurement noise
        self.R = np.eye(2) * 5

        # covariance
        self.P = np.eye(4)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0, 0], self.x[1, 0]

    def update(self, measured_x, measured_y):
        z = np.array([[measured_x], [measured_y]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

# ===================== OPENCV PERFORMANCE SETTINGS  =====================
# ===================== ONLY IF OPENCV-PYTHON IS INSTALLED  ==============
cv2.setNumThreads(1)        # prevent multi-thread jitter on older CPUs
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
VIDEO_PATH = r"D:\Projects\Thesis\LiveFeed\test_720.mp4"  # or set to 0 for webcam

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
kf = KalmanFilter2D()

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


# ---------------- VIDEO THREAD ----------------
def capture_loop_kalman():
    global current_frame, tracker, bbox, last_guidance_print, kf
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

        # === Draw center crosshair ===
        center = (fw // 2, fh // 2)
        cv2.drawMarker(
            frame, center, (255, 255, 255),
            markerType=cv2.MARKER_CROSS, markerSize=14,
            thickness=1, line_type=cv2.LINE_AA
        )

        # -----------------------------------------------------
        #                TRACKING + KALMAN FILTER
        # -----------------------------------------------------
        if tracker is not None and bbox is not None:
            try:
                ok, new_box = tracker.update(frame)

                if ok:
                    # --- RAW TRACKER OUTPUT ---
                    x, y, w_box, h_box = [int(v) for v in new_box]
                    cx = x + w_box / 2
                    cy = y + h_box / 2

                    # --- UPDATE KALMAN FILTER WITH MEASUREMENT ---
                    kf.update(cx, cy)

                    # --- PREDICT NEXT POSITION ---
                    pred_x, pred_y = kf.predict()

                    # convert predicted center to bbox coords
                    px = int(pred_x - w_box / 2)
                    py = int(pred_y - h_box / 2)

                    bbox = (px, py, w_box, h_box)

                    # draw smoothed bbox
                    cv2.rectangle(frame, (px, py), (px + w_box, py + h_box),
                                  (0, 255, 0), 2)

                else:
                    # ------------------------------------------------
                    # TRACKER LOST → rely ONLY on Kalman prediction
                    # ------------------------------------------------
                    pred_x, pred_y = kf.predict()


                    # use old bbox size (if available)
                    bw = bbox[2]
                    bh = bbox[3]

                    px = int(pred_x - bw / 2)
                    py = int(pred_y - bh / 2)

                    bbox = (px, py, bw, bh)

                    cv2.rectangle(frame, (px, py), (px + bw, py + bh),
                                  (0, 128, 128), 2)
                    cv2.putText(frame, "PREDICTING...",
                                (10, fh - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 200, 200), 2)

            except Exception as e:
                print(f"[ERROR] Tracker update failed: {e}")
                #tracker = None
                #bbox = None
                #continue

        # -----------------------------------------------------
        #                GUIDANCE COMPUTATION
        # -----------------------------------------------------
        if bbox is not None:
            cx, cy = bbox_center(bbox)
            yaw_deg, pitch_deg = bbox_to_angles(bbox, fw, fh)
            yaw_cmd, pitch_cmd = format_guidance(yaw_deg, pitch_deg)

            # --- Draw line from center to bbox center ---
            target_center = (int(cx), int(cy))
            cv2.line(frame, center, target_center, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, target_center, 5, (0, 255, 255), -1, cv2.LINE_AA)

            # --- Draw angle numbers ---
            cv2.putText(
                frame,
                f"Yaw: {yaw_deg:+.1f} deg   Pitch: {pitch_deg:+.1f} deg",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            # --- Draw command (signed, Android-safe) ---
            cv2.putText(
                frame, yaw_cmd, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA
            )
            cv2.putText(
                frame, pitch_cmd, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA
            )

            # --- Throttled console printing ---
            now = time.time()
            if now - last_guidance_print > GUIDANCE_PRINT_INTERVAL:
                print(f"[GUIDANCE] {yaw_cmd}, {pitch_cmd}")
                last_guidance_print = now

        # -----------------------------------------------------
        #       UPDATE SHARED FRAME + DISPLAY
        # -----------------------------------------------------
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
    global bbox, tracker, current_frame, last_guidance_print

    data = request.get_json()

    # ---------- LATENCY MEASUREMENTS ----------
    try:
        client_send_ts = float(data.get("client_send_ts", 0))      # when Android sent bbox
        server_frame_ts = float(data.get("server_frame_ts", 0))    # when Python created frame
        client_recv_ts = float(data.get("client_recv_ts", 0))      # when Android got frame
        server_recv_ts = time.time()                               # NOW (Python receiving bbox)

        # Round-trip time (server frame → phone → server)
        rtt = server_recv_ts - server_frame_ts

        # Server → Phone (frame download delay)
        downlink = client_recv_ts - server_frame_ts

        # Phone → Server (bbox upload delay)
        uplink = server_recv_ts - client_send_ts

        # RTT = time for a frame to go phone -> phone sends bbox -> server receives it
        # Downlink latency = How long a frame generated on the python server to ARRIVE at Android phone
        # Uplink latency = How long it takes for the BBOX (drawn by user) to from Android phone back to python server
        print(f"[LATENCY] RTT={rtt*1000:.1f} ms  Down={downlink*1000:.1f} ms  Up={uplink*1000:.1f} ms")

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

    try:
        # tracker = cv2.TrackerKCF_create() # BEST 2
        tracker = cv2.TrackerCSRT_create() # BEST 1
        # tracker = cv2.legacy.TrackerMOSSE_create()
        # tracker = cv2.legacy.TrackerMIL_create() # BEST 3
        # tracker = cv2.legacy.TrackerTLD_create() # BEST 4
        # tracker = cv2.legacy.TrackerMedianFlow_create() # BEST 5
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
        tracker = None
        bbox = None
        return "Tracker init error", 500


# ---------------- MAIN ----------------
if __name__ == '__main__':
    threading.Thread(target=capture_loop_kalman, daemon=True).start()
    app.run(host='0.0.0.0', port=10000, debug=False)
