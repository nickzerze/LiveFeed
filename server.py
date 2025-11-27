import cv2
from flask import Flask, Response, request
import threading
import os
import time
import logging
import numpy as np
import torch
from ultralytics import YOLO


# try:
#     import cv2
#     from flask import Flask, Response, request
#     import threading
#     import os
#     import time
#     import logging
#     import numpy as np
#     import torch
#     from ultralytics import YOLO
#
# except ImportError:
#     print("[INFO] Missing dependencies, running auto-installer...")
#     import subprocess, sys
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "opencv-python", "numpy", "torch", "torchvision", "torchaudio", "ultralytics", "pillow"])


# Quiet Flask access logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# ===================== KALMAN FILTER =====================

class KalmanFilter2D:
    def __init__(self):
        # state: [x, y, vx, vy]
        self.x = np.zeros((4, 1))

        # state transition matrix (constant velocity model)
        dt = 1.0
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)

        # process noise
        self.Q = np.eye(4) * 0.01

        # measurement matrix: we only measure x, y
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # measurement noise
        self.R = np.eye(2) * 5.0

        # covariance
        self.P = np.eye(4)

    def init_state(self, measured_x, measured_y):
        """Initialize state from first measurement."""
        self.x[:] = 0.0
        self.x[0, 0] = measured_x
        self.x[1, 0] = measured_y
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


# ===================== CONFIG =====================
USE_VIDEO = True
VIDEO_PATH = 0  # 0 = default webcam, or path to video file
# VIDEO_PATH = r"D:\Projects\Thesis\LiveFeed\test_720.mp4"  # or set to 0 for webcam

# Camera FOV for bbox->bearing (your measured values)
CAMERA_HFOV_DEG = 67.24  # HFOV = 2 × arctan(W / (2 × D))
CAMERA_VFOV_DEG = 41.12  # VFOV = 2 × arctan(H / (2 × D))

# How often to print guidance in terminal (seconds)
GUIDANCE_PRINT_INTERVAL = 1.0
# ===================================================

# Shared state
lock = threading.Lock()
current_frame = None          # last frame to send to Android
bbox = None                   # current smoothed bbox (x, y, w, h)
roi_bbox = None               # ROI selected from Android (x, y, w, h)


kf = KalmanFilter2D()
kf_initialized = False

# For throttling console prints
last_guidance_print = 0.0

# Load YOLOv8 nano model (COCO pretrained)
# This will auto-download yolov8n.pt the first time.
yolo_model = YOLO("yolov8n.pt")


# ===================== HELPER FUNCTIONS =====================

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
    Uses 'deg' instead of the degree symbol to avoid '??' on Android.
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


def point_in_bbox(cx, cy, box):
    """Check if a point (cx, cy) lies inside bbox (x, y, w, h)."""
    x, y, w, h = box
    return (x <= cx <= x + w) and (y <= cy <= y + h)


# ===================== VIDEO + YOLO + KALMAN LOOP =====================

def capture_loop_yolo_kalman():
    global current_frame, bbox, last_guidance_print, kf_initialized, roi_bbox

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
        #                YOLOv8n DETECTION
        # -----------------------------------------------------
        measurement_box = None  # (x, y, w, h) from YOLO
        measurement_cx = None
        measurement_cy = None

        if roi_bbox is not None:
            # Run YOLO only if user has selected an ROI
            results = yolo_model(frame, imgsz=640, conf=0.4, verbose=False)
            boxes = results[0].boxes

            best_candidate = None
            best_area = 0.0

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()  # [N, 4]

                for box in xyxy:
                    x1, y1, x2, y2 = box
                    w_box = x2 - x1
                    h_box = y2 - y1
                    cx = x1 + w_box / 2.0
                    cy = y1 + h_box / 2.0

                    if point_in_bbox(cx, cy, roi_bbox):
                        area = w_box * h_box
                        # choose largest detection that lies inside ROI
                        if area > best_area:
                            best_area = area
                            best_candidate = (int(x1), int(y1),
                                              int(w_box), int(h_box),
                                              float(cx), float(cy))

            if best_candidate is not None:
                x_meas, y_meas, w_meas, h_meas, cx_meas, cy_meas = best_candidate
                measurement_box = (x_meas, y_meas, w_meas, h_meas)
                measurement_cx = cx_meas
                measurement_cy = cy_meas

        # -----------------------------------------------------
        #                KALMAN FILTER LOGIC
        # -----------------------------------------------------
        if measurement_box is not None:
            # We have a YOLO detection inside the ROI
            if not kf_initialized:
                kf.init_state(measurement_cx, measurement_cy)
                kf_initialized = True

            # Update with measurement, then predict next state
            kf.update(measurement_cx, measurement_cy)
            pred_x, pred_y = kf.predict()

            w_meas, h_meas = measurement_box[2], measurement_box[3]
            px = int(pred_x - w_meas / 2.0)
            py = int(pred_y - h_meas / 2.0)
            bbox = (px, py, int(w_meas), int(h_meas))

            # Draw raw YOLO box (optional, in red)
            x_raw, y_raw, w_raw, h_raw = measurement_box
            cv2.rectangle(frame, (x_raw, y_raw), (x_raw + w_raw, y_raw + h_raw),
                          (0, 0, 255), 1)

            # Draw smoothed (Kalman) box in green
            cv2.rectangle(frame, (px, py), (px + w_meas, py + h_meas),
                          (0, 255, 0), 2)

        else:
            # No YOLO detection this frame
            if kf_initialized and bbox is not None:
                # Predict only
                pred_x, pred_y = kf.predict()
                bw, bh = bbox[2], bbox[3]
                px = int(pred_x - bw / 2.0)
                py = int(pred_y - bh / 2.0)
                bbox = (px, py, bw, bh)

                cv2.rectangle(frame, (px, py), (px + bw, py + bh),
                              (0, 128, 128), 2)
                cv2.putText(frame, "PREDICTING...",
                            (10, fh - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 200, 200), 2)

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

            # --- Draw command strings ---
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


# ===================== FLASK ROUTES =====================

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
    """
    Receive ROI from Android; used to select which YOLO detection to follow.
    Android sends normalized coordinates [0..1] (x, y, w, h).
    """
    global bbox, roi_bbox, current_frame, last_guidance_print, kf_initialized

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

    roi_bbox = (x, y, w, h)
    bbox = None  # will be set once YOLO finds a detection inside ROI

    # reset Kalman
    kf.init_state(x + w / 2.0, y + h / 2.0)
    kf_initialized = False
    last_guidance_print = 0.0

    print(f"[INFO] ROI from Android: {roi_bbox}")
    return "ROI received; YOLO+Kalman tracking will start", 200


# ===================== MAIN =====================

if __name__ == '__main__':
    threading.Thread(target=capture_loop_yolo_kalman, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
