import logging
import threading
import time
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
from flask import Flask, Response, request


# OpenCV runtime tuning
cv2.setNumThreads(1)
cv2.setUseOptimized(True)


# Flask / Werkzeug logging
werkzeug_log = logging.getLogger("werkzeug")
werkzeug_log.setLevel(logging.ERROR)

app = Flask(__name__)

# Configuration
@dataclass(frozen=True)
class Settings:
    # Camera source (0 for the default camera device)
    camera_index: int = 0

    # Requested capture resolution
    frame_width: int = 1280
    frame_height: int = 720

    # Camera field-of-view (degrees) used for yaw/pitch estimation
    camera_hfov_deg: float = 67.24  # HFOV = 2 × arctan( W / (2 × D) )  D=1m camera distance from the wall W=1.33m
    camera_vfov_deg: float = 41.12  # VFOV = 2 × arctan( H / (2 × D) ) D=1m H=0,75M

    # How often to print guidance to the console (seconds)
    guidance_print_interval_sec: float = 1.0

    # Preview window
    show_preview_window: bool = True
    preview_window_name: str = "UAV Edge Preview"

    # HTTP server bind
    host: str = "0.0.0.0"
    port: int = 10000


CFG = Settings()

# Shared state (accessed by capture/tracking thread and Flask request handlers)
BBox = Tuple[int, int, int, int]  # (x, y, w, h) in pixels

state_lock = threading.Lock()

current_frame: Optional["cv2.Mat"] = None
tracker: Optional[object] = None
bbox: Optional[BBox] = None

# Throttle for console guidance output
last_guidance_print_ts: float = 0.0

# Tracker performance statistics
tracker_timing = {
    "name": None,
    "frames": 0,
    "total_time": 0.0,
}


# Helper functions: bbox geometry and guidance computation
def bbox_center(b: BBox) -> Tuple[float, float]:
    """Return bbox center coordinates (cx, cy) in pixel space."""
    x, y, w, h = b
    return x + w / 2.0, y + h / 2.0


def bbox_to_angles(b: BBox, frame_w: int, frame_h: int) -> Tuple[float, float]:
    """
    Convert bbox position to yaw/pitch angles (in degrees) w.r.t. image center.

    Sign convention:
      yaw_deg   > 0 : target is to the right  -> yaw right
      yaw_deg   < 0 : target is to the left   -> yaw left
      pitch_deg > 0 : target is below center  -> pitch down
      pitch_deg < 0 : target is above center  -> pitch up
    """
    cx, cy = bbox_center(b)

    # Normalize error into [-1, 1]
    dx_norm = (cx - frame_w / 2.0) / (frame_w / 2.0)
    dy_norm = (cy - frame_h / 2.0) / (frame_h / 2.0)

    # Map normalized error to angles using camera FOV
    yaw_deg = dx_norm * (CFG.camera_hfov_deg / 2.0)
    pitch_deg = dy_norm * (CFG.camera_vfov_deg / 2.0)
    return yaw_deg, pitch_deg


def format_guidance(yaw_deg: float, pitch_deg: float) -> Tuple[str, str]:
    """Format guidance commands for display and logging."""
    # Yaw
    if abs(yaw_deg) < 1.0:
        yaw_cmd = "YAW 0 deg"
    elif yaw_deg > 0:
        yaw_cmd = f"YAW RIGHT {abs(yaw_deg):.1f} deg"
    else:
        yaw_cmd = f"YAW LEFT {abs(yaw_deg):.1f} deg"

    # Pitch
    if abs(pitch_deg) < 1.0:
        pitch_cmd = "PITCH 0 deg"
    elif pitch_deg > 0:
        pitch_cmd = f"PITCH DOWN {abs(pitch_deg):.1f} deg"
    else:
        pitch_cmd = f"PITCH UP {abs(pitch_deg):.1f} deg"

    return yaw_cmd, pitch_cmd


# Tracker factory: create an OpenCV tracker based on the requested name
def create_tracker(name: str):
    """
    Create and return an OpenCV tracker instance based on a string identifier.
    Note: Some trackers are available under cv2.legacy depending on the OpenCV build.
    """
    #name = (name or "").upper().strip()

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
    if name in ("MEDIAN"):
        return cv2.legacy.TrackerMedianFlow_create()

    raise ValueError(f"Unknown tracker type: {name}")

# Capture + tracking loop (runs in a background thread)
def capture_loop():
    """
    Continuously capture frames from the camera, run tracking if initialized,
    draw overlays, and publish the latest frame to shared state.
    """
    global current_frame, tracker, bbox, last_guidance_print_ts, tracker_timing

    cap = cv2.VideoCapture(CFG.camera_index)

    # Set the desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.frame_height)

    if not cap.isOpened():
        print("[ERROR] Could not open camera source.")
        return

    # Camera FPS may be unknown (0.0). We only need delay for preview mode.
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay_ms = int(1000 / fps) if fps and fps > 0 else 1

    while True:
        ok_read, frame = cap.read()
        if not ok_read or frame is None:
            print("[ERROR] Failed to read frame from camera.")
            time.sleep(0.1)
            continue

        fh, fw = frame.shape[:2]

        # Draw a center crosshair (visual reference)
        center = (fw // 2, fh // 2)
        cv2.drawMarker(frame,center,(255, 255, 255),markerType=cv2.MARKER_CROSS,markerSize=14,thickness=1,line_type=cv2.LINE_AA,
        )

        # If a tracker is active, update tracking state and draw overlays
        if tracker is not None and bbox is not None:
            try:
                t0 = time.perf_counter()
                track_ok, new_box = tracker.update(frame)
                dt = time.perf_counter() - t0

                tracker_timing["frames"] += 1
                tracker_timing["total_time"] += dt

                if track_ok:
                    x, y, w, h = [int(v) for v in new_box]
                    bbox = (x, y, w, h)

                    # Draw bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Guidance (yaw/pitch) from image-space error
                    yaw_deg, pitch_deg = bbox_to_angles(bbox, fw, fh)
                    yaw_cmd, pitch_cmd = format_guidance(yaw_deg, pitch_deg)

                    # Throttle console logging
                    now = time.time()
                    if now - last_guidance_print_ts > CFG.guidance_print_interval_sec:
                        print(f"[GUIDANCE] {yaw_cmd}, {pitch_cmd}")
                        last_guidance_print_ts = now

                    # Draw line from image center to target center + target point
                    cx, cy = bbox_center(bbox)
                    target_center = (int(cx), int(cy))
                    cv2.line(frame, center, target_center, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.circle(frame, target_center, 5, (0, 255, 255), -1, cv2.LINE_AA)

                    # Overlay guidance text
                    cv2.putText(
                        frame,yaw_cmd,(10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2,cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        pitch_cmd,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    # Overlay tracker performance metrics
                    frames = tracker_timing["frames"]
                    total = tracker_timing["total_time"]
                    avg_fps = (frames / total) if total > 0 else 0.0
                    #avg_ms = (total / frames) * 1000.0 if frames > 0 else 0.0

                    perf_text = (
                        f"Tracker selected: {tracker_timing['name']} | "
                        #f"avg frame: {avg_ms:.1f} ms \n   "
                        f"FPS: {avg_fps:.2f}"
                    )
                    cv2.putText(
                        frame,
                        perf_text,
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
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

        # Publish the latest frame for the HTTP handler (thread-safe)
        with state_lock:
            current_frame = frame.copy()

        # Optional local preview (disable for headless production)
        if CFG.show_preview_window:
            cv2.imshow(CFG.preview_window_name, frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(CFG.preview_window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Preview closed; stopping capture loop.")
                break

    cap.release()
    if CFG.show_preview_window:
        cv2.destroyAllWindows()
    os._exit(0)

# -----------------------------------------------------------------------------
# HTTP interface (Flask)
# -----------------------------------------------------------------------------
@app.route("/ping")
def ping():
    """Check endpoint for discovery."""
    return "SERVER UP", 200


@app.route("/frame")
def get_frame():
    """
    Return the latest available frame as a JPEG payload.

    Adds an X-Server-Timestamp header (epoch seconds) to support latency measurements.
    """
    with state_lock:
        if current_frame is None:
            return "No frame available", 503
        ok, buffer = cv2.imencode(".jpg", current_frame)

    if not ok:
        return "Frame encode failed", 500

    resp = Response(buffer.tobytes(), mimetype="image/jpeg")
    resp.headers["X-Server-Timestamp"] = str(time.time())
    return resp


@app.route("/bbox", methods=["POST"])
def set_bbox():
    """
    Initialize the tracker using a normalized bounding box and a tracker name.

    Expected JSON fields (normalized to [0,1] relative to current frame):
      - x, y, width, height
      - tracker (e.g., CSRT, KCF, MOSSE, MIL, TLD, MEDIANFLOW)

    Timestamps for latency measurements:
      - client_send_ts, server_frame_ts, client_recv_ts
    """
    global bbox, tracker, last_guidance_print_ts, tracker_timing

    data = request.get_json(force=True) or {}

    # ---------- LATENCY MEASUREMENTS ----------
    try:
        server_frame_ts = float(data.get("server_frame_ts", 0.0))
        if server_frame_ts > 0.0:
            server_recv_ts = time.time()
            rtt = server_recv_ts - server_frame_ts
            print(f"[LATENCY] RTT={rtt * 1000:.1f} ms")
    except Exception as e:
        print(f"[LATENCY ERROR] {e}")

    # Normalized bbox inputs (0..1)
    try:
        norm_x = float(data.get("x", 0.0))
        norm_y = float(data.get("y", 0.0))
        norm_w = float(data.get("width", 0.0))
        norm_h = float(data.get("height", 0.0))
    except Exception:
        return "Invalid bbox values", 400

    tracker_name = str(data.get("tracker", "CSRT"))

    # Convert normalized bbox to pixel bbox (thread-safe access to current frame)
    with state_lock:
        if current_frame is None:
            return "No frame available", 500

        fh, fw = current_frame.shape[:2]

        x = int(norm_x * fw)
        y = int(norm_y * fh)
        w = int(norm_w * fw)
        h = int(norm_h * fh)

        # Clamp bbox to image bounds
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(1, min(w, fw - x))
        h = max(1, min(h, fh - y))
        bbox = (x, y, w, h)

        # Create and initialize tracker on the current frame
        try:
            tracker = create_tracker(tracker_name)
        except Exception:
            tracker = None
            bbox = None
            return f"Unknown tracker: {tracker_name}", 400

        try:
            tracker.init(current_frame, bbox)

            # Reset tracker performance counters
            tracker_timing["name"] = tracker_name
            tracker_timing["frames"] = 0
            tracker_timing["total_time"] = 0.0
            last_guidance_print_ts = 0.0

            # Print initial guidance for debugging/telemetry
            yaw_deg, pitch_deg = bbox_to_angles(bbox, fw, fh)
            yaw_cmd, pitch_cmd = format_guidance(yaw_deg, pitch_deg)
            print(f"[INFO] Tracker '{tracker_name}' initialized at {bbox}")
            print(f"[GUIDANCE-INITIAL] {yaw_cmd}, {pitch_cmd}")

            return f"Tracker {tracker_name} started", 200
        except Exception as e:
            print(f"[ERROR] Tracker init error: {e}")
            tracker = None
            bbox = None
            return "Tracker init error", 500


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Start capture/tracking loop in a background thread
    threading.Thread(target=capture_loop, daemon=True).start()

    # Start HTTP server
    app.run(host=CFG.host, port=CFG.port, debug=False)
