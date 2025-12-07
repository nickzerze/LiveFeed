import cv2
from flask import Flask, Response, request
import threading
import os
import time
import logging

# ===================== OPENCV PERFORMANCE SETTINGS =====================
cv2.setNumThreads(1)
cv2.setUseOptimized(True)
# ======================================================================

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

app = Flask(__name__)

# ========================= CONFIG =========================
USE_VIDEO = True
VIDEO_PATH = r"D:\Projects\Thesis\LiveFeed\test1.mp4"  # or set to 0

ALL_TRACKER_NAMES = ["CSRT", "KCF", "MOSSE", "MIL", "TLD", "MEDIAN"]

TRACKER_COLORS = {
    "CSRT": (0, 255, 0),
    "KCF": (255, 0, 0),
    "MOSSE": (0, 255, 255),
    "MIL": (255, 0, 255),
    "TLD": (0, 128, 255),
    "MEDIAN": (255, 255, 0)
}

trackers = {}
trackers_bbox = {}
trackers_stats = {}

bbox = None
current_frame = None
lock = threading.Lock()

start_wall_time = None   # For wall-clock FPS


# ========================= Tracker Factory =========================
def create_tracker(name):
    name = name.upper()
    if name == "CSRT": return cv2.TrackerCSRT_create()
    if name == "KCF": return cv2.TrackerKCF_create()
    if name == "MOSSE": return cv2.legacy.TrackerMOSSE_create()
    if name == "MIL": return cv2.legacy.TrackerMIL_create()
    if name == "TLD": return cv2.legacy.TrackerTLD_create()
    if name == "MEDIAN": return cv2.legacy.TrackerMedianFlow_create()
    raise ValueError(f"Unknown tracker: {name}")


# ========================= SAVE BENCHMARK RESULTS =========================
def save_results_to_files(total_wall_frames):
    """Writes one .txt file per tracker into benchmark_results/ folder."""

    os.makedirs("benchmark_results", exist_ok=True)

    total_duration_wall_sec = time.time() - start_wall_time
    wall_clock_fps = total_wall_frames / total_duration_wall_sec if total_duration_wall_sec > 0 else 0

    for name, stats in trackers_stats.items():
        frames = stats["frames"]
        lost = stats["lost"]
        total_time = stats["total"]    # seconds
        avg_fps = frames / total_time if total_time > 0 else 0
        avg_ms = (total_time / frames * 1000) if frames > 0 else 0

        path = f"benchmark_results/{name}.txt"

        with open(path, "w") as f:
            f.write(f"========= {name} =========\n")
            f.write(f"Processed frames: {frames}\n")
            f.write(f"Lost frames: {lost}\n")
            f.write(f"Total tracking time: {total_time:.4f} s\n")
            f.write(f"Average tracker FPS: {avg_fps:.2f}\n")
            f.write(f"Average tracker time per frame: {avg_ms:.2f} ms\n")
            f.write(f"Wall-clock FPS: {wall_clock_fps:.2f}\n")
            f.write(f"Total wall time: {total_duration_wall_sec:.2f} s\n")

        print(f"[SAVED] {path}")


# ========================= Capture Loop =========================
def capture_loop():
    global current_frame, trackers, trackers_stats, trackers_bbox, bbox, start_wall_time

    cap = cv2.VideoCapture(VIDEO_PATH if USE_VIDEO else 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 33

    total_wall_frames = 0
    start_wall_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            if USE_VIDEO:
                break
            else:
                continue

        total_wall_frames += 1

        # Run benchmark tracking
        if bbox is not None and len(trackers) > 0:
            overlay_y = 30
            for name, tracker_obj in list(trackers.items()):

                t0 = time.perf_counter()
                ok, new_box = tracker_obj.update(frame)
                dt = time.perf_counter() - t0

                trackers_stats[name]["frames"] += 1
                trackers_stats[name]["total"] += dt

                if ok:
                    x, y, w, h = [int(v) for v in new_box]
                    color = TRACKER_COLORS[name]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                else:
                    trackers_stats[name]["lost"] += 1

                avg_fps = trackers_stats[name]["frames"] / trackers_stats[name]["total"]
                avg_ms = (trackers_stats[name]["total"] / trackers_stats[name]["frames"]) * 1000

                cv2.putText(
                    frame,
                    f"{name:<6} avg: {avg_ms:5.1f} ms  FPS: {avg_fps:6.2f}  lost:{trackers_stats[name]['lost']}",
                    (10, overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    TRACKER_COLORS[name],
                    2
                )
                overlay_y += 22

        with lock:
            current_frame = frame.copy()

        cv2.imshow("BENCHMARK", frame)
        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            print("[BENCHMARK] Stopping benchmark…")
            break

    # SAVE REPORT FILES HERE
    save_results_to_files(total_wall_frames)

    cap.release()
    cv2.destroyAllWindows()
    os._exit(0)


# ========================= Flask Routes =========================
@app.route("/ping")
def ping():
    return "OK", 200


@app.route("/frame")
def get_frame():
    global current_frame
    with lock:
        if current_frame is None:
            return "No frame available", 503

        _, buffer = cv2.imencode(".jpg", current_frame)
        response = Response(buffer.tobytes(), mimetype="image/jpeg")
        response.headers["X-Server-Timestamp"] = str(time.time())
        return response


@app.route("/bbox", methods=["POST"])
def set_bbox():
    global bbox, trackers, trackers_bbox, trackers_stats

    data = request.get_json()

    if current_frame is None:
        return "No frame", 500

    fh, fw = current_frame.shape[:2]

    x = int(float(data["x"]) * fw)
    y = int(float(data["y"]) * fh)
    w = int(float(data["width"]) * fw)
    h = int(float(data["height"]) * fh)

    bbox = (x, y, w, h)
    print(f"[BENCHMARK] Received bbox from phone: {bbox}")

    # Reset trackers
    trackers = {}
    trackers_bbox = {}
    trackers_stats = {}

    for name in ALL_TRACKER_NAMES:
        try:
            t = create_tracker(name)
            t.init(current_frame, bbox)
            trackers[name] = t
            trackers_stats[name] = {"frames": 0, "total": 0.0, "lost": 0}
            print(f"[BENCHMARK] {name} initialized.")
        except Exception as e:
            print(f"[ERROR] {name} init failed: {e}")

    return "Benchmark Started", 200


# ========================= MAIN =========================
if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=8000, debug=False)
