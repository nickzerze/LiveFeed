import cv2
from flask import Flask, Response, request, jsonify
import threading
import os
import time
import logging
from collections import OrderedDict

# ===================== OPENCV PERFORMANCE SETTINGS =====================
cv2.setNumThreads(1)
cv2.setUseOptimized(True)
# ======================================================================

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

app = Flask(__name__)

# ========================= CONFIG =========================
USE_VIDEO = True
VIDEO_PATH = r"D:\Projects\Thesis\LiveFeed\test4.mp4"  # <-- set to your 10-second video

ALL_TRACKER_NAMES = ["CSRT", "KCF", "MOSSE", "MIL", "TLD", "MEDIAN"]

JPEG_QUALITY = 80  # lower = less CPU + bandwidth

# ========================= Global state for streaming =========================
state_lock = threading.Lock()
latest_jpeg = None               # bytes
latest_frame_index = None        # int (file frame index)
latest_server_ts = 0.0           # float

latest_tracker_name = ""         # str
latest_track_norm_bbox = None    # (x,y,w,h) normalized 0..1 or None
latest_track_ok = False          # bool
latest_mode = "preview"          # preview | benchmark

# ========================= Selection state =========================
selection_lock = threading.Lock()
selected_norm_bbox = None        # (x,y,w,h) normalized 0..1
selected_frame_index = None      # frame index at which bbox was selected
selection_event = threading.Event()

# ========================= Benchmark results =========================
results_lock = threading.Lock()
benchmark_results = OrderedDict()

bench_lock = threading.Lock()
benchmark_running = False
benchmark_done = False

# ========================= /frame request logging =========================
req_lock = threading.Lock()
req_count = 0
req_last_log = 0.0
req_last_ip = ""


# ========================= Tracker Factory =========================
def create_tracker(name: str):
    name = name.upper()
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
    raise ValueError(f"Unknown tracker: {name}")


def clamp_bbox(bbox_xywh, fw: int, fh: int):
    x, y, w, h = bbox_xywh
    x = max(0, min(int(x), fw - 1))
    y = max(0, min(int(y), fh - 1))
    w = max(1, min(int(w), fw - x))
    h = max(1, min(int(h), fh - y))
    return (x, y, w, h)


def norm_to_px(norm_bbox, fw: int, fh: int):
    nx, ny, nw, nh = norm_bbox
    x = int(float(nx) * fw)
    y = int(float(ny) * fh)
    w = int(float(nw) * fw)
    h = int(float(nh) * fh)
    return clamp_bbox((x, y, w, h), fw, fh)


def px_to_norm(bbox_xywh, fw: int, fh: int):
    x, y, w, h = bbox_xywh
    if fw <= 0 or fh <= 0:
        return None
    return (x / fw, y / fh, w / fw, h / fh)


def publish_frame(
    frame,
    frame_idx: int | None,
    mode: str,
    tracker_name: str = "",
    track_norm_bbox=None,
    track_ok: bool = False,
):
    """Encode once and publish for /frame."""
    global latest_jpeg, latest_frame_index, latest_server_ts
    global latest_tracker_name, latest_track_norm_bbox, latest_track_ok, latest_mode

    ok, buf = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)],
    )
    if not ok:
        return

    with state_lock:
        latest_jpeg = buf.tobytes()
        latest_frame_index = frame_idx
        latest_server_ts = time.time()
        latest_tracker_name = tracker_name or ""
        latest_track_norm_bbox = track_norm_bbox
        latest_track_ok = bool(track_ok)
        latest_mode = mode


# ========================= Streaming loop =========================
def streaming_loop():
    """Two phases:
    1) preview: loop the video so the operator can select bbox
    2) benchmark: sequential trackers, but ALSO stream frames + tracking bbox to Android
    """

    cap = cv2.VideoCapture(VIDEO_PATH if USE_VIDEO else 0)
    fps = cap.get(cv2.CAP_PROP_FPS) if USE_VIDEO else 0
    frame_sleep = 1.0 / fps if fps and fps > 0 else 0.03

    while True:
        # ---------- PREVIEW MODE ----------
        while not selection_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if USE_VIDEO:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                continue

            frame_idx = None
            if USE_VIDEO:
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            publish_frame(frame, frame_idx, mode="preview", tracker_name="", track_norm_bbox=None, track_ok=False)
            time.sleep(frame_sleep)

        # ---------- BENCHMARK MODE (run once per selection) ----------
        with bench_lock:
            global benchmark_running, benchmark_done
            if benchmark_running or benchmark_done:
                # If benchmark already running/done, just keep showing last preview frame
                # (or whatever is in latest_jpeg). Avoid busy-loop.
                pass
            else:
                benchmark_running = True

        if benchmark_running and not benchmark_done:
            try:
                run_sequential_benchmark_streaming(frame_sleep)
            finally:
                with bench_lock:
                    benchmark_running = False
                    benchmark_done = True

        # After benchmark, return to preview mode (keep selection_event set until /reset)
        # Rewind preview video so user sees it again.
        if USE_VIDEO:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        time.sleep(0.05)


# ========================= Benchmark worker (sequential + streaming) =========================
def run_sequential_benchmark_streaming(frame_sleep: float):
    global benchmark_results

    with selection_lock:
        norm_bbox = selected_norm_bbox
        start_idx = selected_frame_index

    if norm_bbox is None or start_idx is None:
        print("[BENCH] No selection info. Skipping benchmark.")
        return

    os.makedirs("benchmark_results", exist_ok=True)
    local_results = OrderedDict()

    print(f"[BENCH] Starting sequential benchmark from frame {start_idx} (norm_bbox={norm_bbox})")

    for name in ALL_TRACKER_NAMES:
        print(f"[BENCH] Tracker -> {name}")

        cap = cv2.VideoCapture(VIDEO_PATH)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_idx))

        ok, first_frame = cap.read()
        if not ok:
            cap.release()
            local_results[name] = {"error": "Could not read start frame"}
            continue

        fh, fw = first_frame.shape[:2]
        bbox_px = norm_to_px(norm_bbox, fw, fh)

        try:
            tracker = create_tracker(name)
            tracker.init(first_frame, bbox_px)
        except Exception as e:
            cap.release()
            local_results[name] = {"error": f"init failed: {e}"}
            continue

        frames = 0
        lost = 0
        total_update_s = 0.0

        # Publish first frame (with initial bbox)
        first_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        publish_frame(
            first_frame,
            first_idx,
            mode="benchmark",
            tracker_name=name,
            track_norm_bbox=px_to_norm(bbox_px, fw, fh),
            track_ok=True,
        )
        time.sleep(frame_sleep)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            t0 = time.perf_counter()
            ok_tr, bbox = tracker.update(frame)
            dt = time.perf_counter() - t0

            frames += 1
            total_update_s += dt

            if ok_tr:
                bbox = clamp_bbox(bbox, fw, fh)
                track_norm = px_to_norm(bbox, fw, fh)
            else:
                lost += 1
                track_norm = None

            publish_frame(
                frame,
                frame_idx,
                mode="benchmark",
                tracker_name=name,
                track_norm_bbox=track_norm,
                track_ok=ok_tr,
            )

            # Keep preview smooth-ish (doesn't affect update timing)
            time.sleep(frame_sleep)

        cap.release()

        avg_fps = (frames / total_update_s) if total_update_s > 0 else 0.0
        avg_ms = (total_update_s / frames * 1000.0) if frames > 0 else 0.0

        local_results[name] = {
            "start_frame_index": int(start_idx),
            "processed_frames": int(frames),
            "lost_frames": int(lost),
            "total_update_time_s": float(total_update_s),
            "avg_tracker_fps": float(avg_fps),
            "avg_update_ms": float(avg_ms),
        }

        path = f"benchmark_results/{name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"========= {name} =========\n")
            f.write(f"Start frame index: {start_idx}\n")
            f.write(f"Processed frames: {frames}\n")
            f.write(f"Lost frames: {lost}\n")
            f.write(f"Total update time: {total_update_s:.4f} s\n")
            f.write(f"Average tracker FPS (update-only): {avg_fps:.2f}\n")
            f.write(f"Average update time per frame: {avg_ms:.2f} ms\n")

        print(f"[BENCH] {name} done. avg {avg_ms:.2f} ms | {avg_fps:.2f} FPS | lost {lost}")

    with results_lock:
        benchmark_results.clear()
        benchmark_results.update(local_results)

    print("[BENCH] All trackers completed.")


# ========================= Flask Routes =========================
@app.route("/ping")
def ping():
    return "OK", 200


@app.route("/frame")
def get_frame():
    # --- terminal feedback that Android is fetching ---
    global req_count, req_last_log, req_last_ip
    with req_lock:
        req_count += 1
        req_last_ip = request.remote_addr or ""
        now = time.time()
        if now - req_last_log > 1.0:
            with state_lock:
                idx = latest_frame_index
                tr = latest_tracker_name
                mode = latest_mode
            print(f"[STREAM] Serving frames -> last_idx={idx} mode={mode} tracker={tr} (requests/s ~ {req_count}) ip={req_last_ip}")
            req_count = 0
            req_last_log = now

    with state_lock:
        if latest_jpeg is None:
            return "No frame available", 503

        resp = Response(latest_jpeg, mimetype="image/jpeg")
        resp.headers["X-Server-Timestamp"] = str(latest_server_ts)
        if latest_frame_index is not None:
            resp.headers["X-Frame-Index"] = str(latest_frame_index)
        resp.headers["X-Mode"] = latest_mode
        resp.headers["X-Tracker-Name"] = latest_tracker_name
        resp.headers["X-Track-Ok"] = "1" if latest_track_ok else "0"
        if latest_track_norm_bbox is not None:
            x, y, w, h = latest_track_norm_bbox
            resp.headers["X-Track-BBox"] = f"{x:.6f},{y:.6f},{w:.6f},{h:.6f}"
        return resp


@app.route("/bbox", methods=["POST"])
def set_bbox():
    global selected_norm_bbox, selected_frame_index
    global benchmark_done

    data = request.get_json(force=True) or {}

    try:
        norm_bbox = (
            float(data["x"]),
            float(data["y"]),
            float(data["width"]),
            float(data["height"]),
        )
    except Exception:
        return "Bad bbox payload", 400

    frame_idx = None
    if "frame_idx" in data:
        try:
            frame_idx = int(data["frame_idx"])
        except Exception:
            frame_idx = None

    if frame_idx is None:
        with state_lock:
            frame_idx = latest_frame_index

    if frame_idx is None:
        return "No frame index available", 500

    with selection_lock:
        selected_norm_bbox = norm_bbox
        selected_frame_index = frame_idx

    # allow rerun on new selection
    with bench_lock:
        benchmark_done = False

    selection_event.set()

    print(f"[SELECT] bbox selected at frame {frame_idx}: norm={norm_bbox}")
    return "OK", 200


@app.route("/results")
def results():
    with results_lock:
        return jsonify(benchmark_results), 200


@app.route("/reset", methods=["POST"])
def reset():
    """Clear selection and allow a fresh run without restarting the script."""
    global selected_norm_bbox, selected_frame_index
    global benchmark_done

    with selection_lock:
        selected_norm_bbox = None
        selected_frame_index = None
    selection_event.clear()
    with bench_lock:
        benchmark_done = False
    with results_lock:
        benchmark_results.clear()
    print("[RESET] Cleared selection + results. Back to preview mode.")
    return "OK", 200


# ========================= MAIN =========================
if __name__ == "__main__":
    threading.Thread(target=streaming_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=8000, debug=False)
