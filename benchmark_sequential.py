#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import threading
import logging
from typing import Optional, Tuple, Dict, Any, List

import cv2
from flask import Flask, Response, request, jsonify

# -------------------- CONFIG --------------------
VIDEO_PATH = "test2.mp4"          # <-- βάλε εδώ το 10s video σου
HOST = "0.0.0.0"
PORT = 8000                      # ταιριάζει με Android αν χρησιμοποιεί :8000

TRACKER_ORDER = ["CSRT", "KCF", "MOSSE", "MIL", "TLD", "MEDIANFLOW"]

IDLE_TARGET_FPS = 5.0           # πριν το bbox (μόνο για preview/επιλογή)
PREVIEW_ENABLED = True
PREVIEW_WINDOW = "Android Stream Preview (Laptop) | ESC closes preview only"

RESULTS_DIR = "benchmark_results"
SELECTION_SHOW_SECONDS = 0.2      # δείξε selection μόνο για λίγο, τη στιγμή που έγινε
JPEG_QUALITY = 100

# --------- NEW: SAVE OUTPUT VIDEO ---------
SAVE_OUTPUT_VIDEO = True
OUTPUT_VIDEO_CODEC = "mp4v"       # δοκίμασε "avc1" ή "H264" αν έχεις, αλλιώς mp4v
OUTPUT_VIDEO_EXT = ".mp4"         # αν έχεις θέμα, άλλαξέ το σε ".avi"
# ------------------------------------------


# -------------------- FLASK APP --------------------
app = Flask(__name__)

# κόψε τα "GET /frame HTTP/1.1" logs
logging.getLogger("werkzeug").setLevel(logging.ERROR)
app.logger.disabled = True


# -------------------- GLOBAL STATE --------------------
lock = threading.Lock()
stop_all = threading.Event()
selection_ready = threading.Event()

latest_frame_bgr: Optional[Any] = None
latest_jpeg_bytes: Optional[bytes] = None
latest_frame_index: int = -1

current_tracker_name: str = ""
track_bbox_norm: Optional[Tuple[float, float, float, float]] = None
track_ok: bool = False

# Selection data (ίδιο για όλους)
selection_bbox_norm: Optional[Tuple[float, float, float, float]] = None
selection_frame_idx: Optional[int] = None
selection_show_until_ts: float = 0.0

# stream stats
_req_count = 0
_req_last_log = time.time()
_req_last_count = 0

VIDEO_FPS_FALLBACK = 30.0


# -------------------- UTILITIES --------------------
def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def format_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    mm = int(seconds // 60)
    ss = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{mm:02d}:{ss:02d}.{ms:03d}"


def norm_to_px(b: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int, int, int]:
    x, y, bw, bh = b
    x = int(clamp01(x) * w)
    y = int(clamp01(y) * h)
    bw = int(clamp01(bw) * w)
    bh = int(clamp01(bh) * h)

    bw = max(2, bw)
    bh = max(2, bh)
    x = min(max(0, x), max(0, w - 2))
    y = min(max(0, y), max(0, h - 2))
    bw = min(bw, w - x)
    bh = min(bh, h - y)
    return (x, y, bw, bh)


def px_to_norm(x: int, y: int, bw: int, bh: int, w: int, h: int) -> Tuple[float, float, float, float]:
    return (x / w, y / h, bw / w, bh / h)


def create_tracker(name: str):
    n = name.strip().upper()

    def _create(fn_name: str):
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, fn_name):
            return getattr(cv2.legacy, fn_name)()
        if hasattr(cv2, fn_name):
            return getattr(cv2, fn_name)()
        raise RuntimeError(f"Tracker factory {fn_name} not found in your OpenCV build.")

    if n == "CSRT":
        return _create("TrackerCSRT_create")
    if n == "KCF":
        return _create("TrackerKCF_create")
    if n == "MOSSE":
        return _create("TrackerMOSSE_create")
    if n == "MIL":
        return _create("TrackerMIL_create")
    if n == "TLD":
        return _create("TrackerTLD_create")
    if n == "MEDIANFLOW":
        return _create("TrackerMedianFlow_create")

    raise ValueError(f"Unknown tracker: {name}")


def draw_overlay(frame_bgr,
                 tracker_name: str,
                 idx: int,
                 fps: float,
                 sel_norm: Optional[Tuple[float, float, float, float]],
                 tr_norm: Optional[Tuple[float, float, float, float]],
                 ok: bool) -> Any:
    """
    Αυτό είναι που βλέπει ΚΑΙ το Android ΚΑΙ το Laptop preview ΚΑΙ γράφεται στο video.
    - πάνω αριστερά: χρόνος video (idx/fps)
    - tracking bbox (και label)
    - selection bbox μόνο αν sel_norm != None (ελεγχόμενο χρονικά)
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # time overlay (top-left)
    t = (idx / fps) if fps > 0 else (idx / VIDEO_FPS_FALLBACK)
    time_txt = format_time(t)
    cv2.putText(out, time_txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # selection bbox (red) - μόνο στιγμιαία
    if sel_norm is not None:
        sx, sy, sw, sh = norm_to_px(sel_norm, w, h)
        cv2.rectangle(out, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
        cv2.putText(out, "SELECTION", (sx, max(20, sy - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # tracking bbox
    if tr_norm is not None:
        tx, ty, tw, th = norm_to_px(tr_norm, w, h)
        color = (0, 255, 0) if ok else (0, 255, 255)
        cv2.rectangle(out, (tx, ty), (tx + tw, ty + th), color, 2)
        label = f"{tracker_name} | idx={idx} | {'OK' if ok else 'LOST'}"
        cv2.putText(out, label, (tx, max(20, ty - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        if tracker_name:
            cv2.putText(out, f"{tracker_name} | idx={idx}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return out


def publish_frame(frame_bgr, idx: int, fps: float,
                  tracker_name: str,
                  tr_norm: Optional[Tuple[float, float, float, float]],
                  ok: bool) -> Any:
    """
    Δημοσιεύει το frame (Android+Laptop) και επιστρέφει το composed frame,
    ώστε να μπορούμε να το γράψουμε και σε .mp4.
    """
    global latest_frame_bgr, latest_jpeg_bytes, latest_frame_index

    with lock:
        sel = selection_bbox_norm
        show_sel = (sel is not None) and (time.time() <= selection_show_until_ts)

    sel_to_draw = sel if show_sel else None
    composed = draw_overlay(frame_bgr, tracker_name, idx, fps, sel_to_draw, tr_norm, ok)

    ok_enc, jpg = cv2.imencode(".jpg", composed, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if ok_enc:
        with lock:
            latest_frame_bgr = composed
            latest_jpeg_bytes = jpg.tobytes()
            latest_frame_index = idx

    return composed


def write_txt(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for k, v in data.items():
            f.write(f"{k}: {v}\n")


def make_video_writer(path: str, fps: float, width: int, height: int) -> Optional[cv2.VideoWriter]:
    """
    Προσπαθεί να ανοίξει VideoWriter. Αν αποτύχει, επιστρέφει None.
    """
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        return None
    return writer


# -------------------- FLASK ENDPOINTS --------------------
@app.get("/ping")
def ping():
    return "OK", 200


@app.get("/frame")
def get_frame():
    global _req_count, _req_last_log, _req_last_count

    with lock:
        data = latest_jpeg_bytes
        idx = latest_frame_index
        tname = current_tracker_name
        tb = track_bbox_norm
        tok = track_ok

    _req_count += 1
    now = time.time()
    if now - _req_last_log >= 1.0:
        rps = _req_count - _req_last_count
        _req_last_count = _req_count
        _req_last_log = now
        log(f"[STREAM] /frame  rps≈{rps}  idx={idx}  tracker={tname}")

    if not data:
        return Response(status=204)

    resp = Response(data, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"

    resp.headers["X-Server-Timestamp"] = f"{time.time():.6f}"
    resp.headers["X-Frame-Index"] = str(idx)
    resp.headers["X-Tracker-Name"] = tname
    resp.headers["X-Track-Ok"] = "1" if tok else "0"
    if tb is not None:
        resp.headers["X-Track-BBox"] = f"{tb[0]:.6f},{tb[1]:.6f},{tb[2]:.6f},{tb[3]:.6f}"
    return resp


@app.post("/bbox")
def set_bbox():
    global selection_bbox_norm, selection_frame_idx, selection_show_until_ts
    global track_bbox_norm, track_ok, current_tracker_name

    payload = request.get_json(silent=True) or {}
    try:
        x = float(payload["x"])
        y = float(payload["y"])
        w = float(payload["width"])
        h = float(payload["height"])
        fidx = int(payload["frame_idx"])
    except Exception:
        return jsonify({"ok": False, "error": "Expected JSON: {x,y,width,height,frame_idx}"}), 400

    x, y, w, h = clamp01(x), clamp01(y), clamp01(w), clamp01(h)
    if w <= 0.001 or h <= 0.001:
        return jsonify({"ok": False, "error": "bbox too small"}), 400

    with lock:
        selection_bbox_norm = (x, y, w, h)
        selection_frame_idx = fidx
        selection_show_until_ts = time.time() + SELECTION_SHOW_SECONDS

        track_bbox_norm = None
        track_ok = False
        current_tracker_name = ""

    log(f"[BBOX] received bbox_norm=({x:.3f},{y:.3f},{w:.3f},{h:.3f}) at frame_idx={fidx}")
    selection_ready.set()
    return jsonify({"ok": True})


# -------------------- THREADS --------------------
def laptop_preview_loop():
    if not PREVIEW_ENABLED:
        return

    while not stop_all.is_set():
        with lock:
            frame = None if latest_frame_bgr is None else latest_frame_bgr.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        cv2.imshow(PREVIEW_WINDOW, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            try:
                cv2.destroyWindow(PREVIEW_WINDOW)
            except Exception:
                pass
            log("[PREVIEW] closed by user (ESC). Stream/benchmark continues.")
            return

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def idle_stream_loop():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        log(f"[ERROR] Cannot open video: {VIDEO_PATH}")
        stop_all.set()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS_FALLBACK

    idx = 0
    dt_target = 1.0 / max(1.0, IDLE_TARGET_FPS)
    last_t = time.time()

    log("[IDLE] Streaming video to Android for bbox selection...")

    while not stop_all.is_set() and not selection_ready.is_set():
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            idx = 0
            continue

        publish_frame(frame, idx, fps, "", None, False)

        now = time.time()
        elapsed = now - last_t
        if elapsed < dt_target:
            time.sleep(dt_target - elapsed)
        last_t = time.time()
        idx += 1

    cap.release()


def run_single_tracker_from_start(tracker_name: str,
                                 init_frame_idx: int,
                                 sel_norm: Tuple[float, float, float, float]) -> Dict[str, Any]:
    """
    Κάθε tracker:
    - ξεκινάει το video από frame 0 (ώστε να “παίζει από την αρχή”)
    - όταν φτάσει στο init_frame_idx κάνει init(bbox)
    - μετά συνεχίζει tracking μέχρι τέλος
    - αποθηκεύει και overlay video (mp4) αν SAVE_OUTPUT_VIDEO=True
    """
    global current_tracker_name, track_bbox_norm, track_ok

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS_FALLBACK
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    init_frame_idx = max(0, min(init_frame_idx, max(0, total_frames - 1)))

    tracker = create_tracker(tracker_name)
    inited = False
    init_ms = 0.0

    processed_updates = 0
    ok_frames = 0
    lost_frames = 0
    sum_update_ms = 0.0
    max_update_ms = 0.0

    with lock:
        current_tracker_name = tracker_name
        track_bbox_norm = None
        track_ok = False

    writer = None
    out_video_path = ""
    width = height = None

    idx = 0
    init_size = None  # (h,w)

    while not stop_all.is_set():
        ok_read, frame = cap.read()
        if not ok_read:
            break

        if width is None:
            height, width = frame.shape[:2]
            # open writer once we know size
            if SAVE_OUTPUT_VIDEO:
                out_video_path = os.path.join(RESULTS_DIR, f"{tracker_name}{OUTPUT_VIDEO_EXT}")
                writer = make_video_writer(out_video_path, fps, width, height)
                if writer is None:
                    log(f"[WARN] Could not open VideoWriter for {out_video_path}. "
                        f"Try OUTPUT_VIDEO_EXT='.avi' or codec change.")
                else:
                    log(f"[VIDEO] Recording overlay video -> {out_video_path}")

        # πριν το init: απλά “παίζει” από την αρχή
        if idx < init_frame_idx:
            composed = publish_frame(frame, idx, fps, tracker_name, None, False)
            if writer is not None:
                writer.write(composed)
            idx += 1
            continue

        # στο frame init: init(bbox)
        if (idx == init_frame_idx) and (not inited):
            h, w = frame.shape[:2]
            init_size = (h, w)
            init_bbox_px = norm_to_px(sel_norm, w, h)

            t0 = time.perf_counter()
            tracker.init(frame, init_bbox_px)
            init_ms = (time.perf_counter() - t0) * 1000.0

            inited = True
            with lock:
                track_bbox_norm = sel_norm
                track_ok = True

            composed = publish_frame(frame, idx, fps, tracker_name, sel_norm, True)
            if writer is not None:
                writer.write(composed)
            idx += 1
            continue

        # μετά το init: update tracking
        if inited:
            h, w = init_size if init_size else frame.shape[:2]

            t0 = time.perf_counter()
            ok_tr, bbox = tracker.update(frame)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            processed_updates += 1
            sum_update_ms += dt_ms
            max_update_ms = max(max_update_ms, dt_ms)

            tr_norm = None
            if ok_tr:
                x, y, bw, bh = bbox
                x = int(max(0, x)); y = int(max(0, y))
                bw = int(max(2, bw)); bh = int(max(2, bh))
                tr_norm = px_to_norm(x, y, bw, bh, w, h)
                ok_frames += 1
            else:
                lost_frames += 1

            with lock:
                track_bbox_norm = tr_norm
                track_ok = bool(ok_tr)

            composed = publish_frame(frame, idx, fps, tracker_name, tr_norm, bool(ok_tr))
            if writer is not None:
                writer.write(composed)

            idx += 1
            continue

        idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        log(f"[VIDEO] Saved overlay video: {out_video_path}")

    avg_update_ms = (sum_update_ms / processed_updates) if processed_updates > 0 else 0.0
    approx_fps = (1000.0 / avg_update_ms) if avg_update_ms > 0 else 0.0

    return {
        "tracker": tracker_name,
        "video_path": VIDEO_PATH,
        "init_frame_idx": init_frame_idx,
        "total_video_frames": total_frames,
        "init_ms": f"{init_ms:.3f}",
        "processed_update_frames": processed_updates,
        "ok_frames": ok_frames,
        "lost_frames": lost_frames,
        "avg_update_ms": f"{avg_update_ms:.3f}",
        "max_update_ms": f"{max_update_ms:.3f}",
        "approx_fps": f"{approx_fps:.2f}",
        "overlay_video_path": out_video_path if out_video_path else "",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def benchmark_sequence_loop():
    global current_tracker_name, track_bbox_norm, track_ok

    ensure_results_dir()

    # 1) δείξε video μέχρι να έρθει bbox
    idle_stream_loop()
    if stop_all.is_set():
        return

    with lock:
        sel = selection_bbox_norm
        start_idx = selection_frame_idx

    if sel is None or start_idx is None:
        log("[ERROR] selection_ready but missing selection data.")
        stop_all.set()
        return

    log(f"[BENCH] Sequential benchmark will init at frame_idx={start_idx} for ALL trackers.")
    all_results: List[Dict[str, Any]] = []

    for i, tname in enumerate(TRACKER_ORDER, start=1):
        if stop_all.is_set():
            break

        log(f"[TRACKER {i}/{len(TRACKER_ORDER)}] START -> {tname}")

        try:
            res = run_single_tracker_from_start(tname, start_idx, sel)
        except Exception as e:
            log(f"[TRACKER {tname}] ERROR: {e}")
            res = {"tracker": tname, "error": str(e), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

        out_path = os.path.join(RESULTS_DIR, f"{tname}.txt")
        write_txt(out_path, res)
        log(f"[TRACKER {i}/{len(TRACKER_ORDER)}] END -> {tname} | results saved: {out_path}")

        all_results.append(res)

        with lock:
            track_bbox_norm = None
            track_ok = False
            current_tracker_name = ""

        time.sleep(0.2)

    combined_path = os.path.join(RESULTS_DIR, "ALL_RESULTS.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        for res in all_results:
            f.write("========================================\n")
            f.write(f"TRACKER: {res.get('tracker')}\n")
            for k, v in res.items():
                if k == "tracker":
                    continue
                f.write(f"{k}: {v}\n")
            f.write("\n")

    log(f"[BENCH] ALL trackers finished. Combined results saved: {combined_path}")
    log("[EXIT] Auto-closing Python now...")

    stop_all.set()
    time.sleep(0.5)
    os._exit(0)


def run_flask():
    app.run(host=HOST, port=PORT, threaded=True, debug=False, use_reloader=False)


def main():
    if not os.path.exists(VIDEO_PATH):
        log(f"[ERROR] VIDEO_PATH not found: {VIDEO_PATH}")
        sys.exit(1)

    log(f"Video: {VIDEO_PATH}")
    log(f"Server: http://{HOST}:{PORT}")
    log("Waiting for Android bbox selection (POST /bbox)...")

    t_flask = threading.Thread(target=run_flask, daemon=True)
    t_flask.start()

    t_prev = threading.Thread(target=laptop_preview_loop, daemon=True)
    t_prev.start()

    t_bench = threading.Thread(target=benchmark_sequence_loop, daemon=False)
    t_bench.start()
    t_bench.join()


if __name__ == "__main__":
    main()
