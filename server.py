import cv2
from flask import Flask, Response, request
import threading
import os

app = Flask(__name__)

USE_VIDEO = True  # 🔁 Set to False for webcam
<<<<<<< HEAD
VIDEO_PATH = r"D:\Projects\Thesis\LiveFeed\test_720.mp4"  # 🔁 Replace with your video path
=======
VIDEO_PATH = r"D:\Projects\Thesis\LiveFeed\test_720.mp4"
>>>>>>> 805e3c97518865e116741a80c917e5803e86adfe
#cap = cv2.VideoCapture(0)

# Shared variables
tracker = None
bbox = None
lock = threading.Lock()
current_frame = None

# Background thread: reads frames and shows OpenCV window
def capture_loop():
    global current_frame, tracker, bbox

    cap = cv2.VideoCapture(VIDEO_PATH if USE_VIDEO else 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 33  # milliseconds

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # or 1920
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # or 1080

    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        os._exit(0)
        #return

    while True:
        ret, frame = cap.read()
        # Loop video if at end
        if not ret:
            if USE_VIDEO:
                print("[INFO] Streaming video in the loop...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[ERROR] Failed to read frame from camera")
                os._exit(0)
                #continue


        #current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if tracker is not None and bbox is not None:
            try:
                ok, new_box = tracker.update(frame)
                if ok:
                    (x, y, w, h) = [int(v) for v in new_box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    print("[INFO] Tracking failure detected")
            except Exception as e:
                print(f"[ERROR] Tracker update failed: {e}")
                tracker = None
                bbox = None

        current_frame = frame.copy()

        cv2.imshow("Live Feed", frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        key = cv2.waitKey(delay) & 0xFF  # ← match video FPS here
        if key == ord('q') or cv2.getWindowProperty("Live Feed", cv2.WND_PROP_VISIBLE) < 1:
            break

        # If 'q' key is pressed or window is closed, break loop
        if key == ord('q') or cv2.getWindowProperty("Live Feed", cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Exiting: Window closed or 'q' pressed")
            break
    cap.release()
    cv2.destroyAllWindows()

    # Exit the Flask server by terminating the whole process
    os._exit(0)



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
    global bbox, tracker

    data = request.get_json()
    norm_x = float(data['x'])
    norm_y = float(data['y'])
    norm_w = float(data['width'])
    norm_h = float(data['height'])

    print(f"[DEBUG] Normalized bbox: {norm_x}, {norm_y}, {norm_w}, {norm_h}")

    if current_frame is None:
        return "No frame available", 500

    frame_h, frame_w = current_frame.shape[:2]

    x = int(norm_x * frame_w)
    y = int(norm_y * frame_h)
    w = int(norm_w * frame_w)
    h = int(norm_h * frame_h)

    # Clamp
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))

    bbox = (x, y, w, h)

    try:
        #tracker = cv2.TrackerKCF_create()  #Use KCF tracker
        #tracker = cv2.TrackerCSRT_create()
        #tracker = cv2.legacy.TrackerMOSSE_create()
        #tracker = cv2.legacy.TrackerMIL_create()
        #tracker = cv2.legacy.TrackerTLD_create()
        tracker = cv2.legacy.TrackerMedianFlow_create()
        tracker.init(current_frame, bbox)
        print(f"[INFO] Tracker initialized at {bbox}")
        return "Bounding box received", 200
    except Exception as e:
        print(f"[ERROR] Tracker init error: {e}")
        tracker = None
        bbox = None
        return "Tracker init error", 500





if __name__ == '__main__':
    # Start the background capture thread
    threading.Thread(target=capture_loop, daemon=True).start()

    # Start the Flask server
    app.run(host='0.0.0.0', port=5000)
