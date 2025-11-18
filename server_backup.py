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
VIDEO_PATH = r"D:\Projects\Thesis\LiveFeed\test_720.mp4"  # or set to 0 for webcam

# MAVProxy must output to this port (e.g. in MAVProxy: `output add 127.0.0.1:14552`)
UDP_IN = 'udpin:0.0.0.0:14550'   # keep QGC on 14550

# Flight profile
LAUNCH_ALT = 1000        # start high
INITIAL_RANGE_M = 3000   # assume ~3 km to target at first lock
IMPACT_RADIUS_M = 15
UPDATE_DT = 0.5

# Camera FOV for bbox->bearing
CAMERA_HFOV_DEG = 60
CAMERA_VFOV_DEG = 35
# ===================================================

# Shared state
tracker = None
bbox = None
lock = threading.Lock()
current_frame = None

# MAV state
mav = None
launch_once_lock = threading.Lock()
has_launched = False
guidance_running = False

# ---------------- MAVLINK HELPERS ----------------
def mav_connect():
    global mav
    if mav is not None:
        return mav
    print("[MAVLINK] Connecting:", UDP_IN)
    mav = mavutil.mavlink_connection(UDP_IN)
    mav.wait_heartbeat(timeout=30)
    print(f"[MAVLINK] Heartbeat sys={mav.target_system} comp={mav.target_component}")
    return mav

def get_global_position(timeout=1.0):
    msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=timeout)
    if not msg:
        return None
    return (msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000.0)

def get_groundspeed(timeout=0.5):
    hud = mav.recv_match(type='VFR_HUD', blocking=True, timeout=timeout)
    return float(getattr(hud, 'groundspeed', 0.0)) if hud else 0.0

def set_mode(mode_name):
    modes = mav.mode_mapping()
    if mode_name not in modes:
        raise RuntimeError(f"Mode {mode_name} not available. Modes: {list(modes.keys())}")
    mode_id = modes[mode_name]
    mav.mav.set_mode_send(mav.target_system,
                          mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                          mode_id)
    t0 = time.time()
    while time.time() - t0 < 10:
        hb = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if hb and hb.custom_mode == mode_id:
            print(f"[MAVLINK] Mode -> {mode_name}")
            return
    raise TimeoutError(f"Timeout waiting for {mode_name}")

def set_takeoff_params():
    """Make AUTO takeoff roll immediately in SITL."""
    def norm_param_id(pid):
        if isinstance(pid, bytes):
            pid = pid.decode(errors='ignore')
        return pid.replace('\x00', '')

    def p(name, value):
        mav.mav.param_set_send(
            mav.target_system, mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1,
            name.encode('utf-8'), float(value), mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )
        t0 = time.time()
        while time.time() - t0 < 2.0:
            msg = mav.recv_match(type='PARAM_VALUE', blocking=False)
            if not msg:
                time.sleep(0.05); continue
            if norm_param_id(msg.param_id) == name:
                break

    p('ARSPD_USE',        0)
    p('TKOFF_THR_MAX',   100)
    p('TKOFF_THR_MINACC', 0)
    p('TKOFF_THR_DELAY',  0)
    p('TKOFF_ROTATE_SPD', 15)
    p('TKOFF_TDRAG_ELEV', 0)
    p('TERRAIN_ENABLE',   0)

def meters_to_latlon(lat, lon, north_m, east_m):
    R = 6378137.0
    dlat = north_m / R
    dlon = east_m / (R * math.cos(math.radians(lat)))
    return lat + (dlat * 180 / math.pi), lon + (dlon * 180 / math.pi)

def send_guided_waypoint(lat, lon, alt_rel):
    """GUIDED point (ArduPlane: mission item with current=2)."""
    mav.mav.mission_item_int_send(
        mav.target_system,
        mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1,
        0,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        2,  # current=2 => guided
        0,
        0,0,0,0,
        int(lat * 1e7),
        int(lon * 1e7),
        float(alt_rel)
    )

def mission_upload_autotakeoff(takeoff_alt=LAUNCH_ALT, climb_ahead_m=200):
    """
    Robust upload:
      0: TAKEOFF (cmd=22) at home, GLOBAL_RELATIVE_ALT(_INT)
      1: WP ahead at same relative alt
    Replies with INT if MISSION_REQUEST_INT, otherwise classic MISSION_ITEM.
    """
    # position & heading
    pos = get_global_position(timeout=5)
    if not pos:
        raise RuntimeError("No GLOBAL_POSITION_INT")
    lat, lon, _ = pos

    hud = mav.recv_match(type='VFR_HUD', blocking=True, timeout=1)
    heading = getattr(hud, 'heading', 0) if hud else 0
    hdg = math.radians(heading)
    north = climb_ahead_m * math.cos(hdg)
    east  = climb_ahead_m * math.sin(hdg)
    wlat, wlon = meters_to_latlon(lat, lon, north, east)

    sysid = mav.target_system
    comp  = mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1

    # Clear & announce
    mav.mav.mission_clear_all_send(sysid, comp)
    time.sleep(0.1)  # tiny pause avoids races on some builds
    mav.mav.mission_count_send(sysid, comp, 2)
    print("[MAVLINK] Sent MISSION_COUNT (2)")

    sent = 0
    while sent < 2:
        req = mav.recv_match(type=['MISSION_REQUEST_INT','MISSION_REQUEST'],
                             blocking=True, timeout=5)
        if not req:
            raise TimeoutError("MISSION_REQUEST timeout")

        seq = req.seq

        if req.get_type() == 'MISSION_REQUEST_INT':
            frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
            if seq == 0:  # TAKEOFF (22)
                mav.mav.mission_item_int_send(
                    sysid, comp, 0, frame,
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    1, 1, 0,0,0,0, int(lat*1e7), int(lon*1e7), float(takeoff_alt)
                )
                print("[MAVLINK] -> TAKEOFF INT")
            elif seq == 1:  # WP ahead (16)
                mav.mav.mission_item_int_send(
                    sysid, comp, 1, frame,
                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                    0, 1, 0,0,0,0, int(wlat*1e7), int(wlon*1e7), float(takeoff_alt)
                )
                print("[MAVLINK] -> WP ahead INT")
        else:  # classic MISSION_REQUEST
            frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
            if seq == 0:
                mav.mav.mission_item_send(
                    sysid, comp, 0, frame,
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    1, 1, 0,0,0,0, float(lat), float(lon), float(takeoff_alt)
                )
                print("[MAVLINK] -> TAKEOFF (classic)")
            elif seq == 1:
                mav.mav.mission_item_send(
                    sysid, comp, 1, frame,
                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                    0, 1, 0,0,0,0, float(wlat), float(wlon), float(takeoff_alt)
                )
                print("[MAVLINK] -> WP ahead (classic)")
        sent += 1

    ack = mav.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
    if not ack or getattr(ack, 'type', None) != mavutil.mavlink.MAV_MISSION_ACCEPTED:
        raise RuntimeError(f"Mission ACK not accepted: {getattr(ack,'type',None)}")
    print("[MAVLINK] Mission ACK: ACCEPTED")




def wait_until_alt(target_alt):
    t0 = time.time()
    while time.time() - t0 < 120:
        pos = get_global_position(timeout=2)
        alt = (pos[2] if pos else 0.0)
        print(f"    --> Alt: {alt:.1f} m")
        if alt >= target_alt * 0.95:
            print("[MAVLINK] Launch altitude reached.")
            return
    raise TimeoutError("Timeout reaching launch altitude")

# ------------- GUIDANCE FROM BBOX -------------
def bbox_center(b):
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)

def bbox_to_angles(b, frame_w, frame_h):
    cx, cy = bbox_center(b)
    dx = (cx - frame_w / 2.0) / frame_w
    dy = (cy - frame_h / 2.0) / frame_h
    h_angle = math.radians(dx * CAMERA_HFOV_DEG)
    v_angle = math.radians(dy * CAMERA_VFOV_DEG)
    return h_angle, v_angle

def guidance_loop():
    """GUIDED pursuit with descending altitude: 1000 m -> 0 m as range closes."""
    global guidance_running
    if guidance_running:
        return
    guidance_running = True0

    try:
        set_mode("GUIDED")
        print("[GUIDANCE] GUIDED mode set. Starting pursuit with descending profile…")

        forward_range = float(INITIAL_RANGE_M)

        while True:
            if tracker is None or bbox is None or current_frame is None:
                time.sleep(0.05); continue

            frame = current_frame
            fh, fw = frame.shape[:2]
            b = bbox

            # horizontal angle from bbox
            h_ang, _ = bbox_to_angles(b, fw, fh)

            lateral = forward_range * math.tan(h_ang)
            forward = max(50.0, forward_range)

            pos = get_global_position(timeout=1)
            if not pos:
                time.sleep(0.05); continue
            lat, lon, _ = pos

            hud = mav.recv_match(type='VFR_HUD', blocking=True, timeout=0.2)
            heading = getattr(hud, 'heading', 0) if hud else 0
            hdg = math.radians(heading)

            # rotate forward/lateral into N/E
            north =  forward * math.cos(hdg) - lateral * math.sin(hdg)
            east  =  forward * math.sin(hdg) + lateral * math.cos(hdg)

            # descend linearly with range: 3km→1000m, 0km→0m
            alt_cmd = max(0.0, LAUNCH_ALT * (forward_range / max(1.0, INITIAL_RANGE_M)))

            tgt_lat, tgt_lon = meters_to_latlon(lat, lon, north, east)
            send_guided_waypoint(tgt_lat, tgt_lon, alt_cmd)

            gs = get_groundspeed(timeout=0.2)   # m/s
            forward_range = max(0.0, forward_range - gs * UPDATE_DT)

            if forward_range < 10.0 and abs(lateral) < IMPACT_RADIUS_M and alt_cmd <= 5.0:
                print("💥 IMPACT (simulated at sea level).")
                break

            time.sleep(UPDATE_DT)

        print("[GUIDANCE] Pursuit ended.")

    except Exception as e:
        print(f"[GUIDANCE ERROR] {e}")
    finally:
        guidance_running = False

# ------------- LAUNCH SEQUENCE -------------
def auto_launch_once():
    """Upload TAKEOFF mission, set params, start AUTO at WP0, force mission start + throttle kick, wait to 1000 m, then GUIDED pursuit."""
    global has_launched
    with launch_once_lock:
        if has_launched:
            print("[LAUNCH] Already launched; skipping."); return
        has_launched = True

    try:
        mav_connect()
        mission_upload_autotakeoff(LAUNCH_ALT, climb_ahead_m=200)
        set_takeoff_params()

        # Start from TAKEOFF (WP0), AUTO
        mav.mav.mission_set_current_send(mav.target_system, mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1, 0)
        set_mode("AUTO")

        # ARM
        mav.mav.command_long_send(
            mav.target_system, mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0,0,0,0,0,0
        )
        print("[MAVLINK] Armed (AUTO + current=0)")

        # ✅ Force mission start (some builds need this)
        mav.mav.command_long_send(
            mav.target_system, mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1,
            mavutil.mavlink.MAV_CMD_MISSION_START,
            0, 0,0,0,0,0,0,0
        )
        print("[MAVLINK] Mission start sent")

        # ✅ Throttle kick via RC override (channel 3). 1600–1800 works in SITL.
        mav.mav.rc_channels_override_send(
            mav.target_system, mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1,
            0, 0, 1700, 0, 0, 0, 0, 0
        )
        # Let it spool up for a bit, then release override
        time.sleep(3.0)
        mav.mav.rc_channels_override_send(
            mav.target_system, mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1,
            0, 0, 0, 0, 0, 0, 0, 0
        )

        wait_until_alt(LAUNCH_ALT)

        # Switch to GUIDED and start pursuit loop
        threading.Thread(target=guidance_loop, daemon=True).start()

    except Exception as e:
        print(f"[LAUNCH ERROR] {e}")
        with launch_once_lock:
            has_launched = False

# ---------------- VIDEO THREAD ----------------
def capture_loop():
    global current_frame, tracker, bbox
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

        # Update tracker
        if tracker is not None and bbox is not None:
            try:
                ok, new_box = tracker.update(frame)
                if ok:
                    x, y, w, h = [int(v) for v in new_box]
                    bbox = (x, y, w, h)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    print("[INFO] Tracking lost")
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
    """Init tracker from Android bbox, then auto-launch once."""
    global bbox, tracker, current_frame

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
        print(f"[INFO] Tracker initialized at {bbox}")
        #threading.Thread(target=auto_launch_once, daemon=True).start()
        return "Bounding box received; launch & guidance started", 200
    except Exception as e:
        print(f"[ERROR] Tracker init error: {e}")
        tracker = None; bbox = None
        return "Tracker init error", 500

# ---------------- MAIN ----------------
if __name__ == '__main__':
    threading.Thread(target=capture_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
