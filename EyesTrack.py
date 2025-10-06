import os
import time
import math
import queue
import csv
import threading
import collections
import json
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import mediapipe as mp
import winsound as _winsound
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# try TFLite Interpreter (prefer tensorflow if present)
try:
    Interpreter = tf.lite.Interpreter
except Exception:
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        Interpreter = None

# winsound optional (Windows)
try:
    import winsound
except Exception:
    winsound = None

# Default config (close to your Step3)
cfg = {
    "EYE_AR_THRESHOLD": 0.25,
    "MOUTH_AR_THRESHOLD": 0.55,
    "HAND_TOUCH_MARGIN_RATIO": 0.08,
    "HAND_TOUCH_DURATION": 0.5,
    "DISTRACT_CENTER_THRESHOLD_RATIO": 0.28,
    "DISTRACT_DURATION": 1.0,
    "EYE_CLOSED_DURATION": 2.0,
    "MOUTH_OPEN_DURATION": 1.5,
    "CONFIRM_TIME": 1.0,
    "GUI_FPS": 12,
    "PROCESS_RESIZE_W": 320,
    "PROCESS_RESIZE_H": 240,
    "VIDEO_W": 640,
    "VIDEO_H": 480,
    "SMOOTHING_WINDOW": 5,
    "REPEAT_BEEP": False,
    "REPEAT_BEEP_INTERVAL": 3.0,
    "ENABLE_BEEP": True,
    "MODEL_WEIGHT": 0.3,
    "SKIP_FRAMES": 1,
    "FRAMES": 8
}

# MediaPipe init
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(static_image_mode=False,
                                max_num_hands=2,
                                min_detection_confidence=0.6,
                                min_tracking_confidence=0.5)

face_detector = mp_face_mesh.FaceMesh(max_num_faces=1,
                                     refine_landmarks=True,
                                     min_detection_confidence=0.6,
                                     min_tracking_confidence=0.5)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14
LEFT_MOUTH_IDX = 78
RIGHT_MOUTH_IDX = 308

# threading / queues / globals
gui_queue = queue.Queue(maxsize=6)
stop_event = threading.Event()
worker_thread = None
interpreter = None
interpreter_lock = threading.Lock()
input_details = None
output_details = None

# recording / logging
video_writer = None
recording = False
record_lock = threading.Lock()
log_file_path = None
log_lock = threading.Lock()

# history
prob_history = {"Focus": [], "Distracted": [], "Fatigued": []}

# frame buffer for model
frame_buffer = collections.deque(maxlen=cfg["FRAMES"])

# calibration baseline for face center
calibration = {"baseline_offset": 0.0, "calibrated": False}

# Step 9: Dynamic thresholds and calibration reset
dynamic_thresholds = {
    "EAR_FATIGUE": cfg["EYE_AR_THRESHOLD"],
    "MAR_YAWN": cfg["MOUTH_AR_THRESHOLD"],
    "DISTRACT_CENTER_RATIO": cfg["DISTRACT_CENTER_THRESHOLD_RATIO"],
    "DISTRACTED_FRAMES": int(max(1, cfg.get("DISTRACT_DURATION", 1.0) * cfg.get("GUI_FPS", 12)))
}
threshold_lock = threading.Lock()

def reset_calibration():
    """Reset calibration baseline_offset and mark uncalibrated."""
    global calibration
    with threshold_lock:
        calibration = {"baseline_offset": 0.0, "calibrated": False}
    print("[INFO] Calibration has been reset.")

# Step 11: Save / Load settings (profiles)
settings_lock = threading.Lock()
settings_file_path = None

def save_settings(path=None):
    """Save dynamic thresholds, calibration and a few cfg items to JSON."""
    global settings_file_path
    if path is None:
        return False
    with settings_lock:
        payload = {
            "dynamic_thresholds": dynamic_thresholds,
            "calibration": calibration,
            "cfg_subset": {
                "MODEL_WEIGHT": cfg.get("MODEL_WEIGHT"),
                "SKIP_FRAMES": cfg.get("SKIP_FRAMES"),
                "EYE_AR_THRESHOLD": cfg.get("EYE_AR_THRESHOLD"),
                "MOUTH_AR_THRESHOLD": cfg.get("MOUTH_AR_THRESHOLD"),
                "DISTRACT_CENTER_THRESHOLD_RATIO": cfg.get("DISTRACT_CENTER_THRESHOLD_RATIO")
            },
            "meta": {"saved_at": time.time()}
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            settings_file_path = path
            print(f"[Settings] Saved to {path}")
            return True
        except Exception as e:
            print("Save settings error:", e)
            return False

def load_settings(path):
    """Load settings from JSON and apply to globals. Returns dict or None."""
    global settings_file_path, dynamic_thresholds, calibration
    if not path or not os.path.exists(path):
        return None
    with settings_lock:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            # apply
            if "dynamic_thresholds" in payload:
                dynamic_thresholds.update(payload["dynamic_thresholds"])
            if "calibration" in payload:
                calibration.update(payload["calibration"])
            if "cfg_subset" in payload:
                cs = payload["cfg_subset"]
                for k in ["MODEL_WEIGHT", "SKIP_FRAMES", "EYE_AR_THRESHOLD", "MOUTH_AR_THRESHOLD", "DISTRACT_CENTER_THRESHOLD_RATIO"]:
                    if k in cs:
                        cfg[k] = cs[k]
            settings_file_path = path
            print(f"[Settings] Loaded from {path}")
            return payload
        except Exception as e:
            print("Load settings error:", e)
            return None

# continuous fatigue beep machinery
fatigue_beep_event = threading.Event()
fatigue_beep_thread = None
fatigue_beep_lock = threading.Lock()

def _fatigue_beep_loop(freq=1200, on_ms=400, off_ms=100):
    """Loop in daemon thread while event is set."""
    try:
        has_winsound = True
    except Exception:
        has_winsound = False
    while fatigue_beep_event.is_set():
        if has_winsound:
            try:
                _winsound.Beep(freq, on_ms)
            except Exception:
                time.sleep(on_ms / 1000.0)
        else:
            # fallback audible bell (may not work everywhere)
            print("\a", end="", flush=True)
            time.sleep(on_ms / 1000.0)
        time.sleep(off_ms / 1000.0)

def start_fatigue_beep():
    """Start continuous beep safely."""
    global fatigue_beep_thread
    with fatigue_beep_lock:
        fatigue_beep_event.set()
        if fatigue_beep_thread is None or not fatigue_beep_thread.is_alive():
            fatigue_beep_thread = threading.Thread(target=_fatigue_beep_loop, daemon=True)
            fatigue_beep_thread.start()

def stop_fatigue_beep():
    """Stop continuous beep."""
    fatigue_beep_event.clear()
    # thread will exit by itself

# beep once helper
def beep_once_for(status):
    if not cfg.get("ENABLE_BEEP", True):
        return
    if winsound is not None:
        try:
            if status == 'Distracted':
                winsound.Beep(800, 220)
            elif status == 'Fatigued':
                winsound.Beep(1200, 420)
            else:
                winsound.Beep(1000, 150)
        except Exception:
            print("BEEP once:", status)
    else:
        print("BEEP once:", status)

# helper clamp
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

# EAR/MAR helpers
def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    coords = [(landmarks[i].x * image_w, landmarks[i].y * image_h) for i in eye_indices]
    A = math.dist(coords[1], coords[5])
    B = math.dist(coords[2], coords[4])
    C = math.dist(coords[0], coords[3])
    ear = (A + B) / (2.0 * C) if C != 0 else 0.0
    return ear

def mouth_aspect_ratio(landmarks, image_w, image_h):
    p_up = (landmarks[UPPER_LIP_IDX].x * image_w, landmarks[UPPER_LIP_IDX].y * image_h)
    p_down = (landmarks[LOWER_LIP_IDX].x * image_w, landmarks[LOWER_LIP_IDX].y * image_h)
    vertical = math.dist(p_up, p_down)
    p_left = (landmarks[LEFT_MOUTH_IDX].x * image_w, landmarks[LEFT_MOUTH_IDX].y * image_h)
    p_right = (landmarks[RIGHT_MOUTH_IDX].x * image_w, landmarks[RIGHT_MOUTH_IDX].y * image_h)
    horizontal = math.dist(p_left, p_right)
    mar = (vertical / horizontal) if horizontal != 0 else 0.0
    return mar

# TFLite helpers (load/run)
def load_tflite_model(path):
    global interpreter, input_details, output_details
    if Interpreter is None:
        messagebox.showerror("TFLite", "No TFLite interpreter available. Install tensorflow or tflite-runtime.")
        return False
    with interpreter_lock:
        try:
            interpreter = Interpreter(model_path=path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print("Loaded TFLite model:", path)
            return True
        except Exception as e:
            interpreter = None
            print("TFLite load error:", e)
            messagebox.showerror("TFLite load error", str(e))
            return False

def preprocess_for_tflite(frames, inp_shape, inp_dtype):
    arr = np.array(frames)
    # determine target size
    if len(inp_shape) == 5:
        target_h = int(inp_shape[-3]); target_w = int(inp_shape[-2])
    elif len(inp_shape) == 4:
        target_h = int(inp_shape[-3]); target_w = int(inp_shape[-2])
    else:
        target_h = cfg["PROCESS_RESIZE_H"]; target_w = cfg["PROCESS_RESIZE_W"]

    resized = []
    for f in arr:
        im = cv2.resize(f, (target_w, target_h))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        resized.append(im)
    resized = np.array(resized).astype(np.float32) / 255.0

    if len(inp_shape) == 5:
        batch = np.expand_dims(resized, axis=0)
        expected = int(inp_shape[1])
        if batch.shape[1] < expected:
            pad = np.repeat(batch[:, -1:, :, :, :], expected - batch.shape[1], axis=1)
            batch = np.concatenate([batch, pad], axis=1)
        elif batch.shape[1] > expected:
            batch = batch[:, -expected:, :, :, :]
        if np.issubdtype(inp_dtype, np.uint8):
            batch = (batch * 255.0).astype(np.uint8)
        else:
            batch = batch.astype(np.float32)
        return batch
    elif len(inp_shape) == 4:
        single = resized[-1]
        batch = np.expand_dims(single, axis=0)
        if np.issubdtype(inp_dtype, np.uint8):
            batch = (batch * 255.0).astype(np.uint8)
        else:
            batch = batch.astype(np.float32)
        return batch
    else:
        raise ValueError("Unsupported input shape for tflite")

def run_tflite(frames):
    with interpreter_lock:
        if interpreter is None:
            return None
        try:
            inp = interpreter.get_input_details()[0]
            out = interpreter.get_output_details()[0]
            batch = preprocess_for_tflite(frames, inp['shape'], inp['dtype'])
            interpreter.set_tensor(inp['index'], batch)
            interpreter.invoke()
            preds = interpreter.get_tensor(out['index'])
            probs = np.squeeze(preds).astype(np.float32)
            s = probs.sum()
            if s > 0:
                probs = probs / s
            else:
                probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            return probs[:3] if probs.shape[0] >= 3 else None
        except Exception as e:
            print("TFLite inference error:", e)
            return None

# heuristic (improved)
def compute_heuristic(ear, mar, hand_touching, face_cx, frame_w, baseline_offset=0.0):
    fatigue_raw = 0.0
    fatigue_raw += clamp((cfg["EYE_AR_THRESHOLD"] - ear) / max(cfg["EYE_AR_THRESHOLD"], 1e-6))
    fatigue_raw += clamp((mar - cfg["MOUTH_AR_THRESHOLD"]) / max(cfg["MOUTH_AR_THRESHOLD"], 1e-6))
    fatigue_raw += (1.0 if hand_touching else 0.0)
    fatigue_score = clamp(fatigue_raw / 3.0)

    if face_cx is None:
        distraction_score = 1.0
    else:
        off = abs(face_cx - (frame_w / 2.0)) / (frame_w / 2.0)
        adj = max(0.0, off - baseline_offset)
        distraction_score = clamp(adj / max(cfg["DISTRACT_CENTER_THRESHOLD_RATIO"] * 1.2, 1e-3))

    focus_score = clamp(1.0 - max(fatigue_score, distraction_score))
    probs = np.array([focus_score, distraction_score, fatigue_score], dtype=np.float32)
    s = probs.sum()
    if s > 0:
        probs = probs / s
    else:
        probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return probs

# Step 10: Auto-tune thresholds (online)
auto_tune_enabled = False
auto_tune_thread = None
auto_tune_lock = threading.Lock()

def _auto_tune_loop(window_sec=10, interval=5):
    """Monitor running EAR/MAR averages and adjust dynamic_thresholds conservatively."""
    global auto_tune_thread
    # local circular buffers
    ear_buf = collections.deque(maxlen=int(window_sec * cfg["GUI_FPS"]))
    mar_buf = collections.deque(maxlen=int(window_sec * cfg["GUI_FPS"]))
    while auto_tune_enabled:
        # populate from latest frames using face detector pass (cheap approximation):
        # we will sample some frames from frame_buffer
        samples = list(frame_buffer)[-int(cfg["GUI_FPS"] * min(window_sec, 3)):]  # take up to last 3s
        local_ears = []
        local_mars = []
        for f in samples:
            try:
                proc = cv2.resize(f, (cfg["PROCESS_RESIZE_W"], cfg["PROCESS_RESIZE_H"]))
                image_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
                res = face_detector.process(image_rgb)
                if not res.multi_face_landmarks:
                    continue
                lm = res.multi_face_landmarks[0]
                w_proc = cfg["PROCESS_RESIZE_W"]; h_proc = cfg["PROCESS_RESIZE_H"]
                le = eye_aspect_ratio(lm.landmark, LEFT_EYE_IDX, w_proc, h_proc)
                re = eye_aspect_ratio(lm.landmark, RIGHT_EYE_IDX, w_proc, h_proc)
                local_ears.append((le + re) / 2.0)
                local_mars.append(mouth_aspect_ratio(lm.landmark, w_proc, h_proc))
            except Exception:
                continue
        if len(local_ears) > 0:
            mean_ear = float(np.mean(local_ears))
            mean_mar = float(np.mean(local_mars)) if len(local_mars)>0 else dynamic_thresholds["MAR_YAWN"]
            with threshold_lock:
                # conservative update: shift threshold slightly toward measured mean with floor/ceiling
                old_ear = dynamic_thresholds["EAR_FATIGUE"]
                new_ear = max(0.12, min(old_ear * 0.98 + mean_ear * 0.02, 0.4))
                dynamic_thresholds["EAR_FATIGUE"] = new_ear
                old_mar = dynamic_thresholds["MAR_YAWN"]
                new_mar = max(0.3, min(old_mar * 0.98 + mean_mar * 0.02, 1.0))
                dynamic_thresholds["MAR_YAWN"] = new_mar
            # Debug print
            print(f"[AutoTune] mean_ear={mean_ear:.3f} mean_mar={mean_mar:.3f} -> EARthr={new_ear:.3f} MARthr={new_mar:.3f}")
        time.sleep(interval)

def start_auto_tune():
    global auto_tune_thread, auto_tune_enabled
    with auto_tune_lock:
        if auto_tune_enabled:
            return
        auto_tune_enabled = True
        auto_tune_thread = threading.Thread(target=_auto_tune_loop, daemon=True)
        auto_tune_thread.start()

def stop_auto_tune():
    global auto_tune_enabled
    with auto_tune_lock:
        auto_tune_enabled = False

# detection thread (with timing alerts and continuous beep)
def detection_thread(device_index=0, record_path=None):
    global video_writer, recording
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print("Cannot open camera", device_index)
        try:
            gui_queue.put({"frame": None, "status": "STOP"})
        except:
            pass
        return

    # timers / flags (must exist)
    eye_closed_start = None
    mouth_open_start = None
    hand_touch_start = None
    distract_start = None

    last_reported_status = None
    last_confirm_time = 0.0
    last_beep_time = 0.0

    frame_counter = 0
    skip = max(1, int(cfg.get("SKIP_FRAMES", 1)))

    # initialize writer if record_path passed
    if record_path:
        with record_lock:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vw = cfg["VIDEO_W"]; vh = cfg["VIDEO_H"]
                video_writer = cv2.VideoWriter(record_path, fourcc, 20.0, (vw, vh))
                recording = True
            except Exception as e:
                print("Cannot start video writer:", e)
                video_writer = None
                recording = False

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue
        frame_counter += 1

        # processing size
        proc = cv2.resize(frame, (cfg["PROCESS_RESIZE_W"], cfg["PROCESS_RESIZE_H"]))
        h_proc, w_proc, _ = proc.shape

        # display-sized frame and buffer
        disp_frame = cv2.resize(frame, (cfg["VIDEO_W"], cfg["VIDEO_H"]))
        frame_buffer.append(disp_frame)

        # skip heavy work on some frames
        if frame_counter % skip != 0:
            with record_lock:
                if recording and video_writer is not None:
                    try:
                        video_writer.write(disp_frame)
                    except Exception as e:
                        print("Video writer error:", e)
            try:
                if gui_queue.full():
                    _ = gui_queue.get_nowait()
                gui_queue.put_nowait({"frame": disp_frame, "status": last_reported_status or "Initializing", "probs": np.array([1.0,0.0,0.0])})
            except queue.Full:
                pass
            continue

        image_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        results_face = face_detector.process(image_rgb)
        results_hand = hands_detector.process(image_rgb)

        eyes_closed = False
        mouth_open = False
        hand_touching = False
        face_bbox = None
        ear = 0.0
        mar = 0.0

        # use dynamic thresholds (read under lock)
        with threshold_lock:
            ear_threshold = dynamic_thresholds.get("EAR_FATIGUE", cfg["EYE_AR_THRESHOLD"])
            mar_threshold = dynamic_thresholds.get("MAR_YAWN", cfg["MOUTH_AR_THRESHOLD"])
            distract_center_ratio = dynamic_thresholds.get("DISTRACT_CENTER_RATIO", cfg["DISTRACT_CENTER_THRESHOLD_RATIO"])

        if results_face.multi_face_landmarks:
            face_landmarks = results_face.multi_face_landmarks[0]
            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]
            minx = int(min(xs) * w_proc); maxx = int(max(xs) * w_proc)
            miny = int(min(ys) * h_proc); maxy = int(max(ys) * h_proc)
            face_bbox = (minx, miny, maxx, maxy)

            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_IDX, w_proc, h_proc)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_IDX, w_proc, h_proc)
            ear = (left_ear + right_ear) / 2.0
            # compare to dynamic ear_threshold
            if ear < ear_threshold:
                eyes_closed = True

            mar = mouth_aspect_ratio(face_landmarks.landmark, w_proc, h_proc)
            # compare to dynamic mar_threshold
            if mar > mar_threshold:
                mouth_open = True

        if results_hand.multi_hand_landmarks and face_bbox is not None:
            minx, miny, maxx, maxy = face_bbox
            margin = int(cfg["HAND_TOUCH_MARGIN_RATIO"] * w_proc)
            box_minx, box_miny = minx - margin, miny - margin
            box_maxx, box_maxy = maxx + margin, maxy + margin
            for hand_landmarks in results_hand.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    hx = int(lm.x * w_proc); hy = int(lm.y * h_proc)
                    if box_minx <= hx <= box_maxx and box_miny <= hy <= box_maxy:
                        hand_touching = True
                        break
                if hand_touching:
                    break

        # compute face_cx in display coordinates
        face_cx = None
        if face_bbox is not None:
            minx, miny, maxx, maxy = face_bbox
            face_cx_proc = (minx + maxx) / 2.0
            scale_x = cfg["VIDEO_W"] / float(w_proc)
            face_cx = face_cx_proc * scale_x

        # heuristic probs (pass baseline_offset)
        heuristic_probs = compute_heuristic(ear, mar, hand_touching, face_cx, cfg["VIDEO_W"], calibration.get("baseline_offset", 0.0))

        # TIMING ALERTS (restore logic from Step3)
        now = time.time()
        # hand alert
        hand_alert = False
        if hand_touching:
            if hand_touch_start is None:
                hand_touch_start = now
            elif now - hand_touch_start >= cfg["HAND_TOUCH_DURATION"]:
                hand_alert = True
        else:
            hand_touch_start = None

        # mouth alert
        mouth_alert = False
        if mouth_open:
            if mouth_open_start is None:
                mouth_open_start = now
            elif now - mouth_open_start >= cfg["MOUTH_OPEN_DURATION"]:
                mouth_alert = True
        else:
            mouth_open_start = None

        # eye alert
        eye_alert = False
        if eyes_closed and not (hand_alert or mouth_alert):
            if eye_closed_start is None:
                eye_closed_start = now
            elif now - eye_closed_start >= cfg["EYE_CLOSED_DURATION"]:
                eye_alert = True
        else:
            if not eyes_closed:
                eye_closed_start = None

        # distract timing (use dynamic distract_center_ratio)
        distract_alert = False
        if face_cx is None:
            if distract_start is None:
                distract_start = now
            elif now - distract_start >= cfg["DISTRACT_DURATION"]:
                distract_alert = True
        else:
            off = abs(face_cx - (cfg["VIDEO_W"] / 2.0)) / (cfg["VIDEO_W"] / 2.0)
            if off > distract_center_ratio:
                if distract_start is None:
                    distract_start = now
                elif now - distract_start >= cfg["DISTRACT_DURATION"]:
                    distract_alert = True
            else:
                distract_start = None

        # model inference (if available) using last N frames
        model_probs = None
        with interpreter_lock:
            if interpreter is not None:
                try:
                    frames_for_model = list(frame_buffer)[-cfg["FRAMES"]:] if len(frame_buffer) > 0 else [disp_frame]
                    model_probs = run_tflite(frames_for_model)
                except Exception as e:
                    print("Model inference error:", e)
                    model_probs = None

        # Combine logic but prioritize timing alerts for Fatigued
        if hand_alert or mouth_alert or eye_alert:
            # immediate fatigued
            probs = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            weight = clamp(cfg.get("MODEL_WEIGHT", 0.3), 0.0, 1.0)
            if model_probs is not None:
                probs = weight * model_probs + (1.0 - weight) * heuristic_probs
                s = probs.sum()
                probs = probs / s if s > 0 else heuristic_probs
            else:
                probs = heuristic_probs

            # if distract_alert triggered, force Distracted
            if distract_alert:
                probs = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # HYSTERESIS / CONFIRM
        status_candidate = ["Focus", "Distracted", "Fatigued"][int(np.argmax(probs))]
        confirmed = False
        if status_candidate != last_reported_status:
            if last_confirm_time == 0.0:
                last_confirm_time = now
                confirmed = False
            else:
                if now - last_confirm_time >= cfg["CONFIRM_TIME"]:
                    confirmed = True
                else:
                    confirmed = False
        else:
            confirmed = True
            last_confirm_time = 0.0

        # confirmed handling (with continuous fatigue beep)
        if confirmed:
            now = time.time()
            if status_candidate != last_reported_status:
                # status changed
                last_reported_status = status_candidate

                # short beep for change
                if cfg.get("ENABLE_BEEP", True):
                    beep_once_for(last_reported_status)

                # continuous beep if fatigued
                if last_reported_status == "Fatigued":
                    start_fatigue_beep()
                else:
                    stop_fatigue_beep()

                # log change
                with log_lock:
                    if log_file_path:
                        header_needed = not os.path.exists(log_file_path)
                        with open(log_file_path, "a", newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            if header_needed:
                                writer.writerow(["timestamp", "status", "ear", "mar", "hand_touch", "p_focus", "p_distracted", "p_fatigued"])
                            writer.writerow([now, last_reported_status, f"{ear:.4f}", f"{mar:.4f}", int(hand_touching), f"{probs[0]:.4f}", f"{probs[1]:.4f}", f"{probs[2]:.4f}"])
            else:
                # same status
                if last_reported_status == "Fatigued":
                    # ensure continuous beep is on
                    if not fatigue_beep_event.is_set():
                        start_fatigue_beep()
                else:
                    stop_fatigue_beep()

                # repeat beep (optional)
                if cfg.get("REPEAT_BEEP", False) and cfg.get("ENABLE_BEEP", True):
                    if now - last_beep_time >= cfg.get("REPEAT_BEEP_INTERVAL", 3.0):
                        beep_once_for(last_reported_status)
                        last_beep_time = now

            # prepare frame for GUI
            out_frame = disp_frame.copy()
            color = (0,255,0) if last_reported_status=='Focus' else (0,165,255) if last_reported_status=='Distracted' else (0,0,255)
            cv2.putText(out_frame, f"Status: {last_reported_status}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(out_frame, f"p:{np.round(probs,2)} EAR:{ear:.2f} MAR:{mar:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            try:
                if gui_queue.full():
                    _ = gui_queue.get_nowait()
                gui_queue.put_nowait({"frame": out_frame, "status": last_reported_status, "probs": probs})
            except queue.Full:
                pass

            with record_lock:
                if recording and video_writer is not None:
                    try:
                        video_writer.write(out_frame)
                    except Exception as e:
                        print("Video writer error:", e)

        # small delay
        time.sleep(0.005)

    # cleanup
    cap.release()
    with record_lock:
        if video_writer is not None:
            try:
                video_writer.release()
            except:
                pass
            video_writer = None
            recording = False
    # stop fatigue beep on graceful exit
    stop_fatigue_beep()
    try:
        gui_queue.put_nowait({"frame": None, "status": "STOP"})
    except:
        pass

# Step1: Collect Samples helper (simple capture to npz)
def collect_samples_to_npz(duration_sec=4, out_path=None, device_index=0):
    """Open camera, collect frames for duration_sec, and save as npz (frames array)."""
    if out_path is None:
        out_path = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NPZ","*.npz")])
        if not out_path:
            return None
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        messagebox.showerror("Collect Samples", f"Cannot open camera {device_index}")
        return None
    frames = []
    t0 = time.time()
    while time.time() - t0 < duration_sec:
        ret, f = cap.read()
        if not ret:
            time.sleep(0.02)
            continue
        frames.append(cv2.resize(f, (cfg["PROCESS_RESIZE_W"], cfg["PROCESS_RESIZE_H"])))
        time.sleep(0.04)
    cap.release()
    np.savez(out_path, frames=np.array(frames))
    messagebox.showinfo("Collect Samples", f"Saved {len(frames)} frames to {out_path}")
    return out_path

# GUI (layout like Step 3) + Step1/2/3/10/11 controls
def run_gui():
    global worker_thread, video_writer, recording, log_file_path, interpreter, auto_tune_enabled

    root = tk.Tk()
    root.title("Tracker — Step3 UI + TFLite (fixed) + Steps 1/2/3/10/11")
    root.geometry("1100x720")
    # SCROLLABLE TOP TOOLBAR (REPLACES previous 'top' frame) ==========
    # A horizontal Canvas + horizontal scrollbar that hosts an inner_frame for controls.
    toolbar_canvas = tk.Canvas(root, height=72, highlightthickness=0)
    toolbar_canvas.pack(side=tk.TOP, fill=tk.X, expand=False)

    h_scroll = ttk.Scrollbar(root, orient=tk.HORIZONTAL, command=toolbar_canvas.xview)
    h_scroll.pack(side=tk.TOP, fill=tk.X)

    toolbar_canvas.configure(xscrollcommand=h_scroll.set)

    inner_frame = ttk.Frame(toolbar_canvas)
    inner_window = toolbar_canvas.create_window((0, 0), window=inner_frame, anchor='nw')

    def _on_inner_config(event):
        # update scroll region when inner_frame size changes
        toolbar_canvas.configure(scrollregion=toolbar_canvas.bbox("all"))
    inner_frame.bind("<Configure>", _on_inner_config)

    def _on_canvas_config(event):
        # optional: adjust inner window size if you want fixed height behavior
        pass
    toolbar_canvas.bind("<Configure>", _on_canvas_config)

    # drag-to-scroll (click+drag) and shift+wheel horizontal scrolling
    def _start_drag(event):
        toolbar_canvas.scan_mark(event.x, event.y)
    def _drag_scroll(event):
        toolbar_canvas.scan_dragto(event.x, event.y, gain=1)
    toolbar_canvas.bind("<ButtonPress-1>", _start_drag)
    toolbar_canvas.bind("<B1-Motion>", _drag_scroll)

    def _on_mousewheel(event):
        # support Shift+Wheel for horizontal scroll in many platforms
        try:
            if event.state & 0x0001:  # Shift pressed? (platform dependent)
                if hasattr(event, 'delta'):
                    if event.delta < 0:
                        toolbar_canvas.xview_scroll(1, "units")
                    else:
                        toolbar_canvas.xview_scroll(-1, "units")
                else:
                    if event.num == 5:
                        toolbar_canvas.xview_scroll(1, "units")
                    elif event.num == 4:
                        toolbar_canvas.xview_scroll(-1, "units")
        except Exception:
            pass

    # Make shift+wheel scroll work (bind_all to be safer across platforms)
    toolbar_canvas.bind_all("<Shift-MouseWheel>", _on_mousewheel)
    toolbar_canvas.bind_all("<Shift-Button-4>", _on_mousewheel)
    toolbar_canvas.bind_all("<Shift-Button-5>", _on_mousewheel)

    # NOTE: all controls that were previously packed into 'top' are now packed into inner_frame
    ttk.Label(inner_frame, text="Camera Index:").pack(side=tk.LEFT, padx=4, pady=14)
    cam_var = tk.IntVar(value=0)
    cam_spin = ttk.Spinbox(inner_frame, from_=0, to=10, width=4, textvariable=cam_var)
    cam_spin.pack(side=tk.LEFT, padx=4, pady=10)

    # Step1: Collect Samples
    def on_collect_samples():
        # ask for duration and filename in simple way
        d = 4
        try:
            d = int(simple_duration_var.get())
        except Exception:
            d = 4
        p = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NPZ","*.npz")])
        if not p:
            return
        threading.Thread(target=lambda: collect_samples_to_npz(duration_sec=d, out_path=p, device_index=cam_var.get()), daemon=True).start()

    ttk.Label(inner_frame, text="Collect sec:").pack(side=tk.LEFT, padx=4, pady=14)
    simple_duration_var = tk.StringVar(value="4")
    ttk.Entry(inner_frame, textvariable=simple_duration_var, width=4).pack(side=tk.LEFT, padx=2, pady=10)
    ttk.Button(inner_frame, text="Step 1: Collect Samples", command=on_collect_samples).pack(side=tk.LEFT, padx=6, pady=10)

    # Step2: Run Model Test (single inference on last frames)
    def on_run_model_test():
        if interpreter is None:
            messagebox.showwarning("Model Test", "No TFLite model loaded.")
            return
        frames_for_model = list(frame_buffer)[-cfg["FRAMES"]:] if len(frame_buffer) > 0 else []
        if len(frames_for_model) == 0:
            messagebox.showwarning("Model Test", "No frames in buffer yet. Start camera/detection or collect samples.")
            return
        probs = run_tflite(frames_for_model)
        if probs is None:
            messagebox.showerror("Model Test", "Model returned no output or error. See console.")
            return
        messagebox.showinfo("Model Test", f"Probs: Focus={probs[0]:.3f}, Distracted={probs[1]:.3f}, Fatigued={probs[2]:.3f}")

    ttk.Button(inner_frame, text="Step 2: Run Model Test", command=on_run_model_test).pack(side=tk.LEFT, padx=6, pady=10)

    # load tflite button
    def on_load_tflite():
        p = filedialog.askopenfilename(title="Load TFLite model", filetypes=[("TFLite","*.tflite;*.tflite.zip"),("All","*.*")])
        if not p:
            return
        ok = load_tflite_model(p)
        if ok:
            messagebox.showinfo("TFLite", f"Loaded: {os.path.basename(p)}")

    ttk.Button(inner_frame, text="Load TFLite", command=on_load_tflite).pack(side=tk.LEFT, padx=6, pady=10)

    # model weight
    ttk.Label(inner_frame, text="Model Weight:").pack(side=tk.LEFT, padx=4, pady=14)
    model_weight_var = tk.DoubleVar(value=cfg["MODEL_WEIGHT"])
    model_weight_scale = ttk.Scale(inner_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=model_weight_var, length=140)
    model_weight_scale.pack(side=tk.LEFT, padx=4, pady=10)

    # skip frames control
    ttk.Label(inner_frame, text="Skip Frames:").pack(side=tk.LEFT, padx=4, pady=14)
    skip_var = tk.IntVar(value=cfg["SKIP_FRAMES"])
    skip_spin = ttk.Spinbox(inner_frame, from_=1, to=10, width=4, textvariable=skip_var)
    skip_spin.pack(side=tk.LEFT, padx=4, pady=10)

    # record
    def on_record():
        global video_writer, recording
        if not worker_thread or not worker_thread.is_alive():
            messagebox.showwarning("Start first", "ابتدا detection را Start کنید.")
            return
        with record_lock:
            if not recording:
                p = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4","*.mp4")])
                if not p:
                    return
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vw = cfg["VIDEO_W"]; vh = cfg["VIDEO_H"]
                    video_writer = cv2.VideoWriter(p, fourcc, 20.0, (vw, vh))
                    recording = True
                    record_btn.config(text="Stop Recording")
                    messagebox.showinfo("Recording", f"Recording: {p}")
                except Exception as e:
                    messagebox.showerror("Record", f"Cannot start recording: {e}")
            else:
                try:
                    if video_writer is not None:
                        video_writer.release()
                except:
                    pass
                video_writer = None
                recording = False
                record_btn.config(text="Start Recording")
                messagebox.showinfo("Recording", "Stopped")

    record_btn = ttk.Button(inner_frame, text="Start Recording", command=on_record)
    record_btn.pack(side=tk.LEFT, padx=6, pady=10)

    # log path chooser
    def on_choose_log():
        global log_file_path
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if p:
            log_file_path = p
            lbl_log.config(text=os.path.basename(p))
    ttk.Button(inner_frame, text="Choose Log CSV", command=on_choose_log).pack(side=tk.LEFT, padx=6, pady=10)

    lbl_log = ttk.Label(inner_frame, text="No log")
    lbl_log.pack(side=tk.LEFT, padx=6, pady=14)

    # Calibrate center baseline
    def on_calibrate():
        messagebox.showinfo("Calibration", "لطفاً روبه‌روی دوربین قرار بگیرید. ۳ ثانیه صبر کنید...")
        samples = []
        t0 = time.time()
        while time.time() - t0 < 3.0:
            try:
                item = gui_queue.get(timeout=0.5)
                if item and item.get("frame") is not None:
                    samples.append(item["frame"])
            except queue.Empty:
                pass
        if len(samples) == 0:
            messagebox.showwarning("Calibration", "فریم پیدا نشد. دوباره تلاش کنید.")
            return
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        offs = []
        for f in samples:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
            if len(faces) > 0:
                x,y,wf,hf = faces[0]
                face_cx = x + wf/2.0
                offs.append(abs(face_cx - (f.shape[1]/2.0)) / (f.shape[1]/2.0))
        if len(offs) == 0:
            messagebox.showwarning("Calibration", "چهره در نمونه‌ها تشخیص داده نشد.")
            return
        baseline = float(np.median(offs))
        calibration["baseline_offset"] = baseline
        calibration["calibrated"] = True
        messagebox.showinfo("Calibration", f"Calibration done. baseline_offset={baseline:.3f}")

    ttk.Button(inner_frame, text="Calibrate (center)", command=on_calibrate).pack(side=tk.LEFT, padx=6, pady=10)

    # Reset calibration
    def on_reset_calibration():
        reset_calibration()
        messagebox.showinfo("Calibration", "Calibration reset.")

    ttk.Button(inner_frame, text="Reset Calibration", command=on_reset_calibration).pack(side=tk.LEFT, padx=6, pady=10)

    # Save / Load settings (Step 11 UI buttons)
    def on_save_settings():
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if not p:
            return
        ok = save_settings(p)
        if ok:
            messagebox.showinfo("Settings", f"Settings saved: {os.path.basename(p)}")
        else:
            messagebox.showerror("Settings", "Failed to save settings. See console.")

    def on_load_settings():
        p = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not p:
            return
        payload = load_settings(p)
        if payload is None:
            messagebox.showerror("Settings", "Failed to load settings.")
            return
        # apply some UI updates for loaded cfg values
        try:
            model_weight_var.set(float(cfg.get("MODEL_WEIGHT", model_weight_var.get())))
            # update skip_spin (Spinbox)
            try:
                skip_spin.delete(0, "end")
                skip_spin.insert(0, str(int(cfg.get("SKIP_FRAMES", skip_var.get()))))
            except Exception:
                pass
            lbl_log.config(text=os.path.basename(log_file_path) if log_file_path else "No log")
        except Exception:
            pass
        messagebox.showinfo("Settings", f"Loaded: {os.path.basename(p)}")

    ttk.Button(inner_frame, text="Save Settings", command=on_save_settings).pack(side=tk.LEFT, padx=6, pady=10)
    ttk.Button(inner_frame, text="Load Settings", command=on_load_settings).pack(side=tk.LEFT, padx=6, pady=10)

    # Auto-tune
    auto_tune_var = tk.BooleanVar(value=False)
    def on_toggle_auto_tune():
        if auto_tune_var.get():
            start_auto_tune()
            messagebox.showinfo("Auto-tune", "Auto-tune thresholds enabled.")
        else:
            stop_auto_tune()
            messagebox.showinfo("Auto-tune", "Auto-tune thresholds disabled.")
    ttk.Checkbutton(inner_frame, text="Step 10: Auto-tune thresholds", variable=auto_tune_var, command=on_toggle_auto_tune).pack(side=tk.LEFT, padx=8, pady=10)

    # Start / Stop detection (Step 3)
    def start_detection():
        global worker_thread
        if worker_thread and worker_thread.is_alive():
            messagebox.showinfo("Already running", "Detection already running.")
            return
        cfg["MODEL_WEIGHT"] = float(model_weight_var.get())
        # ensure SKIP_FRAMES updated from spinbox widget
        try:
            cfg["SKIP_FRAMES"] = int(skip_spin.get())
        except Exception:
            cfg["SKIP_FRAMES"] = int(skip_var.get())
        stop_event.clear()
        worker_thread = threading.Thread(target=detection_thread, args=(cam_var.get(), None), daemon=True)
        worker_thread.start()
        start_btn.config(state=tk.DISABLED)
        stop_btn.config(state=tk.NORMAL)

    def stop_detection():
        stop_event.set()
        start_btn.config(state=tk.NORMAL)
        stop_btn.config(state=tk.DISABLED)

    start_btn = ttk.Button(inner_frame, text="Step 3: Start", command=start_detection)
    start_btn.pack(side=tk.LEFT, padx=6, pady=10)
    stop_btn = ttk.Button(inner_frame, text="Stop", command=stop_detection, state=tk.DISABLED)
    stop_btn.pack(side=tk.LEFT, padx=6, pady=10)

    # main layout (left video, right plot)
    main = ttk.Frame(root)
    main.pack(fill=tk.BOTH, expand=True)

    left = ttk.Frame(main)
    left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

    img_lbl = tk.Label(left)
    img_lbl.pack(fill=tk.BOTH, expand=True)

    right = ttk.Frame(main)
    right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)

    # probability plot using matplotlib (like step3)
    fig, ax = plt.subplots(figsize=(5,4))
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    status_lbl = ttk.Label(right, text="Status: Idle", font=("Arial", 14))
    status_lbl.pack(pady=6)

    # Keyboard shortcuts (Step 11)
    def _on_key(event):
        # handle letters regardless of caps lock
        key = event.keysym.lower()
        if key == 's':
            start_detection()
        elif key == 'x':
            stop_detection()
        elif key == 'r':
            on_record()
        elif key == 'c':
            on_calibrate()
        elif key == 't':
            auto_tune_var.set(not auto_tune_var.get())
            on_toggle_auto_tune()
        elif key == 'l':
            # quick load settings without dialog (if settings_file_path exists)
            if settings_file_path:
                load_settings(settings_file_path)
                messagebox.showinfo("Settings", f"Loaded last settings: {os.path.basename(settings_file_path)}")
    root.bind_all("<Key>", _on_key)
    print("[Shortcuts] s=start, x=stop, r=record, c=calibrate, t=toggle auto-tune, l=load-last-settings")

    # GUI update loop
    def gui_loop():
        cfg["MODEL_WEIGHT"] = float(model_weight_var.get())
        try:
            cfg["SKIP_FRAMES"] = int(skip_spin.get())
        except Exception:
            cfg["SKIP_FRAMES"] = int(skip_var.get())
        # drain gui_queue
        while not gui_queue.empty():
            item = gui_queue.get_nowait()
            if item is None:
                continue
            if item.get("status") == "STOP":
                status_lbl.config(text="Status: Stopped")
                start_btn.config(state=tk.NORMAL)
                stop_btn.config(state=tk.DISABLED)
                continue
            frame = item.get("frame")
            status = item.get("status")
            probs = item.get("probs", np.array([1.0,0.0,0.0]))

            # update histories
            for i, name in enumerate(["Focus","Distracted","Fatigued"]):
                prob_history[name].append(float(probs[i]))
                if len(prob_history[name]) > 400:
                    prob_history[name].pop(0)

            if frame is not None:
                disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(Image.fromarray(disp))
                img_lbl.imgtk = imgtk
                img_lbl.config(image=imgtk)

            status_lbl.config(text=f"Status: {status}  probs: {np.round(probs,2)}")

        # update plot
        ax.clear()
        win = int(cfg.get("SMOOTHING_WINDOW",5))
        for name, col in zip(["Focus","Distracted","Fatigued"], ["g","y","r"]):
            arr = prob_history[name][-400:]
            if len(arr) > 0:
                if win > 1 and len(arr) >= win:
                    sm = np.convolve(arr, np.ones(win)/win, mode='valid')
                    ax.plot(sm, label=name, color=col)
                else:
                    ax.plot(arr, label=name, color=col)
        ax.set_ylim(0,1)
        ax.set_title("Probability History")
        ax.legend(loc='upper right')
        canvas.draw()

        root.after(int(1000/cfg["GUI_FPS"]), gui_loop)

    root.after(0, gui_loop)
    root.protocol("WM_DELETE_WINDOW", lambda: (stop_event.set(), stop_fatigue_beep(), stop_auto_tune(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    run_gui()