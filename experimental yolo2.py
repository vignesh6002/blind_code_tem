from ultralytics import YOLOWorld
import cv2
from threading import Thread
import torch
import numpy as np
from collections import deque
import win32com.client
import pythoncom
import threading
import queue
import time


# ===========================
# TTS Engine — win32com SAPI5
# ===========================
tts_queue = queue.Queue()
last_spoken = ""
last_spoken_time = 0
COOLDOWN = 2.5

def speak(command: str):
    global last_spoken, last_spoken_time
    now = time.time()
    if command == last_spoken and (now - last_spoken_time) < COOLDOWN:
        return
    last_spoken = command
    last_spoken_time = now
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
        except queue.Empty:
            break
    tts_queue.put(command)

def tts_worker():
    pythoncom.CoInitialize()
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    speaker.Rate = 1
    speaker.Volume = 100
    while True:
        command = tts_queue.get()
        if command is None:
            break
        speaker.Speak(command)
    pythoncom.CoUninitialize()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()


# ===========================
# Threaded Webcam Reader
# ===========================
ip_address = input("Enter IP address of IP webcamera or type 0 if using laptop cam: ")
if ip_address == "0":
    ip_address = 0

class WebcamStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


# ===========================
# OpenCV Optimization
# ===========================
cv2.setNumThreads(6)
cv2.ocl.setUseOpenCL(True)


# ===========================
# Load YOLO-World VLM
# ===========================
model = YOLOWorld("yolov8s-world.pt")

with open(r"C:\Users\avikv\.vscode\yolo_worlds_objects.txt", "r") as f:
    VOCAB = eval(f.read())

model.set_classes(VOCAB)


# ===========================
# Load MiDaS Depth Model
# ===========================
device = torch.device("cpu")

print("[INFO] Loading MiDaS...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

midas_transform = torch.hub.load(
    "intel-isl/MiDaS", "transforms"
).small_transform

print("[INFO] MiDaS loaded")


# ===========================
# MiDaS Background Thread
# ===========================
depth_map = None           # normalized [0,1] at YOLO resolution (480x480)
depth_lock = threading.Lock()
depth_running = False

def depth_worker(small_frame):
    """
    Runs MiDaS on a 320x320 crop, then resizes the result back to
    480x480 so it aligns 1-to-1 with YOLO's coordinate space.
    """
    global depth_map, depth_running
    img_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    input_tensor = midas_transform(img_rgb).to(device)
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)
    with torch.no_grad():
        d = midas(input_tensor)
    dm = d.squeeze().cpu().numpy()
    dm = (dm - dm.min()) / (dm.max() - dm.min() + 1e-6)

    # ── KEY CHANGE: upsample depth map to match YOLO frame (480×480) ──
    dm_resized = cv2.resize(dm, (480, 480), interpolation=cv2.INTER_LINEAR)

    with depth_lock:
        depth_map = dm_resized
    depth_running = False


# ===========================
# Camera Stream
# ===========================
stream = WebcamStream(ip_address)


# ===========================
# State
# ===========================
command_history  = deque(maxlen=5)
last_command     = None
last_yolo_result = None   # cache last YOLO detections for skipped frames
frame_counter    = 0

# How often to run each expensive model
YOLO_EVERY_N  = 2   # run YOLO every 2nd frame  → effectively halves YOLO load
DEPTH_EVERY_N = 5   # run MiDaS every 5th frame → smooth depth, low CPU spike

# Depth threshold: values above this are considered "close / blocking"
# MiDaS is inverse depth — higher value = closer object
CLOSE_THRESHOLD = 0.65

COMMAND_PHRASES = {
    "go":    "Go forward",
    "left":  "Turn left",
    "right": "Turn right",
    "duck":  "Duck down",
    "uturn": "Make a U-turn",
    "stop":  "Stop. No way ahead",
}


# ===========================
# Depth Fusion Helper
# ===========================
# ===========================
# Navigation Decision — pure MiDaS zones
# ===========================
def decide_command(depth, yolo_result, frame_w=480, frame_h=480):
    """
    Navigation driven entirely by MiDaS depth zones.
    YOLO is only used to gate the 'stop' condition (no objects = stop).

    Zones (on the 480x480 depth map):
      left   : left third
      center : middle third, split into top/bottom halves
      right  : right third
    """
    # No objects visible at all — stop
    any_objects_detected = (
        yolo_result is not None and len(yolo_result.boxes) > 0
    )
    if not any_objects_detected:
        return "stop"

    h, w = depth.shape
    d_left         = np.mean(depth[:, :w//3])
    d_right        = np.mean(depth[:, 2*w//3:])
    center         = depth[:, w//3:2*w//3]
    d_center_top   = np.mean(center[:h//2, :])
    d_center_bot   = np.mean(center[h//2:, :])

    # Overhead obstacle: top-center is close but bottom-center is clear
    if d_center_top > CLOSE_THRESHOLD and d_center_bot < CLOSE_THRESHOLD:
        return "duck"

    # Center is clear — go forward
    if d_center_bot < CLOSE_THRESHOLD:
        return "go"

    # Center blocked — pick clearest side
    if d_left < CLOSE_THRESHOLD and d_right < CLOSE_THRESHOLD:
        # Both sides open — pick the less-blocked one
        return "left" if d_left < d_right else "right"

    if d_left < CLOSE_THRESHOLD:
        return "left"

    if d_right < CLOSE_THRESHOLD:
        return "right"

    # Everything blocked
    return "uturn"


# ===========================
# Main Loop
# ===========================
while True:
    frame = stream.read()
    if frame is None:
        continue

    frame_counter += 1

    # Resize once; share between YOLO and display
    frame_small = cv2.resize(frame, (480, 480))
    frame_depth = cv2.resize(frame, (320, 320))   # smaller for MiDaS speed

    # ---------- YOLO — skip every other frame ----------
    if frame_counter % YOLO_EVERY_N == 0:
        last_yolo_result = model.predict(frame_small, conf=0.3, verbose=False)[0]

    yolo_result = last_yolo_result

    # ---------- MiDaS — fire every N frames, non-blocking ----------
    if frame_counter % DEPTH_EVERY_N == 0 and not depth_running:
        depth_running = True
        Thread(target=depth_worker, args=(frame_depth,), daemon=True).start()

    # ---------- Annotate (uses cached result on skipped frames) ----------
    if yolo_result is not None:
        annotated = yolo_result.plot()
    else:
        annotated = frame_small.copy()

    # ---------- Navigation ----------
    with depth_lock:
        current_depth = depth_map

    if current_depth is None:
        cv2.imshow("YOLO + MiDaS Navigation", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    command = decide_command(current_depth, yolo_result)

    command_history.append(command)
    stable_command = max(set(command_history), key=command_history.count)

    # ---------- Speak + Print on change ----------
    if stable_command != last_command:
        print(f"[NAV] {stable_command.upper()}")
        speak(COMMAND_PHRASES.get(stable_command, stable_command))
        last_command = stable_command

    # ---------- Debug overlay: show all YOLO boxes in green ----------
    if yolo_result is not None and len(yolo_result.boxes):
        for box in yolo_result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        annotated,
        f"CMD: {stable_command.upper()}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("YOLO + MiDaS Navigation", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ===========================
# Cleanup
# ===========================
tts_queue.put(None)
stream.stop()
cv2.destroyAllWindows()