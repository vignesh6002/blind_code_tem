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
# TTS Engine — win32com SAPI5 (truly thread-safe on Windows)
# ===========================
tts_queue = queue.Queue()
last_spoken = ""
last_spoken_time = 0
COOLDOWN = 2.5

def speak(command: str):
    """Enqueue a TTS command — never blocks the main loop."""
    global last_spoken, last_spoken_time
    now = time.time()
    if command == last_spoken and (now - last_spoken_time) < COOLDOWN:
        return
    last_spoken = command
    last_spoken_time = now
    # Flush stale commands so we always speak the latest
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
        except queue.Empty:
            break
    tts_queue.put(command)

def tts_worker():
    """Dedicated thread — owns the SAPI5 COM object exclusively."""
    pythoncom.CoInitialize()          # required: init COM on this thread
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    speaker.Rate = 1                  # -10 (slowest) to 10 (fastest)
    speaker.Volume = 100
    while True:
        command = tts_queue.get()
        if command is None:           # shutdown signal
            break
        speaker.Speak(command)        # blocking within this thread — safe
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
depth_map = None
depth_lock = threading.Lock()
depth_running = False

def depth_worker(small_frame):
    global depth_map, depth_running
    img_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    input_tensor = midas_transform(img_rgb).to(device)
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)
    with torch.no_grad():
        d = midas(input_tensor)
    dm = d.squeeze().cpu().numpy()
    dm = (dm - dm.min()) / (dm.max() - dm.min() + 1e-6)
    dm = dm[:, ::-1]
    with depth_lock:
        depth_map = dm
    depth_running = False


# ===========================
# Camera Stream
# ===========================
stream = WebcamStream(ip_address)


# ===========================
# Command Smoothing
# ===========================
command_history = deque(maxlen=5)
last_command = None
frame_counter = 0
DEPTH_EVERY_N = 3   # run MiDaS every 3rd frame

COMMAND_PHRASES = {
    "go":    "Go forward",
    "left":  "Turn left",
    "right": "Turn right",
    "duck":  "Duck down",
    "uturn": "Make a U-turn",
}


# ===========================
# Main Loop
# ===========================
while True:
    frame = stream.read()
    if frame is None:
        continue

    frame_counter += 1
    frame_small = cv2.resize(frame, (480, 480))   # YOLO at 480
    frame_depth = cv2.resize(frame, (320, 320))   # MiDaS at 320

    # ---------- YOLO Detection ----------
    result = model.predict(frame_small, conf=0.3, verbose=False)[0]
    annotated = result.plot()

    # ---------- MiDaS — fire every N frames, non-blocking ----------
    if frame_counter % DEPTH_EVERY_N == 0 and not depth_running:
        depth_running = True
        Thread(target=depth_worker, args=(frame_depth,), daemon=True).start()

    # ---------- Navigation Logic (uses latest depth) ----------
    with depth_lock:
        current_depth = depth_map

    if current_depth is None:
        cv2.imshow("YOLO + MiDaS Navigation", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    h, w = current_depth.shape
    left          = current_depth[:, :w//3]
    center        = current_depth[:, w//3:2*w//3]
    right         = current_depth[:, 2*w//3:]
    center_top    = center[:h//2, :]
    center_bottom = center[h//2:, :]

    d_left  = np.mean(left)
    d_right = np.mean(right)
    d_top   = np.mean(center_top)
    d_bot   = np.mean(center_bottom)

    command = "go"

    if d_top > d_bot and d_top > d_left and d_top > d_right:
        command = "duck"
    elif d_bot > d_left and d_bot > d_right:
        if d_left > d_right:
            command = "left"
        elif d_right > d_left:
            command = "right"
        else:
            command = "uturn"

    command_history.append(command)
    stable_command = max(set(command_history), key=command_history.count)

    # ---------- Speak + Print on change ----------
    if stable_command != last_command:
        print(f"[NAV] {stable_command.upper()}")
        speak(COMMAND_PHRASES.get(stable_command, stable_command))
        last_command = stable_command

    # ---------- Display ----------
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
tts_queue.put(None)   # gracefully shut down TTS worker
stream.stop()
cv2.destroyAllWindows()