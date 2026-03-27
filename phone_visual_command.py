from ultralytics import YOLOWorld
import cv2
from threading import Thread
import torch
import numpy as np
from collections import deque
import threading
import time

from flask import Flask, render_template_string
from flask_socketio import SocketIO


# ===========================
# Flask + SocketIO Setup
# ===========================
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

def update_command(cmd):
    """Emit the latest navigation command to all connected clients."""
    socketio.emit("command", cmd)


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>Navigation Command</title>
  <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
  <style>
    body {
      background: #0d0d0d;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100vh;
      margin: 0;
      font-family: monospace;
    }
    #cmd {
      font-size: 5rem;
      font-weight: bold;
      color: #00ff99;
      letter-spacing: 0.1em;
      transition: color 0.2s;
    }
    #dot {
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: #ff3333;
      margin-top: 1.5rem;
    }
    #dot.connected { background: #00ff99; }
    #status { color: #555; font-size: 0.9rem; margin-top: 0.5rem; }
  </style>
</head>
<body>
  <div id="cmd">WAITING...</div>
  <div id="dot"></div>
  <div id="status">connecting...</div>

  <script>
    const socket = io();
    const cmdEl   = document.getElementById('cmd');
    const dotEl   = document.getElementById('dot');
    const statEl  = document.getElementById('status');

    socket.on('connect', () => {
      dotEl.classList.add('connected');
      statEl.textContent = 'connected';
    });

    socket.on('disconnect', () => {
      dotEl.classList.remove('connected');
      statEl.textContent = 'disconnected';
    });

    socket.on('command', (cmd) => {
      console.log('NAV CMD:', cmd);
      cmdEl.textContent = cmd.toUpperCase();
    });
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


# ===========================
# Threaded Webcam Reader
# ===========================
class WebcamStream:
    def __init__(self, src):
        if isinstance(src, str):
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
# Load YOLO-World
# ===========================
model = YOLOWorld("yolov8s-world.pt")

with open(r"C:\Users\avikv\.vscode\yolo_worlds_objects.txt", "r") as f:
    VOCAB = eval(f.read())

model.set_classes(VOCAB)


# ===========================
# Load MiDaS
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
depth_map    = None
depth_lock   = threading.Lock()
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
    dm_resized = cv2.resize(dm, (480, 480), interpolation=cv2.INTER_LINEAR)
    with depth_lock:
        depth_map = dm_resized
    depth_running = False


# ===========================
# State
# ===========================
command_history  = deque(maxlen=5)
last_command     = None
last_yolo_result = None
frame_counter    = 0

YOLO_EVERY_N    = 2
DEPTH_EVERY_N   = 5
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
# Navigation Decision
# ===========================
def decide_command(depth, yolo_result, frame_w=480, frame_h=480):
    any_objects_detected = (
        yolo_result is not None and len(yolo_result.boxes) > 0
    )
    if not any_objects_detected:
        return "stop"

    h, w = depth.shape
    d_left       = np.mean(depth[:, :w//3])
    d_right      = np.mean(depth[:, 2*w//3:])
    center       = depth[:, w//3:2*w//3]
    d_center_top = np.mean(center[:h//2, :])
    d_center_bot = np.mean(center[h//2:, :])

    if d_center_top > CLOSE_THRESHOLD and d_center_bot < CLOSE_THRESHOLD:
        return "duck"

    if d_center_bot < CLOSE_THRESHOLD:
        return "go"

    if d_left < CLOSE_THRESHOLD and d_right < CLOSE_THRESHOLD:
        return "left" if d_left < d_right else "right"

    if d_left < CLOSE_THRESHOLD:
        return "left"

    if d_right < CLOSE_THRESHOLD:
        return "right"

    return "uturn"


# ===========================
# Main Vision Loop (runs in its own thread)
# ===========================
def vision_loop(stream):
    global last_yolo_result, frame_counter, depth_running, last_command

    while True:
        frame = stream.read()
        if frame is None:
            continue

        frame_counter += 1
        frame_small = cv2.resize(frame, (480, 480))
        frame_depth = cv2.resize(frame, (320, 320))

        # YOLO — every N frames
        if frame_counter % YOLO_EVERY_N == 0:
            last_yolo_result = model.predict(
                frame_small, conf=0.3, verbose=False
            )[0]

        yolo_result = last_yolo_result

        # MiDaS — every N frames, non-blocking
        if frame_counter % DEPTH_EVERY_N == 0 and not depth_running:
            depth_running = True
            Thread(
                target=depth_worker, args=(frame_depth,), daemon=True
            ).start()

        # Annotate
        if yolo_result is not None:
            annotated = yolo_result.plot()
        else:
            annotated = frame_small.copy()

        # Navigation
        with depth_lock:
            current_depth = depth_map

        if current_depth is None:
            cv2.imshow("YOLO + MiDaS Navigation", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        command = decide_command(current_depth, yolo_result)
        command_history.append(command)
        stable_command = max(set(command_history), key=command_history.count)

        # Emit on change
        if stable_command != last_command:
            print(f"[NAV] {stable_command.upper()}")
            update_command(stable_command)          # → SocketIO
            last_command = stable_command

        # Draw YOLO boxes + command
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
            2,
        )

        cv2.imshow("YOLO + MiDaS Navigation", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.stop()
    cv2.destroyAllWindows()


# ===========================
# Entry Point
# ===========================
if __name__ == "__main__":
    ip_address = input(
        "Enter IP address of IP webcamera or type 0 if using laptop cam: "
    )
    if ip_address == "0":
        ip_address = 0

    stream = WebcamStream(ip_address)

    # Vision runs in a background thread so Flask can own the main thread
    vision_thread = Thread(target=vision_loop, args=(stream,), daemon=True)
    vision_thread.start()

    print("[INFO] Dashboard → http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, use_reloader=False)