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
    }
    #start-btn {
      font-size: 1.5rem;
      padding: 1rem 2.5rem;
      background: #00ff99;
      color: #0d0d0d;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-family: monospace;
      font-weight: bold;
      margin-top: 2rem;
    }
    #start-btn:disabled {
      background: #1a1a1a;
      color: #444;
      cursor: default;
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
  <button id="start-btn">TAP TO ENABLE AUDIO</button>
  <div id="dot"></div>
  <div id="status">connecting...</div>

  <script>
    const socket   = io();
    const cmdEl    = document.getElementById('cmd');
    const dotEl    = document.getElementById('dot');
    const statEl   = document.getElementById('status');
    const startBtn = document.getElementById('start-btn');

    let audioUnlocked = false;

    startBtn.addEventListener('click', () => {
      let unlock = new SpeechSynthesisUtterance('');
      speechSynthesis.speak(unlock);
      audioUnlocked = true;
      startBtn.textContent = 'AUDIO ENABLED';
      startBtn.disabled = true;
    });

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
      if (audioUnlocked) {
        speechSynthesis.cancel();
        speechSynthesis.speak(new SpeechSynthesisUtterance(cmd));
      }
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
depth_map     = None
depth_lock    = threading.Lock()
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
    dm_resized = cv2.resize(dm, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)
    with depth_lock:
        depth_map = dm_resized
    depth_running = False


# ===========================
# 4x6 Grid Constants (720x480 frame)
# ===========================
FRAME_W = 720
FRAME_H = 480
COLS    = 4
ROWS    = 6
CW      = FRAME_W // COLS   # 180px per cell
CH      = FRAME_H // ROWS   #  80px per cell

# Danger zone: C2R5, C3R5, C2R6, C3R6 (0-indexed: cols 1-2, rows 4-5)
DANGER_COL_START = 1
DANGER_COL_END   = 3   # exclusive
DANGER_ROW_START = 4
DANGER_ROW_END   = 6   # exclusive

DANGER_PX_X1 = DANGER_COL_START * CW   # 180
DANGER_PX_X2 = DANGER_COL_END   * CW   # 540
DANGER_PX_Y1 = DANGER_ROW_START * CH   # 320
DANGER_PX_Y2 = DANGER_ROW_END   * CH   # 480

DANGER_THRESHOLD  = 0.60
WARNING_THRESHOLD = 0.50


# ===========================
# State
# ===========================
command_history  = deque(maxlen=5)
last_command     = None
last_yolo_result = None
last_scaled_boxes = []          # <-- plain tuples (x1,y1,x2,y2,cls_id,conf)
frame_counter    = 0

YOLO_EVERY_N  = 2
DEPTH_EVERY_N = 5


# ===========================
# Helpers
# ===========================
def get_cell_depth(depth, row, col):
    """Mean depth for a single grid cell (0-indexed)."""
    y1 = row * CH
    y2 = y1 + CH
    x1 = col * CW
    x2 = x1 + CW
    return float(np.mean(depth[y1:y2, x1:x2]))


# ===========================
# Grid-Based Navigation Decision
# ===========================
def decide_command(depth, scaled_boxes):
    """
    scaled_boxes: list of (x1, y1, x2, y2, cls_id, conf) in 720x480 coords.
    Returns a navigation command string.
    """

    # --- Check if any YOLO box overlaps danger zone ---
    object_in_danger = False
    for (x1, y1, x2, y2, cls_id, conf) in scaled_boxes:
        overlap_x = x1 < DANGER_PX_X2 and x2 > DANGER_PX_X1
        overlap_y = y1 < DANGER_PX_Y2 and y2 > DANGER_PX_Y1
        if overlap_x and overlap_y:
            object_in_danger = True
            break

    # --- Danger zone depth (all 4 cells) ---
    d_c2r5 = get_cell_depth(depth, 4, 1)
    d_c3r5 = get_cell_depth(depth, 4, 2)
    d_c2r6 = get_cell_depth(depth, 5, 1)
    d_c3r6 = get_cell_depth(depth, 5, 2)
    danger_depth = np.mean([d_c2r5, d_c3r5, d_c2r6, d_c3r6])

    # --- Escape route depth: C1 (left) and C4 (right), bottom 2 rows ---
    left_escape  = np.mean([get_cell_depth(depth, 4, 0),
                             get_cell_depth(depth, 5, 0)])
    right_escape = np.mean([get_cell_depth(depth, 4, 3),
                             get_cell_depth(depth, 5, 3)])

    # --- Overhead depth: top 2 rows, center cols ---
    overhead_depth = np.mean([
        get_cell_depth(depth, 0, 1), get_cell_depth(depth, 0, 2),
        get_cell_depth(depth, 1, 1), get_cell_depth(depth, 1, 2),
    ])

    # --- Decision tree ---

    # Priority 1: overhead obstacle
    if overhead_depth > DANGER_THRESHOLD:
        return "duck down"

    # Priority 2: object in danger zone AND confirmed close by depth
    if object_in_danger and danger_depth > DANGER_THRESHOLD:
        if left_escape > WARNING_THRESHOLD and right_escape > WARNING_THRESHOLD:
            return "stop"
        if left_escape > right_escape:
            return "turn right"
        return "turn left"

    # Priority 3: object approaching (depth warning only)
    if danger_depth > WARNING_THRESHOLD:
        if left_escape < right_escape:
            return "move left"
        return "move right"

    # Priority 4: clear path
    return "go forward"


# ===========================
# Draw Grid Overlay
# ===========================
def draw_grid(frame):
    h, w = frame.shape[:2]
    cw = w // COLS
    ch = h // ROWS

    for c in range(1, COLS):
        cv2.line(frame, (c * cw, 0), (c * cw, h), (80, 80, 80), 1)
    for r in range(1, ROWS):
        cv2.line(frame, (0, r * ch), (w, r * ch), (80, 80, 80), 1)

    # Red overlay on danger cells
    for (row, col) in [(4, 1), (4, 2), (5, 1), (5, 2)]:
        x1 = col * cw
        y1 = row * ch
        x2 = x1 + cw
        y2 = y1 + ch
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 220), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return frame


# ===========================
# Main Vision Loop
# ===========================
def vision_loop(stream):
    global last_yolo_result, last_scaled_boxes, frame_counter, depth_running, last_command

    while True:
        frame = stream.read()
        if frame is None:
            continue

        frame_counter += 1
        frame_resized = cv2.resize(frame, (FRAME_W, FRAME_H))
        frame_depth   = cv2.resize(frame, (320, 320))

        # --- YOLO every N frames ---
        if frame_counter % YOLO_EVERY_N == 0:
            frame_yolo = cv2.resize(frame_resized, (480, 480))
            result = model.predict(frame_yolo, conf=0.3, verbose=False)[0]
            last_yolo_result = result

            # Rescale boxes 480x480 → 720x480 into plain tuples (no xyxy write)
            scaled_boxes = []
            if result.boxes is not None and len(result.boxes):
                for box in result.boxes:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    x1 = x1 * FRAME_W / 480
                    x2 = x2 * FRAME_W / 480
                    y1 = y1 * FRAME_H / 480
                    y2 = y2 * FRAME_H / 480
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    scaled_boxes.append((x1, y1, x2, y2, cls_id, conf))
            last_scaled_boxes = scaled_boxes

        # --- MiDaS every N frames, non-blocking ---
        if frame_counter % DEPTH_EVERY_N == 0 and not depth_running:
            depth_running = True
            Thread(target=depth_worker, args=(frame_depth,), daemon=True).start()

        # --- Draw YOLO boxes ---
        annotated = frame_resized.copy()
        for (x1, y1, x2, y2, cls_id, conf) in last_scaled_boxes:
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            label = model.names[cls_id] if cls_id < len(model.names) else ""
            cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{label} {conf:.2f}",
                        (ix1, max(iy1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # --- Draw 4x6 grid ---
        annotated = draw_grid(annotated)

        # --- Navigation ---
        with depth_lock:
            current_depth = depth_map

        if current_depth is None:
            cv2.putText(annotated, "Waiting for depth...", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("YOLO + MiDaS Navigation", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        command = decide_command(current_depth, last_scaled_boxes)
        command_history.append(command)
        stable_command = max(set(command_history), key=command_history.count)

        if stable_command != last_command:
            print(f"[NAV] {stable_command.upper()}")
            update_command(stable_command)
            last_command = stable_command

        # --- HUD ---
        cv2.putText(annotated, f"CMD: {stable_command.upper()}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        d_c2r5 = get_cell_depth(current_depth, 4, 1)
        d_c3r5 = get_cell_depth(current_depth, 4, 2)
        d_c2r6 = get_cell_depth(current_depth, 5, 1)
        d_c3r6 = get_cell_depth(current_depth, 5, 2)
        cv2.putText(annotated,
                    f"DZ: C2R5={d_c2r5:.2f} C3R5={d_c3r5:.2f} "
                    f"C2R6={d_c2r6:.2f} C3R6={d_c3r6:.2f}",
                    (10, FRAME_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

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

    vision_thread = Thread(target=vision_loop, args=(stream,), daemon=True)
    vision_thread.start()

    print("[INFO] Dashboard → http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, use_reloader=False)