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
    socketio.emit("command", {"cmd": cmd})


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
      font-size: 4rem;
      font-weight: bold;
      color: #00ff99;
      letter-spacing: 0.1em;
      text-align: center;
    }
    #zone-grid {
      display: grid;
      grid-template-columns: repeat(4, 100px);
      grid-template-rows: repeat(6, 50px);
      gap: 3px;
      margin-top: 1.5rem;
    }
    .zone-cell {
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.6rem;
      color: #888;
      border: 1px solid #222;
      border-radius: 3px;
      background: #111;
      transition: background 0.2s, color 0.2s;
    }
    .zone-cell.clear    { background: #0a2e0a; color: #00ff99; }
    .zone-cell.slight   { background: #2e2200; color: #ffaa00; }
    .zone-cell.danger   { background: #2e0000; color: #ff4444; }
    .zone-cell.critical { background: #550000; color: #ff0000; border: 1px solid #ff0000; }
    .zone-cell.dz       { border: 2px solid #ff0000 !important; }
    #start-btn {
      font-size: 1.2rem;
      padding: 0.8rem 2rem;
      background: #00ff99;
      color: #0d0d0d;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-family: monospace;
      font-weight: bold;
      margin-top: 1.5rem;
    }
    #start-btn:disabled { background: #1a1a1a; color: #444; cursor: default; }
    #dot { width: 12px; height: 12px; border-radius: 50%; background: #ff3333; margin-top: 1rem; }
    #dot.connected { background: #00ff99; }
    #status { color: #555; font-size: 0.85rem; margin-top: 0.4rem; }
  </style>
</head>
<body>
  <div id="cmd">WAITING...</div>

  <div id="zone-grid">
    <div class="zone-cell" id="z_c1r1">C1R1</div>
    <div class="zone-cell" id="z_c2r1">C2R1</div>
    <div class="zone-cell" id="z_c3r1">C3R1</div>
    <div class="zone-cell" id="z_c4r1">C4R1</div>

    <div class="zone-cell" id="z_c1r2">C1R2</div>
    <div class="zone-cell" id="z_c2r2">C2R2</div>
    <div class="zone-cell" id="z_c3r2">C3R2</div>
    <div class="zone-cell" id="z_c4r2">C4R2</div>

    <div class="zone-cell" id="z_c1r3">C1R3</div>
    <div class="zone-cell" id="z_c2r3">C2R3</div>
    <div class="zone-cell" id="z_c3r3">C3R3</div>
    <div class="zone-cell" id="z_c4r3">C4R3</div>

    <div class="zone-cell" id="z_c1r4">C1R4</div>
    <div class="zone-cell" id="z_c2r4">C2R4</div>
    <div class="zone-cell" id="z_c3r4">C3R4</div>
    <div class="zone-cell" id="z_c4r4">C4R4</div>

    <div class="zone-cell" id="z_c1r5">C1R5</div>
    <div class="zone-cell dz" id="z_c2r5">C2R5</div>
    <div class="zone-cell dz" id="z_c3r5">C3R5</div>
    <div class="zone-cell" id="z_c4r5">C4R5</div>

    <div class="zone-cell" id="z_c1r6">C1R6</div>
    <div class="zone-cell dz" id="z_c2r6">C2R6</div>
    <div class="zone-cell dz" id="z_c3r6">C3R6</div>
    <div class="zone-cell" id="z_c4r6">C4R6</div>
  </div>

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
      speechSynthesis.speak(new SpeechSynthesisUtterance(''));
      audioUnlocked = true;
      startBtn.textContent = 'AUDIO ENABLED';
      startBtn.disabled = true;
    });

    socket.on('connect',    () => { dotEl.classList.add('connected');    statEl.textContent = 'connected'; });
    socket.on('disconnect', () => { dotEl.classList.remove('connected'); statEl.textContent = 'disconnected'; });

    socket.on('command', (data) => {
      cmdEl.textContent = data.cmd.toUpperCase();
      if (audioUnlocked) {
        speechSynthesis.cancel();
        speechSynthesis.speak(new SpeechSynthesisUtterance(data.cmd));
      }
    });

    socket.on('zones', (grid) => {
      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 4; c++) {
          const id  = `z_c${c+1}r${r+1}`;
          const el  = document.getElementById(id);
          if (!el) continue;
          const val     = grid[r][c];
          const isDZ    = el.classList.contains('dz');
          el.className  = 'zone-cell' + (isDZ ? ' dz' : '');
          if      (val >= 0.75) el.classList.add('critical');
          else if (val >= 0.60) el.classList.add('danger');
          else if (val >= 0.45) el.classList.add('slight');
          else                  el.classList.add('clear');
          el.title = val.toFixed(2);
        }
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
midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
print("[INFO] MiDaS loaded")


# ===========================
# MiDaS Background Thread
# ===========================
depth_map     = None
depth_lock    = threading.Lock()
depth_running = False

def depth_worker(small_frame):
    global depth_map, depth_running
    try:
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
    finally:
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

# Danger zone pixel bounds (C2R5, C3R5, C2R6, C3R6)
DANGER_PX_X1 = 1 * CW   # 180
DANGER_PX_X2 = 3 * CW   # 540
DANGER_PX_Y1 = 4 * CH   # 320
DANGER_PX_Y2 = 6 * CH   # 480

# ===========================
# Proximity Thresholds
# Calibrate by watching DZ depth HUD while walking toward a wall.
# ===========================
CRITICAL_THRESHOLD = 0.75   # very close  → "turn {side} now"
DANGER_THRESHOLD   = 0.60   # close       → "turn {side}"
WARNING_THRESHOLD  = 0.45   # approaching → "slight {side}"

COMMAND_PHRASES = {
    "go forward":     "Go forward",
    "slight left":    "Slightly left",
    "slight right":   "Slightly right",
    "turn left":      "Turn left",
    "turn right":     "Turn right",
    "turn left now":  "Turn left now",
    "turn right now": "Turn right now",
    "duck down":      "Duck down",
    "stop":           "Stop, obstacle ahead",
}


# ===========================
# State
# ===========================
last_scaled_boxes = []
frame_counter     = 0

YOLO_EVERY_N  = 2
DEPTH_EVERY_N = 5


# ===========================
# Helpers
# ===========================
def get_cell_depth(depth, row, col):
    y1 = row * CH;  y2 = y1 + CH
    x1 = col * CW;  x2 = x1 + CW
    return float(np.mean(depth[y1:y2, x1:x2]))

def compute_grid(depth):
    grid = np.zeros((ROWS, COLS), dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            grid[r, c] = get_cell_depth(depth, r, c)
    return grid

def proximity_direction(danger_depth, left_escape, right_escape):
    """Proximity-scaled directional command. Lower escape = safer side."""
    side = "right" if left_escape >= right_escape else "left"
    if danger_depth >= CRITICAL_THRESHOLD:
        return f"turn {side} now"
    elif danger_depth >= DANGER_THRESHOLD:
        return f"turn {side}"
    elif danger_depth >= WARNING_THRESHOLD:
        return f"slight {side}"
    return None


# ===========================
# Grid-Based Navigation Decision
# ===========================
def decide_command(depth, scaled_boxes):
    """
    4x6 matrix danger zone logic (C2R5, C3R5, C2R6, C3R6) with
    proximity-scaled commands layered on top.
    """

    # --- YOLO overlap with danger zone ---
    object_in_danger = False
    for (x1, y1, x2, y2, cls_id, conf) in scaled_boxes:
        if x1 < DANGER_PX_X2 and x2 > DANGER_PX_X1 and \
           y1 < DANGER_PX_Y2 and y2 > DANGER_PX_Y1:
            object_in_danger = True
            break

    # --- Danger zone depth (4 cells averaged) ---
    danger_depth = float(np.mean([
        get_cell_depth(depth, 4, 1),   # C2R5
        get_cell_depth(depth, 4, 2),   # C3R5
        get_cell_depth(depth, 5, 1),   # C2R6
        get_cell_depth(depth, 5, 2),   # C3R6
    ]))

    # --- Escape corridors: C1 (left) and C4 (right), rows 5-6 ---
    left_escape  = float(np.mean([get_cell_depth(depth, 4, 0),
                                   get_cell_depth(depth, 5, 0)]))
    right_escape = float(np.mean([get_cell_depth(depth, 4, 3),
                                   get_cell_depth(depth, 5, 3)]))

    # --- Overhead: rows 1-2, cols C2+C3 ---
    overhead_depth = float(np.mean([
        get_cell_depth(depth, 0, 1), get_cell_depth(depth, 0, 2),
        get_cell_depth(depth, 1, 1), get_cell_depth(depth, 1, 2),
    ]))

    # Priority 1: overhead obstacle
    if overhead_depth >= DANGER_THRESHOLD:
        return "duck down"

    # Priority 2: YOLO-confirmed object in danger zone + depth
    if object_in_danger and danger_depth >= WARNING_THRESHOLD:
        if left_escape >= DANGER_THRESHOLD and right_escape >= DANGER_THRESHOLD:
            return "stop"
        cmd = proximity_direction(danger_depth, left_escape, right_escape)
        if cmd:
            return cmd

    # Priority 3: depth-only warning (object approaching, no YOLO box yet)
    if danger_depth >= WARNING_THRESHOLD:
        if left_escape >= DANGER_THRESHOLD and right_escape >= DANGER_THRESHOLD:
            return "stop"
        cmd = proximity_direction(danger_depth, left_escape, right_escape)
        if cmd:
            return cmd

    # Priority 4: clear
    return "go forward"


# ===========================
# Stability Filter
# ===========================
class CommandFilter:
    def __init__(self, stable_frames=3, cooldown_sec=2.5):
        self.stable_frames  = stable_frames
        self.cooldown_sec   = cooldown_sec
        self.history        = deque(maxlen=stable_frames)
        self.last_emitted   = None
        self.last_emit_time = 0.0

    def update(self, command):
        self.history.append(command)
        if len(self.history) < self.stable_frames:
            return None
        if len(set(self.history)) != 1:
            return None
        now = time.time()
        if command == self.last_emitted and \
           (now - self.last_emit_time) < self.cooldown_sec:
            return None
        self.last_emitted   = command
        self.last_emit_time = now
        return command


# ===========================
# Draw Grid Overlay
# ===========================
def draw_grid(frame, grid):
    h, w = frame.shape[:2]
    cw = w // COLS
    ch = h // ROWS

    for r in range(ROWS):
        for c in range(COLS):
            val = grid[r, c]
            x1, y1 = c * cw, r * ch
            x2, y2 = x1 + cw, y1 + ch
            if   val >= CRITICAL_THRESHOLD: color = (0,   0, 255)
            elif val >= DANGER_THRESHOLD:   color = (0,  80, 255)
            elif val >= WARNING_THRESHOLD:  color = (0, 200, 255)
            else:                           color = (0, 180,   0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, f"{val:.2f}", (x1+4, y1+16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Bold red border on danger cells
    for (row, col) in [(4,1),(4,2),(5,1),(5,2)]:
        x1, y1 = col*cw, row*ch
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x1+cw, y1+ch), (0,0,200), -1)
        cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x1+cw, y1+ch), (0,0,255), 2)
    return frame


# ===========================
# Main Vision Loop
# ===========================
def vision_loop(stream):
    global last_scaled_boxes, frame_counter, depth_running

    cmd_filter = CommandFilter(stable_frames=3, cooldown_sec=2.5)

    while True:
        frame = stream.read()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_counter += 1
        frame_resized = cv2.resize(frame, (FRAME_W, FRAME_H))
        frame_depth   = cv2.resize(frame, (320, 320))

        # --- YOLO every N frames ---
        if frame_counter % YOLO_EVERY_N == 0:
            frame_yolo = cv2.resize(frame_resized, (480, 480))
            result = model.predict(frame_yolo, conf=0.3, verbose=False)[0]
            scaled_boxes = []
            if result.boxes is not None and len(result.boxes):
                for box in result.boxes:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    scaled_boxes.append((
                        x1 * FRAME_W / 480, y1 * FRAME_H / 480,
                        x2 * FRAME_W / 480, y2 * FRAME_H / 480,
                        int(box.cls[0]), float(box.conf[0])
                    ))
            last_scaled_boxes = scaled_boxes

        # --- MiDaS every N frames, non-blocking ---
        if frame_counter % DEPTH_EVERY_N == 0 and not depth_running:
            depth_running = True
            Thread(target=depth_worker, args=(frame_depth,), daemon=True).start()

        # --- Draw YOLO boxes ---
        annotated = frame_resized.copy()
        for (x1, y1, x2, y2, cls_id, conf) in last_scaled_boxes:
            label = model.names[cls_id] if cls_id < len(model.names) else ""
            cv2.rectangle(annotated, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            cv2.putText(annotated, f"{label} {conf:.2f}",
                        (int(x1), max(int(y1)-6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

        # --- Navigation ---
        with depth_lock:
            current_depth = depth_map

        if current_depth is None:
            cv2.putText(annotated, "Waiting for depth...", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.imshow("Navigation", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        grid     = compute_grid(current_depth)
        raw_cmd  = decide_command(current_depth, last_scaled_boxes)
        emit_cmd = cmd_filter.update(raw_cmd)

        if emit_cmd is not None:
            phrase = COMMAND_PHRASES.get(emit_cmd, emit_cmd)
            print(f"[NAV] {emit_cmd.upper():20s} → {phrase}")
            update_command(phrase)
            socketio.emit("zones", grid.tolist())

        # --- Grid overlay + HUD ---
        annotated = draw_grid(annotated, grid)
        phrase    = COMMAND_PHRASES.get(raw_cmd, raw_cmd)
        cv2.putText(annotated, f"CMD: {phrase}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        dz = float(np.mean([grid[4,1], grid[4,2], grid[5,1], grid[5,2]]))
        cv2.putText(annotated,
                    f"DZ:{dz:.2f}  L-esc:{grid[4,0]:.2f}  R-esc:{grid[4,3]:.2f}",
                    (10, FRAME_H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,0), 1)

        cv2.imshow("Navigation", annotated)
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