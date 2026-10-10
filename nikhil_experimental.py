from ultralytics import YOLOWorld
import cv2
from threading import Thread, Lock
import queue
import torch
import numpy as np
from collections import deque
import time
import os

from flask import Flask, render_template_string
from flask_socketio import SocketIO


# ===========================
# Flask + SocketIO Setup
# ===========================
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

def update_command(cmd, alert_type="info"):
    socketio.emit("command", {"cmd": cmd, "type": alert_type})


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>Blind Navigation Aid - Rush Mode</title>
  <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
  <style>
    body {
      background: #000;
      color: #fff;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100vh;
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
      overflow: hidden;
    }
    #hud {
      display: flex;
      flex-direction: column;
      align-items: center;
      width: 90%;
      max-width: 600px;
    }
    #cmd {
      font-size: 3.5rem;
      font-weight: 900;
      color: #00ff99;
      letter-spacing: 0.05em;
      text-align: center;
      min-height: 5rem;
      display: flex;
      align-items: center;
      justify-content: center;
      text-transform: uppercase;
    }
    #cmd.danger {
      color: #ff3333;
      text-shadow: 0 0 20px rgba(255, 51, 51, 0.8);
      animation: pulse 0.4s infinite alternate;
    }
    #cmd.warning {
      color: #ffaa00;
    }
    @keyframes pulse {
      from { transform: scale(1); }
      to { transform: scale(1.05); }
    }
    #corridor-bar {
      display: flex;
      width: 100%;
      height: 40px;
      margin: 1.5rem 0;
      border: 2px solid #333;
      border-radius: 8px;
      overflow: hidden;
    }
    .corridor-seg {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.85rem;
      font-weight: bold;
      background: #111;
      color: #666;
      border-right: 1px solid #222;
      transition: background 0.15s, color 0.15s;
    }
    .corridor-seg:last-child { border-right: none; }
    .corridor-seg.open   { background: #00441b; color: #00ff99; }
    .corridor-seg.blocked { background: #550000; color: #ff5555; }

    #zone-grid {
      display: grid;
      grid-template-columns: repeat(4, 80px);
      grid-template-rows: repeat(6, 40px);
      gap: 3px;
      margin-top: 1rem;
    }
    .zone-cell {
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.65rem;
      color: #666;
      border: 1px solid #222;
      border-radius: 3px;
      background: #0d0d0d;
    }
    .zone-cell.clear    { background: #062406; color: #00ff99; }
    .zone-cell.slight   { background: #261d00; color: #ffaa00; }
    .zone-cell.danger   { background: #330000; color: #ff4444; }
    .zone-cell.critical { background: #660000; color: #ff0000; font-weight: bold; border: 1px solid #ff3333; }
    .zone-cell.dz       { border: 2px solid #ff0000 !important; }

    #start-btn {
      font-size: 1.4rem;
      padding: 1rem 2.5rem;
      background: #00ff99;
      color: #000;
      border: none;
      border-radius: 12px;
      cursor: pointer;
      font-weight: bold;
      margin-top: 2rem;
      letter-spacing: 0.05em;
      box-shadow: 0 0 25px rgba(0, 255, 153, 0.4);
    }
    #start-btn:disabled { background: #222; color: #555; box-shadow: none; cursor: default; }
    #status { color: #555; font-size: 0.85rem; margin-top: 0.8rem; }
  </style>
</head>
<body>
  <div id="hud">
    <div id="cmd">READY</div>

    <div id="corridor-bar">
      <div class="corridor-seg" id="cor_left">LEFT</div>
      <div class="corridor-seg" id="cor_mid_left">C-LEFT</div>
      <div class="corridor-seg" id="cor_mid_right">C-RIGHT</div>
      <div class="corridor-seg" id="cor_right">RIGHT</div>
    </div>

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

    <button id="start-btn">ENABLE AUDIO ASSIST</button>
    <div id="status">Audio Engine Standby</div>
  </div>

  <script>
    const socket   = io();
    const cmdEl    = document.getElementById('cmd');
    const statEl   = document.getElementById('status');
    const startBtn = document.getElementById('start-btn');
    let audioUnlocked = false;

    // Web Audio Synthesizer for Instant Low-Latency Earcons (<5ms)
    let audioCtx = null;

    function playTone(freq, durationMs, type="sine") {
      if (!audioCtx) return;
      try {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + durationMs / 1000);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + durationMs / 1000);
      } catch (e) {}
    }

    function triggerEarcon(alertType) {
      if (!audioCtx) return;
      if (alertType === "critical") {
        // High double-beep for urgent danger
        playTone(900, 80, "sawtooth");
        setTimeout(() => playTone(900, 100, "sawtooth"), 100);
      } else if (alertType === "warning") {
        // Warning chime
        playTone(550, 120, "triangle");
      }
    }

    startBtn.addEventListener('click', () => {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      speechSynthesis.speak(new SpeechSynthesisUtterance(''));
      audioUnlocked = true;
      startBtn.textContent = 'RUSH AUDIO ACTIVE';
      startBtn.disabled = true;
      statEl.textContent = 'Active - Headphone ready';
      playTone(440, 150);
    });

    socket.on('command', (data) => {
      cmdEl.textContent = data.cmd.toUpperCase();
      cmdEl.className = data.type === 'critical' ? 'danger' : (data.type === 'warning' ? 'warning' : '');

      if (audioUnlocked) {
        triggerEarcon(data.type);

        speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(data.cmd);
        // Fast speech rate (1.25x) so blind user gets instant information without latency
        utterance.rate = 1.25;
        utterance.pitch = data.type === 'critical' ? 1.2 : 1.0;
        speechSynthesis.speak(utterance);
      }
    });

    socket.on('corridors', (c) => {
      // 4 corridors: left, c_left, c_right, right
      const ids = ['cor_left', 'cor_mid_left', 'cor_mid_right', 'cor_right'];
      for (let i = 0; i < 4; i++) {
        const el = document.getElementById(ids[i]);
        if (el) {
          el.className = 'corridor-seg ' + (c[i] ? 'open' : 'blocked');
        }
      }
    });

    socket.on('zones', (grid) => {
      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 4; c++) {
          const id = `z_c${c+1}r${r+1}`;
          const el = document.getElementById(id);
          if (!el) continue;
          const val  = grid[r][c];
          const isDZ = el.classList.contains('dz');
          el.className = 'zone-cell' + (isDZ ? ' dz' : '');
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
# Hardware & OpenCV Optimization
# ===========================
cv2.setNumThreads(cv2.getNumberOfCPUs())
cv2.ocl.setUseOpenCL(True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_HALF = (DEVICE.type == "cuda")
print(f"[INFO] Rush Navigation running on {DEVICE} (FP16: {USE_HALF})")


# ===========================
# Grid & Spatial Layout Constants
# ===========================
FRAME_W = 720
FRAME_H = 480
COLS    = 4
ROWS    = 6
CW      = FRAME_W // COLS   # 180px per corridor
CH      = FRAME_H // ROWS   #  80px per depth slice

# Central Walking Corridor (Danger Zone)
DANGER_PX_X1 = 1 * CW   # 180px
DANGER_PX_X2 = 3 * CW   # 540px
DANGER_PX_Y1 = 3 * CH   # 240px
DANGER_PX_Y2 = 6 * CH   # 480px

# Overhead Zone (Head-strike region, impossible for white-cane to detect)
OVERHEAD_PX_X1 = 1 * CW
OVERHEAD_PX_X2 = 3 * CW
OVERHEAD_PX_Y1 = 0
OVERHEAD_PX_Y2 = 2 * CH

# Proximity Thresholds
CRITICAL_THRESHOLD = 0.75   # Immediate collision threat
DANGER_THRESHOLD   = 0.58   # Obstacle in path requiring maneuver
WARNING_THRESHOLD  = 0.42   # Approaching obstacle
OPEN_PATH_MAX      = 0.40   # Maximum depth considered a completely clear corridor

YOLO_CONF_THRESHOLD = 0.45


# ===========================
# Safety-Critical Vocabulary for Rush / Urban Environments
# ===========================
SAFETY_CLASSES = [
    # High-impact traffic
    "car", "bus", "truck", "motorcycle", "bicycle", "scooter", "auto rickshaw",

    # Drop-offs & fall hazards (unseen by distracted walking)
    "open manhole", "pothole", "stairs", "step", "open drain",

    # Overhead clearance (White-cane blindspots)
    "low hanging branch", "overhead obstacle", "low ceiling", "hanging wire",

    # Heavy urban path blockers
    "person", "pedestrian", "tree trunk", "electric pole", "pillar",
    "road barrier", "barricade", "traffic cone", "concrete block", "wall", "gate",
    "dog", "stray dog", "cow"
]

VEHICLE_SET = {"car", "bus", "truck", "motorcycle", "bicycle", "scooter", "auto rickshaw"}
GROUND_HAZARD_SET = {"open manhole", "pothole", "stairs", "step", "open drain"}
OVERHEAD_SET = {"low hanging branch", "overhead obstacle", "low ceiling", "hanging wire"}


# ===========================
# Threaded Video Stream
# ===========================
class WebcamStream:
    def __init__(self, src):
        if isinstance(src, str):
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(src)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.lock = Lock()
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.lock:
                    self.grabbed = grabbed
                    self.frame = frame

    def read(self):
        with self.lock:
            if not self.grabbed or self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()


# ===========================
# Model Initialization
# ===========================
print("[INFO] Initializing YOLO-World...")
yolo_model = YOLOWorld("yolov8s-world.pt")
yolo_model.set_classes(SAFETY_CLASSES)
yolo_model.to(DEVICE)

print("[INFO] Initializing MiDaS Depth...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(DEVICE)
if USE_HALF:
    midas = midas.half()
midas.eval()
midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform


# ===========================
# Threaded Pipeline & Async Queues
# ===========================
latest_grid = None
grid_lock = Lock()

latest_boxes = []
boxes_lock = Lock()

depth_queue = queue.Queue(maxsize=1)
yolo_queue = queue.Queue(maxsize=1)


def depth_worker_loop():
    global latest_grid
    while True:
        frame_for_depth = depth_queue.get()
        try:
            img_rgb = cv2.cvtColor(frame_for_depth, cv2.COLOR_BGR2RGB)
            input_tensor = midas_transform(img_rgb).to(DEVICE)
            if input_tensor.ndim == 3:
                input_tensor = input_tensor.unsqueeze(0)
            if USE_HALF:
                input_tensor = input_tensor.half()

            with torch.no_grad():
                d = midas(input_tensor)

            dm = d.squeeze().cpu().numpy()
            d_min, d_max = dm.min(), dm.max()
            dm = (dm - d_min) / (d_max - d_min + 1e-6)

            # Direct 4x6 matrix downsampling
            grid = cv2.resize(dm, (COLS, ROWS), interpolation=cv2.INTER_AREA)

            with grid_lock:
                latest_grid = grid
        except Exception as e:
            print(f"[ERROR in Depth Worker]: {e}")
        finally:
            depth_queue.task_done()


def yolo_worker_loop():
    global latest_boxes
    while True:
        frame_for_yolo = yolo_queue.get()
        try:
            results = yolo_model.predict(
                frame_for_yolo,
                conf=YOLO_CONF_THRESHOLD,
                verbose=False,
                device=DEVICE,
                half=USE_HALF,
                imgsz=480
            )[0]

            filtered_boxes = []
            if results.boxes is not None and len(results.boxes):
                orig_h, orig_w = frame_for_yolo.shape[:2]
                scale_x = FRAME_W / orig_w
                scale_y = FRAME_H / orig_h
                for box in results.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = yolo_model.names[cls_id] if cls_id < len(yolo_model.names) else ""

                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    bx1, by1, bx2, by2 = x1 * scale_x, y1 * scale_y, x2 * scale_y, y2 * scale_y

                    in_dz = (bx1 < DANGER_PX_X2 and bx2 > DANGER_PX_X1 and by1 < DANGER_PX_Y2 and by2 > DANGER_PX_Y1)
                    in_oh = (bx1 < OVERHEAD_PX_X2 and bx2 > OVERHEAD_PX_X1 and by1 < OVERHEAD_PX_Y2 and by2 > OVERHEAD_PX_Y1)

                    if in_dz or in_oh or label in VEHICLE_SET:
                        filtered_boxes.append((bx1, by1, bx2, by2, cls_id, conf, label, in_dz, in_oh))

            with boxes_lock:
                latest_boxes = filtered_boxes
        except Exception as e:
            print(f"[ERROR in YOLO Worker]: {e}")
        finally:
            yolo_queue.task_done()


Thread(target=depth_worker_loop, daemon=True).start()
Thread(target=yolo_worker_loop, daemon=True).start()


# ===========================
# Blind Person Rush-Navigation Logic
# - Punchy 1-2 word commands (low cognitive load)
# - Dynamic Safe Corridor Finding (Navigating crowd bottlenecks)
# - Looming Approach Detection
# ===========================
class RushNavigator:
    def __init__(self):
        self.prev_dz_depth = 0.0
        self.prev_time = time.time()

    def evaluate(self, grid, detected_boxes):
        now = time.time()
        dt = max(now - self.prev_time, 0.05)

        # Danger zone depth (Rows 4-5, Cols 1-2)
        danger_depth = float((grid[4, 1] + grid[4, 2] + grid[5, 1] + grid[5, 2]) * 0.25)

        # Looming velocity: Rate of obstacle approach
        looming_rate = (danger_depth - self.prev_dz_depth) / dt
        self.prev_dz_depth = danger_depth
        self.prev_time = now

        # Overhead clearance (White cane cannot sense this!)
        overhead_depth = float((grid[0, 1] + grid[0, 2] + grid[1, 1] + grid[1, 2]) * 0.25)

        # Corridor depths across 4 walking columns (average of forward rows 3-5)
        c0_depth = float(np.mean(grid[3:6, 0]))  # Far Left
        c1_depth = float(np.mean(grid[3:6, 1]))  # Center Left
        c2_depth = float(np.mean(grid[3:6, 2]))  # Center Right
        c3_depth = float(np.mean(grid[3:6, 3]))  # Far Right

        corridor_status = [
            c0_depth < OPEN_PATH_MAX,
            c1_depth < OPEN_PATH_MAX,
            c2_depth < OPEN_PATH_MAX,
            c3_depth < OPEN_PATH_MAX
        ]

        # Scan detected semantic boxes
        has_overhead_box = False
        has_ground_drop = False
        has_vehicle = False
        hazard_label = ""

        for (bx1, by1, bx2, by2, cls_id, conf, label, in_dz, in_oh) in detected_boxes:
            if in_oh and label in OVERHEAD_SET:
                has_overhead_box = True
                hazard_label = label
                break
            if in_dz:
                if label in GROUND_HAZARD_SET:
                    has_ground_drop = True
                    hazard_label = label
                    break
                elif label in VEHICLE_SET:
                    has_vehicle = True
                    hazard_label = "vehicle"
                    break

        # ========================================================
        # THREAT LEVEL 1: IMMEDIATE CRITICAL HAZARDS (1-word speech)
        # ========================================================
        # 1. Overhead Head-Strike
        if overhead_depth >= DANGER_THRESHOLD or has_overhead_box:
            return "DUCK!", "critical", corridor_status

        # 2. Drop-off (Stairs / Platform Edge / Manhole)
        if has_ground_drop:
            cue = "STAIRS!" if ("stair" in hazard_label or "step" in hazard_label) else "DROP AHEAD!"
            return cue, "critical", corridor_status

        # 3. Fast Looming Threat / Rapid Vehicle Approach
        if (has_vehicle and danger_depth >= WARNING_THRESHOLD) or (looming_rate > 0.8 and danger_depth >= WARNING_THRESHOLD):
            # Find fastest open escape corridor
            if c0_depth < c3_depth and c0_depth < DANGER_THRESHOLD:
                return "VEHICLE! STEP LEFT", "critical", corridor_status
            elif c3_depth < DANGER_THRESHOLD:
                return "VEHICLE! STEP RIGHT", "critical", corridor_status
            return "STOP! VEHICLE", "critical", corridor_status

        # 4. Immediate Walking Collision (Very Close)
        if danger_depth >= CRITICAL_THRESHOLD:
            if c0_depth < c3_depth and c0_depth < DANGER_THRESHOLD:
                return "RIGHT BLOCKED! GO LEFT", "critical", corridor_status
            elif c3_depth < DANGER_THRESHOLD:
                return "LEFT BLOCKED! GO RIGHT", "critical", corridor_status
            return "STOP!", "critical", corridor_status

        # ========================================================
        # THREAT LEVEL 2: CROWD STEERING & CORRIDOR SELECTION
        # ========================================================
        center_blocked = (c1_depth >= WARNING_THRESHOLD or c2_depth >= WARNING_THRESHOLD)

        if center_blocked:
            # Find best open corridor
            if c0_depth < c3_depth and c0_depth < DANGER_THRESHOLD:
                return "TURN LEFT", "warning", corridor_status
            elif c3_depth < DANGER_THRESHOLD:
                return "TURN RIGHT", "warning", corridor_status
            elif c0_depth < DANGER_THRESHOLD:
                return "TURN LEFT", "warning", corridor_status
            else:
                return "STOP! PATH BLOCKED", "critical", corridor_status

        # Path is clear or slightly crowded
        if c1_depth > OPEN_PATH_MAX and c3_depth < c0_depth:
            return "SLIGHT RIGHT", "info", corridor_status
        elif c2_depth > OPEN_PATH_MAX and c0_depth < c3_depth:
            return "SLIGHT LEFT", "info", corridor_status

        return "CLEAR", "info", corridor_status


# ===========================
# Audio Stream Filter (Zero-Chatter Debouncer)
# ===========================
class RushAudioFilter:
    def __init__(self):
        self.last_cmd = None
        self.last_time = 0.0
        self.clear_announced_time = 0.0

    def should_announce(self, cmd, alert_type):
        now = time.time()

        # CRITICAL commands interrupt instantly
        if alert_type == "critical":
            if cmd != self.last_cmd or (now - self.last_time) > 1.0:
                self.last_cmd = cmd
                self.last_time = now
                return True
            return False

        # WARNING commands (turns)
        if alert_type == "warning":
            if cmd != self.last_cmd:
                self.last_cmd = cmd
                self.last_time = now
                return True
            elif (now - self.last_time) > 2.2:
                self.last_time = now
                return True
            return False

        # CLEAR: Quiet in rush mode so blind person can listen to ambient sounds.
        # Only chime 'Clear' once after transitioning out of danger
        if cmd == "CLEAR":
            if self.last_cmd is not None and self.last_cmd != "CLEAR":
                self.last_cmd = "CLEAR"
                self.last_time = now
                return True
            return False

        # Standard Info
        if cmd != self.last_cmd and (now - self.last_time) > 2.0:
            self.last_cmd = cmd
            self.last_time = now
            return True

        return False


# ===========================
# Draw HUD & Visual Overlay
# ===========================
def draw_hud(frame, grid, corridors, cmd, alert_type):
    h, w = frame.shape[:2]
    cw = w // COLS
    ch = h // ROWS

    # Grid overlay
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

    # Highlight danger walking corridor
    for (row, col) in [(4,1),(4,2),(5,1),(5,2)]:
        x1, y1 = col*cw, row*ch
        cv2.rectangle(frame, (x1, y1), (x1+cw, y1+ch), (0, 0, 255), 2)

    # Top status banner
    banner_color = (0, 0, 255) if alert_type == "critical" else ((0, 160, 255) if alert_type == "warning" else (0, 200, 0))
    cv2.rectangle(frame, (0, 0), (w, 45), banner_color, -1)
    cv2.putText(frame, f"NAV: {cmd}", (15, 32), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2)

    # Bottom corridor indicator
    cor_text = "Corridors: " + " | ".join(["[L:OPEN]" if corridors[0] else "[L:BLOCKED]",
                                            "[CL:OPEN]" if corridors[1] else "[CL:BLOCKED]",
                                            "[CR:OPEN]" if corridors[2] else "[CR:BLOCKED]",
                                            "[R:OPEN]" if corridors[3] else "[R:BLOCKED]"])
    cv2.putText(frame, cor_text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return frame


# ===========================
# Main Vision Loop
# ===========================
def vision_loop(stream):
    navigator = RushNavigator()
    audio_filter = RushAudioFilter()
    frame_counter = 0

    YOLO_EVERY_N  = 2
    DEPTH_EVERY_N = 3

    while True:
        frame = stream.read()
        if frame is None:
            time.sleep(0.005)
            continue

        frame_counter += 1
        h, w = frame.shape[:2]
        if (w, h) != (FRAME_W, FRAME_H):
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        # Push to async queues
        if frame_counter % YOLO_EVERY_N == 0 and yolo_queue.empty():
            yolo_queue.put_nowait(frame)

        if frame_counter % DEPTH_EVERY_N == 0 and depth_queue.empty():
            depth_queue.put_nowait(frame)

        # Get latest async detections
        with boxes_lock:
            current_boxes = list(latest_boxes)

        with grid_lock:
            current_grid = None if latest_grid is None else latest_grid.copy()

        annotated = frame.copy()

        # Draw detected threat boxes
        for (x1, y1, x2, y2, cls_id, conf, label, in_dz, in_oh) in current_boxes:
            color = (0, 0, 255) if (in_dz or in_oh) else (0, 255, 0)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (int(x1), max(int(y1)-6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if current_grid is None:
            cv2.putText(annotated, "CALIBRATING RUSH RADAR...", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Blind Assist - Rush Mode", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Evaluate rush navigation
        cmd, alert_type, corridors = navigator.evaluate(current_grid, current_boxes)

        if audio_filter.should_announce(cmd, alert_type):
            print(f"[RUSH AUDIO] -> {cmd} ({alert_type})")
            update_command(cmd, alert_type)
            socketio.emit("corridors", corridors)
            socketio.emit("zones", current_grid.tolist())

        annotated = draw_hud(annotated, current_grid, corridors, cmd, alert_type)
        cv2.imshow("Blind Assist - Rush Mode", annotated)
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

    print("[INFO] Rush Assist Dashboard → http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, use_reloader=False)
