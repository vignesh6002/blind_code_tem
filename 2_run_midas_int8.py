"""
Run the INT8 MiDaS small model on a live camera / IP camera stream.

Requirements:
    pip install openvino opencv-python numpy

Run:
    python 2_run_midas_int8.py

Notes:
  * MiDaS outputs RELATIVE inverse depth (bigger = closer). It is NOT metric
    distance, so the "nearness" values below are for relative comparison only.
  * Depth changes slowly, so it only runs every DEPTH_EVERY_N frames.
    Your YOLO model can still run on every frame.
"""

import time

import cv2
import numpy as np
import openvino as ov

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
MODEL_PATH = "./optimized_models/midas_small_int8.xml"
CAMERA_SOURCE = 0            # 0 = webcam, or "http://<ip>:<port>/video" for an IP camera
INPUT_SIZE = 256
DEPTH_EVERY_N = 2            # run depth every Nth frame (1 = every frame)
SHOW_WINDOW = True

# ---------------------------------------------------------------------
# Preprocessing (KEEP IDENTICAL to the one in 1_convert_midas_int8.py)
# ---------------------------------------------------------------------
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr(frame_bgr):
    """BGR uint8 image -> float32 tensor of shape (1, 3, 256, 256)."""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(img, dtype=np.float32)


# ---------------------------------------------------------------------
# Load the INT8 model
# ---------------------------------------------------------------------
core = ov.Core()
model = core.read_model(MODEL_PATH)
compiled = core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
output_layer = compiled.output(0)


def get_depth(frame_bgr):
    """Returns a 256x256 relative inverse-depth map (float32)."""
    x = preprocess_bgr(frame_bgr)
    out = compiled(x)[output_layer]
    return out.squeeze()          # (256, 256)


def normalize01(depth):
    """Per-frame scale to 0..1 (1 = nearest). Relative values only."""
    return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)


def zone_nearness(depth01):
    """
    Average nearness in the lower half of the view (where obstacles that
    matter for walking appear), split into left / center / right.
    """
    h, w = depth01.shape
    lower = depth01[h // 2:, :]
    third = w // 3
    return {
        "left": float(lower[:, :third].mean()),
        "center": float(lower[:, third:2 * third].mean()),
        "right": float(lower[:, 2 * third:].mean()),
    }


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera source: {CAMERA_SOURCE}")

frame_idx = 0
last_depth01 = None
last_zones = None
fps_t = time.perf_counter()
fps = 0.0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed, stopping.")
            break

        # ---- Depth (every Nth frame, reuse last result otherwise) ----
        if frame_idx % DEPTH_EVERY_N == 0:
            depth = get_depth(frame)
            last_depth01 = normalize01(depth)
            last_zones = zone_nearness(last_depth01)

        # ---- TODO: run YOLO on `frame` here (every frame) and combine ----
        # Example idea: for each detected box, average last_depth01 inside the
        # box (after resizing the box coordinates to 256x256) to rank how
        # close each object is relative to the others.

        # ---- FPS counter ----
        frame_idx += 1
        if frame_idx % 10 == 0:
            now = time.perf_counter()
            fps = 10.0 / (now - fps_t)
            fps_t = now

        # ---- Display ----
        if SHOW_WINDOW and last_depth01 is not None:
            depth_vis = cv2.resize(
                (last_depth01 * 255).astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
            )
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
            view = np.hstack([frame, depth_vis])

            z = last_zones
            text = f"FPS {fps:.1f} | L {z['left']:.2f}  C {z['center']:.2f}  R {z['right']:.2f}"
            cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.imshow("RGB | Depth (INT8)", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    cap.release()
    cv2.destroyAllWindows()
