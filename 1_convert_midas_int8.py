"""
MiDaS small: FP32 -> INT8 conversion (OpenVINO + NNCF Post-Training Quantization)

Requirements:
    pip install openvino nncf torch torchvision timm opencv-python numpy

Folder layout:
    ./calibration_images/   <- 100-500+ frames from REAL walking footage
                               (indoor/outdoor, low light, glare, motion blur;
                                avoid near-duplicate consecutive frames)

Output:
    ./optimized_models/midas_small_fp32.onnx  (intermediate ONNX)
    ./optimized_models/midas_small_fp32.xml   (reference FP32 IR)
    ./optimized_models/midas_small_int8.xml   (quantized INT8 IR)

NOTE on conversion path:
    MiDaS small uses EfficientNet-Lite (geffnet) as its backbone, which
    computes SAME-style padding *dynamically at runtime*.  When OpenVINO
    tries to trace the model directly, it bakes input-size-specific
    constants and the tracer detects the mismatch and fails.
    Exporting to ONNX opset-11 first represents the padding symbolically,
    so the resulting OpenVINO IR is clean and input-size-agnostic.
"""

import os
import random
import time

import cv2
import numpy as np
import torch
import openvino as ov
import nncf

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
IMAGE_DIR   = "./calibration_images"
OUTPUT_DIR  = "./optimized_models"
INPUT_SIZE  = 256              # MiDaS small native size
SUBSET_SIZE = 300              # max calibration images NNCF will use
VAL_FRACTION = 0.2             # held out for validation, NOT used in calibration
PRESET      = nncf.QuantizationPreset.PERFORMANCE   # try MIXED if accuracy drops
SEED        = 42

# ---------------------------------------------------------------------
# Preprocessing (KEEP IDENTICAL to the one in 2_run_midas_int8.py)
# ---------------------------------------------------------------------
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr(frame_bgr):
    """BGR uint8 image -> float32 tensor of shape (1, 3, 256, 256)."""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(img, dtype=np.float32)


def load_and_preprocess(path):
    frame = cv2.imread(path)
    if frame is None:
        raise ValueError(f"Could not read image: {path}")
    return preprocess_bgr(frame)


# ---------------------------------------------------------------------
# 1. Load MiDaS small (FP32) and convert → ONNX → OpenVINO FP32 IR
# ---------------------------------------------------------------------
print("Loading MiDaS small from PyTorch Hub...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
midas.eval()

os.makedirs(OUTPUT_DIR, exist_ok=True)
onnx_path = os.path.join(OUTPUT_DIR, "midas_small_fp32.onnx")
dummy     = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

print("Step 1/4  Exporting PyTorch → ONNX (opset 11)...")
with torch.no_grad():
    torch.onnx.export(
        midas, dummy, onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
print(f"           ONNX saved: {onnx_path}")

print("Step 2/4  Converting ONNX → OpenVINO FP32 IR...")
ov_fp32   = ov.convert_model(onnx_path)
fp32_path = os.path.join(OUTPUT_DIR, "midas_small_fp32.xml")
ov.save_model(ov_fp32, fp32_path, compress_to_fp16=False)
print(f"           FP32 IR saved: {fp32_path}")

# ---------------------------------------------------------------------
# 2. Build calibration and validation sets (kept separate)
# ---------------------------------------------------------------------
image_paths = sorted(
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
)
print(f"\nFound {len(image_paths)} images in '{IMAGE_DIR}'")

if len(image_paths) < 20:
    raise ValueError("Too few images. Add real camera frames (aim for 300+).")
if len(image_paths) < 100:
    print("WARNING: fewer than 100 images. INT8 ranges may be poorly estimated.")

random.Random(SEED).shuffle(image_paths)
n_val       = max(5, int(len(image_paths) * VAL_FRACTION))
val_paths   = image_paths[:n_val]
calib_paths = image_paths[n_val:]
print(f"Calibration: {len(calib_paths)} images | Validation (held out): {len(val_paths)} images")

calibration_dataset = nncf.Dataset(calib_paths, load_and_preprocess)

# ---------------------------------------------------------------------
# 3. INT8 Post-Training Quantization
# ---------------------------------------------------------------------
print("\nStep 3/4  Quantizing to INT8 (this can take a few minutes)...")
ov_int8 = nncf.quantize(
    ov_fp32,
    calibration_dataset,
    preset=PRESET,
    subset_size=min(SUBSET_SIZE, len(calib_paths)),
)

int8_path = os.path.join(OUTPUT_DIR, "midas_small_int8.xml")
ov.save_model(ov_int8, int8_path)
print(f"           Saved INT8 model: {int8_path}")

# ---------------------------------------------------------------------
# 4. Validate INT8 vs FP32 on held-out images
# ---------------------------------------------------------------------
print("\nStep 4/4  Validating accuracy on held-out images...")
core   = ov.Core()
cfg    = {"PERFORMANCE_HINT": "LATENCY"}
fp32_c = core.compile_model(ov_fp32, "CPU", cfg)
int8_c = core.compile_model(ov_int8, "CPU", cfg)


def norm01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


corrs, maes = [], []
for p in val_paths:
    x = load_and_preprocess(p)
    a = fp32_c(x)[0].squeeze()
    b = int8_c(x)[0].squeeze()
    corrs.append(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    maes.append(np.abs(norm01(a) - norm01(b)).mean())

print("\n=== Accuracy (INT8 vs FP32, held-out images) ===")
print(f"Correlation  mean: {np.mean(corrs):.4f}   worst: {np.min(corrs):.4f}")
print(f"Normalized MAE mean: {np.mean(maes):.4f}   worst: {np.max(maes):.4f}")
if np.mean(corrs) < 0.98:
    print("Correlation below ~0.98: try PRESET = nncf.QuantizationPreset.MIXED, "
          "more/better calibration images, or nncf.quantize_with_accuracy_control.")

# ---------------------------------------------------------------------
# 5. Quick speed benchmark
# ---------------------------------------------------------------------
def bench(compiled, runs=100, warmup=10):
    x = load_and_preprocess(val_paths[0])
    for _ in range(warmup):
        compiled(x)
    t = time.perf_counter()
    for _ in range(runs):
        compiled(x)
    return (time.perf_counter() - t) / runs * 1000.0


t_fp32 = bench(fp32_c)
t_int8 = bench(int8_c)
print("\n=== Speed (single-frame latency, CPU) ===")
print(f"FP32: {t_fp32:.1f} ms  ({1000 / t_fp32:.1f} FPS)")
print(f"INT8: {t_int8:.1f} ms  ({1000 / t_int8:.1f} FPS)")
print(f"Speedup: {t_fp32 / t_int8:.2f}x")
print("\nDone. For a more rigorous benchmark, also run `benchmark_app` on both .xml files.")
