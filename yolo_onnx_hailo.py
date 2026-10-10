import os
import cv2
import numpy as np
from hailo_sdk_client import ClientRunner

ONNX = "yolov4_1_3_608_608_static.onnx"
OUTPUT = "yolov4_1_3_608_608_static_hailo8.hef"
CALIB_DIR = "/home/anaim/pytorch-YOLOv4/calibration_images/validation/images/"

runner = ClientRunner(hw_arch="hailo8")

# 1. Parse ONNX, cutting before YOLO decoding
runner.translate_onnx_model(
    ONNX,
    "yolov4_608",
    start_node_names=["input"],
    end_node_names=[
        "/head/conv2/conv.0/Conv",
        "/head/conv10/conv.0/Conv",
        "/head/conv18/conv.0/Conv",
    ],
    net_input_shapes={"input": [1, 3, 608, 608]},
)

runner.save_har("yolov4_608_parsed.har")

# 2. Representative calibration images
image_files = [
    os.path.join(CALIB_DIR, f)
    for f in os.listdir(CALIB_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not image_files:
    raise RuntimeError("No calibration images found")

calib = []

for path in image_files[:500]:
    img = cv2.imread(path)
    if img is None:
        continue

    img = cv2.resize(img, (608, 608))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Hailo calibration input is normally NHWC.
    calib.append(img.astype(np.float32))

calib = np.asarray(calib, dtype=np.float32)

print("Calibration set:", calib.shape)

# 3. Quantize / optimize FP32 -> INT8
runner.optimize(calib)
runner.save_har("yolov4_608_quantized.har")

# 4. Compile for Hailo-8
hef = runner.compile()

with open(OUTPUT, "wb") as f:
    f.write(hef)

print("Created:", OUTPUT)