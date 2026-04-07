import cv2
import numpy as np
import onnxruntime as ort
import time
import os
from pathlib import Path

# ================= CONFIG =================
MODEL_PATHS = [
    r"C:\Users\Taboka\Documents\others\personal final project\updated_model\my_model.onnx",
    r"C:\Users\Taboka\Documents\others\personal final project\updated_model\my_model_v2.onnx",
]
CAMERA_SOURCE = 0
USE_GPU = True
FAILURE_DIR = "failure_cases"
RECORD_DIR = "recordings"
# =========================================

os.makedirs(FAILURE_DIR, exist_ok=True)
os.makedirs(RECORD_DIR, exist_ok=True)

# -------- Labels (Only for Display) --------
CLASSES = ["metal", "paper", "plastic", "plastic-bottle"]

# -------- Provider Auto-Detection --------
available_providers = ort.get_available_providers()
if USE_GPU and "CUDAExecutionProvider" in available_providers:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("✅ Using GPU (CUDAExecutionProvider)")
else:
    providers = ["CPUExecutionProvider"]
    print("⚠️ GPU not available, using CPU")

# -------- Model Loader --------
sessions = []
valid_model_paths = []
model_num_classes = []

for mp in MODEL_PATHS:
    print(f"🔎 Checking model path: {mp}")
    if not Path(mp).exists():
        print(f"⚠️ Model not found, skipping: {mp}")
        continue

    try:
        s = ort.InferenceSession(mp, providers=providers)
        out_shape = s.get_outputs()[0].shape
        num_classes = out_shape[-1] - 5  # YOLO format: 4 box + 1 obj + C classes

        sessions.append(s)
        valid_model_paths.append(mp)
        model_num_classes.append(num_classes)

        print(f"✅ Loaded model: {mp}")
        print(f"   → Model classes: {num_classes}")

    except Exception as e:
        print(f"❌ Failed to load model {mp}: {e}")

if not sessions:
    raise RuntimeError("❌ No valid ONNX models found.")

current_model_idx = 0
session = sessions[current_model_idx]
NUM_MODEL_CLASSES = model_num_classes[current_model_idx]

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
INPUT_H, INPUT_W = input_shape[2], input_shape[3]

# -------- UI State --------
ui = {"conf": 25, "iou": 45, "paused": False, "show_labels": 1, "use_nms": 1}

# -------- Trackbars --------
def nothing(x): pass

cv2.namedWindow("YOLOv11 Tester", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv11 Tester", 1300, 750)
cv2.createTrackbar("Confidence %", "YOLOv11 Tester", ui["conf"], 100, nothing)
cv2.createTrackbar("IoU %", "YOLOv11 Tester", ui["iou"], 100, nothing)
cv2.createTrackbar("Show Labels", "YOLOv11 Tester", ui["show_labels"], 1, nothing)
cv2.createTrackbar("Use NMS", "YOLOv11 Tester", ui["use_nms"], 1, nothing)

# -------- Preprocess --------
def preprocess(frame, size):
    img = cv2.resize(frame, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img

def nms(boxes, scores, conf, iou):
    if len(boxes) == 0:
        return []
    return cv2.dnn.NMSBoxes(boxes, scores, conf, iou)

# -------- Video --------
cap = cv2.VideoCapture(CAMERA_SOURCE)
writer = None
recording = False
prev_time = time.time()

print("\nControls:")
print("  1/2/3   = switch models")
print("  SPACE   = pause")
print("  F       = save failure case")
print("  R       = start/stop recording")
print("  ESC     = quit")

# -------- Main Loop --------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    conf = cv2.getTrackbarPos("Confidence %", "YOLOv11 Tester") / 100
    iou = cv2.getTrackbarPos("IoU %", "YOLOv11 Tester") / 100
    show_labels = cv2.getTrackbarPos("Show Labels", "YOLOv11 Tester")
    use_nms = cv2.getTrackbarPos("Use NMS", "YOLOv11 Tester")

    h, w = frame.shape[:2]
    inp = preprocess(frame, INPUT_W)

    outputs = session.run(None, {input_name: inp})[0][0]

    boxes, scores, class_ids = [], [], []

    for det in outputs:
        obj = det[4]
        cls_scores = det[5:]
        cls_id = int(np.argmax(cls_scores))
        score = float(obj * cls_scores[cls_id])

        # 🔒 SAFE GUARD
        if cls_id >= len(CLASSES):
            continue

        if score > conf:
            cx, cy, bw, bh = det[:4]
            x = int((cx - bw / 2) * w)
            y = int((cy - bh / 2) * h)
            boxes.append([x, y, int(bw * w), int(bh * h)])
            scores.append(score)
            class_ids.append(cls_id)

    indices = nms(boxes, scores, conf, iou) if use_nms else np.arange(len(boxes)).reshape(-1, 1)

    for i in indices.flatten() if len(indices) else []:
        x, y, bw, bh = boxes[i]
        label = CLASSES[class_ids[i]]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        if show_labels:
            cv2.putText(frame, f"{label} {scores[i]:.2f}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    fps = 1 / max(1e-6, (time.time() - prev_time))
    prev_time = time.time()

    cv2.putText(frame, f"FPS: {fps:.1f} | Model {current_model_idx+1}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("YOLOv11 Tester", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    elif key in [ord("1"), ord("2"), ord("3")]:
        idx = int(chr(key)) - 1
        if idx < len(sessions):
            current_model_idx = idx
            session = sessions[current_model_idx]
            print(f"🔁 Switched to model {current_model_idx + 1}")

cap.release()
cv2.destroyAllWindows()