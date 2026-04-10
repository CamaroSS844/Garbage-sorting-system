import asyncio
import cv2
import numpy as np
import uuid
import shutil
from pathlib import Path
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import threading
from queue import Queue
import time
import math
from collections import OrderedDict
import queue
import sqlite3
import aiosqlite

# --- Inference libraries ---
import torch
import onnxruntime as ort
from ultralytics import YOLO

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceHTTPClient
from dataclasses import dataclass, asdict, field

# ------------------------------
# Config
# ------------------------------

ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_API_KEY = "kKM7L2QWjyHfhTodDavs"
WORKSPACE_NAME = "trash-sorting-system-lxi4a"
WORKFLOW_ID = "detect-count-and-visualize"

UPLOAD_DIR = Path("./uploads")
FAILED_DIR = Path("./failed")
UPLOAD_DIR.mkdir(exist_ok=True)
FAILED_DIR.mkdir(exist_ok=True)

DB_PATH = Path("./ecosort.db")

HEARTBEAT_TIMEOUT = timedelta(seconds=10)
CONTROL_DEADMAN_TIMEOUT = timedelta(milliseconds=5000)

DEFAULT_CAMERA_URL = "http://192.168.43.133:9000/video"

LOCAL_INFERENCE_FPS = 10
CLOUD_INFERENCE_INTERVAL = 30

LOCAL_MODEL_PATH = r"C:\Users\Taboka\Documents\others\personal final project\withOnnx\latest.pt"
LOCAL_MODEL_TYPE = "ultralytics"

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
TRACKING_ENABLED = False
TRACKER_MAX_DISAPPEARED = 30
TRACKER_MAX_DISTANCE = 50

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# How often (in seconds) to flush an aggregated stats snapshot to DB.
# Keeps DB writes infrequent so they don't affect inference latency.
STATS_FLUSH_INTERVAL = 30  # seconds

TRIGGER_ZONES = [
    {
        "id": "zone_a",
        "label": "Zone A",
        "actuator_id": "actuator_1",
        "x_position": 180,
        "color": "#10b981",
        "cooldown_frames": 45,
    },
    {
        "id": "zone_b",
        "label": "Zone B",
        "actuator_id": "actuator_2",
        "x_position": 350,
        "color": "#3b82f6",
        "cooldown_frames": 45,
    },
    {
        "id": "zone_c",
        "label": "Zone C",
        "actuator_id": "actuator_3",
        "x_position": 520,
        "color": "#f59e0b",
        "cooldown_frames": 45,
    },
]

# ------------------------------
# App Init
# ------------------------------

app = FastAPI(title="EcoSort Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

capture_queue: queue.Queue = queue.Queue(maxsize=1)
result_queue_async: asyncio.Queue = None

capture_thread: Optional[threading.Thread] = None
inference_thread: Optional[threading.Thread] = None
pipeline_running = False

main_event_loop: Optional[asyncio.AbstractEventLoop] = None

# ------------------------------
# In-memory stats accumulator
# (written by inference thread, flushed to DB periodically by async task)
# ------------------------------

class StatsAccumulator:
    """
    Thread-safe counters incremented during inference.
    Periodically snapshotted and written to DB without blocking inference.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.total_processed: int = 0
        self.total_failures: int = 0

    def record_detection(self, count: int = 1):
        with self._lock:
            self.total_processed += count

    def record_failure(self, count: int = 1):
        with self._lock:
            self.total_failures += count

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "total_processed": self.total_processed,
                "total_failures": self.total_failures,
            }


stats_accumulator = StatsAccumulator()

# ------------------------------
# Data Classes
# ------------------------------

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str

@dataclass
class TrackedObject(Detection):
    track_id: int = -1

# ------------------------------
# Database helpers
# ------------------------------

async def init_db():
    """Create tables and seed demo failed_inference rows if empty."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS inference_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       REAL    NOT NULL,
                class_name      TEXT,
                confidence      REAL,
                zone_id         TEXT,
                actuator_id     TEXT,
                is_failure      INTEGER DEFAULT 0,
                image_path      TEXT,
                device_id       TEXT,
                inference_time_ms REAL
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS failed_inferences (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp           REAL    NOT NULL,
                image_path          TEXT,
                device_id           TEXT,
                confidence          REAL,
                original_guess      TEXT,
                assigned_category   TEXT,
                reviewed            INTEGER DEFAULT 0,
                notes               TEXT
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS system_stats (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        REAL    NOT NULL,
                total_processed  INTEGER DEFAULT 0,
                total_failures   INTEGER DEFAULT 0,
                online_devices   INTEGER DEFAULT 0,
                accuracy_rate    REAL    DEFAULT 0.0
            )
        """)

        await db.commit()

        # Seed 3 demo failed inference rows only if table is empty
        cursor = await db.execute("SELECT COUNT(*) FROM failed_inferences")
        row = await cursor.fetchone()
        if row[0] == 0:
            demo_rows = [
                (
                    time.time() - 3600,          # 1 hour ago
                    None,                         # image_path
                    "esp32-node-01",              # device_id
                    0.21,                         # confidence
                    "Plastic",                    # original_guess
                    None,                         # assigned_category (not reviewed yet)
                    0,                            # reviewed
                    "Low confidence on conveyor belt edge",  # notes
                ),
                (
                    time.time() - 1800,
                    None,
                    "esp32-node-02",
                    0.18,
                    "Metal",
                    None,
                    0,
                    "Blurry frame during high-speed pass",
                ),
                (
                    time.time() - 600,
                    None,
                    "esp32-node-01",
                    0.09,
                    "Glass",
                    None,
                    0,
                    "Occlusion — item partially behind belt guide",
                ),
            ]
            await db.executemany(
                """INSERT INTO failed_inferences
                   (timestamp, image_path, device_id, confidence,
                    original_guess, assigned_category, reviewed, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                demo_rows,
            )
            await db.commit()
            print("[db] Seeded 3 demo failed inference rows")


async def log_zone_event(event: Dict[str, Any], inference_time_ms: float = 0.0):
    """Write a single zone-fire detection to inference_logs (non-blocking)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO inference_logs
               (timestamp, class_name, confidence, zone_id, actuator_id,
                is_failure, image_path, device_id, inference_time_ms)
               VALUES (?, ?, ?, ?, ?, 0, NULL, NULL, ?)""",
            (
                event.get("timestamp", time.time()),
                event.get("class_name"),
                None,
                event.get("zone_id"),
                event.get("actuator_id"),
                inference_time_ms,
            ),
        )
        await db.commit()


async def flush_stats_to_db():
    """Snapshot in-memory counters to system_stats table."""
    snap = stats_accumulator.snapshot()
    total = snap["total_processed"]
    failures = snap["total_failures"]
    accuracy = ((total - failures) / total * 100) if total > 0 else 0.0
    online = len([d for d, t in devices.items()
                  if utc_now() - t < HEARTBEAT_TIMEOUT])

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO system_stats
               (timestamp, total_processed, total_failures, online_devices, accuracy_rate)
               VALUES (?, ?, ?, ?, ?)""",
            (time.time(), total, failures, online, accuracy),
        )
        await db.commit()


# ------------------------------
# Zone Trigger State
# ------------------------------

class ZoneTriggerManager:
    def __init__(self):
        self.fired_pairs: set = set()
        self.zone_last_fired: Dict[str, int] = {}
        self.frame_counter: int = 0
        self.pending_events: List[Dict] = []

    def reset(self):
        self.fired_pairs.clear()
        self.zone_last_fired.clear()
        self.frame_counter = 0
        self.pending_events.clear()

    def process(self, tracked_objects: List[Any]) -> List[Dict]:
        self.frame_counter += 1
        events = []

        for obj in tracked_objects:
            if isinstance(obj, dict):
                bbox = obj.get("bbox", [0, 0, 0, 0])
                track_id = obj.get("track_id", None)
                class_name = obj.get("class_name") or obj.get("class", "Unknown")
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            else:
                x1, y1, x2, y2 = obj.bbox
                track_id = getattr(obj, "track_id", None)
                class_name = obj.class_name

            leading_edge = x2

            for zone in TRIGGER_ZONES:
                zone_id = zone["id"]
                cooldown = zone["cooldown_frames"]

                pair_key = (track_id, zone_id)
                if track_id is not None and pair_key in self.fired_pairs:
                    continue

                last_fired = self.zone_last_fired.get(zone_id, -9999)
                if self.frame_counter - last_fired < cooldown:
                    continue

                if leading_edge >= zone["x_position"]:
                    if track_id is not None:
                        self.fired_pairs.add(pair_key)
                    self.zone_last_fired[zone_id] = self.frame_counter

                    event = {
                        "zone_id": zone_id,
                        "zone_label": zone["label"],
                        "actuator_id": zone["actuator_id"],
                        "class_name": class_name,
                        "track_id": track_id,
                        "x_position": zone["x_position"],
                        "timestamp": time.time(),
                    }
                    events.append(event)

                    print(
                        f"[zone] {class_name} (track #{track_id}) "
                        f"→ {zone['label']} ({zone['actuator_id']}) FIRED "
                        f"[frame {self.frame_counter}]"
                    )

        return events


zone_trigger_manager = ZoneTriggerManager()

# ------------------------------
# Centroid Tracker
# ------------------------------

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_id = 0
        self.objects = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, class_name, confidence):
        self.objects[self.next_id] = (centroid, bbox, class_name, confidence, 0)
        self.next_id += 1

    def deregister(self, track_id):
        del self.objects[track_id]

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        if len(detections) == 0:
            for track_id in list(self.objects.keys()):
                centroid, bbox, class_name, confidence, disappeared = self.objects[track_id]
                disappeared += 1
                if disappeared > self.max_disappeared:
                    self.deregister(track_id)
                else:
                    self.objects[track_id] = (centroid, bbox, class_name, confidence, disappeared)
            return []

        detection_centroids = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            detection_centroids.append((centroid, det))

        if len(self.objects) == 0:
            for centroid, det in detection_centroids:
                self.register(centroid, det.bbox, det.class_name, det.confidence)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[oid][0] for oid in object_ids]

            D = np.zeros((len(object_centroids), len(detection_centroids)))
            for i, oc in enumerate(object_centroids):
                for j, (dc, _) in enumerate(detection_centroids):
                    D[i, j] = math.hypot(oc[0] - dc[0], oc[1] - dc[1])

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)

            used_rows = set()
            used_cols = set()

            for row in rows:
                if row in used_rows:
                    continue
                col = cols[row]
                if col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                centroid, det = detection_centroids[col]
                track_id = object_ids[row]
                self.objects[track_id] = (centroid, det.bbox, det.class_name, det.confidence, 0)
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(len(object_centroids))) - used_rows
            for row in unused_rows:
                track_id = object_ids[row]
                centroid, bbox, class_name, confidence, disappeared = self.objects[track_id]
                disappeared += 1
                if disappeared > self.max_disappeared:
                    self.deregister(track_id)
                else:
                    self.objects[track_id] = (centroid, bbox, class_name, confidence, disappeared)

            unused_cols = set(range(len(detection_centroids))) - used_cols
            for col in unused_cols:
                centroid, det = detection_centroids[col]
                self.register(centroid, det.bbox, det.class_name, det.confidence)

        tracked = []
        for track_id, (centroid, bbox, class_name, confidence, _) in self.objects.items():
            x1, y1, x2, y2 = bbox
            tracked.append(TrackedObject(
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                class_id=-1,
                class_name=class_name,
                track_id=track_id
            ))
        return tracked


# ------------------------------
# Class Names Loader
# ------------------------------

class ClassNamesLoader:
    def __init__(self, model_path: str):
        model_path = Path(model_path)
        self.classes_file = model_path.with_name("classes.txt")
        if not self.classes_file.exists():
            raise FileNotFoundError(f"classes.txt not found next to model: {self.classes_file}")
        self.class_names = self.load_classes()

    def load_classes(self) -> List[str]:
        with open(self.classes_file, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        if len(classes) == 0:
            raise ValueError("classes.txt is empty")
        return classes

    def get(self) -> List[str]:
        return self.class_names


# ------------------------------
# Model Wrappers
# ------------------------------

class UltralyticsModel:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def infer(self, frame: np.ndarray, conf_thresh: float, iou_thresh: float):
        start = time.time()
        results = self.model(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)
        inference_time = (time.time() - start) * 1000
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append(Detection((x1, y1, x2, y2), conf, cls_id, self.class_names[cls_id]))
        return detections, inference_time


class ONNXModel:
    def __init__(self, model_path: str):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        loader = ClassNamesLoader(model_path)
        self.class_names = loader.get()
        print(f"Loaded {len(self.class_names)} classes from classes.txt")

    def infer(self, frame: np.ndarray, conf_thresh: float, iou_thresh: float):
        start = time.time()
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        outputs = self.session.run(None, {self.input_name: img})
        inference_time = (time.time() - start) * 1000
        detections = []
        preds = outputs[0][0]
        for pred in preds:
            obj_conf = pred[4]
            if obj_conf < conf_thresh:
                continue
            class_scores = pred[5:]
            class_id = int(np.argmax(class_scores))
            class_conf = class_scores[class_id]
            score = obj_conf * class_conf
            if score < conf_thresh:
                continue
            cx, cy, w, h = pred[:4]
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            detections.append(Detection((x1, y1, x2, y2), float(score), class_id, self.class_names[class_id]))
        return detections, inference_time


class LocalInference:
    def __init__(self, model_path: str, model_type: str = "ultralytics"):
        self.model_type = model_type
        if model_type == "ultralytics":
            self.model = UltralyticsModel(model_path)
        elif model_type == "onnx":
            self.model = ONNXModel(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class CloudInference:
    def __init__(self):
        self.client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)

    def infer_sync(self, frame: np.ndarray) -> List[Dict]:
        file_id = f"{uuid.uuid4()}.jpg"
        file_path = UPLOAD_DIR / file_id
        cv2.imwrite(str(file_path), frame)
        try:
            result = self.client.run_workflow(
                workspace_name=WORKSPACE_NAME,
                workflow_id=WORKFLOW_ID,
                images={"image": str(file_path)}
            )
            return self.parse_roboflow_result(result)
        finally:
            if file_path.exists():
                file_path.unlink()

    def parse_roboflow_result(self, result) -> List[Dict]:
        detections = []
        if isinstance(result, list) and len(result) > 0:
            predictions = result[0].get("predictions", [])
            for pred in predictions:
                detections.append({
                    "bbox": [pred["x"], pred["y"], pred["width"], pred["height"]],
                    "class": pred["class"],
                    "class_name": pred["class"],
                    "confidence": pred["confidence"]
                })
        return detections


# ------------------------------
# Global State
# ------------------------------

class InferenceMode(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"

class SystemMode(str, Enum):
    RUN = "RUN"
    TEST = "TEST"
    FAULT = "FAULT"

current_mode = InferenceMode.LOCAL
camera_url = DEFAULT_CAMERA_URL
local_model: Optional[LocalInference] = None
cloud_model = CloudInference()

conf_threshold = CONF_THRESHOLD
iou_threshold = IOU_THRESHOLD
tracking_enabled = TRACKING_ENABLED
tracker: Optional[CentroidTracker] = None

camera_connected = False
camera_lock = threading.Lock()

system_state = {"mode": SystemMode.RUN, "fault": None, "last_control": None}
devices: Dict[str, datetime] = {}
latest_control_command: Dict[str, Any] = {
    "arm": {"azimuth": 0.0, "elevation": 0.0},
    "conveyor": 1.0,
    "vacuum": False
}

capture_task: Optional[asyncio.Task] = None
stop_capture_event = asyncio.Event()
frame_queue: Queue = Queue(maxsize=1)

rf_client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)

# ------------------------------
# WebSocket Manager
# ------------------------------

class WSManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, message: dict):
        for ws in list(self.connections):
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(ws)

status_ws = WSManager()
control_ws = WSManager()
inference_ws = WSManager()

# ------------------------------
# Utility
# ------------------------------

def utc_now():
    return datetime.now(timezone.utc)

def clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min(value, max_v), min_v)

def require_test_mode():
    if system_state["mode"] != SystemMode.TEST:
        raise HTTPException(403, "System is not in TEST mode")

# =========================================================
# PIPELINE THREADS
# =========================================================

def capture_loop():
    global pipeline_running, camera_connected

    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    with camera_lock:
        camera_connected = cap.isOpened()

    if not cap.isOpened():
        print(f"[capture] ERROR: Cannot open camera {camera_url}")
        return

    print("[capture] Camera opened OK")

    while pipeline_running:
        ret, frame = cap.read()

        if not ret:
            with camera_lock:
                camera_connected = False
            time.sleep(0.05)
            continue

        with camera_lock:
            camera_connected = True

        if capture_queue.full():
            try:
                capture_queue.get_nowait()
            except Exception:
                pass

        capture_queue.put(frame)

    cap.release()
    with camera_lock:
        camera_connected = False
    print("[capture] Stopped")


def inference_loop_thread():
    global pipeline_running
    last_cloud_time = 0

    while pipeline_running:
        if capture_queue.empty():
            time.sleep(0.001)
            continue

        frame = capture_queue.get()

        message: Dict[str, Any] = {
            "timestamp": time.time(),
            "mode": current_mode.value,
        }

        # ---- LOCAL ----
        if current_mode == InferenceMode.LOCAL and local_model:
            try:
                detections, inference_time = local_model.model.infer(
                    frame, conf_threshold, iou_threshold
                )

                # Count detections for stats
                if detections:
                    stats_accumulator.record_detection(len(detections))

                if tracking_enabled and tracker:
                    tracked = tracker.update(detections)
                    zone_events = zone_trigger_manager.process(tracked)
                    message["type"] = "tracked"
                    message["objects"] = [asdict(t) for t in tracked]
                else:
                    zone_events = zone_trigger_manager.process(detections)
                    message["type"] = "detection"
                    message["objects"] = [asdict(d) for d in detections]

                message["inference_time_ms"] = inference_time
                message["zone_events"] = zone_events

                # Post zone events to DB via event loop (fire-and-forget, non-blocking)
                if zone_events and main_event_loop is not None:
                    for ev in zone_events:
                        asyncio.run_coroutine_threadsafe(
                            log_zone_event(ev, inference_time),
                            main_event_loop
                        )

            except Exception as e:
                print(f"[inference] Local model error: {e}")
                stats_accumulator.record_failure()
                continue

        # ---- CLOUD ----
        elif current_mode == InferenceMode.CLOUD:
            now = time.time()
            if now - last_cloud_time >= CLOUD_INFERENCE_INTERVAL:
                try:
                    detections = cloud_model.infer_sync(frame)
                    last_cloud_time = now

                    if detections:
                        stats_accumulator.record_detection(len(detections))

                    zone_events = zone_trigger_manager.process(detections)
                    message["type"] = "detection"
                    message["objects"] = detections
                    message["zone_events"] = zone_events

                    if zone_events and main_event_loop is not None:
                        for ev in zone_events:
                            asyncio.run_coroutine_threadsafe(
                                log_zone_event(ev, 0.0),
                                main_event_loop
                            )
                except Exception as e:
                    print(f"[inference] Cloud error: {e}")
                    stats_accumulator.record_failure()
                    continue
            else:
                continue

        else:
            continue

        if main_event_loop is not None and result_queue_async is not None:
            if result_queue_async.full():
                try:
                    result_queue_async.get_nowait()
                except Exception:
                    pass
            asyncio.run_coroutine_threadsafe(
                result_queue_async.put(message),
                main_event_loop
            )


# =========================================================
# BROADCAST LOOP
# =========================================================

async def broadcast_loop():
    print("[broadcast] Loop started")
    while pipeline_running:
        try:
            msg = await asyncio.wait_for(result_queue_async.get(), timeout=0.1)
            await inference_ws.broadcast(msg)
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            print(f"[broadcast] Error: {e}")
            await asyncio.sleep(0.01)
    print("[broadcast] Loop stopped")


# =========================================================
# PIPELINE START / STOP
# =========================================================

@app.post("/inference/start")
async def start_pipeline():
    global capture_thread, inference_thread, pipeline_running

    if pipeline_running:
        return {"status": "already running"}

    if local_model is None:
        try:
            await load_local_model()
        except Exception as e:
            raise HTTPException(500, f"Model load failed: {e}")

    zone_trigger_manager.reset()

    pipeline_running = True

    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    inference_thread = threading.Thread(target=inference_loop_thread, daemon=True)

    capture_thread.start()
    inference_thread.start()

    asyncio.create_task(broadcast_loop())

    return {"status": "started"}


@app.post("/inference/stop")
async def stop_pipeline():
    global pipeline_running

    if not pipeline_running:
        return {"status": "not running"}

    pipeline_running = False

    if capture_thread and capture_thread.is_alive():
        capture_thread.join(timeout=2.0)
    if inference_thread and inference_thread.is_alive():
        inference_thread.join(timeout=2.0)

    return {"status": "stopped"}


# =========================================================
# ZONE ENDPOINTS
# =========================================================

@app.get("/zones")
async def get_zones():
    return {"zones": TRIGGER_ZONES}

@app.post("/zones/reset")
async def reset_zones():
    zone_trigger_manager.reset()
    return {"status": "zone trigger state cleared"}


# =========================================================
# INFERENCE CONTROL ENDPOINTS
# =========================================================

@app.post("/inference/set_camera")
async def set_camera(url: str):
    global camera_url
    camera_url = url
    return {"status": "ok", "camera_url": camera_url}

@app.get("/inference/get_camera")
async def get_camera():
    return {"camera_url": camera_url}

@app.get("/inference/camera_status")
async def get_camera_status():
    with camera_lock:
        return {"connected": camera_connected}

@app.post("/inference/set_mode/{mode}")
async def set_inference_mode(mode: InferenceMode):
    global current_mode
    current_mode = mode
    return {"mode": current_mode}

@app.get("/inference/get_mode")
async def get_inference_mode():
    return {"mode": current_mode}

@app.post("/inference/load_local_model")
async def load_local_model(model_path: str = LOCAL_MODEL_PATH, model_type: str = LOCAL_MODEL_TYPE):
    global local_model
    try:
        loop = asyncio.get_event_loop()
        local_model = await loop.run_in_executor(
            None,
            lambda: LocalInference(model_path, model_type)
        )
        return {"status": "loaded", "model_path": model_path, "type": model_type}
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")

@app.post("/inference/set_thresholds")
async def set_thresholds(conf: float = CONF_THRESHOLD, iou: float = IOU_THRESHOLD):
    global conf_threshold, iou_threshold
    conf_threshold = clamp(conf, 0.01, 1.0)
    iou_threshold = clamp(iou, 0.01, 1.0)
    return {"conf": conf_threshold, "iou": iou_threshold}

@app.get("/inference/get_thresholds")
async def get_thresholds():
    return {"conf": conf_threshold, "iou": iou_threshold}

@app.post("/inference/tracking")
async def set_tracking(enable: bool, max_disappeared: int = 30, max_distance: int = 50):
    global tracking_enabled, tracker
    tracking_enabled = enable
    tracker = CentroidTracker(max_disappeared=max_disappeared, max_distance=max_distance) if enable else None
    return {"tracking_enabled": tracking_enabled}

@app.post("/inference/tracking/reset")
async def reset_tracker():
    global tracker
    if not tracking_enabled:
        raise HTTPException(400, "Tracking is not enabled")
    tracker = CentroidTracker(max_disappeared=TRACKER_MAX_DISAPPEARED, max_distance=TRACKER_MAX_DISTANCE)
    zone_trigger_manager.reset()
    return {"status": "tracker and zone state reset"}

@app.post("/inference/set_cloud_interval")
async def set_cloud_interval(seconds: int):
    global CLOUD_INFERENCE_INTERVAL
    if seconds < 1:
        raise HTTPException(400, "Interval must be at least 1 second")
    CLOUD_INFERENCE_INTERVAL = seconds
    return {"interval": CLOUD_INFERENCE_INTERVAL}


# =========================================================
# REPORTS ENDPOINTS
# =========================================================

@app.get("/reports")
async def get_reports():
    """
    Returns the latest system_stats row plus live in-memory counters merged together.
    The in-memory counters are always more current than the DB snapshot.
    """
    snap = stats_accumulator.snapshot()
    total = snap["total_processed"]
    failures = snap["total_failures"]
    online = len([d for d, t in devices.items() if utc_now() - t < HEARTBEAT_TIMEOUT])
    accuracy = ((total - failures) / total * 100) if total > 0 else 0.0

    return {
        "total_items_processed": total,
        "failed_inferences": failures,
        "online_devices": online,
        "accuracy_rate": round(accuracy, 2),
    }

@app.get("/reports/history")
async def get_reports_history(limit: int = 50):
    """Returns the last N system_stats snapshots for trend graphs."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM system_stats ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
    return {"history": [dict(r) for r in reversed(rows)]}


# =========================================================
# FAILED INFERENCES ENDPOINTS
# =========================================================

@app.get("/failed-inferences")
async def get_failed_inferences(limit: int = 100, reviewed: Optional[int] = None):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if reviewed is not None:
            cursor = await db.execute(
                "SELECT * FROM failed_inferences WHERE reviewed=? ORDER BY timestamp DESC LIMIT ?",
                (reviewed, limit)
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM failed_inferences ORDER BY timestamp DESC LIMIT ?", (limit,)
            )
        rows = await cursor.fetchall()
    return {"items": [dict(r) for r in rows]}


@app.patch("/failed-inferences/{item_id}/review")
async def review_failed_inference(item_id: int, assigned_category: str, notes: str = ""):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE failed_inferences
               SET assigned_category=?, reviewed=1, notes=?
               WHERE id=?""",
            (assigned_category, notes, item_id)
        )
        await db.commit()
        changes = db.total_changes
    if changes == 0:
        raise HTTPException(404, "Item not found")
    return {"status": "reviewed", "id": item_id, "assigned_category": assigned_category}


@app.post("/failed-inferences/{item_id}/retry")
async def retry_failed_inference(item_id: int):
    """Reset a reviewed item back to unreviewed so it can be re-labelled."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE failed_inferences SET reviewed=0, assigned_category=NULL WHERE id=?",
            (item_id,)
        )
        await db.commit()
    return {"status": "reset", "id": item_id}


# =========================================================
# DATABASE CLEAR ENDPOINTS
# =========================================================

class ClearLevel(str, Enum):
    STATS_ONLY = "stats_only"          # Clear system_stats history only
    FAILED_ONLY = "failed_only"        # Clear failed_inferences only
    LOGS_ONLY = "logs_only"            # Clear inference_logs only
    ALL = "all"                        # Nuclear — clears everything


@app.delete("/database/clear")
async def clear_database(level: ClearLevel, confirm: str = ""):
    """
    Clear database records at the specified level.
    Requires confirm="CONFIRM" to prevent accidental wipes.
    """
    if confirm != "CONFIRM":
        raise HTTPException(
            400,
            "Pass confirm=CONFIRM in query params to proceed. This action is irreversible."
        )

    cleared = []
    async with aiosqlite.connect(DB_PATH) as db:
        if level in (ClearLevel.STATS_ONLY, ClearLevel.ALL):
            await db.execute("DELETE FROM system_stats")
            cleared.append("system_stats")

        if level in (ClearLevel.FAILED_ONLY, ClearLevel.ALL):
            await db.execute("DELETE FROM failed_inferences")
            cleared.append("failed_inferences")

        if level in (ClearLevel.LOGS_ONLY, ClearLevel.ALL):
            await db.execute("DELETE FROM inference_logs")
            cleared.append("inference_logs")

        await db.commit()

    # If clearing all, also reset in-memory counters
    if level == ClearLevel.ALL:
        stats_accumulator.__init__()

    print(f"[db] Cleared tables: {cleared}")
    return {"status": "cleared", "tables": cleared}


# =========================================================
# WEBSOCKET ENDPOINTS
# =========================================================

@app.websocket("/ws/status")
async def websocket_status(ws: WebSocket):
    await status_ws.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        status_ws.disconnect(ws)

@app.websocket("/ws/control")
async def websocket_control(ws: WebSocket):
    await control_ws.connect(ws)
    try:
        while True:
            cmd = await ws.receive_json()
            require_test_mode()
            system_state["last_control"] = utc_now()
            apply_control_command(cmd)
    except WebSocketDisconnect:
        control_ws.disconnect(ws)

@app.websocket("/ws/inference")
async def websocket_inference(ws: WebSocket):
    await inference_ws.connect(ws)
    await ws.send_json({"type": "zone_config", "zones": TRIGGER_ZONES})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        inference_ws.disconnect(ws)


# =========================================================
# SYSTEM ENDPOINTS
# =========================================================

def apply_control_command(cmd: Dict[str, Any]):
    arm = cmd.get("arm", {})
    latest_control_command.update({
        "arm": {
            "azimuth": clamp(arm.get("azimuth", 0.0), -1.0, 1.0),
            "elevation": clamp(arm.get("elevation", 0.0), -1.0, 1.0),
        },
        "conveyor": clamp(cmd.get("conveyor", 0.0), -1.0, 1.0),
        "vacuum": bool(cmd.get("vacuum", False))
    })

@app.get("/system/state")
async def get_system_state():
    return system_state

@app.post("/system/mode/{mode}")
async def set_system_mode(mode: SystemMode):
    if system_state["mode"] == SystemMode.FAULT and mode != SystemMode.RUN:
        raise HTTPException(403, "Clear fault before changing mode")
    system_state["mode"] = mode
    system_state["fault"] = None
    await status_ws.broadcast({"type": "mode_change", "mode": mode})
    return {"mode": mode}

@app.post("/system/emergency-stop")
async def emergency_stop():
    system_state["mode"] = SystemMode.FAULT
    system_state["fault"] = "EMERGENCY_STOP"
    latest_control_command.update({
        "arm": {"azimuth": 0.0, "elevation": 0.0},
        "conveyor": 0.0,
        "vacuum": False
    })
    return {"status": "FAULT", "reason": "EMERGENCY_STOP"}

@app.post("/system/reset")
async def reset_fault():
    if system_state["mode"] != SystemMode.FAULT:
        raise HTTPException(400, "System is not in FAULT state")
    system_state["mode"] = SystemMode.RUN
    system_state["fault"] = None
    return {"status": "RESET"}

@app.post("/test/control")
async def test_control(cmd: Dict[str, Any]):
    require_test_mode()
    system_state["last_control"] = utc_now()
    apply_control_command(cmd)
    return {"status": "ok"}

@app.get("/test/control")
async def get_test_control():
    # DEADMAN: if no recent control, return safe state
    last = system_state.get("last_control")

    if last and utc_now() - last > CONTROL_DEADMAN_TIMEOUT:
        return {
            "arm": {"azimuth": 0.0, "elevation": 0.0},
            "conveyor": 0.0,
            "vacuum": False
        }

    return latest_control_command


# =========================================================
# ESP32 ENDPOINTS
# =========================================================

@app.post("/esp32/heartbeat")
async def heartbeat(device_id: str):
    devices[device_id] = utc_now()
    return {"status": "ok"}

@app.get("/esp32/status")
async def esp32_status():
    now = utc_now()
    return {d: "online" if now - t < HEARTBEAT_TIMEOUT else "offline" for d, t in devices.items()}

@app.post("/esp32/capture")
async def capture_image(device_id: str, file: UploadFile = File(...)):
    devices[device_id] = utc_now()
    file_id = f"{uuid.uuid4()}.jpg"
    file_path = UPLOAD_DIR / file_id

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: rf_client.run_workflow(
                workspace_name=WORKSPACE_NAME,
                workflow_id=WORKFLOW_ID,
                images={"image": str(file_path)}
            )
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if file_path.exists():
            file_path.unlink()


# =========================================================
# BACKGROUND TASKS
# =========================================================

async def device_status_cleaner():
    while True:
        now = utc_now()
        offline = [d for d, t in devices.items() if now - t > HEARTBEAT_TIMEOUT]
        for d in offline:
            devices.pop(d)
        await asyncio.sleep(5)

async def control_deadman_watchdog():
    while True:
        if system_state["mode"] == SystemMode.TEST:
            last = system_state.get("last_control")
            if last and utc_now() - last > CONTROL_DEADMAN_TIMEOUT:
                latest_control_command.update({
                    "arm": {"azimuth": 0.0, "elevation": 0.0},
                    "conveyor": 0.0,
                    "vacuum": False
                })
        await asyncio.sleep(0.05)

async def stats_flush_task():
    """Periodically write in-memory stats to DB. Runs every STATS_FLUSH_INTERVAL seconds."""
    while True:
        await asyncio.sleep(STATS_FLUSH_INTERVAL)
        try:
            await flush_stats_to_db()
            print(f"[stats] Flushed snapshot: {stats_accumulator.snapshot()}")
        except Exception as e:
            print(f"[stats] Flush error: {e}")


@app.on_event("startup")
async def startup_event():
    global main_event_loop, result_queue_async

    main_event_loop = asyncio.get_running_loop()
    result_queue_async = asyncio.Queue(maxsize=2)

    await init_db()

    asyncio.create_task(device_status_cleaner())
    asyncio.create_task(control_deadman_watchdog())
    asyncio.create_task(stats_flush_task())

    print("[startup] EcoSort backend ready")
