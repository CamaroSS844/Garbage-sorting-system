import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import threading
import queue
import time
from PIL import Image, ImageTk
from dataclasses import dataclass, field
from typing import List, Tuple
from collections import OrderedDict
import onnxruntime as ort
import os

# =========================================================
# DATA CLASSES
# =========================================================
@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str

@dataclass
class TrackedObject(Detection):
    track_id: int = -1

# =========================================================
# SIMPLE CENTROID TRACKER
# =========================================================
class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_id = 0
        self.objects = OrderedDict()   # id -> centroid
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, dets: List[Detection]) -> List[TrackedObject]:
        if not dets:
            for tid in list(self.disappeared):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.objects[tid]
                    del self.disappeared[tid]
            return []

        input_centroids = [self._centroid(d.bbox) for d in dets]

        if not self.objects:
            for c in input_centroids:
                self.objects[self.next_id] = c
                self.disappeared[self.next_id] = 0
                self.next_id += 1
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Build distance matrix
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i, oc in enumerate(object_centroids):
                for j, ic in enumerate(input_centroids):
                    D[i, j] = np.linalg.norm(np.array(oc) - np.array(ic))

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                if D[r, c] > self.max_distance:
                    continue
                tid = object_ids[r]
                self.objects[tid] = input_centroids[c]
                self.disappeared[tid] = 0
                used_rows.add(r)
                used_cols.add(c)

            for r in range(len(object_centroids)):
                if r not in used_rows:
                    tid = object_ids[r]
                    self.disappeared[tid] += 1
                    if self.disappeared[tid] > self.max_disappeared:
                        del self.objects[tid]
                        del self.disappeared[tid]

            for c in range(len(input_centroids)):
                if c not in used_cols:
                    self.objects[self.next_id] = input_centroids[c]
                    self.disappeared[self.next_id] = 0
                    self.next_id += 1

        # Map detections to track IDs by nearest centroid
        tracked = []
        for d in dets:
            dc = self._centroid(d.bbox)
            best_id, best_dist = -1, float("inf")
            for tid, tc in self.objects.items():
                dist = np.linalg.norm(np.array(dc) - np.array(tc))
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            tracked.append(TrackedObject(d.bbox, d.confidence, d.class_id, d.class_name, best_id))
        return tracked

# =========================================================
# ULTRALYTICS MODEL
# =========================================================
class UltralyticsModel:
    def __init__(self, path):
        self.model = YOLO(path)
        self.class_names = self.model.names

    def infer(self, frame, conf, iou):
        start = time.time()
        results = self.model(frame, conf=conf, iou=iou, verbose=False)
        t = (time.time() - start) * 1000
        out = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                confv = float(b.conf[0])
                cid = int(b.cls[0])
                out.append(Detection((x1, y1, x2, y2), confv, cid, self.class_names[cid]))
        return out, t

# =========================================================
# ONNX MODEL
# =========================================================
class ONNXModel:
    def __init__(self, model_path, class_file):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        with open(class_file) as f:
            self.class_names = [l.strip() for l in f.readlines()]

    def infer(self, frame, conf_thresh, iou_thresh):
        h, w = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]

        start = time.time()
        preds = self.session.run(None, {self.input_name: img})[0][0].T
        t = (time.time() - start) * 1000

        boxes = preds[:, :4]
        scores = preds[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confs = np.max(scores, axis=1)

        mask = confs > conf_thresh
        boxes, confs, class_ids = boxes[mask], confs[mask], class_ids[mask]

        if len(boxes) == 0:
            return [], t

        x, y, w0, h0 = boxes.T
        x1 = (x - w0 / 2) * w / 640
        y1 = (y - h0 / 2) * h / 640
        x2 = (x + w0 / 2) * w / 640
        y2 = (y + h0 / 2) * h / 640
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        idxs = cv2.dnn.NMSBoxes(boxes.tolist(), confs.tolist(), conf_thresh, iou_thresh)
        out = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                out.append(
                    Detection(tuple(map(int, boxes[i])),
                              float(confs[i]),
                              int(class_ids[i]),
                              self.class_names[int(class_ids[i])])
                )
        return out, t

# =========================================================
# LOW-LATENCY CAPTURE THREAD
# =========================================================
class CaptureThread(threading.Thread):
    """
    Dedicated thread that drains the decoder's internal ring buffer
    using grab() so we always deliver the LATEST frame, not a buffered one.

    Key latency fixes:
    - CAP_PROP_BUFFERSIZE=1  → shrinks OpenCV's internal queue
    - Tight grab() loop       → empties the decoder backlog each cycle
    - Drop policy on queue    → inference always sees the newest frame
    """

    # FFmpeg options injected via environment for low-delay RTSP
    RTSP_ENV = {
        "OPENCV_FFMPEG_CAPTURE_OPTIONS":
            "rtsp_transport;tcp|"
            "fflags;nobuffer|"
            "flags;low_delay|"
            "framedrop;1|"
            "max_delay;0|"
            "reorder_queue_size;0"
    }

    def __init__(self, source_type, source_val, out_queue: queue.Queue):
        super().__init__(daemon=True)
        self.source_type = source_type
        self.source_val = source_val
        self.out_queue = out_queue
        self.running = True
        self.cap = None
        self.error = None

    def _open(self):
        st = self.source_type
        sv = self.source_val

        if st == "webcam":
            cap = cv2.VideoCapture(int(sv) if sv else 0, cv2.CAP_DSHOW)
        elif st == "ipcam":
            # Inject low-latency FFmpeg options via environment variable
            for k, v in self.RTSP_ENV.items():
                os.environ[k] = v
            cap = cv2.VideoCapture(sv, cv2.CAP_FFMPEG)
        elif st == "video":
            cap = cv2.VideoCapture(sv)
        else:
            return None  # image handled separately

        # Minimise internal OpenCV buffer (1 frame)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        if self.source_type == "image":
            img = cv2.imread(self.source_val)
            if img is None:
                self.error = "Image load failed"
                return
            self._push(img)
            return

        self.cap = self._open()
        if self.cap is None or not self.cap.isOpened():
            self.error = "Failed to open source"
            return

        while self.running:
            # ── Drain the decoder's internal queue ──────────────────────────
            # grab() decodes header only (cheap). We call it in a tight loop
            # until no more frames are waiting, then retrieve() the last one.
            grabbed = False
            for _ in range(10):          # drain up to 10 buffered frames
                if not self.cap.grab():
                    break
                grabbed = True

            if not grabbed:
                time.sleep(0.001)
                continue

            ret, frame = self.cap.retrieve()
            if not ret:
                continue

            self._push(frame)

        if self.cap:
            self.cap.release()

    def _push(self, frame):
        # Drop oldest if queue is full — always keep newest
        if self.out_queue.full():
            try:
                self.out_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.out_queue.put_nowait(frame)
        except queue.Full:
            pass

    def stop(self):
        self.running = False

# =========================================================
# GUI APPLICATION
# =========================================================
class ModelTesterApp:

    def __init__(self, root):
        self.root = root
        self.root.title("AI Vision Tester")
        self.root.geometry("1200x720")

        self.model = None
        self.running = False

        self.capture_queue = queue.Queue(maxsize=1)
        self.display_queue = queue.Queue(maxsize=1)

        self.capture_thread: CaptureThread = None
        self.tracker = CentroidTracker()
        self.tracking = False

        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        self.last_inf = 0

        self.create_widgets()
        self.update_display()

    # =====================================================
    # UI BUILD
    # =====================================================
    def create_widgets(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # ---------- LEFT PANEL ----------
        left = ttk.LabelFrame(main, text="Controls", padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # MODEL
        ttk.Label(left, text="Model File").grid(row=0, column=0, sticky=tk.W)
        self.model_path = tk.StringVar()
        ttk.Entry(left, textvariable=self.model_path, width=28).grid(row=0, column=1)
        ttk.Button(left, text="Browse", command=self.browse_model).grid(row=0, column=2)

        ttk.Label(left, text="Type").grid(row=1, column=0, sticky=tk.W)
        self.model_type = tk.StringVar(value="ultralytics")
        ttk.Combobox(left, textvariable=self.model_type,
                     values=["ultralytics", "onnx"],
                     state="readonly", width=15).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(left, text="Classes (onnx)").grid(row=2, column=0, sticky=tk.W)
        self.class_path = tk.StringVar()
        ttk.Entry(left, textvariable=self.class_path, width=28).grid(row=2, column=1)
        ttk.Button(left, text="Browse", command=self.browse_classes).grid(row=2, column=2)

        ttk.Button(left, text="Load Model", command=self.load_model) \
            .grid(row=3, column=0, columnspan=3, pady=10)

        # SOURCE
        ttk.Label(left, text="Source").grid(row=4, column=0, sticky=tk.W)
        self.source = tk.StringVar(value="webcam")
        combo = ttk.Combobox(left, textvariable=self.source,
                             values=["webcam", "image", "video", "ipcam"],
                             state="readonly", width=15)
        combo.grid(row=4, column=1)
        combo.bind("<<ComboboxSelected>>", self.source_changed)

        self.source_path = tk.StringVar()
        self.src_entry = ttk.Entry(left, textvariable=self.source_path, width=35)
        self.src_entry.grid(row=5, column=0, columnspan=3, pady=5)

        self.src_btn = ttk.Button(left, text="Browse", command=self.browse_source)
        self.src_btn.grid(row=5, column=2)

        # SLIDERS
        ttk.Label(left, text="Confidence").grid(row=6, column=0, sticky=tk.W)
        self.conf = tk.DoubleVar(value=0.5)
        ttk.Scale(left, from_=0, to=1, variable=self.conf, length=160) \
            .grid(row=6, column=1, columnspan=2, sticky=tk.W)
        ttk.Label(left, textvariable=self.conf).grid(row=6, column=2, sticky=tk.E)

        ttk.Label(left, text="IoU").grid(row=7, column=0, sticky=tk.W)
        self.iou = tk.DoubleVar(value=0.45)
        ttk.Scale(left, from_=0, to=1, variable=self.iou, length=160) \
            .grid(row=7, column=1, columnspan=2, sticky=tk.W)
        ttk.Label(left, textvariable=self.iou).grid(row=7, column=2, sticky=tk.E)

        # INFER RESOLUTION (lower = faster inference & less lag)
        ttk.Label(left, text="Infer Size").grid(row=8, column=0, sticky=tk.W)
        self.infer_size = tk.IntVar(value=640)
        ttk.Combobox(left, textvariable=self.infer_size,
                     values=[320, 416, 640],
                     state="readonly", width=8).grid(row=8, column=1, sticky=tk.W)

        # STREAM (sub-stream toggle for IP cams)
        ttk.Label(left, text="Sub-stream").grid(row=9, column=0, sticky=tk.W)
        self.use_substream = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, variable=self.use_substream,
                        command=self._apply_substream) \
            .grid(row=9, column=1, sticky=tk.W)
        ttk.Label(left, text="(/stream2)").grid(row=9, column=2, sticky=tk.W)

        # TRACKER
        track_frame = ttk.LabelFrame(left, text="Tracking", padding=5)
        track_frame.grid(row=10, column=0, columnspan=3, sticky=tk.EW, pady=10)

        self.track_var = tk.BooleanVar()
        ttk.Checkbutton(track_frame, text="Enable",
                        variable=self.track_var,
                        command=self.toggle_tracking) \
            .grid(row=0, column=0, sticky=tk.W)

        ttk.Label(track_frame, text="Max disappear").grid(row=1, column=0)
        self.max_dis = tk.IntVar(value=30)
        ttk.Spinbox(track_frame, from_=1, to=200,
                    textvariable=self.max_dis, width=8).grid(row=1, column=1)

        ttk.Label(track_frame, text="Max distance").grid(row=2, column=0)
        self.max_dist = tk.IntVar(value=50)
        ttk.Spinbox(track_frame, from_=1, to=200,
                    textvariable=self.max_dist, width=8).grid(row=2, column=1)

        ttk.Button(track_frame, text="Reset",
                   command=self.reset_tracker).grid(row=3, column=0, columnspan=2, pady=5)

        # START / STOP
        self.start_btn = ttk.Button(left, text="Start", command=self.start)
        self.start_btn.grid(row=11, column=0, pady=10)

        self.stop_btn = ttk.Button(left, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.grid(row=11, column=1)

        # STATS
        stats = ttk.LabelFrame(left, text="Stats", padding=5)
        stats.grid(row=12, column=0, columnspan=3, sticky=tk.EW, pady=10)

        ttk.Label(stats, text="Inference").grid(row=0, column=0)
        self.inf_lbl = ttk.Label(stats, text="0 ms")
        self.inf_lbl.grid(row=0, column=1)

        ttk.Label(stats, text="FPS").grid(row=1, column=0)
        self.fps_lbl = ttk.Label(stats, text="0")
        self.fps_lbl.grid(row=1, column=1)

        ttk.Label(stats, text="Frames").grid(row=2, column=0)
        self.frames_lbl = ttk.Label(stats, text="0")
        self.frames_lbl.grid(row=2, column=1)

        ttk.Label(stats, text="Tracks").grid(row=3, column=0)
        self.tracks_lbl = ttk.Label(stats, text="0")
        self.tracks_lbl.grid(row=3, column=1)

        ttk.Label(stats, text="Latency").grid(row=4, column=0)
        self.lat_lbl = ttk.Label(stats, text="-- ms")
        self.lat_lbl.grid(row=4, column=1)

        # LOG
        log_frame = ttk.LabelFrame(left, text="Log", padding=5)
        log_frame.grid(row=13, column=0, columnspan=3, sticky=tk.EW)
        self.log_box = tk.Text(log_frame, height=8, width=40, state=tk.DISABLED)
        self.log_box.pack(fill=tk.BOTH, expand=True)

        # ---------- RIGHT PANEL ----------
        right = ttk.LabelFrame(main, text="Display", padding=5)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.video_label = ttk.Label(right)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.source_changed()

    # =====================================================
    # UI HELPERS
    # =====================================================
    def log(self, msg):
        self.log_box.config(state=tk.NORMAL)
        self.log_box.insert(tk.END, time.strftime("%H:%M:%S") + " " + msg + "\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    def browse_model(self):
        p = filedialog.askopenfilename(filetypes=[("Models", "*.pt *.onnx")])
        if p: self.model_path.set(p)

    def browse_classes(self):
        p = filedialog.askopenfilename(filetypes=[("Text", "*.txt")])
        if p: self.class_path.set(p)

    def browse_source(self):
        src = self.source.get()
        if src == "image":
            p = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        elif src == "video":
            p = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mkv")])
        else:
            return
        if p: self.source_path.set(p)

    def source_changed(self, e=None):
        s = self.source.get()
        if s in ("image", "video"):
            self.src_btn.config(state=tk.NORMAL)
        elif s == "webcam":
            self.source_path.set("0")
            self.src_btn.config(state=tk.DISABLED)
        else:
            self.source_path.set("rtsp://admin:password@<camera-ip>/stream1")
            self.src_btn.config(state=tk.DISABLED)

    def _apply_substream(self):
        """Switch between /stream1 and /stream2 on the URL."""
        url = self.source_path.get()
        if self.use_substream.get():
            url = url.replace("/stream1", "/stream2")
            if "/stream2" not in url:
                url = url.rstrip("/") + "/stream2"
        else:
            url = url.replace("/stream2", "/stream1")
            if "/stream1" not in url:
                url = url.rstrip("/") + "/stream1"
        self.source_path.set(url)

    # =====================================================
    # MODEL
    # =====================================================
    def load_model(self):
        try:
            if self.model_type.get() == "ultralytics":
                self.model = UltralyticsModel(self.model_path.get())
            else:
                self.model = ONNXModel(self.model_path.get(), self.class_path.get())
            self.log("Model loaded")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # =====================================================
    # TRACKER
    # =====================================================
    def toggle_tracking(self):
        self.tracking = self.track_var.get()
        self.reset_tracker()

    def reset_tracker(self):
        self.tracker = CentroidTracker(self.max_dis.get(), self.max_dist.get())
        self.log("Tracker reset")

    # =====================================================
    # RUN CONTROL
    # =====================================================
    def start(self):
        if not self.model:
            messagebox.showerror("Error", "Load model first")
            return

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        # Fresh queues — maxsize=1 ensures no frame accumulation
        self.capture_queue = queue.Queue(maxsize=1)
        self.display_queue = queue.Queue(maxsize=1)

        # Low-latency capture thread
        self.capture_thread = CaptureThread(
            self.source.get(),
            self.source_path.get(),
            self.capture_queue
        )
        self.capture_thread.start()

        # Inference thread
        threading.Thread(target=self.inference_loop, daemon=True).start()

        # Poll for capture errors after 2 s
        self.root.after(2000, self._check_capture_error)
        self.log("Started")

    def stop(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log("Stopped")

    def _check_capture_error(self):
        if self.capture_thread and self.capture_thread.error:
            self.log("Capture error: " + self.capture_thread.error)
            messagebox.showerror("Capture Error", self.capture_thread.error)
            self.stop()

    # =====================================================
    # INFERENCE LOOP
    # =====================================================
    def inference_loop(self):
        """
        Latency improvements vs original:
        1. Resize frame to `infer_size` BEFORE inference — smaller input = faster.
        2. Draw annotations on the resized copy, scale bbox back for display.
        3. Also resize to display panel size here (off UI thread) to avoid
           doing it in update_display() on the main thread.
        4. Timestamp frames so we can show end-to-end latency.
        """
        panel_w = self.video_label.winfo_width() or 800
        panel_h = self.video_label.winfo_height() or 500

        while self.running:
            try:
                frame = self.capture_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            t_capture = time.time()

            orig_h, orig_w = frame.shape[:2]
            sz = self.infer_size.get()

            # ── Scale down for inference ─────────────────────────────────────
            scale_x = sz / orig_w
            scale_y = sz / orig_h
            if orig_w != sz or orig_h != sz:
                infer_frame = cv2.resize(frame, (sz, sz))
            else:
                infer_frame = frame

            dets, inf = self.model.infer(infer_frame, self.conf.get(), self.iou.get())
            self.last_inf = inf

            # ── Scale bboxes back to original resolution ─────────────────────
            for d in dets:
                x1, y1, x2, y2 = d.bbox
                d.bbox = (
                    int(x1 / scale_x), int(y1 / scale_y),
                    int(x2 / scale_x), int(y2 / scale_y)
                )

            if self.tracking:
                dets = self.tracker.update(dets)

            # ── Draw on original-resolution frame ────────────────────────────
            for d in dets:
                x1, y1, x2, y2 = d.bbox
                label = f"{d.class_name} {d.confidence:.2f}"
                if isinstance(d, TrackedObject):
                    label = f"ID {d.track_id} " + label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # ── FPS counter ──────────────────────────────────────────────────
            self.frame_count += 1
            now = time.time()
            if now - self.last_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_time = now

            latency_ms = (time.time() - t_capture) * 1000

            cv2.putText(frame, f"FPS {self.fps}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Inf {inf:.0f}ms", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            # ── Pre-scale to display panel size (off UI thread) ──────────────
            ph = self.video_label.winfo_height() or panel_h
            pw = self.video_label.winfo_width() or panel_w
            if ph > 10 and pw > 10:
                scale = min(pw / orig_w, ph / orig_h)
                disp = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)),
                                  interpolation=cv2.INTER_LINEAR)
            else:
                disp = frame

            # Drop stale display frame, push newest
            if self.display_queue.full():
                try:
                    self.display_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.display_queue.put_nowait((disp, len(dets), latency_ms))
            except queue.Full:
                pass

    # =====================================================
    # DISPLAY LOOP (main thread, driven by after())
    # =====================================================
    def update_display(self):
        if hasattr(self, "display_queue") and not self.display_queue.empty():
            try:
                frame, count, latency_ms = self.display_queue.get_nowait()
            except queue.Empty:
                frame = None

            if frame is not None:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(img)
                self.video_label.configure(image=imgtk)
                self.video_label.imgtk = imgtk

                self.inf_lbl.config(text=f"{self.last_inf:.1f} ms")
                self.fps_lbl.config(text=str(self.fps))
                self.frames_lbl.config(text=str(self.frame_count))
                self.tracks_lbl.config(text=str(count))
                self.lat_lbl.config(text=f"{latency_ms:.0f} ms")

        # 10 ms poll = 100 Hz update cap, plenty for 30 fps feeds
        self.root.after(10, self.update_display)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = ModelTesterApp(root)
    root.mainloop()