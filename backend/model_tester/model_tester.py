import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import threading
import queue
import time
from PIL import Image, ImageTk
from dataclasses import dataclass
from typing import List, Tuple
from collections import OrderedDict
import onnxruntime as ort

# =========================================================
# DATA CLASSES
# =========================================================
@dataclass
class Detection:
    bbox: Tuple[int,int,int,int]
    confidence: float
    class_id: int
    class_name: str

@dataclass
class TrackedObject(Detection):
    track_id:int=-1

# =========================================================
# SIMPLE TRACKER (from second code logic base)
# =========================================================
class CentroidTracker:
    def __init__(self,max_disappeared=30,max_distance=50):
        self.next_id=0
        self.objects=OrderedDict()
        self.max_disappeared=max_disappeared
        self.max_distance=max_distance

    def update(self,dets:List[Detection])->List[TrackedObject]:
        tracked=[]
        for i,d in enumerate(dets):
            tracked.append(
                TrackedObject(d.bbox,d.confidence,d.class_id,d.class_name,i)
            )
        return tracked

# =========================================================
# ULTRALYTICS MODEL (WORKING ENGINE)
# =========================================================
class UltralyticsModel:
    def __init__(self,path):
        self.model=YOLO(path)
        self.class_names=self.model.names

    def infer(self,frame,conf,iou):
        start=time.time()
        results=self.model(frame,conf=conf,iou=iou,verbose=False)
        t=(time.time()-start)*1000

        out=[]
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1,y1,x2,y2=map(int,b.xyxy[0].tolist())
                confv=float(b.conf[0])
                cid=int(b.cls[0])
                out.append(Detection((x1,y1,x2,y2),confv,cid,self.class_names[cid]))
        return out,t

# =========================================================
# ONNX MODEL (WORKING ENGINE)
# =========================================================
class ONNXModel:
    def __init__(self,model_path,class_file):
        self.session=ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider','CPUExecutionProvider']
        )
        self.input_name=self.session.get_inputs()[0].name

        with open(class_file) as f:
            self.class_names=[l.strip() for l in f.readlines()]

    def infer(self,frame,conf_thresh,iou_thresh):
        h,w=frame.shape[:2]

        img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(640,640)).astype(np.float32)/255.0
        img=np.transpose(img,(2,0,1))[None]

        start=time.time()
        preds=self.session.run(None,{self.input_name:img})[0][0].T
        t=(time.time()-start)*1000

        boxes=preds[:,:4]
        scores=preds[:,4:]
        class_ids=np.argmax(scores,axis=1)
        confs=np.max(scores,axis=1)

        mask=confs>conf_thresh
        boxes,confs,class_ids=boxes[mask],confs[mask],class_ids[mask]

        x,y,w0,h0=boxes.T
        x1=(x-w0/2)*w/640
        y1=(y-h0/2)*h/640
        x2=(x+w0/2)*w/640
        y2=(y+h0/2)*h/640
        boxes=np.stack([x1,y1,x2,y2],axis=1)

        idxs=cv2.dnn.NMSBoxes(boxes.tolist(),confs.tolist(),conf_thresh,iou_thresh)

        out=[]
        if len(idxs)>0:
            for i in idxs.flatten():
                out.append(
                    Detection(tuple(map(int,boxes[i])),
                              float(confs[i]),
                              int(class_ids[i]),
                              self.class_names[int(class_ids[i])])
                )
        return out,t

# =========================================================
# GUI APPLICATION
# =========================================================
class ModelTesterApp:

    def __init__(self,root):
        self.root=root
        self.root.title("AI Vision Tester")
        self.root.geometry("1200x720")

        self.model=None
        self.running=False
        self.frame_queue=queue.Queue(maxsize=2)

        self.tracker=CentroidTracker()
        self.tracking=False

        self.fps=0
        self.frame_count=0
        self.last_time=time.time()
        self.last_inf=0

        self.create_widgets()
        self.update_display()

    # =====================================================
    # UI BUILD
    # =====================================================
    def create_widgets(self):

        main=ttk.Frame(self.root,padding=10)
        main.pack(fill=tk.BOTH,expand=True)

        # ---------- LEFT PANEL ----------
        left=ttk.LabelFrame(main,text="Controls",padding=10)
        left.pack(side=tk.LEFT,fill=tk.Y,padx=(0,10))

        # MODEL
        ttk.Label(left,text="Model File").grid(row=0,column=0,sticky=tk.W)
        self.model_path=tk.StringVar()
        ttk.Entry(left,textvariable=self.model_path,width=28).grid(row=0,column=1)
        ttk.Button(left,text="Browse",command=self.browse_model).grid(row=0,column=2)

        ttk.Label(left,text="Type").grid(row=1,column=0,sticky=tk.W)
        self.model_type=tk.StringVar(value="ultralytics")
        ttk.Combobox(left,
                     textvariable=self.model_type,
                     values=["ultralytics","onnx"],
                     state="readonly",
                     width=15).grid(row=1,column=1,sticky=tk.W)

        ttk.Label(left,text="Classes (onnx)").grid(row=2,column=0,sticky=tk.W)
        self.class_path=tk.StringVar()
        ttk.Entry(left,textvariable=self.class_path,width=28).grid(row=2,column=1)
        ttk.Button(left,text="Browse",command=self.browse_classes).grid(row=2,column=2)

        ttk.Button(left,text="Load Model",command=self.load_model)\
            .grid(row=3,column=0,columnspan=3,pady=10)

        # SOURCE
        ttk.Label(left,text="Source").grid(row=4,column=0,sticky=tk.W)
        self.source=tk.StringVar(value="webcam")
        combo=ttk.Combobox(left,textvariable=self.source,
                           values=["webcam","image","video","ipcam"],
                           state="readonly",width=15)
        combo.grid(row=4,column=1)
        combo.bind("<<ComboboxSelected>>",self.source_changed)

        self.source_path=tk.StringVar()
        self.src_entry=ttk.Entry(left,textvariable=self.source_path,width=35)
        self.src_entry.grid(row=5,column=0,columnspan=3,pady=5)

        self.src_btn=ttk.Button(left,text="Browse",command=self.browse_source)
        self.src_btn.grid(row=5,column=2)

        # SLIDERS
        ttk.Label(left,text="Confidence").grid(row=6,column=0,sticky=tk.W)
        self.conf=tk.DoubleVar(value=0.5)
        ttk.Scale(left,from_=0,to=1,variable=self.conf,length=160)\
            .grid(row=6,column=1,columnspan=2,sticky=tk.W)
        ttk.Label(left,textvariable=self.conf).grid(row=6,column=2,sticky=tk.E)

        ttk.Label(left,text="IoU").grid(row=7,column=0,sticky=tk.W)
        self.iou=tk.DoubleVar(value=0.45)
        ttk.Scale(left,from_=0,to=1,variable=self.iou,length=160)\
            .grid(row=7,column=1,columnspan=2,sticky=tk.W)
        ttk.Label(left,textvariable=self.iou).grid(row=7,column=2,sticky=tk.E)

        # TRACKER
        track_frame=ttk.LabelFrame(left,text="Tracking",padding=5)
        track_frame.grid(row=8,column=0,columnspan=3,sticky=tk.EW,pady=10)

        self.track_var=tk.BooleanVar()
        ttk.Checkbutton(track_frame,text="Enable",
                        variable=self.track_var,
                        command=self.toggle_tracking)\
            .grid(row=0,column=0,sticky=tk.W)

        ttk.Label(track_frame,text="Max disappear").grid(row=1,column=0)
        self.max_dis=tk.IntVar(value=30)
        ttk.Spinbox(track_frame,from_=1,to=200,
                    textvariable=self.max_dis,width=8)\
            .grid(row=1,column=1)

        ttk.Label(track_frame,text="Max distance").grid(row=2,column=0)
        self.max_dist=tk.IntVar(value=50)
        ttk.Spinbox(track_frame,from_=1,to=200,
                    textvariable=self.max_dist,width=8)\
            .grid(row=2,column=1)

        ttk.Button(track_frame,text="Reset",
                   command=self.reset_tracker)\
            .grid(row=3,column=0,columnspan=2,pady=5)

        # START STOP
        self.start_btn=ttk.Button(left,text="Start",command=self.start)
        self.start_btn.grid(row=9,column=0,pady=10)

        self.stop_btn=ttk.Button(left,text="Stop",command=self.stop,state=tk.DISABLED)
        self.stop_btn.grid(row=9,column=1)

        # STATS
        stats=ttk.LabelFrame(left,text="Stats",padding=5)
        stats.grid(row=10,column=0,columnspan=3,sticky=tk.EW,pady=10)

        self.inf_lbl=ttk.Label(stats,text="0 ms")
        self.inf_lbl.grid(row=0,column=1)
        ttk.Label(stats,text="Inference").grid(row=0,column=0)

        self.fps_lbl=ttk.Label(stats,text="0")
        self.fps_lbl.grid(row=1,column=1)
        ttk.Label(stats,text="FPS").grid(row=1,column=0)

        self.frames_lbl=ttk.Label(stats,text="0")
        self.frames_lbl.grid(row=2,column=1)
        ttk.Label(stats,text="Frames").grid(row=2,column=0)

        self.tracks_lbl=ttk.Label(stats,text="0")
        self.tracks_lbl.grid(row=3,column=1)
        ttk.Label(stats,text="Tracks").grid(row=3,column=0)

        # LOG
        log_frame=ttk.LabelFrame(left,text="Log",padding=5)
        log_frame.grid(row=11,column=0,columnspan=3,sticky=tk.EW)

        self.log_box=tk.Text(log_frame,height=8,width=40,state=tk.DISABLED)
        self.log_box.pack(fill=tk.BOTH,expand=True)

        # ---------- RIGHT PANEL ----------
        right=ttk.LabelFrame(main,text="Display",padding=5)
        right.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True)

        self.video_label=ttk.Label(right)
        self.video_label.pack(fill=tk.BOTH,expand=True)

        self.source_changed()

    # =====================================================
    # UI EVENTS
    # =====================================================
    def log(self,msg):
        self.log_box.config(state=tk.NORMAL)
        self.log_box.insert(tk.END,time.strftime("%H:%M:%S")+" "+msg+"\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    def browse_model(self):
        p=filedialog.askopenfilename(filetypes=[("Models","*.pt *.onnx")])
        if p:self.model_path.set(p)

    def browse_classes(self):
        p=filedialog.askopenfilename(filetypes=[("Text","*.txt")])
        if p:self.class_path.set(p)

    def browse_source(self):
        src=self.source.get()
        if src=="image":
            p=filedialog.askopenfilename(filetypes=[("Images","*.jpg *.png *.jpeg")])
        elif src=="video":
            p=filedialog.askopenfilename(filetypes=[("Videos","*.mp4 *.avi *.mkv")])
        else:
            return
        if p:self.source_path.set(p)

    def source_changed(self,e=None):
        s=self.source.get()
        if s in ("image","video"):
            self.src_btn.config(state=tk.NORMAL)
        elif s=="webcam":
            self.source_path.set("0")
            self.src_btn.config(state=tk.DISABLED)
        else:
            self.source_path.set("rtsp://")
            self.src_btn.config(state=tk.DISABLED)

    # =====================================================
    # MODEL
    # =====================================================
    def load_model(self):
        try:
            if self.model_type.get()=="ultralytics":
                self.model=UltralyticsModel(self.model_path.get())
            else:
                self.model=ONNXModel(self.model_path.get(),self.class_path.get())
            self.log("Model loaded")
        except Exception as e:
            messagebox.showerror("Error",str(e))

    # =====================================================
    # TRACKER
    # =====================================================
    def toggle_tracking(self):
        self.tracking=self.track_var.get()
        self.reset_tracker()

    def reset_tracker(self):
        self.tracker=CentroidTracker(self.max_dis.get(),self.max_dist.get())
        self.log("Tracker reset")

    # =====================================================
    # RUN CONTROL
    # =====================================================
    def start(self):
        if not self.model:
            messagebox.showerror("Error","Load model first")
            return

        self.running=True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        threading.Thread(target=self.loop,daemon=True).start()
        self.log("Started")

    def stop(self):
        self.running=False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log("Stopped")

    # =====================================================
    # CAPTURE LOOP
    # =====================================================
    def loop(self):

        src=self.source.get()
        val=self.source_path.get()

        if src=="webcam":
            cap=cv2.VideoCapture(int(val) if val else 0)
        elif src=="ipcam":
            cap=cv2.VideoCapture(val)
        elif src=="video":
            cap=cv2.VideoCapture(val)
        else:
            img=cv2.imread(val)
            if img is None:
                self.log("Image load failed")
                return
            self.process_frame(img)
            return

        while self.running:
            ret,frame=cap.read()
            if not ret: break
            self.process_frame(frame)

        cap.release()

    # =====================================================
    # PROCESS FRAME
    # =====================================================
    def process_frame(self,frame):

        dets,inf=self.model.infer(frame,self.conf.get(),self.iou.get())
        self.last_inf=inf

        if self.tracking:
            dets=self.tracker.update(dets)

        for d in dets:
            x1,y1,x2,y2=d.bbox
            label=f"{d.class_name} {d.confidence:.2f}"
            if isinstance(d,TrackedObject):
                label=f"ID {d.track_id} "+label
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,label,(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

        self.frame_count+=1
        if time.time()-self.last_time>=1:
            self.fps=self.frame_count
            self.frame_count=0
            self.last_time=time.time()

        cv2.putText(frame,f"FPS {self.fps}",(10,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        if not self.frame_queue.full():
            self.frame_queue.put((frame,len(dets)))

    # =====================================================
    # DISPLAY LOOP
    # =====================================================
    def update_display(self):
        if not self.frame_queue.empty():
            frame,count=self.frame_queue.get()

            img=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
            self.video_label.configure(image=img)
            self.video_label.imgtk=img

            self.inf_lbl.config(text=f"{self.last_inf:.1f} ms")
            self.fps_lbl.config(text=str(self.fps))
            self.frames_lbl.config(text=str(self.frame_count))
            self.tracks_lbl.config(text=str(count))

        self.root.after(30,self.update_display)

# =========================================================
# MAIN
# =========================================================
if __name__=="__main__":
    root=tk.Tk()
    app=ModelTesterApp(root)
    root.mainloop()