import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
from ultralytics import YOLO

# ================= ENHANCED CONFIG =================
MODEL_PATH = "yolov8n_ncnn_model"  # The folder you just created
RTSP_USER = "admin"
RTSP_PASS = ""

# Matches your export size for maximum speed
INFERENCE_SIZE = 256 

# COCO IDs: 0:person, 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck, 15:cat, 16:dog
TARGET_CLASSES = [0, 1, 2, 3, 5, 7, 15, 16]

CAMERAS = [
    {"ip": "192.168.18.2", "name": "Front Gate"},
    {"ip": "192.168.18.113", "name": "Driveway"},
]

# ================= TURBO CPU ENGINE =================
class TurboCPUEngine:
    def __init__(self, model_path):
        # Load the NCNN version
        self.model = YOLO(model_path, task='detect')
        self.input_queue = Queue(maxsize=1) # Keep only the freshest frame
        self.output_results = {}
        self.running = True
        self.fps = 0

    def start(self):
        threading.Thread(target=self._inference_worker, daemon=True).start()

    def _inference_worker(self):
        last_time = time.time()
        frame_count = 0

        while self.running:
            try:
                key, frame = self.input_queue.get(timeout=0.1)
            except Empty:
                continue

            # Run NCNN inference
            # half=True uses FP16 math which is faster on Pi 5
            results = self.model.predict(
                frame, 
                imgsz=INFERENCE_SIZE, 
                conf=0.35, 
                classes=TARGET_CLASSES,
                verbose=False,
                half=True 
            )
            
            if len(results) > 0:
                boxes = results[0].boxes
                self.output_results[key] = [
                    [*(box.xyxyn[0].tolist()), float(box.conf[0]), int(box.cls[0])] 
                    for box in boxes
                ]

            frame_count += 1
            if time.time() - last_time >= 1.0:
                self.fps = frame_count / (time.time() - last_time)
                frame_count = 0
                last_time = time.time()

# ================= STREAM WORKER =================
class StreamWorker:
    def __init__(self, ip, name, engine):
        self.name = name
        self.url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{ip}:554/h264"
        self.engine = engine
        self.latest_frame = None
        self.latest_dets = []
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(2)
                cap.open(self.url)
                continue
            
            key = f"{self.name}_{f_idx}"
            
            # If engine is busy, skip this frame to keep the stream "Live"
            if self.engine.input_queue.empty():
                try:
                    self.engine.input_queue.put_nowait((key, frame))
                except:
                    pass

            # Update results if ready
            if key in self.engine.output_results:
                dets = self.engine.output_results.pop(key)
                with self.lock:
                    self.latest_dets = dets
            
            with self.lock:
                self.latest_frame = frame
            f_idx += 1

# ================= MAIN DISPLAY =================
def main():
    print(f"🚀 Pi 5 NCNN Turbo Running...")
    print(f"Detecting: {', '.join([YOLO('yolov8n.pt').names[i] for i in TARGET_CLASSES])}")
    
    engine = TurboCPUEngine(MODEL_PATH)
    engine.start()
    
    workers = [StreamWorker(c['ip'], c['name'], engine) for c in CAMERAS]

    while True:
        display_frames = []
        for w in workers:
            with w.lock:
                if w.latest_frame is None: continue
                img = w.latest_frame.copy()
                dets = w.latest_dets
            
            h, w_img = img.shape[:2]
            for d in dets:
                # d: [x1, y1, x2, y2, conf, cls_id]
                x1, y1, x2, y2 = int(d[0]*w_img), int(d[1]*h), int(d[2]*w_img), int(d[3]*h)
                label = engine.model.names[d[5]]
                
                # Draw stylized box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y1-25), (x1+100, y1), (0, 255, 0), -1)
                cv2.putText(img, f"{label}", (x1+5, y1-7), 0, 0.6, (0, 0, 0), 2)

            # Dashboard Info
            cv2.putText(img, f"SYSTEM FPS: {engine.fps:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
            
            # Resize for a clean dashboard view
            display_frames.append(cv2.resize(img, (854, 480)))

        if display_frames:
            # Combine camera views
            combined = cv2.hconcat(display_frames) if len(display_frames) > 1 else display_frames[0]
            cv2.imshow("Pi 5 CPU Turbo Dashboard", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
