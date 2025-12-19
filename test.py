# CRITICAL: Import cv2 first to initialize the display driver
import cv2
import os

# Force XCB to avoid the Wayland crash
os.environ["QT_QPA_PLATFORM"] = "xcb"

import numpy as np
import threading
import time
from queue import Queue, Empty
from ultralytics import YOLO

# ================= CONFIGURATION =================
MODEL_PATH = "yolov8n_ncnn_model" 
RTSP_USER = "admin"
RTSP_PASS = ""

INFERENCE_SIZE = 256 
TARGET_CLASSES = [0, 1, 2, 3, 5, 7, 15, 16] # Person, Vehicle, Pet

CAMERAS = [
    {"ip": "192.168.18.2", "name": "Front Gate"},
    {"ip": "192.168.18.113", "name": "Driveway"},
]

# ================= ENGINE =================
class TurboCPUEngine:
    def __init__(self, model_path):
        print(f"Loading {model_path} for NCNN inference...")
        self.model = YOLO(model_path, task='detect')
        self.input_queue = Queue(maxsize=1) 
        self.output_results = {}
        self.running = True
        self.fps = 0

    def start(self):
        t = threading.Thread(target=self._inference_worker, daemon=True)
        t.start()

    def _inference_worker(self):
        last_time = time.time()
        frame_count = 0
        while self.running:
            try:
                key, frame = self.input_queue.get(timeout=0.1)
                results = self.model.predict(
                    frame, imgsz=INFERENCE_SIZE, conf=0.35, 
                    classes=TARGET_CLASSES, verbose=False, half=True
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
            except Empty: continue
            except Exception as e:
                print(f"Inference Error: {e}")

# ================= WORKER =================
class StreamWorker:
    def __init__(self, ip, name, engine):
        self.name, self.engine = name, engine
        self.url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{ip}:554/h264"
        self.latest_frame, self.latest_dets = None, []
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
            if self.engine.input_queue.empty():
                try: self.engine.input_queue.put_nowait((key, frame))
                except: pass
            if key in self.engine.output_results:
                dets = self.engine.output_results.pop(key)
                with self.lock: self.latest_dets = dets
            with self.lock: self.latest_frame = frame
            f_idx += 1

# ================= MAIN =================
def main():
    print(f"🚀 Pi 5 NCNN Turbo Starting...")
    engine = TurboCPUEngine(MODEL_PATH)
    engine.start()
    
    workers = [StreamWorker(c['ip'], c['name'], engine) for c in CAMERAS]

    try:
        while True:
            display_frames = []
            for w in workers:
                with w.lock:
                    if w.latest_frame is None: continue
                    img = w.latest_frame.copy()
                    dets = w.latest_dets
                
                h, w_img = img.shape[:2]
                for d in dets:
                    x1, y1, x2, y2 = int(d[0]*w_img), int(d[1]*h), int(d[2]*w_img), int(d[3]*h)
                    label = engine.model.names[d[5]]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1-10), 0, 0.6, (0, 255, 0), 2)

                cv2.putText(img, f"FPS: {engine.fps:.1f}", (20, 40), 0, 0.7, (255, 255, 255), 2)
                display_frames.append(cv2.resize(img, (640, 360)))

            if display_frames:
                combined = cv2.hconcat(display_frames) if len(display_frames) > 1 else display_frames[0]
                cv2.imshow("Detection Dashboard", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        engine.running = False
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
