import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
from ultralytics import YOLO

# ================= CONFIGURATION =================
MODEL_PATH = "yolov8n.pt"  # Nano model is fastest for CPU
RTSP_USER = "admin"
RTSP_PASS = ""

# List of COCO IDs to detect: 
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 15: bird, 16: cat, 17: dog
TARGET_CLASSES = [0, 1, 2, 3, 15, 16, 17]

CAMERAS = [
    {"ip": "192.168.18.2", "name": "Front Gate"},
    {"ip": "192.168.18.113", "name": "Driveway"},
]

# ================= CPU ENGINE =================
class CPUTurboEngine:
    def __init__(self, model_path):
        # Loads the standard PyTorch model (runs on CPU)
        self.model = YOLO(model_path)
        self.input_queue = Queue(maxsize=64)
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
                # Get latest frame from any camera
                key, frame = self.input_queue.get(timeout=0.01)
            except Empty:
                continue

            # 1. Run Inference
            # imgsz=320 makes it much faster on Pi 5 CPU than default 640
            results = self.model.predict(
                frame, 
                conf=0.4, 
                imgsz=320, 
                classes=TARGET_CLASSES, 
                verbose=False
            )
            
            # 2. Extract Data
            if len(results) > 0:
                boxes = results[0].boxes
                raw_dets = []
                for box in boxes:
                    # Normalized coords [x1, y1, x2, y2]
                    coords = box.xyxyn[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]
                    
                    # Store as [y1, x1, y2, x2, conf, label]
                    raw_dets.append([coords[1], coords[0], coords[3], coords[2], conf, label])
                
                self.output_results[key] = raw_dets

            # FPS Tracking
            frame_count += 1
            if time.time() - last_time >= 1.0:
                self.fps = frame_count / (time.time() - last_time)
                frame_count = 0
                last_time = time.time()

# ================= CAMERA WORKER =================
class CameraWorker:
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
            
            # Non-blocking put: if queue is full, skip frame to stay real-time
            try:
                self.engine.input_queue.put_nowait((key, frame))
            except:
                pass

            # Check for results belonging to this camera
            if key in self.engine.output_results:
                dets = self.engine.output_results.pop(key)
                with self.lock:
                    self.latest_dets = dets
            
            with self.lock:
                self.latest_frame = frame
            f_idx += 1

# ================= MAIN LOOP =================
def main():
    print("🚀 Starting CPU Object Detection (Person/Vehicle/Pet)")
    engine = CPUTurboEngine(MODEL_PATH)
    engine.start()
    
    workers = [CameraWorker(c['ip'], c['name'], engine) for c in CAMERAS]

    while True:
        display_frames = []
        for w in workers:
            with w.lock:
                if w.latest_frame is None: continue
                img = w.latest_frame.copy()
                dets = w.latest_dets
            
            fh, fw = img.shape[:2]
            
            # Draw detections
            for d in dets:
                # Format: [y1, x1, y2, x2, conf, label]
                y1, x1, y2, x2 = int(d[0]*fh), int(d[1]*fw), int(d[2]*fh), int(d[3]*fw)
                label = d[5]
                conf = d[4]

                color = (0, 255, 127) # Bright Green
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Label background
                label_txt = f"{label} {conf:.2f}"
                cv2.putText(img, label_txt, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Dashboard Info
            cv2.putText(img, f"{w.name} | Total CPU FPS: {engine.fps:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Resize for display grid (SD Resolution)
            display_frames.append(cv2.resize(img, (854, 480)))

        if display_frames:
            # Combine camera views side-by-side
            combined = cv2.hconcat(display_frames) if len(display_frames) > 1 else display_frames[0]
            cv2.imshow("Multi-Object CPU Dashboard", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    engine.running = False
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
