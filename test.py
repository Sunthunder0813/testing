import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
from ultralytics import YOLO

# ================= HIGH-SPEED CONFIG =================
# Use 'yolov8n.pt' for the fastest CPU performance
MODEL_PATH = "yolov8n.pt" 
BATCH_SIZE = 1 # CPU usually performs better with batch 1 for real-time
RTSP_USER = "admin"
RTSP_PASS = ""

CAMERAS = [
    {"ip": "192.168.18.2", "name": "Front Gate"},
    {"ip": "192.168.18.113", "name": "Driveway"},
]

# ================= CPU INFERENCE ENGINE =================
class CPUTurboEngine:
    def __init__(self, model_path):
        # Load the standard PyTorch model (runs on CPU by default)
        self.model = YOLO(model_path)
        self.input_queue = Queue(maxsize=128)
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
                # CPU is often faster at individual frames than small batches
                key, frame = self.input_queue.get(timeout=0.01)
            except Empty:
                continue

            # 1. Run Inference (CPU)
            # classes=[0] filters for 'person' only in COCO dataset
            results = self.model.predict(frame, conf=0.45, classes=[0], verbose=False)
            
            # 2. Extract Detections
            # Results are in format: [y1, x1, y2, x2, conf] to match your original logic
            if len(results) > 0:
                boxes = results[0].boxes
                raw_dets = []
                for box in boxes:
                    coords = box.xyxyn[0].tolist() # Normalized [x1, y1, x2, y2]
                    conf = float(box.conf[0])
                    # Reorder to match your original loop: [y1, x1, y2, x2, conf]
                    raw_dets.append([coords[1], coords[0], coords[3], coords[2], conf])
                
                self.output_results[key] = raw_dets

            # Update FPS tracking
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
                time.sleep(1)
                cap.open(self.url)
                continue
            
            # Use a smaller size for CPU inference to maintain FPS
            # YOLOv8n default is 640x640
            key = f"{self.name}_{f_idx}"
            self.engine.input_queue.put((key, frame))

            # Fetch results
            if key in self.engine.output_results:
                dets = self.engine.output_results.pop(key)
                with self.lock:
                    self.latest_dets = dets
            
            with self.lock:
                self.latest_frame = frame
            f_idx += 1

# ================= MAIN =================
def main():
    print("🚀 Running YOLOv8 on CPU Mode")
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
            if dets:
                for d in dets:
                    # d format: [y1, x1, y2, x2, conf]
                    y1, x1, y2, x2 = int(d[0]*fh), int(d[1]*fw), int(d[2]*fh), int(d[3]*fw)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 127), 2)

            cv2.putText(img, f"{w.name} | CPU FPS: {engine.fps:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            display_frames.append(cv2.resize(img, (854, 480)))

        if display_frames:
            combined = cv2.hconcat(display_frames) if len(display_frames) > 1 else display_frames[0]
            cv2.imshow("CPU Detection Dashboard", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    engine.running = False
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
