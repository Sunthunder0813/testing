import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
from ultralytics import YOLO

# ================= CONFIGURATION =================
MODEL_PATH = "yolov8n_ncnn_model"
INFERENCE_SIZE = 160  # Lowering to 160/192 significantly boosts FPS on Pi 5
TARGET_CLASSES = [0]  # Focus on Person (0) for max speed. Add others if needed.
NUM_THREADS = 4       # Utilize Pi 5's 4 cores

# ================= MULTI-THREADED ENGINE =================
class TurboEngine:
    def __init__(self, model_path):
        # Load model once
        self.model = YOLO(model_path, task='detect')
        self.input_queue = Queue(maxsize=1) # Only care about the absolute latest frame
        self.results = {}
        self.fps = 0
        self.running = True

    def start(self):
        # Dedicated thread for the NCNN inference loop
        threading.Thread(target=self._inference_loop, daemon=True).start()

    def _inference_loop(self):
        last_time = time.time()
        cnt = 0
        while self.running:
            try:
                cam_name, frame = self.input_queue.get(timeout=1)
                # CRITICAL: Run inference on a small, resized frame
                # verbose=False and half=True are mandatory for speed
                res = self.model.predict(
                    frame, 
                    imgsz=INFERENCE_SIZE, 
                    half=True, 
                    verbose=False, 
                    conf=0.4,
                    classes=TARGET_CLASSES
                )
                
                if res:
                    # Store only normalized coordinates to save memory/time
                    self.results[cam_name] = res[0].boxes.data.cpu().numpy()
                
                cnt += 1
                if time.time() - last_time > 1:
                    self.fps = cnt
                    cnt = 0
                    last_time = time.time()
            except Empty:
                continue

# ================= CAMERA WORKER =================
class CamStream:
    def __init__(self, url, name, engine):
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce RTSP lag
        self.name = name
        self.engine = engine
        self.latest_frame = None
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            self.latest_frame = frame
            # Try to push to engine, but don't block (skip if engine is busy)
            if self.engine.input_queue.empty():
                try:
                    self.engine.input_queue.put_nowait((self.name, frame))
                except: pass

# ================= MAIN DASHBOARD =================
def run_dashboard():
    engine = TurboEngine(MODEL_PATH)
    engine.start()

    # Update with your actual RTSP links
    streams = [
        CamStream("rtsp://admin:@192.168.18.2:554/h264", "Gate", engine),
        CamStream("rtsp://admin:@192.168.18.113:554/h264", "Drive", engine)
    ]

    while True:
        display_list = []
        for s in streams:
            frame = s.latest_frame
            if frame is None: continue
            
            # Draw detections if they exist for this camera
            if s.name in engine.results:
                dets = engine.results[s.name]
                h, w = frame.shape[:2]
                for d in dets:
                    x1, y1, x2, y2, conf, cls = d
                    # Scale normalized to pixel coordinates
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            cv2.putText(frame, f"{s.name} | Inference FPS: {engine.fps}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Resize for dashboard view
            display_list.append(cv2.resize(frame, (640, 480)))

        if display_list:
            combined = np.hstack(display_list)
            cv2.imshow("Pi 5 Turbo Monitor", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    run_dashboard()
