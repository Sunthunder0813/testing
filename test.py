import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
from ultralytics import YOLO

# ================= ⚡ PERFORMANCE CONFIG ⚡ =================
MODEL_PATH = "yolov8n_ncnn_model"  # Path to your NCNN export folder
INFERENCE_SIZE = 160               # Critical: Lowering to 160px enables 20+ FPS
TARGET_CLASS = 0                   # 0 is the COCO class ID for "person"
CONF_THRESHOLD = 0.40              # Minimum confidence to show a box

# Camera list: Add as many as needed (Pi 5 handles 2-3 comfortably at high FPS)
CAMERAS = [
    {"ip": "192.168.18.2", "name": "Front Gate"},
    {"ip": "192.168.18.113", "name": "Driveway"},
]

# ================= 🧠 TURBO INFERENCE ENGINE =================
class TurboInferenceEngine:
    def __init__(self, model_path):
        # Load NCNN model (optimized for ARM NEON instructions)
        self.model = YOLO(model_path, task='detect')
        self.input_queue = Queue(maxsize=1)  # Only process the single freshest frame
        self.results = {}
        self.fps = 0
        self.running = True

    def start(self):
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        last_time = time.time()
        frame_count = 0
        
        while self.running:
            try:
                # Grab the freshest frame from any camera
                cam_name, frame = self.input_queue.get(timeout=0.01)
            except Empty:
                continue

            # Run person-only inference
            # half=True and imgsz=160 are the performance keys here
            results = self.model.predict(
                frame, 
                imgsz=INFERENCE_SIZE, 
                classes=[TARGET_CLASS], 
                half=True, 
                verbose=False,
                conf=CONF_THRESHOLD
            )

            if results:
                # Store normalized results to map back to original frame size
                self.results[cam_name] = results[0].boxes.data.cpu().numpy()

            # Calculate actual Inference FPS
            frame_count += 1
            if time.time() - last_time >= 1.0:
                self.fps = frame_count
                frame_count = 0
                last_time = time.time()

# ================= 📹 STREAM HANDLER =================
class CameraStream:
    def __init__(self, camera_info, engine):
        self.name = camera_info['name']
        self.url = f"rtsp://admin:@{camera_info['ip']}:554/h264"
        self.engine = engine
        self.latest_frame = None
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Force minimal buffer for real-time
        
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(2) # Retry connection
                cap.open(self.url)
                continue
            
            self.latest_frame = frame
            
            # Feed the engine if it's ready for a new frame
            if self.engine.input_queue.empty():
                try:
                    self.engine.input_queue.put_nowait((self.name, frame))
                except: pass

# ================= 🖥️ MAIN DASHBOARD =================
def main():
    print(f"🚀 Starting Person-Only Dashboard on Pi 5...")
    engine = TurboInferenceEngine(MODEL_PATH)
    engine.start()

    # Initialize all cameras
    streams = [CameraStream(c, engine) for c in CAMERAS]

    while True:
        display_frames = []
        
        for s in streams:
            frame = s.latest_frame
            if frame is None: continue
            
            # Draw detections for this specific camera
            if s.name in engine.results:
                dets = engine.results[s.name]
                h, w_img = frame.shape[:2]
                
                for d in dets:
                    # x1, y1, x2, y2, confidence, class_id
                    x1, y1, x2, y2, conf, cls = d
                    
                    # Draw green box for person
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"PERSON {conf:.2f}", (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Dashboard Overlay
            cv2.putText(frame, f"{s.name} | Engine: {engine.fps} FPS", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Resize for dashboard grid
            display_frames.append(cv2.resize(frame, (854, 480)))

        if display_frames:
            # Layout: Horizontal stack of all cameras
            combined = np.hstack(display_frames) if len(display_frames) > 1 else display_frames[0]
            cv2.imshow("Multi-Cam Person Detection", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
