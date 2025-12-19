import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
from ultralytics import YOLO

# ================= ⚡ ACCURACY & SPEED CONFIG ⚡ =================
MODEL_PATH = "yolov8n_ncnn_model"  
# Use a tuple for aspect ratio: (Height, Width) 
# 160x288 or 192x320 maintains the 16:9 ratio of most RTSP cameras
INFERENCE_SIZE = (192, 320)         
TARGET_CLASS = 0                   
CONF_THRESHOLD = 0.35              # Lower = more detections, Higher = fewer "ghosts"
IOU_THRESHOLD = 0.45               # Helps clean up overlapping boxes

CAMERAS = [
    {"ip": "192.168.18.2", "name": "Front Gate"},
    {"ip": "192.168.18.113", "name": "Driveway"},
]

class TurboInferenceEngine:
    def __init__(self, model_path):
        self.model = YOLO(model_path, task='detect')
        self.input_queue = Queue(maxsize=1) 
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
                cam_name, frame = self.input_queue.get(timeout=0.01)
            except Empty:
                continue

            # --- ACCURACY IMPROVEMENTS ---
            # 1. Use 'rect=True' for rectangular aspect ratios
            # 2. Set 'imgsz' to your specific tuple
            results = self.model.predict(
                frame, 
                imgsz=INFERENCE_SIZE, 
                classes=[TARGET_CLASS], 
                half=True, 
                verbose=False,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                augment=False # Keep False for speed
            )

            if results and len(results[0].boxes) > 0:
                # Store pixel-relative coordinates for accurate drawing
                self.results[cam_name] = results[0].boxes.data.cpu().numpy()
            else:
                self.results[cam_name] = []

            frame_count += 1
            if time.time() - last_time >= 1.0:
                self.fps = frame_count
                frame_count = 0
                last_time = time.time()

class CameraStream:
    def __init__(self, camera_info, engine):
        self.name = camera_info['name']
        self.url = f"rtsp://admin:@{camera_info['ip']}:554/h264"
        self.engine = engine
        self.latest_frame = None
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(2) 
                cap.open(self.url)
                continue
            
            self.latest_frame = frame
            
            if self.engine.input_queue.empty():
                try:
                    self.engine.input_queue.put_nowait((self.name, frame))
                except: pass

def main():
    print(f"🚀 Dashboard Running. Optimized for Person Detection Accuracy.")
    engine = TurboInferenceEngine(MODEL_PATH)
    engine.start()

    streams = [CameraStream(c, engine) for c in CAMERAS]

    while True:
        display_frames = []
        
        for s in streams:
            frame = s.latest_frame
            if frame is None: continue
            
            # Use a local copy to avoid drawing on the frame being sent to inference
            draw_frame = frame.copy()
            
            if s.name in engine.results:
                dets = engine.results[s.name]
                for d in dets:
                    x1, y1, x2, y2, conf, cls = d
                    # Draw with confidence for accuracy verification
                    cv2.rectangle(draw_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(draw_frame, f"PER {conf:.2f}", (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(draw_frame, f"{s.name} | Inference: {engine.fps} FPS", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            display_frames.append(cv2.resize(draw_frame, (854, 480)))

        if display_frames:
            combined = np.hstack(display_frames) if len(display_frames) > 1 else display_frames[0]
            cv2.imshow("Multi-Cam Accurate Person Detection", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
