import cv2
import threading
import time
from queue import Queue
from ultralytics import YOLO

# ================= CONFIGURATION =================
MODEL_PATH = "yolo11n_ncnn_model"  # The folder created by export
RTSP_URL = "rtsp://admin:@192.168.1.100:554/stream" # Your IP Camera
INPUT_SIZE = 256  # Must match the export size

class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = Queue(maxsize=2)
        self.stopped = False
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret: break
            if not self.q.full():
                self.q.put(frame)
            else:
                self.q.get(); self.q.put(frame) # Keep latest frame

    def read(self):
        return self.q.get() if not self.q.empty() else None

# ================= MAIN LOOP =================
def run_detector():
    print("🚀 Initializing NCNN Turbo Mode...")
    model = YOLO(MODEL_PATH, task='detect')
    stream = VideoStream(RTSP_URL)
    
    prev_time = 0
    
    while True:
        frame = stream.read()
        if frame is None: continue

        # Optimized Inference
        results = model.predict(
            frame, 
            imgsz=INPUT_SIZE, 
            conf=0.4, 
            verbose=False, 
            half=True, 
            stream=True
        )

        # Draw Green Borders
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Target", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(frame, f"CPU FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Pi 5 CPU Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detector()
