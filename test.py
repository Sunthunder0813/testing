#!/usr/bin/env python3
import os
import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
import signal

# ================= 🛠️ ENVIRONMENT & PATHS 🛠️ =================
# Force X11 for the Pi 5 to avoid Wayland-related window crashes
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Update this path to your confirmed location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HEF_MODEL_PATH = os.path.join(SCRIPT_DIR, "yolov8n_person.hef")

# ================= ⚙️ CONFIGURATION ⚙️ =================
CONF_THRESHOLD = 0.45
TARGET_CLASS_ID = 0  # 0 is 'person' in standard COCO models

# List your RTSP Cameras here
CAMERAS = [
    {"ip": "192.168.18.2", "name": "Front Gate"},
    {"ip": "192.168.18.71", "name": "Driveway"},
]

# ================= 🧠 HAILO INFERENCE ENGINE 🧠 =================
try:
    from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False

class HailoInferenceEngine:
    def __init__(self, hef_path):
        if not os.path.exists(hef_path):
            print(f"❌ ERROR: HEF file not found at {hef_path}")
            exit(1)
            
        self.hef = HEF(hef_path)
        self.target = VDevice(VDevice.create_params())
        
        # Configure the NPU pipeline
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()
        
        # Determine model input dimensions (usually 640x640 for YOLOv8)
        input_vstream_info = self.network_group.get_input_vstream_infos()[0]
        self.input_h, self.input_w = input_vstream_info.shape[1], input_vstream_info.shape[2]
        
        self.input_queue = Queue(maxsize=1)
        self.results = {}
        self.fps = 0
        self.running = True

    def start(self):
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        last_time = time.time()
        frame_count = 0
        
        # The InferVStreams context manager handles the NPU data flow
        with InferVStreams(self.network_group, self.network_group_params) as infer_pipeline:
            input_vstream_name = self.network_group.get_input_vstream_infos()[0].name
            
            while self.running:
                try:
                    cam_name, frame = self.input_queue.get(timeout=0.1)
                except Empty:
                    continue

                # Pre-processing: Resize to match model input
                resized_frame = cv2.resize(frame, (self.input_w, self.input_h))
                
                # Inference
                input_data = {input_vstream_name: np.expand_dims(resized_frame, axis=0)}
                raw_output = infer_pipeline.infer(input_data)
                
                # Extract detections (assumes model has HailoRT post-processing included)
                # Output format: [x1, y1, x2, y2, confidence, class_id]
                output_name = list(raw_output.keys())[0]
                detections = raw_output[output_name][0]
                
                valid_detections = []
                for det in detections:
                    if len(det) >= 6:
                        conf, cls_id = det[4], int(det[5])
                        if conf > CONF_THRESHOLD and cls_id == TARGET_CLASS_ID:
                            valid_detections.append(det)
                
                self.results[cam_name] = valid_detections

                # Performance calculation
                frame_count += 1
                if time.time() - last_time >= 1.0:
                    self.fps = frame_count
                    frame_count = 0
                    last_time = time.time()

# ================= 📹 STREAM HANDLER 📹 =================
class CameraStream:
    def __init__(self, cam_info, engine):
        self.name = cam_info['name']
        self.url = f"rtsp://admin:@{cam_info['ip']}:554/h264"
        self.engine = engine
        self.latest_frame = None
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Keep latency low
        
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(2) # Retry connection if dropped
                cap.open(self.url)
                continue
            
            self.latest_frame = frame
            
            # Update the inference engine with the newest frame
            if self.engine.input_queue.empty():
                try:
                    self.engine.input_queue.put_nowait((self.name, frame))
                except: pass

# ================= 🖥️ MAIN EXECUTION 🖥️ =================
def main():
    if not HAILO_AVAILABLE:
        print("❌ Hailo Platform library not installed. Run: sudo apt install hailo-all")
        return

    print(f"🚀 Initializing Hailo-8L with model: {os.path.basename(HEF_MODEL_PATH)}")
    engine = HailoInferenceEngine(HEF_MODEL_PATH)
    engine.start()

    streams = [CameraStream(c, engine) for c in CAMERAS]

    try:
        while True:
            display_frames = []
            
            for s in streams:
                frame = s.latest_frame
                if frame is None: continue
                
                # Draw detections on a copy for display
                draw_frame = frame.copy()
                if s.name in engine.results:
                    h_img, w_img = frame.shape[:2]
                    for d in engine.results[s.name]:
                        # Hailo coordinates are typically normalized (0 to 1)
                        x1, y1, x2, y2 = int(d[0]*w_img), int(d[1]*h_img), int(d[2]*w_img), int(d[3]*h_img)
                        conf = d[4]
                        
                        cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(draw_frame, f"PERSON {conf:.2f}", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Dashboard Info
                cv2.putText(draw_frame, f"{s.name} | NPU: {engine.fps} FPS", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Resize for side-by-side dashboard view
                display_frames.append(cv2.resize(draw_frame, (854, 480)))

            if display_frames:
                combined = np.hstack(display_frames) if len(display_frames) > 1 else display_frames[0]
                cv2.imshow("Hailo-8L Person Detection Dashboard", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        engine.running = False
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
