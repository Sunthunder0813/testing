import os
import cv2
import numpy as np
import threading
import time
import signal
import sys

# --- 1. HAILO IMPORTS & SAFETY ---
try:
    from hailo_platform import (
        HEF, VDevice, ConfigureParams, InputVStreamParams, 
        OutputVStreamParams, HailoStreamInterface, InferVStreams
    )
except ImportError:
    print("❌ Error: hailo_platform not found. Run: source hailo-venv/bin/activate")
    sys.exit(1)

# --- 2. OPTIMIZATION ENVIRONMENT VARIABLES ---
# Forces zero-latency and prevents frame buffering
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.35  # Lowered to 0.35 to catch more people
GREEN = (0, 255, 0)
CAMERAS = [
    {"ip": "192.168.18.2", "name": "Cam 1"},
    {"ip": "192.168.18.113", "name": "Cam 2"},
]
RTSP_USER = "admin"
RTSP_PASS = "" # <--- ENTER YOUR CAMERA PASSWORD HERE

# ================= GLOBAL STATE =================
shutdown_requested = False
def signal_handler(sig, frame):
    global shutdown_requested
    shutdown_requested = True
signal.signal(signal.SIGINT, signal_handler)

# ================= CAMERA THREAD =================
class StreamWorker:
    def __init__(self, url, name):
        self.name = name
        self.url = url
        self.latest_frame = None
        self.detections = [] 
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        while self.running and not shutdown_requested:
            # .grab() is critical: it keeps the buffer empty for 0 delay
            if not cap.grab():
                time.sleep(0.1)
                continue
            ret, frame = cap.retrieve()
            if ret:
                with self.lock:
                    self.latest_frame = frame
        cap.release()

# ================= AI WORKER THREAD =================
class AIWorker:
    def __init__(self, model_path, streams):
        self.hef = HEF(model_path)
        self.device = VDevice()
        
        # Configure the Hailo-8L hardware
        params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.device.configure(self.hef, params)[0]
        
        # Identify model input/output names
        self.input_name = self.hef.get_input_vstream_infos()[0].name
        self.output_name = self.hef.get_output_vstream_infos()[0].name
        self.target_shape = self.hef.get_input_vstream_infos()[0].shape[:2] # Usually (640, 640)
        
        self.streams = streams
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        # Create params in thread scope
        input_vparams = InputVStreamParams.make(self.network_group)
        output_vparams = OutputVStreamParams.make(self.network_group)

        with self.network_group.activate():
            with InferVStreams(self.network_group, input_vparams, output_vparams) as infer_pipeline:
                while self.running and not shutdown_requested:
                    for s in self.streams:
                        with s.lock:
                            if s.latest_frame is None: continue
                            frame_to_ai = s.latest_frame.copy()
                        
                        # Pre-process: Resize to match Hailo model input
                        h, w = self.target_shape
                        resized = cv2.resize(frame_to_ai, (w, h), interpolation=cv2.INTER_LINEAR)
                        
                        # Inference
                        try:
                            results = infer_pipeline.infer({self.input_name: np.expand_dims(resized, axis=0)})
                            raw_data = results[self.output_name][0]
                            
                            current_dets = []
                            # MANUAL PARSING: Checking for Person (Class 0)
                            if raw_data is not None:
                                for obj in raw_data:
                                    # Typical YOLOv8-NMS structure: [ymin, xmin, ymax, xmax, confidence, class_id]
                                    if len(obj) >= 6:
                                        conf = obj[4]
                                        cls_id = int(obj[5])
                                        
                                        if cls_id == 0 and conf >= CONF_THRESH:
                                            fh, fw = frame_to_ai.shape[:2]
                                            # Scale normalized (0-1) coordinates to pixel size
                                            x1 = int(obj[1] * fw)
                                            y1 = int(obj[0] * fh)
                                            x2 = int(obj[3] * fw)
                                            y2 = int(obj[2] * fh)
                                            current_dets.append([x1, y1, x2, y2, conf])
                            
                            with s.lock:
                                s.detections = current_dets
                        except Exception as e:
                            pass # Prevent thread crash if inference fails once
                    time.sleep(0.001)

# ================= MAIN EXECUTION =================
def main():
    print("🚀 Starting Raspberry Pi 5 + Hailo-8L Detection...")
    
    # 1. Start camera streams
    streams = []
    for c in CAMERAS:
        url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{c['ip']}:554/h264"
        streams.append(StreamWorker(url, c['name']))
    
    # 2. Start AI engine
    ai = AIWorker(HEF_MODEL, streams)
    ai.thread.start()

    print("📺 Display is open. Press 'q' to quit.")

    while not shutdown_requested:
        canvases = []
        for s in streams:
            with s.lock:
                frame = s.latest_frame.copy() if s.latest_frame is not None else np.zeros((480, 640, 3), np.uint8)
                dets = s.detections.copy()
            
            # 3. DRAWING PHASE: Green frames for persons
            for (x1, y1, x2, y2, score) in dets:
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
                label = f"PERSON {int(score*100)}%"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
            
            # Show name on top left
            cv2.putText(frame, s.name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            canvases.append(cv2.resize(frame, (640, 480)))

        # Combined view
        if canvases:
            cv2.imshow("Hailo-8L Detect", cv2.hconcat(canvases))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # CLEANUP
    print("🧹 Cleaning up...")
    ai.running = False
    for s in streams: s.running = False
    cv2.destroyAllWindows()
    ai.device.release()

if __name__ == "__main__":
    main()
