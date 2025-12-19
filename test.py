#!/usr/bin/env python3
import os
import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
import signal

# --- ENVIRONMENT FIXES ---
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HEF_MODEL_PATH = os.path.join(SCRIPT_DIR, "yolov8n_person.hef")

CONF_THRESHOLD = 0.45
TARGET_CLASS_ID = 0  # Person

# Updated to use the IPs mentioned in your previous successful logs
CAMERAS = [
    {"ip": "192.168.18.2", "name": "Front Gate"},
    {"ip": "192.168.18.113", "name": "Driveway"}, 
]

# --- HAILO INFERENCE ENGINE ---
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
        
        # Get vstream info and dimensions
        input_vstream_infos = self.network_group.get_input_vstream_infos()
        self.input_h, self.input_w = input_vstream_infos[0].shape[1], input_vstream_infos[0].shape[2]
        
        self.input_queue = Queue(maxsize=1)
        self.results = {}
        self.fps = 0
        self.running = True

    def start(self):
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        last_time = time.time()
        frame_count = 0
        
        # 1. Get vstream infos
        input_vstream_infos = self.network_group.get_input_vstream_infos()
        output_vstream_infos = self.network_group.get_output_vstream_infos()

        # 2. Correctly create BOTH input and output params (fixes the TypeError)
        input_vstreams_params = InferVStreams.get_params(self.network_group, input_vstream_infos)
        output_vstreams_params = InferVStreams.get_params(self.network_group, output_vstream_infos)
        
        # 3. Initialize the pipeline with all required positional arguments
        with InferVStreams(self.network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            input_vstream_name = input_vstream_infos[0].name
            
            while self.running:
                try:
                    cam_name, frame = self.input_queue.get(timeout=0.1)
                except Empty:
                    continue

                # Pre-processing
                resized_frame = cv2.resize(frame, (self.input_w, self.input_h))
                input_data = {input_vstream_name: np.expand_dims(resized_frame, axis=0)}
                
                # Inference
                raw_output = infer_pipeline.infer(input_data)
                
                # Parsing results
                output_name = list(raw_output.keys())[0]
                detections = raw_output[output_name][0]
                
                valid_dets = []
                for det in detections:
                    if len(det) >= 6:
                        # [x1, y1, x2, y2, confidence, class_id]
                        conf, cls_id = det[4], int(det[5])
                        if conf > CONF_THRESHOLD and cls_id == TARGET_CLASS_ID:
                            valid_dets.append(det)
                
                self.results[cam_name] = valid_dets

                frame_count += 1
                if time.time() - last_time >= 1.0:
                    self.fps = frame_count
                    frame_count = 0
                    last_time = time.time()

# --- STREAM HANDLER ---
class CameraStream:
    def __init__(self, cam_info, engine):
        self.name = cam_info['name']
        self.url = f"rtsp://admin:@{cam_info['ip']}:554/h264"
        self.engine = engine
        self.latest_frame = None
        self.is_connected = False
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        while True:
            cap = cv2.VideoCapture(self.url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                self.is_connected = False
                print(f"⚠️ Failed to connect to {self.name}. Retrying in 5s...")
                time.sleep(5)
                continue

            self.is_connected = True
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.is_connected = False
                    break
                
                self.latest_frame = frame
                if self.engine.input_queue.empty():
                    try:
                        self.engine.input_queue.put_nowait((self.name, frame))
                    except: pass
            
            cap.release()
            time.sleep(2)

# --- MAIN DASHBOARD ---
def main():
    if not HAILO_AVAILABLE:
        print("❌ Hailo library missing.")
        return

    print(f"🚀 Initializing Hailo-8L | Model: {os.path.basename(HEF_MODEL_PATH)}")
    engine = HailoInferenceEngine(HEF_MODEL_PATH)
    engine.start()

    streams = [CameraStream(c, engine) for c in CAMERAS]

    try:
        while True:
            display_frames = []
            for s in streams:
                if s.latest_frame is None:
                    # Black frame for offline cameras
                    frame = np.zeros((480, 854, 3), dtype=np.uint8)
                    cv2.putText(frame, f"{s.name} OFFLINE", (250, 240), 0, 1, (0, 0, 255), 2)
                else:
                    frame = s.latest_frame.copy()
                    h_img, w_img = frame.shape[:2]
                    
                    if s.name in engine.results:
                        for d in engine.results[s.name]:
                            x1, y1, x2, y2 = int(d[0]*w_img), int(d[1]*h_img), int(d[2]*w_img), int(d[3]*h_img)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"PERSON {d[4]:.2f}", (x1, y1-10), 0, 0.5, (0, 255, 0), 2)

                    cv2.putText(frame, f"{s.name} | {engine.fps} FPS", (20, 40), 0, 0.7, (255, 255, 255), 2)
                    frame = cv2.resize(frame, (854, 480))
                
                display_frames.append(frame)

            if display_frames:
                combined = np.hstack(display_frames)
                cv2.imshow("Hailo-8L Dashboard", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        engine.running = False
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
