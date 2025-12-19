#!/usr/bin/env python3
import os
import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty

# --- SYSTEM & DIRECTORY SETUP ---
# Fixes potential Wayland/X11 display issues on Pi 5
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Points exactly to the person model in your ~/testing folder
HEF_MODEL_PATH = os.path.join(SCRIPT_DIR, "yolov8n_person.hef")

# --- CAMERA & DETECTION CONFIG ---
CONF_THRESHOLD = 0.45
TARGET_CLASS_ID = 0  # 0 is "person" in standard COCO-based YOLO models

CAMERAS = [
    {"ip": "192.168.18.2", "name": "Front Gate"},
    {"ip": "192.168.18.113", "name": "Driveway"}, 
]

# --- HAILO NPU ENGINE ---
from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams

class HailoInferenceEngine:
    def __init__(self, hef_path):
        if not os.path.exists(hef_path):
            print(f"❌ ERROR: HEF not found: {hef_path}")
            exit(1)
            
        self.hef = HEF(hef_path)
        self.target = VDevice(VDevice.create_params())
        
        # Configure the NPU
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        
        # Get input dimensions (usually 640x640)
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
        
        # FIX: Correct way to generate vstream params for your SDK version
        vstream_params = self.network_group.create_vstreams_params()
        
        with InferVStreams(self.network_group, vstream_params) as infer_pipeline:
            input_vstream_name = self.network_group.get_input_vstream_infos()[0].name
            
            while self.running:
                try:
                    cam_name, frame = self.input_queue.get(timeout=0.1)
                except Empty:
                    continue

                # Prepare frame for NPU
                resized = cv2.resize(frame, (self.input_w, self.input_h))
                input_data = {input_vstream_name: np.expand_dims(resized, axis=0)}
                
                # Inference
                raw_output = infer_pipeline.infer(input_data)
                
                # Parse detections (Format: [x1, y1, x2, y2, confidence, class_id])
                output_name = list(raw_output.keys())[0]
                detections = raw_output[output_name][0]
                
                valid_persons = []
                for det in detections:
                    if len(det) >= 6:
                        conf, cls_id = det[4], int(det[5])
                        if conf > CONF_THRESHOLD and cls_id == TARGET_CLASS_ID:
                            valid_persons.append(det)
                
                self.results[cam_name] = valid_persons

                frame_count += 1
                if time.time() - last_time >= 1.0:
                    self.fps = frame_count
                    frame_count = 0
                    last_time = time.time()

# --- RTSP STREAMING ---
class CameraStream:
    def __init__(self, cam_info, engine):
        self.name = cam_info['name']
        self.url = f"rtsp://admin:@{cam_info['ip']}:554/h264"
        self.engine = engine
        self.latest_frame = None
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        while True:
            cap = cv2.VideoCapture(self.url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                time.sleep(5)
                continue
            while True:
                ret, frame = cap.read()
                if not ret: break
                self.latest_frame = frame
                if self.engine.input_queue.empty():
                    try:
                        self.engine.input_queue.put_nowait((self.name, frame))
                    except: pass
            cap.release()
            time.sleep(1)

# --- DISPLAY ---
def main():
    print(f"🚀 Initializing Hailo-8L Person Detection...")
    engine = HailoInferenceEngine(HEF_MODEL_PATH)
    engine.start()

    streams = [CameraStream(c, engine) for c in CAMERAS]

    try:
        while True:
            display_list = []
            for s in streams:
                if s.latest_frame is not None:
                    frame = s.latest_frame.copy()
                    h, w = frame.shape[:2]
                    
                    if s.name in engine.results:
                        for d in engine.results[s.name]:
                            x1, y1, x2, y2 = int(d[0]*w), int(d[1]*h), int(d[2]*w), int(d[3]*h)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"PERSON {d[4]:.2f}", (x1, y1-10), 0, 0.6, (0, 255, 0), 2)

                    cv2.putText(frame, f"{s.name} | NPU: {engine.fps} FPS", (20, 30), 0, 0.8, (255, 255, 255), 2)
                    display_list.append(cv2.resize(frame, (854, 480)))
            
            if display_list:
                cv2.imshow("Hailo-8L Dashboard", np.hstack(display_list))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        engine.running = False
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
