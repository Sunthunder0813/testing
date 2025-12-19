import os
import cv2
import numpy as np
import threading
import time
from hailo_platform import (
    HEF, VDevice, ConfigureParams, InputVStreamParams, 
    OutputVStreamParams, HailoStreamInterface, InferVStreams
)

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
RTSP_USER = "admin"
RTSP_PASS = "" 
MAX_FPS = 25  # Matches your camera's hardware limit

CAMERAS = [
    {"ip": "192.168.18.2", "name": "Front Gate"},
    {"ip": "192.168.18.113", "name": "Driveway"},
]

# ================= CAMERA CLASS =================
class Pi5Camera:
    def __init__(self, ip, name):
        self.name = name
        # Optimization: Clear buffers and force TCP for 25fps stability
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        self.url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{ip}:554/h264"
        
        self.frame = None
        self.new_frame_ready = False
        self.detections = []
        self.lock = threading.Lock()
        self.running = True
        
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                cap.open(self.url)
                continue
            with self.lock:
                self.frame = frame
                self.new_frame_ready = True

# ================= HAILO ENGINE =================
class HailoEngine:
    def __init__(self, model_path):
        self.device = VDevice()
        self.hef = HEF(model_path)
        self.target_shape = self.hef.get_input_vstream_infos()[0].shape[:2] 
        
        params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.device.configure(self.hef, params)[0]
        self.input_v = self.hef.get_input_vstream_infos()[0].name
        self.output_v = self.hef.get_output_vstream_infos()[0].name
        self.fps = 0

    def infer_loop(self, cameras):
        prev_time = time.time()
        frame_duration = 1.0 / MAX_FPS
        
        with self.network_group.activate():
            with InferVStreams(self.network_group, 
                               InputVStreamParams.make(self.network_group), 
                               OutputVStreamParams.make(self.network_group)) as pipeline:
                while True:
                    loop_start = time.time()
                    
                    for cam in cameras:
                        # Only infer if there is a new frame to avoid redundant work
                        with cam.lock:
                            if cam.frame is None: continue
                            raw_f = cam.frame.copy()
                            cam.new_frame_ready = False
                        
                        h, w = raw_f.shape[:2]
                        resized = cv2.resize(raw_f, (self.target_shape[1], self.target_shape[0]))
                        
                        # Inference
                        res = pipeline.infer({self.input_v: np.expand_dims(resized, axis=0)})
                        
                        # Robust List/Array Parsing
                        raw_out = res[self.output_v]
                        data = raw_out[0] if isinstance(raw_out, list) else raw_out

                        new_dets = []
                        if data is not None and hasattr(data, 'shape') and len(data.shape) >= 2:
                            for d in data:
                                if len(d) >= 6:
                                    try:
                                        conf = float(d[4])
                                        cls = int(d[5])
                                        if conf > 0.45 and cls == 0:
                                            new_dets.append([
                                                int(d[1]*w), int(d[0]*h), 
                                                int(d[3]*w), int(d[2]*h)
                                            ])
                                    except: continue

                        with cam.lock:
                            cam.detections = new_dets

                    # Sync logic: Calculate how long to sleep to maintain 25 FPS
                    elapsed = time.time() - loop_start
                    if elapsed < frame_duration:
                        time.sleep(frame_duration - elapsed)

                    # Update Dashboard FPS
                    curr_time = time.time()
                    self.fps = 1 / (curr_time - prev_time)
                    prev_time = curr_time

# ================= MAIN DASHBOARD =================
def main():
    print(f"💎 Pi 5 (Gen 3) + Hailo-8L: Locked to {MAX_FPS} FPS")
    cams = [Pi5Camera(c['ip'], c['name']) for c in CAMERAS]
    engine = None
    
    try:
        engine = HailoEngine(HEF_MODEL)
        threading.Thread(target=engine.infer_loop, args=(cams,), daemon=True).start()

        while True:
            display_frames = []
            for cam in cams:
                with cam.lock:
                    if cam.frame is None: continue
                    frame = cam.frame.copy()
                    boxes = list(cam.detections)
                
                for box in boxes:
                    # Drawing box with high-visibility neon green
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 127), 2)
                    # Label background
                    cv2.putText(frame, "PERSON", (box[0], box[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 127), 2)
                
                # Camera Dashboard Info
                cv2.putText(frame, f"{cam.name} | NPU: {engine.fps:.1f} FPS", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                display_frames.append(cv2.resize(frame, (854, 480)))

            if display_frames:
                combined = cv2.hconcat(display_frames) if len(display_frames) > 1 else display_frames[0]
                cv2.imshow("Hailo Surveillance", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if engine: engine.device.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
