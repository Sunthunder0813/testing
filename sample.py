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

CAMERAS = [
    {"ip": "192.168.18.2", "name": "Front Gate"},
    {"ip": "192.168.18.113", "name": "Driveway"},
]

# ================= CAMERA CLASS =================
class Pi5Camera:
    def __init__(self, ip, name):
        self.name = name
        # Force low-latency settings for Pi 5
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        self.url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{ip}:554/h264"
        
        self.frame = None
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
                self.frame = frame # Always overwrite with newest frame

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
        with self.network_group.activate():
            with InferVStreams(self.network_group, 
                               InputVStreamParams.make(self.network_group), 
                               OutputVStreamParams.make(self.network_group)) as pipeline:
                while True:
                    for cam in cameras:
                        with cam.lock:
                            if cam.frame is None: continue
                            raw_f = cam.frame.copy()
                        
                        h, w = raw_f.shape[:2]
                        resized = cv2.resize(raw_f, (self.target_shape[1], self.target_shape[0]))
                        
                        # Inference
                        res = pipeline.infer({self.input_v: np.expand_dims(resized, axis=0)})
                        # Res format can be {name: [array]} or {name: array}
                        raw_out = res[self.output_v]
                        
                        # Handle potential batch list wrapping
                        if isinstance(raw_out, list):
                            data = raw_out[0]
                        else:
                            data = raw_out

                        new_dets = []
                        # Correcting the "length-1 array" conversion error
                        if data is not None and len(data.shape) >= 2:
                            for d in data:
                                # Logic: Access values as scalars even if wrapped in arrays
                                try:
                                    conf = float(d[4]) if d[4].size == 1 else float(d[4][0])
                                    cls = int(d[5]) if d[5].size == 1 else int(d[5][0])
                                    
                                    if conf > 0.45 and cls == 0: # Person detection
                                        new_dets.append([
                                            int(d[1]*w), int(d[0]*h), # x1, y1
                                            int(d[3]*w), int(d[2]*h)  # x2, y2
                                        ])
                                except (IndexError, TypeError):
                                    continue

                        with cam.lock:
                            cam.detections = new_dets

                        # FPS Calculation
                        curr_time = time.time()
                        self.fps = 1 / (curr_time - prev_time)
                        prev_time = curr_time

# ================= MAIN DISPLAY =================
def main():
    print("💎 Pi 5 (Gen 3) + Hailo-8L: Running Dashboard...")
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
                    boxes = cam.detections
                
                for box in boxes:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 127), 2)
                
                cv2.putText(frame, f"{cam.name} | FPS: {engine.fps:.1f}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
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
