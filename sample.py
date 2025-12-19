import os
import cv2
import numpy as np
import threading
import time
from hailo_platform import (
    HEF, VDevice, ConfigureParams, InputVStreamParams, 
    OutputVStreamParams, HailoStreamInterface, InferVStreams
)

# ================= PERFORMANCE CONFIG =================
HEF_MODEL = "yolov8n_person.hef"
RTSP_USER = "admin"
RTSP_PASS = "" 

# Smoothing: 0.1 (Liquid) to 0.9 (Instant)
SMOOTH_ALPHA = 0.25 

CAMERAS = [
    {"ip": "192.168.18.2", "name": "Front Gate"},
    {"ip": "192.168.18.113", "name": "Driveway"},
]

# ================= PI 5 OPTIMIZED CAPTURE =================
class Pi5Camera:
    def __init__(self, ip, name):
        self.name = name
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        self.url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{ip}:554/h264"
        
        self.frame = None
        self.detections = []
        self.smoothed_boxes = {} 
        self.lock = threading.Lock()
        self.running = True
        
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                cap.open(self.url)
                continue
            with self.lock:
                self.frame = frame

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
        prev_time = 0
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
                        raw_out = res[self.output_v][0]
                        
                        # Calculate FPS
                        curr_time = time.time()
                        self.fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
                        prev_time = curr_time

                        new_dets = []
                        # Robust Parsing to fix IndexError
                        if raw_out is not None and len(raw_out.shape) >= 2:
                            for d in raw_out:
                                if len(d) >= 6: # Ensure index 4 and 5 exist
                                    if d[4] > 0.45 and int(d[5]) == 0:
                                        new_dets.append([int(d[1]*w), int(d[0]*h), int(d[3]*w), int(d[2]*h)])
                        
                        with cam.lock:
                            cam.detections = new_dets

# ================= UI & SMOOTHING =================
def smooth_ui(cam):
    with cam.lock:
        raw_boxes = cam.detections
        if not raw_boxes:
            return []
        for i, target in enumerate(raw_boxes):
            if i not in cam.smoothed_boxes:
                cam.smoothed_boxes[i] = np.array(target, dtype=float)
            else:
                curr = np.array(target)
                prev = cam.smoothed_boxes[i]
                cam.smoothed_boxes[i] = (prev * (1 - SMOOTH_ALPHA)) + (curr * SMOOTH_ALPHA)
        return list(cam.smoothed_boxes.values())

def main():
    print("💎 Pi 5 (Gen 3) + Hailo-8L: Industrial Surveillance")
    cams = [Pi5Camera(c['ip'], c['name']) for c in CAMERAS]
    engine = None
    
    try:
        engine = HailoEngine(HEF_MODEL)
        threading.Thread(target=engine.infer_loop, args=(cams,), daemon=True).start()

        while True:
            display_list = []
            for cam in cams:
                with cam.lock:
                    if cam.frame is None: continue
                    frame = cam.frame.copy()
                
                boxes = smooth_ui(cam)
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)
                
                # Overlay Camera Name and Stats
                cv2.putText(frame, f"{cam.name} | FPS: {engine.fps:.1f}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 127), 2)
                display_list.append(cv2.resize(frame, (854, 480)))

            if display_list:
                combined = cv2.hconcat(display_list) if len(display_list)>1 else display_list[0]
                cv2.imshow("Hailo-8L Pi5 Dashboard", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cv2.destroyAllWindows()
        if engine: engine.device.release()

if __name__ == "__main__":
    main()
