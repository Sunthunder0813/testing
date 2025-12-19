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
                        
                        # FIX: Handle raw_out as a list or array
                        raw_out = res[self.output_v]
                        data = raw_out[0] if isinstance(raw_out, list) else raw_out

                        new_dets = []
                        # Verification of shape (e.g., [N, 6])
                        if data is not None and hasattr(data, 'shape') and len(data.shape) >= 2:
                            for d in data:
                                # Ensure it has at least 6 elements [y1, x1, y2, x2, conf, cls]
                                if len(d) >= 6:
                                    try:
                                        conf = float(d[4])
                                        cls = int(d[5])
                                        if conf > 0.45 and cls == 0:
                                            # Scale normalized to pixel coordinates
                                            new_dets.append([
                                                int(d[1]*w), int(d[0]*h), 
                                                int(d[3]*w), int(d[2]*h)
                                            ])
                                    except: continue

                        with cam.lock:
                            cam.detections = new_dets

                        # Update FPS
                        curr_time = time.time()
                        self.fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 0
                        prev_time = curr_time

# ================= MAIN DASHBOARD =================
def main():
    print("💎 Pi 5 (Gen 3) + Hailo-8L: Industrial Surveillance Mode")
    cams = [Pi5Camera(c['ip'], c['name']) for c in CAMERAS]
    engine = None
    
    try:
        engine = HailoEngine(HEF_MODEL)
        # Run inference in a background thread
        t = threading.Thread(target=engine.infer_loop, args=(cams,), daemon=True)
        t.start()

        while True:
            display_frames = []
            for cam in cams:
                with cam.lock:
                    if cam.frame is None: continue
                    frame = cam.frame.copy()
                    boxes = list(cam.detections)
                
                # Draw boxes
                for box in boxes:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 127), 2)
                
                # Dashboard text
                cv2.putText(frame, f"{cam.name} | FPS: {engine.fps:.1f}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                display_frames.append(cv2.resize(frame, (854, 480)))

            if display_frames:
                # Merge cameras side-by-side
                combined = cv2.hconcat(display_frames) if len(display_frames) > 1 else display_frames[0]
                cv2.imshow("Hailo Surveillance", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Main Loop Error: {e}")
    finally:
        print("Cleaning up...")
        if engine: engine.device.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
