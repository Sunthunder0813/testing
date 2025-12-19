import os
import cv2
import numpy as np
import threading
import time
import sys
from datetime import datetime
from hailo_platform import (
    HEF, VDevice, ConfigureParams, InputVStreamParams, 
    OutputVStreamParams, HailoStreamInterface, InferVStreams
)

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.55  
GREEN_TARGET = (0, 255, 127) 
SMOOTH_FACTOR = 0.35 # 0.0 to 1.0 (Lower is smoother/slower, Higher is snappier)
RTSP_USER = "admin"
RTSP_PASS = "" 
SAVE_PATH = "detections"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

CAMERAS = [
    {"ip": "192.168.18.2", "name": "Zone A"},
    {"ip": "192.168.18.113", "name": "Zone B"},
]

# ================= BOX SMOOTHER =================
class BoxSmoother:
    """Prevents the green box from flickering/jumping."""
    def __init__(self, alpha=SMOOTH_FACTOR):
        self.alpha = alpha
        self.smooth_box = None

    def update(self, new_box):
        if self.smooth_box is None:
            self.smooth_box = np.array(new_box, dtype=float)
        else:
            # Weighted average: Smooth = (1-a)*Old + a*New
            self.smooth_box = (1 - self.alpha) * self.smooth_box + self.alpha * np.array(new_box)
        return self.smooth_box.astype(int)

# ================= UI & DRAWING =================
def draw_ui(img, box, obj_id, score):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    label = f"LOCKED ID:{obj_id}"
    
    # 1. Corner Brackets
    c_len = int(w * 0.18)
    t = 4
    cv2.rectangle(img, (x1, y1), (x2, y2), GREEN_TARGET, 1)
    # TL
    cv2.line(img, (x1, y1), (x1+c_len, y1), GREEN_TARGET, t)
    cv2.line(img, (x1, y1), (x1, y1+c_len), GREEN_TARGET, t)
    # BR
    cv2.line(img, (x2, y2), (x2-c_len, y2), GREEN_TARGET, t)
    cv2.line(img, (x2, y2), (x2, y2-c_len), GREEN_TARGET, t)

    # 2. Tech-Style Label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, label, (x1, y1 - 10), font, 0.5, GREEN_TARGET, 1, cv2.LINE_AA)

# ================= STREAM WORKER =================
class CameraWorker:
    def __init__(self, ip, name):
        self.name = name
        self.url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{ip}:554/h264"
        self.latest_frame = None
        self.detections = [] 
        self.smoothers = {} # ID -> BoxSmoother
        self.lock = threading.Lock()
        self.running = True
        # Optimized for RPi: Small buffer to prevent lag
        threading.Thread(target=self._stream, daemon=True).start()

    def _stream(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Force low latency
        while self.running:
            ret, frame = cap.read()
            if ret:
                with self.lock: self.latest_frame = frame
            else:
                time.sleep(0.01)

# ================= AI MANAGER =================
class AIModelManager:
    def __init__(self, model_path):
        self.hef = HEF(model_path)
        self.device = VDevice()
        self.target_shape = self.hef.get_input_vstream_infos()[0].shape[:2]
        params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.device.configure(self.hef, params)[0]
        self.input_name = self.hef.get_input_vstream_infos()[0].name
        self.output_name = self.hef.get_output_vstream_infos()[0].name

    def run_inference(self, streams):
        with self.network_group.activate():
            with InferVStreams(self.network_group, 
                               InputVStreamParams.make(self.network_group), 
                               OutputVStreamParams.make(self.network_group)) as pipeline:
                while True:
                    for s in streams:
                        with s.lock:
                            if s.latest_frame is None: continue
                            img = s.latest_frame.copy()
                        fh, fw = img.shape[:2]
                        resized = cv2.resize(img, (self.target_shape[1], self.target_shape[0]))
                        results = pipeline.infer({self.input_name: np.expand_dims(resized, axis=0)})
                        raw_data = results[self.output_name][0]
                        
                        valid_dets = []
                        if hasattr(raw_data, "__len__"):
                            for d in raw_data:
                                if len(d) >= 6:
                                    ymin, xmin, ymax, xmax, conf, cls_id = d
                                    if float(conf) > CONF_THRESH and int(cls_id) == 0:
                                        valid_dets.append([int(xmin*fw), int(ymin*fh), int(xmax*fw), int(ymax*fh), float(conf)])
                        with s.lock: s.detections = valid_dets
                    time.sleep(0.001)

# ================= MAIN =================
def main():
    print(f"🚀 Smooth Stream Active. Smoothing Factor: {SMOOTH_FACTOR}")
    streams = [CameraWorker(c['ip'], c['name']) for c in CAMERAS]
    ai = AIModelManager(HEF_MODEL)
    threading.Thread(target=ai.run_inference, args=(streams,), daemon=True).start()

    while True:
        display_canvases = []
        for s in streams:
            with s.lock:
                if s.latest_frame is None: continue
                frame = s.latest_frame.copy()
                dets = list(s.detections)

            # Logic to smooth each box
            # To keep it simple for this version, we smooth based on the index of detection
            for i, d in enumerate(dets):
                if i not in s.smoothers:
                    s.smoothers[i] = BoxSmoother()
                
                smooth_box = s.smoothers[i].update(d[:4])
                draw_ui(frame, smooth_box, i+1, d[4])

            # Remove smoothers for objects no longer detected
            if len(dets) < len(s.smoothers):
                s.smoothers = {i: s.smoothers[i] for i in range(len(dets))}

            # HUD Display
            cv2.rectangle(frame, (10, 10), (350, 60), (0,0,0), -1)
            cv2.putText(frame, f"{s.name} | PEOPLE: {len(dets)}", (25, 45), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            display_canvases.append(cv2.resize(frame, (854, 480)))

        if display_canvases:
            combined = cv2.hconcat(display_canvases) if len(display_canvases) > 1 else display_canvases[0]
            cv2.imshow("Hailo-8L Smooth Monitor", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
    ai.device.release()

if __name__ == "__main__":
    main()
