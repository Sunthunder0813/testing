import os
import cv2
import numpy as np
import threading
import time
import sys
from hailo_platform import (
    HEF, VDevice, ConfigureParams, InputVStreamParams, 
    OutputVStreamParams, HailoStreamInterface, InferVStreams
)

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.55  
GREEN_TARGET = (0, 255, 127) 
RTSP_USER = "admin"
RTSP_PASS = "" # YOUR PASSWORD

CAMERAS = [
    {"ip": "192.168.18.2", "name": "Zone A"},
    {"ip": "192.168.18.113", "name": "Zone B"},
]

# ================= UI & DRAWING FUNCTIONS =================
def draw_counter_hud(img, count, cam_name):
    """Draws a professional HUD display for the person counter."""
    text = f"CAM: {cam_name} | PERSON COUNT: {count}"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.8
    thickness = 2
    
    # Get text size for background rectangle
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    # Draw Background Shadow Box (Black, semi-transparent)
    # Positions the HUD at top-left with some padding
    padding = 10
    x, y = 20, 40
    cv2.rectangle(img, (x - padding, y - th - padding), 
                  (x + tw + padding, y + baseline + padding), (0, 0, 0), -1)
    
    # Draw Text
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Status Dot (Blinking effect logic)
    dot_color = (0, 0, 255) if count > 0 else (100, 100, 100)
    cv2.circle(img, (x + tw + 30, y - 10), 8, dot_color, -1)

def draw_target_ui(img, box, score):
    """Draws the Targeting Brackets around the detected person."""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    
    # 1. Target Glow
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), GREEN_TARGET, -1)
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

    # 2. Main Border & Precision Corners
    cv2.rectangle(img, (x1, y1), (x2, y2), GREEN_TARGET, 1)
    c_len = int(w * 0.18) 
    t = 4
    cv2.line(img, (x1, y1), (x1+c_len, y1), GREEN_TARGET, t) # TL
    cv2.line(img, (x1, y1), (x1, y1+c_len), GREEN_TARGET, t)
    cv2.line(img, (x2, y1), (x2-c_len, y1), GREEN_TARGET, t) # TR
    cv2.line(img, (x2, y1), (x2, y1+c_len), GREEN_TARGET, t)
    cv2.line(img, (x1, y2), (x1+c_len, y2), GREEN_TARGET, t) # BL
    cv2.line(img, (x1, y2), (x1, y2-c_len), GREEN_TARGET, t)
    cv2.line(img, (x2, y2), (x2-c_len, y2), GREEN_TARGET, t) # BR
    cv2.line(img, (x2, y2), (x2, y2-c_len), GREEN_TARGET, t)

# ================= STREAM WORKER =================
class CameraWorker:
    def __init__(self, ip, name):
        self.name = name
        self.url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{ip}:554/h264"
        self.latest_frame = None
        self.detections = []
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._stream, daemon=True).start()

    def _stream(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
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
    print("💎 Initializing Hailo-8L Surveillance Mode...")
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

            # 1. Draw Target Brackets
            for d in dets:
                draw_target_ui(frame, d[:4], d[4])

            # 2. Draw the Person Counter HUD
            draw_counter_hud(frame, len(dets), s.name)
            
            display_canvases.append(cv2.resize(frame, (854, 480)))

        if display_canvases:
            combined = cv2.hconcat(display_canvases) if len(display_canvases) > 1 else display_canvases[0]
            cv2.imshow("Hailo-8L Target Monitor", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
    ai.device.release()

if __name__ == "__main__":
    main()
