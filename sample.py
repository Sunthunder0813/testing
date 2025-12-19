import os
import cv2
import numpy as np
import threading
import time
import signal
import sys

# 1. HAILO PLATFORM IMPORTS
try:
    from hailo_platform import (
        HEF, VDevice, ConfigureParams, InputVStreamParams, 
        OutputVStreamParams, HailoStreamInterface, InferVStreams
    )
except ImportError:
    print("❌ Error: hailo_platform not found.")
    sys.exit(1)

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.45  # Increased for higher accuracy
IOU_THRESH = 0.45   # For NMS accuracy
GREEN_BRIGHT = (0, 255, 127)
SHADOW = (0, 40, 0)
CAMERAS = [
    {"ip": "192.168.18.2", "name": "Entrance"},
    {"ip": "192.168.18.113", "name": "Backyard"},
]
RTSP_USER = "admin"
RTSP_PASS = "" # ENTER YOUR PASSWORD

shutdown_requested = False
def signal_handler(sig, frame):
    global shutdown_requested
    shutdown_requested = True
signal.signal(signal.SIGINT, signal_handler)

# ================= ADVANCED DRAWING UI =================
def draw_pro_target(img, box, label):
    x, y, x2, y2 = box
    w, h = x2 - x, y2 - y
    
    # 1. Subtle Background Glow (The "Nice" effect)
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), GREEN_BRIGHT, -1)
    cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)

    # 2. Main Border with Shadow for legibility
    cv2.rectangle(img, (x+1, y+1), (x2+1, y2+1), SHADOW, 2) # Shadow
    cv2.rectangle(img, (x, y), (x2, y2), GREEN_BRIGHT, 2)   # Line

    # 3. Targeting Corners (Professional look)
    t_len = int(w * 0.15) # Length of corner lines
    # Top Left
    cv2.line(img, (x, y), (x + t_len, y), GREEN_BRIGHT, 5)
    cv2.line(img, (x, y), (x, y + t_len), GREEN_BRIGHT, 5)
    # Top Right
    cv2.line(img, (x2, y), (x2 - t_len, y), GREEN_BRIGHT, 5)
    cv2.line(img, (x2, y), (x2, y + t_len), GREEN_BRIGHT, 5)
    # Bottom Left
    cv2.line(img, (x, y2), (x + t_len, y2), GREEN_BRIGHT, 5)
    cv2.line(img, (x, y2), (x, y2 - t_len), GREEN_BRIGHT, 5)
    # Bottom Right
    cv2.line(img, (x2, y2), (x2 - t_len, y2), GREEN_BRIGHT, 5)
    cv2.line(img, (x2, y2), (x2, y2 - t_len), GREEN_BRIGHT, 5)

    # 4. Professional Label Tag
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Label Background
    cv2.rectangle(img, (x, y - th - 15), (x + tw + 20, y), GREEN_BRIGHT, -1)
    # Label Text
    cv2.putText(img, label, (x + 10, y - 8), font, font_scale, (0,0,0), thickness + 1, cv2.LINE_AA)

# ================= STREAM WORKER =================
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
            if not cap.grab(): continue
            ret, frame = cap.retrieve()
            if ret:
                with self.lock:
                    self.latest_frame = frame
        cap.release()

# ================= AI ENGINE =================
class AIWorker:
    def __init__(self, model_path, streams):
        self.hef = HEF(model_path)
        self.device = VDevice()
        params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.device.configure(self.hef, params)[0]
        
        self.input_name = self.hef.get_input_vstream_infos()[0].name
        self.output_name = self.hef.get_output_vstream_infos()[0].name
        self.target_shape = self.hef.get_input_vstream_infos()[0].shape[:2]
        
        self.streams = streams
        self.running = True

    def _run(self):
        input_vparams = InputVStreamParams.make(self.network_group)
        output_vparams = OutputVStreamParams.make(self.network_group)

        with self.network_group.activate():
            with InferVStreams(self.network_group, input_vparams, output_vparams) as infer_pipeline:
                while self.running and not shutdown_requested:
                    for s in self.streams:
                        with s.lock:
                            if s.latest_frame is None: continue
                            raw_frame = s.latest_frame.copy()
                        
                        # Prepare input
                        h, w = self.target_shape
                        resized = cv2.resize(raw_frame, (w, h))
                        
                        # Inference
                        results = infer_pipeline.infer({self.input_name: np.expand_dims(resized, axis=0)})
                        raw_out = results[self.output_name][0]
                        
                        # High-Accuracy Parsing
                        new_dets = []
                        fh, fw = raw_frame.shape[:2]
                        for i in range(0, len(raw_out), 6):
                            ymin, xmin, ymax, xmax, conf, cls_id = raw_out[i:i+6]
                            if conf > CONF_THRESH and int(cls_id) == 0:
                                # Scale to original frame size
                                new_dets.append([
                                    int(xmin * fw), int(ymin * fh), 
                                    int(xmax * fw), int(ymax * fh), conf
                                ])
                        
                        with s.lock:
                            s.detections = new_dets
                    time.sleep(0.001)

# ================= MAIN APP =================
def main():
    print("💎 Initializing Premium Hailo-8L Detection...")
    streams = [StreamWorker(f"rtsp://{RTSP_USER}:{RTSP_PASS}@{c['ip']}:554/h264", c['name']) for c in CAMERAS]
    ai = AIWorker(HEF_MODEL, streams)
    threading.Thread(target=ai._run, daemon=True).start()

    while not shutdown_requested:
        views = []
        for s in streams:
            with s.lock:
                if s.latest_frame is None: continue
                frame = s.latest_frame.copy()
                dets = s.detections

            for d in dets:
                label = f"PERSON | {int(d[4]*100)}%"
                draw_pro_target(frame, d[:4], label)

            # Add Camera Label
            cv2.putText(frame, s.name, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            views.append(cv2.resize(frame, (800, 450))) # Larger view for better detail

        if views:
            combined = cv2.hconcat(views) if len(views) > 1 else views[0]
            cv2.imshow("Hailo-8L Precision Guard", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
    ai.device.release()

if __name__ == "__main__":
    main()
