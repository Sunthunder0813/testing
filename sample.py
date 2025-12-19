import os
import cv2
import numpy as np
import threading
import time
import signal
import sys

# 1. CORE HAILO IMPORTS
try:
    from hailo_platform import (
        HEF, VDevice, ConfigureParams, InputVStreamParams, 
        OutputVStreamParams, HailoStreamInterface, InferVStreams
    )
except ImportError:
    print("❌ Error: hailo_platform not found. Ensure your venv is active.")
    sys.exit(1)

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.50  
GREEN_TARGET = (0, 255, 127) 
RTSP_USER = "admin"
RTSP_PASS = "" # YOUR PASSWORD

CAMERAS = [
    {"ip": "192.168.18.2", "name": "Front Entry"},
    {"ip": "192.168.18.113", "name": "Backyard"},
]

# ================= UTILS & DRAWING =================
def draw_professional_box(img, box, score):
    x, y, x2, y2 = map(int, box)
    label = f"HUMAN {int(score*100)}%"
    
    # Draw Thick Corners
    c_len = int((x2 - x) * 0.2)
    # Box Shadow for contrast
    cv2.rectangle(img, (x-1, y-1), (x2+1, y2+1), (0, 30, 0), 1)
    cv2.rectangle(img, (x, y), (x2, y2), GREEN_TARGET, 2)
    
    # Fancy Corners
    pts = [((x,y),(x+c_len,y)), ((x,y),(x,y+c_len)), 
           ((x2,y),(x2-c_len,y)), ((x2,y),(x2,y+c_len)),
           ((x,y2),(x+c_len,y2)), ((x,y2),(x,y2-c_len)),
           ((x2,y2),(x2-c_len,y2)), ((x2,y2),(x2,y2-c_len))]
    for p1, p2 in pts:
        cv2.line(img, p1, p2, GREEN_TARGET, 5)

    # Label Background
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.6, 1)
    cv2.rectangle(img, (x, y - th - 15), (x + tw + 15, y), GREEN_TARGET, -1)
    cv2.putText(img, label, (x + 7, y - 8), font, 0.6, (0,0,0), 1, cv2.LINE_AA)

# ================= CAMERA WORKER =================
class CameraStream:
    def __init__(self, ip, name):
        self.name = name
        self.url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{ip}:554/h264"
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.frame = None
        self.detections = []
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

# ================= HAILO ENGINE =================
class HailoInference:
    def __init__(self, model_path):
        self.hef = HEF(model_path)
        self.target_shape = self.hef.get_input_vstream_infos()[0].shape[:2]
        self.device = VDevice()
        
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.device.configure(self.hef, configure_params)[0]
        self.input_vstream_params = InputVStreamParams.make(self.network_group)
        self.output_vstream_params = OutputVStreamParams.make(self.network_group)
        self.input_name = self.hef.get_input_vstream_infos()[0].name
        self.output_name = self.hef.get_output_vstream_infos()[0].name

    def process(self, streams):
        with self.network_group.activate():
            with InferVStreams(self.network_group, self.input_vstream_params, self.output_vstream_params) as infer_pipeline:
                while True:
                    for s in streams:
                        with s.lock:
                            if s.frame is None: continue
                            local_frame = s.frame.copy()
                        
                        # Prepare Image
                        fh, fw = local_frame.shape[:2]
                        resized = cv2.resize(local_frame, (self.target_shape[1], self.target_shape[0]))
                        
                        # Run Inference
                        res = infer_pipeline.infer({self.input_name: np.expand_dims(resized, axis=0)})
                        
                        # SAFE PARSING: Avoid the "inhomogeneous" array error
                        # We convert the output directly to a flat list of detections
                        raw_output = res[self.output_name][0]
                        final_dets = []
                        
                        # Handle varied output formats (Flattened vs Structured)
                        for i in range(len(raw_output)):
                            det = raw_output[i]
                            # Check if the detection is a valid array/list of at least 6 items
                            if hasattr(det, "__len__") and len(det) >= 6:
                                ymin, xmin, ymax, xmax, conf, cls_id = det
                                
                                # ONLY PERSON (Class 0)
                                if conf > CONF_THRESH and int(cls_id) == 0:
                                    final_dets.append([
                                        int(xmin * fw), int(ymin * fh), 
                                        int(xmax * fw), int(ymax * fh), conf
                                    ])
                        
                        with s.lock:
                            s.detections = final_dets
                    time.sleep(0.001)

# ================= MAIN LOOP =================
def main():
    print("💎 Initializing Hailo-8L Professional Guard...")
    streams = [CameraStream(c['ip'], c['name']) for c in CAMERAS]
    engine = HailoInference(HEF_MODEL)
    
    # Run AI in Background
    threading.Thread(target=engine.process, args=(streams,), daemon=True).start()

    while True:
        display_frames = []
        for s in streams:
            with s.lock:
                if s.frame is None: continue
                canvas = s.frame.copy()
                current_dets = list(s.detections)

            for d in current_dets:
                draw_professional_box(canvas, d[:4], d[4])

            # Header
            cv2.rectangle(canvas, (0,0), (300, 60), (0,0,0), -1)
            cv2.putText(canvas, f"CAM: {s.name}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            display_frames.append(cv2.resize(canvas, (854, 480)))

        if display_frames:
            # Combine side-by-side
            combined = cv2.hconcat(display_frames) if len(display_frames) > 1 else display_frames[0]
            cv2.imshow("Hailo-8L Monitoring System", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Closing...")
    for s in streams: s.running = False
    cv2.destroyAllWindows()
    engine.device.release()

if __name__ == "__main__":
    main()
