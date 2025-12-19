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
    print("❌ Error: hailo_platform not found.")
    sys.exit(1)

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.55  # High confidence for accuracy
GREEN_TARGET = (0, 255, 127) # Safety Neon Green
RTSP_USER = "admin"
RTSP_PASS = "" # YOUR PASSWORD

CAMERAS = [
    {"ip": "192.168.18.2", "name": "Zone A"},
    {"ip": "192.168.18.113", "name": "Zone B"},
]

# ================= ADVANCED TARGET DRAWING =================
def draw_target_ui(img, box, score):
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    label = f"TARGET: {int(score*100)}%"
    
    # 1. Subtle Glow (Fill)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), GREEN_TARGET, -1)
    cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)

    # 2. Main Thin Border
    cv2.rectangle(img, (x1, y1), (x2, y2), GREEN_TARGET, 1)

    # 3. Weighted Corners (Targeting Effect)
    # This makes the border "look" like it's locking onto the person
    c_len = int(w * 0.15) # Length of the corner lines
    thickness = 4
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + c_len, y1), GREEN_TARGET, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + c_len), GREEN_TARGET, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - c_len, y1), GREEN_TARGET, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + c_len), GREEN_TARGET, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + c_len, y2), GREEN_TARGET, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - c_len), GREEN_TARGET, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - c_len, y2), GREEN_TARGET, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - c_len), GREEN_TARGET, thickness)

    # 4. Label with Background Tag
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 15), (x1 + tw + 15, y1), GREEN_TARGET, -1)
    cv2.putText(img, label, (x1 + 7, y1 - 8), font, 0.5, (0,0,0), 1, cv2.LINE_AA)

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

# ================= HAILO AI MANAGER =================
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
        input_vparams = InputVStreamParams.make(self.network_group)
        output_vparams = OutputVStreamParams.make(self.network_group)

        with self.network_group.activate():
            with InferVStreams(self.network_group, input_vparams, output_vparams) as pipeline:
                while True:
                    for s in streams:
                        with s.lock:
                            if s.latest_frame is None: continue
                            img = s.latest_frame.copy()
                        
                        fh, fw = img.shape[:2]
                        resized = cv2.resize(img, (self.target_shape[1], self.target_shape[0]))
                        
                        # Inference
                        results = pipeline.infer({self.input_name: np.expand_dims(resized, axis=0)})
                        
                        # SAFE PARSING (Avoids ValueError)
                        raw_data = results[self.output_name][0]
                        valid_dets = []
                        
                        # Only loop if raw_data is iterable
                        if hasattr(raw_data, "__getitem__"):
                            for i in range(len(raw_data)):
                                d = raw_data[i]
                                if len(d) >= 6:
                                    ymin, xmin, ymax, xmax, conf, cls_id = d
                                    # STRICT PERSON FILTER (Class 0)
                                    if float(conf) > CONF_THRESH and int(cls_id) == 0:
                                        valid_dets.append([
                                            int(xmin * fw), int(ymin * fh), 
                                            int(xmax * fw), int(ymax * fh), float(conf)
                                        ])
                        
                        with s.lock: s.detections = valid_dets
                    time.sleep(0.001)

# ================= MAIN EXECUTION =================
def main():
    print("🚀 Initializing Hailo-8L Target Guard...")
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

            for d in dets:
                draw_target_ui(frame, d[:4], d[4])

            # Label the Camera
            cv2.putText(frame, f"LIVE | {s.name}", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            display_canvases.append(cv2.resize(frame, (854, 480)))

        if display_canvases:
            combined = cv2.hconcat(display_canvases) if len(display_canvases) > 1 else display_canvases[0]
            cv2.imshow("Hailo-8L Precision Detection", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
    ai.device.release()

if __name__ == "__main__":
    main()
