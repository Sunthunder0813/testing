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
CONF_THRESH = 0.50  
GREEN_TARGET = (0, 255, 127) 

CAMERAS = [
    {"ip": "192.168.18.2", "name": "Front Entry"},
    {"ip": "192.168.18.113", "name": "Backyard"},
]
RTSP_USER = "admin"
RTSP_PASS = "" # YOUR PASSWORD

shutdown_requested = False
def signal_handler(sig, frame):
    global shutdown_requested
    shutdown_requested = True
signal.signal(signal.SIGINT, signal_handler)

# ================= UI DRAWING =================
def draw_target(img, box, label):
    x, y, x2, y2 = map(int, box)
    w, h = x2 - x, y2 - y
    
    cv2.rectangle(img, (x, y), (x2, y2), GREEN_TARGET, 2)
    # Corners
    c_len = int(w * 0.2)
    cv2.line(img, (x, y), (x + c_len, y), GREEN_TARGET, 5)
    cv2.line(img, (x, y), (x, y + c_len), GREEN_TARGET, 5)
    cv2.line(img, (x2, y), (x2 - c_len, y), GREEN_TARGET, 5)
    cv2.line(img, (x2, y), (x2, y + c_len), GREEN_TARGET, 5)
    cv2.line(img, (x, y2), (x + c_len, y2), GREEN_TARGET, 5)
    cv2.line(img, (x, y2), (x, y2 - c_len), GREEN_TARGET, 5)
    cv2.line(img, (x2, y2), (x2 - c_len, y2), GREEN_TARGET, 5)
    cv2.line(img, (x2, y2), (x2, y2 - c_len), GREEN_TARGET, 5)

    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.6, 1)
    cv2.rectangle(img, (x, y - th - 15), (x + tw + 15, y), GREEN_TARGET, -1)
    cv2.putText(img, label, (x + 7, y - 8), font, 0.6, (0,0,0), 1, cv2.LINE_AA)

# ================= STREAM WORKER =================
class StreamWorker:
    def __init__(self, ip, name):
        self.name = name
        self.url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{ip}:554/h264"
        self.latest_frame = None
        self.detections = [] 
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        while self.running and not shutdown_requested:
            ret, frame = cap.read()
            if ret:
                with self.lock: self.latest_frame = frame
            else:
                time.sleep(0.01)
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

    def _run(self):
        input_vparams = InputVStreamParams.make(self.network_group)
        output_vparams = OutputVStreamParams.make(self.network_group)

        with self.network_group.activate():
            with InferVStreams(self.network_group, input_vparams, output_vparams) as infer_pipeline:
                while not shutdown_requested:
                    for s in self.streams:
                        with s.lock:
                            if s.latest_frame is None: continue
                            frame = s.latest_frame.copy()
                        
                        h, w = self.target_shape
                        resized = cv2.resize(frame, (w, h))
                        
                        # Inference
                        infer_results = infer_pipeline.infer({self.input_name: np.expand_dims(resized, axis=0)})
                        
                        # FIX: Extract results as a raw list first to avoid NumPy inhomogeneous error
                        detections_list = infer_results[self.output_name][0]
                        
                        new_dets = []
                        fh, fw = frame.shape[:2]

                        # Handle the data as a list of detections
                        # Hailo NMS results are often padded; we iterate through the valid ones
                        for detection in detections_list:
                            # Depending on your HEF, this might be a flat array or a structured object
                            # We assume flat array [ymin, xmin, ymax, xmax, conf, class_id]
                            if len(detection) >= 6:
                                ymin, xmin, ymax, xmax, conf, cls_id = detection
                                
                                # Strict Person-Only Check
                                if conf > CONF_THRESH and int(cls_id) == 0:
                                    new_dets.append([
                                        int(xmin * fw), int(ymin * fh), 
                                        int(xmax * fw), int(ymax * fh), float(conf)
                                    ])
                        
                        with s.lock: s.detections = new_dets
                    time.sleep(0.001)

# ================= MAIN =================
def main():
    print("💎 Hailo-8L: High-Accuracy Person Detection")
    streams = [StreamWorker(c['ip'], c['name']) for c in CAMERAS]
    ai = AIWorker(HEF_MODEL, streams)
    
    ai_thread = threading.Thread(target=ai._run, daemon=True)
    ai_thread.start()

    while not shutdown_requested:
        canvases = []
        for s in streams:
            with s.lock:
                if s.latest_frame is None: continue
                img = s.latest_frame.copy()
                dets = list(s.detections)

            for d in dets:
                draw_target(img, d[:4], f"PERSON {int(d[4]*100)}%")

            cv2.putText(img, s.name, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            canvases.append(cv2.resize(img, (800, 450)))

        if canvases:
            display = cv2.hconcat(canvases) if len(canvases) > 1 else canvases[0]
            cv2.imshow("Hailo-8L Guard", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    print("\n[!] Shutting down...")
    cv2.destroyAllWindows()
    ai.device.release()

if __name__ == "__main__":
    main()
