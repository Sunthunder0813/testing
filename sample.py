import os
import cv2
import numpy as np
import threading
import time
import signal
import sys

# 1. HAILO IMPORTS
try:
    from hailo_platform import (
        HEF, VDevice, ConfigureParams, InputVStreamParams, 
        OutputVStreamParams, HailoStreamInterface, InferVStreams
    )
except ImportError:
    print("❌ Error: hailo_platform not found. Ensure your venv is active.")
    sys.exit(1)

# 2. ZERO-LATENCY OPTIMIZATIONS
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
# TCP prevents the 'blinking' common with UDP packet loss
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.40  # Balanced for accuracy and speed
GREEN = (0, 255, 0)
CAMERAS = [
    {"ip": "192.168.18.2", "name": "Cam 1"},
    {"ip": "192.168.18.113", "name": "Cam 2"},
]
RTSP_USER = "admin"
RTSP_PASS = "" # ENTER PASSWORD HERE

# ================= GLOBAL STATE =================
shutdown_requested = False
def signal_handler(sig, frame):
    global shutdown_requested
    shutdown_requested = True
signal.signal(signal.SIGINT, signal_handler)

# ================= CAMERA WORKER =================
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
        # We use FFMPEG for best H.264 performance
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        while self.running and not shutdown_requested:
            # .grab() flushes the network buffer to keep the feed "LIVE"
            if not cap.grab():
                time.sleep(0.1)
                continue
            ret, frame = cap.retrieve()
            if ret:
                with self.lock:
                    self.latest_frame = frame
        cap.release()

# ================= AI WORKER =================
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
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        input_vparams = InputVStreamParams.make(self.network_group)
        output_vparams = OutputVStreamParams.make(self.network_group)

        with self.network_group.activate():
            with InferVStreams(self.network_group, input_vparams, output_vparams) as infer_pipeline:
                while self.running and not shutdown_requested:
                    for s in self.streams:
                        with s.lock:
                            if s.latest_frame is None: continue
                            frame_to_ai = s.latest_frame.copy()
                        
                        h, w = self.target_shape
                        resized = cv2.resize(frame_to_ai, (w, h), interpolation=cv2.INTER_LINEAR)
                        
                        try:
                            # Run inference on Hailo hardware
                            results = infer_pipeline.infer({self.input_name: np.expand_dims(resized, axis=0)})
                            raw_data = results[self.output_name][0]
                            
                            current_dets = []
                            if raw_data is not None:
                                for obj in raw_data:
                                    # obj: [ymin, xmin, ymax, xmax, confidence, class_id]
                                    if len(obj) >= 6:
                                        conf, cls_id = obj[4], int(obj[5])
                                        if cls_id == 0 and conf >= CONF_THRESH:
                                            fh, fw = frame_to_ai.shape[:2]
                                            # Scale normalized coords to pixels
                                            x1, y1 = int(obj[1] * fw), int(obj[0] * fh)
                                            x2, y2 = int(obj[3] * fw), int(obj[2] * fh)
                                            current_dets.append([x1, y1, x2, y2, conf])
                            
                            with s.lock:
                                s.detections = current_dets
                        except: pass
                    time.sleep(0.001)

# ================= DISPLAY LOOP =================
def main():
    print(f"🚀 Starting Dual-Cam H.264 AI on Raspberry Pi 5...")
    
    # Init Streams
    streams = [StreamWorker(f"rtsp://{RTSP_USER}:{RTSP_PASS}@{c['ip']}:554/h264", c['name']) for c in CAMERAS]
    
    # Init AI
    ai = AIWorker(HEF_MODEL, streams)
    ai.thread.start()

    print("📺 Display Active. Press 'q' to quit.")

    while not shutdown_requested:
        canvases = []
        for s in streams:
            with s.lock:
                frame = s.latest_frame.copy() if s.latest_frame is not None else np.zeros((480, 640, 3), np.uint8)
                dets = s.detections.copy()
            
            # DRAW GREEN FRAMES
            for (x1, y1, x2, y2, score) in dets:
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
                cv2.putText(frame, f"PERSON {int(score*100)}%", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
            
            # Add camera name and downscale for display side-by-side
            cv2.putText(frame, s.name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            canvases.append(cv2.resize(frame, (640, 480)))

        if canvases:
            # Join the two camera feeds together horizontally
            cv2.imshow("Hailo-8L Dual Cam", cv2.hconcat(canvases))
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # CLEANUP
    ai.running = False
    cv2.destroyAllWindows()
    ai.device.release()
    print("👋 Exit complete.")

if __name__ == "__main__":
    main()
