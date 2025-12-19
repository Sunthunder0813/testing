import os
import cv2
import numpy as np
import threading
import time
import signal
import sys

# CRITICAL IMPORTS
try:
    from hailo_platform import (
        HEF, VDevice, ConfigureParams, InputVStreamParams, 
        OutputVStreamParams, HailoStreamInterface, InferVStreams
    )
except ImportError:
    print("❌ Error: hailo_platform not found. Run: source hailo-venv/bin/activate")
    sys.exit(1)

# Zero-Latency RTSP Config
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.45
GREEN = (0, 255, 0)  # BGR Color for the frame
CAMERAS = [
    {"ip": "192.168.18.2", "name": "Cam 1"},
    {"ip": "192.168.18.113", "name": "Cam 2"},
]
RTSP_USER = "admin"
RTSP_PASS = "" # ENTER YOUR PASSWORD HERE

# ================= GLOBAL STATE =================
shutdown_requested = False
def signal_handler(sig, frame):
    global shutdown_requested
    shutdown_requested = True
signal.signal(signal.SIGINT, signal_handler)

# ================= HIGH-SPEED WORKERS =================
class StreamWorker:
    def __init__(self, url, name):
        self.name = name
        self.url = url
        self.latest_frame = None
        self.detections = [] # Stores [x1, y1, x2, y2, score]
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        while self.running and not shutdown_requested:
            if not cap.grab():
                time.sleep(0.1); continue
            ret, frame = cap.retrieve()
            if ret:
                with self.lock:
                    self.latest_frame = frame
        cap.release()

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
        with self.network_group.activate():
            with InferVStreams(self.network_group, InputVStreamParams.make(self.network_group), 
                               OutputVStreamParams.make(self.network_group)) as infer_pipeline:
                while self.running and not shutdown_requested:
                    for s in self.streams:
                        with s.lock:
                            if s.latest_frame is None: continue
                            frame_to_ai = s.latest_frame.copy()
                        
                        h, w = self.target_shape
                        resized = cv2.resize(frame_to_ai, (w, h))
                        
                        try:
                            results = infer_pipeline.infer({self.input_name: np.expand_dims(resized, axis=0)})
                            raw = results[self.output_name][0]
                            
                            current_dets = []
                            if raw is not None:
                                for det in raw:
                                    # det format: [ymin, xmin, ymax, xmax, score, class_id]
                                    if len(det) >= 6 and det[4] >= CONF_THRESH and int(det[5]) == 0:
                                        fh, fw = frame_to_ai.shape[:2]
                                        x1, y1 = int(det[1]*fw), int(det[0]*fh)
                                        x2, y2 = int(det[3]*fw), int(det[2]*fh)
                                        current_dets.append([x1, y1, x2, y2, det[4]])
                            
                            with s.lock:
                                s.detections = current_dets
                        except: pass
                    time.sleep(0.001)

# ================= MAIN LOOP =================
def main():
    print("🚀 Running Smooth Detection with Green Frames...")
    streams = [StreamWorker(f"rtsp://{RTSP_USER}:{RTSP_PASS}@{c['ip']}:554/h264", c['name']) for c in CAMERAS]
    ai = AIWorker(HEF_MODEL, streams)
    ai.thread.start()

    while not shutdown_requested:
        canvases = []
        for s in streams:
            with s.lock:
                frame = s.latest_frame.copy() if s.latest_frame is not None else np.zeros((480, 640, 3), np.uint8)
                dets = s.detections.copy()
            
            # Draw Green Boxes and Labels
            for (x1, y1, x2, y2, score) in dets:
                # 1. Draw the main green rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
                
                # 2. Draw a small label background and text
                label = f"Person {int(score*100)}%"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
            
            canvases.append(cv2.resize(frame, (640, 480)))

        cv2.imshow("Hailo-8L Green Frame Detection", cv2.hconcat(canvases))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    ai.running = False
    cv2.destroyAllWindows()
    ai.device.release()

if __name__ == "__main__":
    main()
