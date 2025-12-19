import os
import cv2
import numpy as np
import threading
import time
import signal
import sys

# 1. CRITICAL HAILO IMPORTS
try:
    from hailo_platform import (
        HEF, VDevice, ConfigureParams, InputVStreamParams, 
        OutputVStreamParams, HailoStreamInterface, InferVStreams
    )
except ImportError:
    print("❌ Error: hailo_platform not found. Ensure your venv is active.")
    sys.exit(1)

# 2. PERFORMANCE TWEAKS
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.35  # Detection sensitivity (0.0 to 1.0)
GREEN = (0, 255, 0)
CAMERAS = [
    {"ip": "192.168.18.2", "name": "Cam 1"},
    {"ip": "192.168.18.113", "name": "Cam 2"},
]
RTSP_USER = "admin"
RTSP_PASS = "" # ENTER YOUR PASSWORD

# ================= GLOBAL STATE =================
shutdown_requested = False
def signal_handler(sig, frame):
    global shutdown_requested
    shutdown_requested = True
signal.signal(signal.SIGINT, signal_handler)

# ================= CAMERA READER =================
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
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self.running and not shutdown_requested:
            if not cap.grab():
                time.sleep(0.01); continue
            ret, frame = cap.retrieve()
            if ret:
                with self.lock:
                    self.latest_frame = frame
        cap.release()

# ================= HAILO AI ENGINE =================
class AIWorker:
    def __init__(self, model_path, streams):
        self.hef = HEF(model_path)
        self.device = VDevice()
        params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.device.configure(self.hef, params)[0]
        
        # Get Model Info
        self.input_name = self.hef.get_input_vstream_infos()[0].name
        self.output_name = self.hef.get_output_vstream_infos()[0].name
        self.target_shape = self.hef.get_input_vstream_infos()[0].shape[:2] # (640, 640)
        
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
                        
                        # Prepare Image
                        h, w = self.target_shape
                        resized = cv2.resize(frame_to_ai, (w, h))
                        
                        try:
                            # 3. RUN INFERENCE
                            results = infer_pipeline.infer({self.input_name: np.expand_dims(resized, axis=0)})
                            raw_out = results[self.output_name][0]
                            
                            new_dets = []
                            # ROBUST PARSING FOR HAILO NMS FORMAT
                            # Standard Hailo NMS results are often padded with zeros.
                            # We iterate through the raw array in chunks of 6.
                            for i in range(0, len(raw_out), 6):
                                det = raw_out[i:i+6]
                                if len(det) < 6: break
                                
                                # Format: [ymin, xmin, ymax, xmax, confidence, class_id]
                                ymin, xmin, ymax, xmax, conf, cls_id = det
                                
                                # If confidence is 0, we've hit the end of the detections
                                if conf < CONF_THRESH: continue
                                
                                # Class 0 is Person in COCO
                                if int(cls_id) == 0:
                                    fh, fw = frame_to_ai.shape[:2]
                                    # Convert normalized (0.0 - 1.0) to pixel values
                                    px1, py1 = int(xmin * fw), int(ymin * fh)
                                    px2, py2 = int(xmax * fw), int(ymax * fh)
                                    new_dets.append((px1, py1, px2, py2, conf))
                            
                            with s.lock:
                                s.detections = new_dets
                        except Exception as e:
                            # Silently ignore frame drops to keep the stream smooth
                            pass
                    time.sleep(0.001)

# ================= DISPLAY =================
def main():
    print("🚀 Initializing Hailo-8L + Dual H.264 Streams...")
    streams = [StreamWorker(f"rtsp://{RTSP_USER}:{RTSP_PASS}@{c['ip']}:554/h264", c['name']) for c in CAMERAS]
    ai = AIWorker(HEF_MODEL, streams)
    ai.thread.start()

    print("📺 System Live. Press 'q' to quit.")
    while not shutdown_requested:
        canvases = []
        for s in streams:
            with s.lock:
                frame = s.latest_frame.copy() if s.latest_frame is not None else np.zeros((480, 640, 3), np.uint8)
                dets = s.detections
            
            # Draw detections
            for (x1, y1, x2, y2, score) in dets:
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
                cv2.putText(frame, f"PERSON {int(score*100)}%", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
            
            canvases.append(cv2.resize(frame, (640, 480)))

        cv2.imshow("Hailo-8L Dual View", cv2.hconcat(canvases))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    ai.running = False
    cv2.destroyAllWindows()
    ai.device.release()

if __name__ == "__main__":
    main()
