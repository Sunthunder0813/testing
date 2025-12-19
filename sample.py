import os
import cv2
import numpy as np
import threading
import time
import signal

# Zero-Latency Environment Config
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.40  # Slightly lower for more consistent "smooth" tracking
CAMERAS = [
    {"ip": "192.168.18.2", "name": "Cam 1"},
    {"ip": "192.168.18.113", "name": "Cam 2"},
]
RTSP_USER = "admin"
RTSP_PASS = "" # ENTER PASSWORD

# ================= GLOBAL STATE =================
shutdown_requested = False
def signal_handler(sig, frame):
    global shutdown_requested
    shutdown_requested = True
signal.signal(signal.SIGINT, signal_handler)

# ================= HIGH-SPEED WORKERS =================
class StreamWorker:
    """Handles the 25FPS Smooth Display Logic"""
    def __init__(self, url, name):
        self.name = name
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.latest_frame = None
        self.boxes = []
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running and not shutdown_requested:
            # .grab() is critical for zero-latency
            if not self.cap.grab():
                time.sleep(0.01)
                continue
            ret, frame = self.cap.retrieve()
            if ret:
                with self.lock:
                    self.latest_frame = frame

class AIWorker:
    """Dedicated Hailo-8L Loop - Runs as fast as the chip allows"""
    def __init__(self, model_path, streams):
        from hailo_platform import HEF, VDevice, ConfigureParams, InputVStreamParams, OutputVStreamParams, HailoStreamInterface
        self.hef = HEF(model_path)
        self.device = VDevice()
        params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.device.configure(self.hef, params)[0]
        self.input_name = self.hef.get_input_vstream_infos()[0].name
        self.output_name = self.hef.get_output_vstream_infos()[0].name
        self.target_shape = self.hef.get_input_vstream_infos()[0].shape[:2]
        self.streams = streams
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        from hailo_platform import InferVStreams
        with self.network_group.activate():
            with InferVStreams(self.network_group, InputVStreamParams.make(self.network_group), 
                               OutputVStreamParams.make(self.network_group)) as infer_pipeline:
                while not shutdown_requested:
                    for s in self.streams:
                        with s.lock:
                            if s.latest_frame is None: continue
                            frame_to_ai = s.latest_frame.copy()
                        
                        # Pre-processing (Optimized Resize)
                        resized = cv2.resize(frame_to_ai, (self.target_shape[1], self.target_shape[0]), interpolation=cv2.INTER_LINEAR)
                        results = infer_pipeline.infer({self.input_name: np.expand_dims(resized, axis=0)})
                        
                        # Parse boxes
                        raw = results[self.output_name][0]
                        current_boxes = []
                        if raw is not None and len(raw) > 0:
                            for det in raw:
                                if len(det) >= 6 and det[4] >= CONF_THRESH and int(det[5]) == 0:
                                    h, w = frame_to_ai.shape[:2]
                                    current_boxes.append([int(det[1]*w), int(det[0]*h), int(det[3]*w), int(det[2]*h)])
                        
                        with s.lock:
                            s.boxes = current_boxes
                    time.sleep(0.001)

# ================= MAIN DISPLAY LOOP =================
def main():
    print("🚀 Optimizing Pi 5 + Hailo-8L...")
    streams = [StreamWorker(f"rtsp://{RTSP_USER}:{RTSP_PASS}@{c['ip']}:554/h264", c['name']) for c in CAMERAS]
    ai = AIWorker(HEF_MODEL, streams)

    while not shutdown_requested:
        canvases = []
        for s in streams:
            with s.lock:
                frame = s.latest_frame.copy() if s.latest_frame is not None else np.zeros((480, 640, 3), np.uint8)
                boxes = s.boxes.copy()
            
            # Smooth box drawing
            for b in boxes:
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            
            canvases.append(cv2.resize(frame, (640, 480)))

        cv2.imshow("Hailo-8L Optimized Smoothness", cv2.hconcat(canvases))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
    print("👋 Shutdown complete.")

if __name__ == "__main__":
    main()
