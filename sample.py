import os
import cv2
import numpy as np
import threading
import time
import signal
import sys

# 1. CRITICAL IMPORTS FIX
try:
    from hailo_platform import (
        HEF, VDevice, ConfigureParams, InputVStreamParams, 
        OutputVStreamParams, HailoStreamInterface, InferVStreams
    )
except ImportError:
    print("❌ Error: hailo_platform not found. Ensure your virtualenv is active.")
    sys.exit(1)

# Zero-Latency Environment Config
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.40
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
    print("\n👋 Shutting down...")
    shutdown_requested = True
signal.signal(signal.SIGINT, signal_handler)

# ================= HIGH-SPEED WORKERS =================
class StreamWorker:
    """Handles the 25FPS Smooth Display Logic"""
    def __init__(self, url, name):
        self.name = name
        self.url = url
        self.latest_frame = None
        self.boxes = []
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        while self.running and not shutdown_requested:
            if not cap.grab():
                # Reconnect logic if stream drops
                time.sleep(1)
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                continue
            
            ret, frame = cap.retrieve()
            if ret:
                with self.lock:
                    self.latest_frame = frame
        cap.release()

class AIWorker:
    """Dedicated Hailo-8L Loop - Decoupled for smoothness"""
    def __init__(self, model_path, streams):
        self.hef = HEF(model_path)
        self.device = VDevice()
        
        # Configure PCIe interface for RPi5
        params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.device.configure(self.hef, params)[0]
        
        self.input_name = self.hef.get_input_vstream_infos()[0].name
        self.output_name = self.hef.get_output_vstream_infos()[0].name
        self.target_shape = self.hef.get_input_vstream_infos()[0].shape[:2]
        
        self.streams = streams
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        # Create params within the thread to ensure scope safety
        input_vparams = InputVStreamParams.make(self.network_group)
        output_vparams = OutputVStreamParams.make(self.network_group)

        with self.network_group.activate():
            with InferVStreams(self.network_group, input_vparams, output_vparams) as infer_pipeline:
                while self.running and not shutdown_requested:
                    for s in self.streams:
                        with s.lock:
                            if s.latest_frame is None: continue
                            frame_to_ai = s.latest_frame.copy()
                        
                        # Pre-processing (Optimized Resize)
                        h, w = self.target_shape
                        resized = cv2.resize(frame_to_ai, (w, h), interpolation=cv2.INTER_LINEAR)
                        
                        try:
                            results = infer_pipeline.infer({self.input_name: np.expand_dims(resized, axis=0)})
                            raw = results[self.output_name][0]
                            
                            current_boxes = []
                            if raw is not None and len(raw) > 0:
                                for det in raw:
                                    if len(det) >= 6 and det[4] >= CONF_THRESH and int(det[5]) == 0:
                                        fh, fw = frame_to_ai.shape[:2]
                                        # Box format: [xmin, ymin, xmax, ymax]
                                        current_boxes.append([int(det[1]*fw), int(det[0]*fh), int(det[3]*fw), int(det[2]*fh)])
                            
                            with s.lock:
                                s.boxes = current_boxes
                        except Exception as e:
                            print(f"⚠️ Inference Error: {e}")
                    
                    # Small sleep to yield to OS
                    time.sleep(0.001)

# ================= MAIN DISPLAY LOOP =================
def main():
    print("🚀 Initializing Ultra-Smooth Multi-Cam...")
    
    # Init Streams
    streams = []
    for c in CAMERAS:
        url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{c['ip']}:554/h264"
        streams.append(StreamWorker(url, c['name']))
    
    # Init AI
    ai = AIWorker(HEF_MODEL, streams)
    ai.thread.start()

    print("📺 Display Active. Press 'q' to quit.")

    while not shutdown_requested:
        canvases = []
        for s in streams:
            with s.lock:
                # Use a copy to prevent flickering while drawing
                frame = s.latest_frame.copy() if s.latest_frame is not None else np.zeros((480, 640, 3), np.uint8)
                boxes = s.boxes.copy()
            
            # Draw boxes on the current video frame
            for b in boxes:
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            
            # Display resize (Downscale for speed)
            canvases.append(cv2.resize(frame, (640, 480)))

        # Side-by-side view
        if canvases:
            cv2.imshow("Hailo-8L RPi5: No-Delay Person Detection", cv2.hconcat(canvases))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Final Cleanup
    print("🧹 Cleaning up devices...")
    ai.running = False
    cv2.destroyAllWindows()
    ai.device.release()
    print("👋 Finished.")

if __name__ == "__main__":
    main()
