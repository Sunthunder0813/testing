import os
import cv2
import numpy as np
import threading
from queue import Queue, Empty
import time
import signal

# --- OPTIMIZATION FLAGS ---
# Bypasses internal OpenCV buffering for immediate frame access
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.45
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

# ================= CAMERA READER =================
class CameraStream:
    def __init__(self, url, name):
        self.name = name
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running and not shutdown_requested:
            # .grab() is much faster as it doesn't decode the frame immediately
            if not self.cap.grab():
                continue
            # Only decode the frame we actually need to display
            ret, frame = self.cap.retrieve()
            if ret:
                self.frame = frame
        self.cap.release()

# ================= HAILO INFERENCE =================
class HailoWorker:
    def __init__(self, model_path):
        from hailo_platform import HEF, VDevice, ConfigureParams, InputVStreamParams, OutputVStreamParams, HailoStreamInterface
        self.hef = HEF(model_path)
        self.device = VDevice()
        
        # Configure PCIe interface for RPi5
        config_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.device.configure(self.hef, config_params)[0]
        
        self.input_vparams = InputVStreamParams.make(self.network_group)
        self.output_vparams = OutputVStreamParams.make(self.network_group)
        self.input_name = self.hef.get_input_vstream_infos()[0].name
        self.output_name = self.hef.get_output_vstream_infos()[0].name
        self.target_shape = self.hef.get_input_vstream_infos()[0].shape[:2] # (h, w)
        
        self.results = [[] for _ in CAMERAS]
        self.running = True
        self.thread = threading.Thread(target=self.run_inference, daemon=True)

    def run_inference(self):
        from hailo_platform import InferVStreams
        with self.network_group.activate():
            with InferVStreams(self.network_group, self.input_vparams, self.output_vparams) as infer_pipeline:
                while self.running and not shutdown_requested:
                    for i in range(len(CAMERAS)):
                        frame = cam_streams[i].frame
                        if frame is not None:
                            # Resize and Infer
                            resized = cv2.resize(frame, (self.target_shape[1], self.target_shape[0]))
                            infer_results = infer_pipeline.infer({self.input_name: np.expand_dims(resized, axis=0)})
                            raw_boxes = infer_results[self.output_name][0]
                            
                            # Parse for Person (Class 0)
                            current_people = []
                            if raw_boxes is not None:
                                for det in raw_boxes:
                                    if len(det) >= 6:
                                        ymin, xmin, ymax, xmax, score, cls_id = det[:6]
                                        if int(cls_id) == 0 and score >= CONF_THRESH:
                                            h, w = frame.shape[:2]
                                            current_people.append([int(xmin*w), int(ymin*h), int(xmax*w), int(ymax*h)])
                            self.results[i] = current_people
                    time.sleep(0.001)

# ================= MAIN EXECUTION =================
cam_streams = []

def main():
    global cam_streams
    print("🚀 Initializing Hailo-8L and Cameras...")
    
    # Start Cameras
    for cam in CAMERAS:
        url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{cam['ip']}:554/h264"
        cam_streams.append(CameraStream(url, cam['name']))
    
    # Start AI
    hailo = HailoWorker(HEF_MODEL)
    hailo.thread.start()

    print("📺 Display active. Press 'q' to quit.")
    
    while not shutdown_requested:
        display_list = []
        for i, stream in enumerate(cam_streams):
            frame = stream.frame
            if frame is not None:
                # Copy current results to avoid flickering during drawing
                boxes = hailo.results[i].copy()
                for (x1, y1, x2, y2) in boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Overlay Info
                cv2.putText(frame, f"{stream.name} | People: {len(boxes)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                display_list.append(cv2.resize(frame, (640, 480)))
            else:
                display_list.append(np.zeros((480, 640, 3), np.uint8))

        if display_list:
            cv2.imshow("Hailo RPi5 25FPS Zero-Delay", cv2.hconcat(display_list))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    print("🧹 Cleaning up...")
    hailo.running = False
    for s in cam_streams: s.running = False
    cv2.destroyAllWindows()
    hailo.device.release()

if __name__ == "__main__":
    main()
