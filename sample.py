import os
import cv2
import numpy as np
import threading
from queue import Queue
import time
import signal
import sys

# High-performance environment tweaks
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.45
RTSP_USER = "admin"
RTSP_PASS = "" 
CAMERAS = [
    {"ip": "192.168.18.2", "name": "Cam 1"},
    {"ip": "192.168.18.113", "name": "Cam 2"},
]

shutdown_requested = False
def signal_handler(sig, frame):
    global shutdown_requested
    shutdown_requested = True
signal.signal(signal.SIGINT, signal_handler)

class HailoInferenceThread(threading.Thread):
    def __init__(self, network_group, input_vparams, output_vparams, input_name, output_name, target_shape):
        super().__init__(daemon=True)
        self.network_group = network_group
        self.input_vparams = input_vparams
        self.output_vparams = output_vparams
        self.input_name = input_name
        self.output_name = output_name
        self.target_shape = target_shape
        self.frame_to_process = [None] * len(CAMERAS)
        self.results = [[] for _ in CAMERAS]
        self.running = True

    def run(self):
        from hailo_platform import InferVStreams
        with self.network_group.activate():
            with InferVStreams(self.network_group, self.input_vparams, self.output_vparams) as infer_pipeline:
                while self.running and not shutdown_requested:
                    for i in range(len(CAMERAS)):
                        frame = self.frame_to_process[i]
                        if frame is not None:
                            # AI Logic
                            th, tw = self.target_shape
                            resized = cv2.resize(frame, (tw, th))
                            infer_results = infer_pipeline.infer({self.input_name: np.expand_dims(resized, axis=0)})
                            raw_boxes = infer_results[self.output_name][0]
                            
                            current_people = []
                            if raw_boxes is not None and len(raw_boxes) > 0:
                                for det in raw_boxes:
                                    if len(det) >= 6:
                                        ymin, xmin, ymax, xmax, score, cls_id = det[:6]
                                        if int(cls_id) == 0 and score >= CONF_THRESH:
                                            h, w = frame.shape[:2]
                                            current_people.append([int(xmin*w), int(ymin*h), int(xmax*w), int(ymax*h)])
                            self.results[i] = current_people
                            self.frame_to_process[i] = None # Mark as done
                    time.sleep(0.001)

def main():
    global shutdown_requested

    # 1. Hailo Setup
    try:
        from hailo_platform import HEF, VDevice, ConfigureParams, InputVStreamParams, OutputVStreamParams, HailoStreamInterface
        hef = HEF(HEF_MODEL)
        device = VDevice()
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = device.configure(hef, configure_params)[0]
        input_vparams = InputVStreamParams.make(network_group)
        output_vparams = OutputVStreamParams.make(network_group)
        input_name = hef.get_input_vstream_infos()[0].name
        output_name = hef.get_output_vstream_infos()[0].name
        target_shape = hef.get_input_vstream_infos()[0].shape[:2]
        print(f"✅ Hailo Ready: {HEF_MODEL}")
    except Exception as e:
        print(f"❌ Init failed: {e}")
        return

    # 2. Start AI Thread
    ai_thread = HailoInferenceThread(network_group, input_vparams, output_vparams, input_name, output_name, target_shape)
    ai_thread.start()

    # 3. Camera Reader (Raw speed)
    caps = []
    for i, cam in enumerate(CAMERAS):
        url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{cam['ip']}:554/h264"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3) # Allow slight buffer for 25fps smoothness
        caps.append(cap)

    print("🚀 Starting Smooth Detection...")
    
    try:
        while not shutdown_requested:
            display_frames = []
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    display_frames.append(np.zeros((480, 640, 3), np.uint8))
                    continue

                # Pass frame to AI thread if it's ready for a new one
                if ai_thread.frame_to_process[i] is None:
                    ai_thread.frame_to_process[i] = frame.copy()

                # Draw latest available results (Async)
                for box in ai_thread.results[i]:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                display_frames.append(cv2.resize(frame, (640, 480)))

            if display_frames:
                cv2.imshow("25 FPS Smooth Detection", cv2.hconcat(display_frames))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        ai_thread.running = False
        for cap in caps: cap.release()
        cv2.destroyAllWindows()
        device.release()

if __name__ == "__main__":
    main()
