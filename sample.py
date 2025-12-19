import os
import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
from hailo_platform import (
    HEF, VDevice, ConfigureParams, InputVStreamParams, 
    OutputVStreamParams, HailoStreamInterface, InferVStreams
)

# ================= HIGH-SPEED CONFIG =================
HEF_MODEL = "yolov8n_person.hef"
BATCH_SIZE = 8
RTSP_USER = "admin"
RTSP_PASS = ""

CAMERAS = [
    {"ip": "192.168.18.2", "name": "Front Gate"},
    {"ip": "192.168.18.113", "name": "Driveway"},
]

# ================= ASYNC ENGINE =================
class HailoTurboEngine:
    def __init__(self, model_path):
        self.device = VDevice()
        self.hef = HEF(model_path)
        self.input_queue = Queue(maxsize=128)
        self.output_results = {}
        
        # CORRECT WAY TO SET BATCH SIZE:
        # We modify the network_params before calling self.device.configure
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        
        # The network name is typically the first key in the params dictionary
        network_name = list(configure_params.network_group_params.keys())[0]
        configure_params.network_group_params[network_name].batch_size = BATCH_SIZE
        
        self.network_group = self.device.configure(self.hef, configure_params)[0]
        
        self.input_v_info = self.hef.get_input_vstream_infos()[0]
        self.output_v_info = self.hef.get_output_vstream_infos()[0]
        self.target_shape = self.input_v_info.shape[:2] # (H, W)
        
        self.running = True
        self.fps = 0

    def start(self):
        threading.Thread(target=self._inference_worker, daemon=True).start()

    def _inference_worker(self):
        with self.network_group.activate():
            with InferVStreams(self.network_group, 
                               InputVStreamParams.make(self.network_group), 
                               OutputVStreamParams.make(self.network_group)) as pipeline:
                
                last_time = time.time()
                frame_count = 0

                while self.running:
                    batch_frames = []
                    batch_keys = []

                    # 1. Collect Batch
                    for _ in range(BATCH_SIZE):
                        try:
                            key, frame = self.input_queue.get(timeout=0.01)
                            batch_frames.append(frame)
                            batch_keys.append(key)
                        except Empty:
                            break

                    if not batch_frames: continue

                    # 2. Pad batch for NPU stability
                    actual_len = len(batch_frames)
                    while len(batch_frames) < BATCH_SIZE:
                        batch_frames.append(np.zeros_like(batch_frames[0]))

                    # 3. Batch Inference
                    input_data = {self.input_v_info.name: np.array(batch_frames)}
                    infer_results = pipeline.infer(input_data)
                    
                    # 4. Distribute Results
                    raw_detections = infer_results[self.output_v_info.name]
                    for i in range(actual_len):
                        self.output_results[batch_keys[i]] = raw_detections[i]

                    # Update FPS tracking
                    frame_count += actual_len
                    if time.time() - last_time >= 1.0:
                        self.fps = frame_count / (time.time() - last_time)
                        frame_count = 0
                        last_time = time.time()

# ================= CAMERA WORKER =================
class CameraWorker:
    def __init__(self, ip, name, engine):
        self.name = name
        self.url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{ip}:554/h264"
        self.engine = engine
        self.latest_frame = None
        self.latest_dets = []
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        cap = cv2.VideoCapture(self.url)
        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                cap.open(self.url)
                continue
            
            # Resize for NPU inside the camera thread to save time
            resized = cv2.resize(frame, (self.engine.target_shape[1], self.engine.target_shape[0]))
            key = f"{self.name}_{f_idx}"
            self.engine.input_queue.put((key, resized))

            # Look for results
            if key in self.engine.output_results:
                dets = self.engine.output_results.pop(key)
                with self.lock:
                    self.latest_dets = dets
            
            with self.lock:
                self.latest_frame = frame
            f_idx += 1

# ================= MAIN =================
def main():
    print("💎 Pi 5 (Gen 3) + Hailo-8L: 240 FPS Async Mode")
    engine = HailoTurboEngine(HEF_MODEL)
    engine.start()
    
    workers = [CameraWorker(c['ip'], c['name'], engine) for c in CAMERAS]

    while True:
        display_frames = []
        for w in workers:
            with w.lock:
                if w.latest_frame is None: continue
                img = w.latest_frame.copy()
                dets = w.latest_dets
            
            # Draw detections
            fh, fw = img.shape[:2]
            for d in dets:
                if d[4] > 0.5:
                    x1, y1, x2, y2 = int(d[1]*fw), int(d[0]*fh), int(d[3]*fw), int(d[2]*fh)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 127), 2)

            cv2.putText(img, f"{w.name} | NPU FPS: {engine.fps:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            display_frames.append(cv2.resize(img, (854, 480)))

        if display_frames:
            cv2.imshow("Hailo Turbo Dashboard", cv2.hconcat(display_frames))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    engine.running = False
    engine.device.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
