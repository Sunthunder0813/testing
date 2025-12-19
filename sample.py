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
        
        # CORRECTED CONFIGURATION ACCESS:
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        
        # Get the network name from HEF and apply batch size
        network_group_name = self.hef.get_network_group_names()[0]
        
        # Check if configure_params is a dict or an object with attributes
        if hasattr(configure_params, 'network_group_params'):
            configure_params.network_group_params[network_group_name].batch_size = BATCH_SIZE
        else:
            # For older/newer versions where it behaves purely as a dictionary
            configure_params[network_group_name].batch_size = BATCH_SIZE
        
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
        # Pi 5 optimization: Set buffer to 1 to reduce lag
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                cap.open(self.url)
                continue
            
            # Pre-resize in worker thread
            resized = cv2.resize(frame, (self.engine.target_shape[1], self.engine.target_shape[0]))
            key = f"{self.name}_{f_idx}"
            self.engine.input_queue.put((key, resized))

            # Fetch results
            if key in self.engine.output_results:
                dets = self.engine.output_results.pop(key)
                with self.lock:
                    self.latest_dets = dets
            
            with self.lock:
                self.latest_frame = frame
            f_idx += 1

# ================= MAIN =================
def main():
    print("💎 Pi 5 (Gen 3) + Hailo-8L: Industrial Turbo Mode")
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
            
            fh, fw = img.shape[:2]
            # Ensure dets is iterable
            if dets is not None and len(dets) > 0:
                for d in dets:
                    # Robust check for detection array structure
                    if len(d) >= 5 and d[4] > 0.45:
                        x1, y1, x2, y2 = int(d[1]*fw), int(d[0]*fh), int(d[3]*fw), int(d[2]*fh)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 127), 2)

            cv2.putText(img, f"{w.name} | Total NPU FPS: {engine.fps:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            display_frames.append(cv2.resize(img, (854, 480)))

        if display_frames:
            combined = cv2.hconcat(display_frames) if len(display_frames) > 1 else display_frames[0]
            cv2.imshow("Hailo Turbo Dashboard", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    engine.running = False
    engine.device.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
