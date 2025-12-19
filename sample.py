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
BATCH_SIZE = 8  # Matches your CLI test for max throughput
NUM_CAMERAS = 2 

# ================= ASYNC ENGINE =================
class HailoTurboEngine:
    def __init__(self, model_path):
        self.device = VDevice()
        self.hef = HEF(model_path)
        self.input_queue = Queue(maxsize=128) # Large buffer for high FPS
        self.output_results = {}
        
        # Configure PCIe for Batching
        params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.device.configure(self.hef, params)[0]
        self.network_group.set_scheduler_batch_size(BATCH_SIZE)
        
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]
        self.target_shape = self.input_vstream_info.shape[:2]
        
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

                    # 1. Pull frames to fill a batch
                    for _ in range(BATCH_SIZE):
                        try:
                            key, frame = self.input_queue.get(timeout=0.01)
                            batch_frames.append(frame)
                            batch_keys.append(key)
                        except Empty:
                            break

                    if not batch_frames: continue

                    # 2. Pad batch if cameras are slower than NPU
                    actual_len = len(batch_frames)
                    while len(batch_frames) < BATCH_SIZE:
                        batch_frames.append(np.zeros_like(batch_frames[0]))

                    # 3. High-Speed Inference
                    input_data = {self.input_vstream_info.name: np.array(batch_frames)}
                    infer_results = pipeline.infer(input_data)
                    
                    # 4. Parse NMS results (Handles 'list' vs 'array' automatically)
                    raw_detections = infer_results[self.output_vstream_info.name]
                    
                    for i in range(actual_len):
                        # Extract detection array for each frame in the batch
                        self.output_results[batch_keys[i]] = raw_detections[i]

                    # Update Stats
                    frame_count += actual_len
                    if time.time() - last_time >= 1.0:
                        self.fps = frame_count / (time.time() - last_time)
                        frame_count = 0
                        last_time = time.time()

# ================= UI & DISPLAY =================
def process_and_draw(frame, detections):
    h, w = frame.shape[:2]
    # Detections: [y1, x1, y2, x2, conf, cls]
    for d in detections:
        if d[4] > 0.5: # Confidence threshold
            x1, y1, x2, y2 = int(d[1]*w), int(d[0]*h), int(d[3]*w), int(d[2]*h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

def main():
    engine = HailoTurboEngine(HEF_MODEL)
    engine.start()
    
    # Simple example using one camera at max speed
    cap = cv2.VideoCapture(0) # Change to your RTSP URL
    print("🚀 NPU Turbo Started. Press 'q' to stop.")
    
    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Prepare for NPU
        resized = cv2.resize(frame, (640, 640))
        frame_key = f"frame_{f_idx}"
        engine.input_queue.put((frame_key, resized))
        
        # Pull results (may be slightly delayed due to batching)
        if frame_key in engine.output_results:
            dets = engine.output_results.pop(frame_key)
            frame = process_and_draw(frame, dets)
        
        cv2.putText(frame, f"NPU FPS: {engine.fps:.2f}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Hailo 240FPS Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        f_idx += 1

    engine.running = False
    engine.device.release()

if __name__ == "__main__":
    main()
