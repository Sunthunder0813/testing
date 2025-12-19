import os
import cv2
import numpy as np
import threading
from queue import Queue
import time
import signal
import sys

# Disable unnecessary logging
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.45
FRAME_SKIP = 1  # 0 = every frame, 1 = every other frame
# Update these if your cameras use a different login
RTSP_USER = "admin"
RTSP_PASS = "" 
CAMERAS = [
    {"ip": "192.168.18.2", "name": "Cam 1"},
    {"ip": "192.168.18.113", "name": "Cam 2"},
]

# ================= SHUTDOWN HANDLER =================
shutdown_requested = False
def signal_handler(sig, frame):
    global shutdown_requested
    shutdown_requested = True
signal.signal(signal.SIGINT, signal_handler)

def main():
    global shutdown_requested

    # 1. Initialize Hailo Device
    try:
        from hailo_platform import HEF, VDevice, InferVStreams, InputVStreamParams, OutputVStreamParams, ConfigureParams, HailoStreamInterface
        hef = HEF(HEF_MODEL)
        device = VDevice()
        
        # Configure for Raspberry Pi 5 PCIe
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = device.configure(hef, configure_params)[0]
        
        input_vparams = InputVStreamParams.make(network_group)
        output_vparams = OutputVStreamParams.make(network_group)

        # Get layer names automatically
        input_name = hef.get_input_vstream_infos()[0].name
        output_name = hef.get_output_vstream_infos()[0].name
        
        # Get expected dimensions (usually 640x640)
        input_info = hef.get_input_vstream_infos()[0]
        target_h, target_w = input_info.shape[:2]
        
        print(f"✅ Hailo Ready: {HEF_MODEL} (Input: {target_w}x{target_h})")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # 2. Camera Threading Setup
    frame_queues = [Queue(maxsize=1) for _ in CAMERAS]
    caps = []
    
    def reader(cap, q):
        while not shutdown_requested:
            ret, frame = cap.read()
            if not ret: break
            if q.full(): q.get_nowait()
            q.put(frame)
        cap.release()

    for i, cam in enumerate(CAMERAS):
        url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{cam['ip']}:554/h264"
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print(f"✅ Connected to {cam['name']}")
            threading.Thread(target=reader, args=(cap, frame_queues[i]), daemon=True).start()
            caps.append(cap)
        else:
            print(f"❌ {cam['name']} connection failed")

    # 3. Inference Loop
    last_detections = [[] for _ in CAMERAS]
    frame_counters = [0] * len(CAMERAS)

    try:
        # Crucial: Must use 'with' to avoid AttributeError
        with network_group.activate():
            with InferVStreams(network_group, input_vparams, output_vparams) as infer_pipeline:
                print("🚀 Detecting People... Press Ctrl+C or 'Q' to exit.")
                
                while not shutdown_requested:
                    display_frames = []
                    
                    for i, cam in enumerate(CAMERAS):
                        if frame_queues[i].empty():
                            display_frames.append(np.zeros((480, 640, 3), np.uint8))
                            continue
                            
                        frame = frame_queues[i].get()
                        frame_counters[i] += 1
                        
                        # --- INFERENCE STEP ---
                        if frame_counters[i] % (FRAME_SKIP + 1) == 0:
                            # Resize and add batch dimension (1, 640, 640, 3)
                            resized = cv2.resize(frame, (target_w, target_h))
                            input_data = {input_name: np.expand_dims(resized, axis=0)}
                            
                            results = infer_pipeline.infer(input_data)
                            raw_boxes = results[output_name][0] # Get first batch
                            
                            current_people = []
                            for det in raw_boxes:
                                # Standard Hailo NMS: [ymin, xmin, ymax, xmax, score, class_id]
                                if len(det) < 6: continue
                                ymin, xmin, ymax, xmax, score, cls_id = det
                                
                                # Class 0 = Person in COCO/YOLOv8
                                if int(cls_id) == 0 and score >= CONF_THRESH:
                                    h, w = frame.shape[:2]
                                    current_people.append({
                                        "box": [int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)],
                                        "score": score
                                    })
                            last_detections[i] = current_people

                        # --- DRAWING STEP ---
                        for p in last_detections[i]:
                            x1, y1, x2, y2 = p["box"]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"PERSON {p['score']:.2f}", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Add camera name and person count
                        cv2.putText(frame, f"{cam['name']} - Count: {len(last_detections[i])}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        display_frames.append(cv2.resize(frame, (640, 480)))

                    # Show combined view
                    if display_frames:
                        cv2.imshow("Hailo RPi5 Multi-Cam Person Detection", cv2.hconcat(display_frames))
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        shutdown_requested = True

    finally:
        print("🧹 Cleaning up streams and device...")
        cv2.destroyAllWindows()
        device.release()
        print("👋 Finished.")

if __name__ == "__main__":
    main()
