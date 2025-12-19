import os
import cv2
import numpy as np
import threading
from queue import Queue
import time
import signal
import sys

# Forces OpenCV to use FFMPEG and helps with RTSP stability
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp" 

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.45
FRAME_SKIP = 1  # 0 = every frame, 1 = every other frame (saves CPU)

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
    print("\n👋 Shutting down...")
    shutdown_requested = True
signal.signal(signal.SIGINT, signal_handler)

def main():
    global shutdown_requested

    # 1. Initialize Hailo Device & Model
    try:
        from hailo_platform import HEF, VDevice, InferVStreams, InputVStreamParams, OutputVStreamParams, ConfigureParams, HailoStreamInterface
        hef = HEF(HEF_MODEL)
        device = VDevice()
        
        # Configure for RPi5 PCIe Interface
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = device.configure(hef, configure_params)[0]
        
        input_vparams = InputVStreamParams.make(network_group)
        output_vparams = OutputVStreamParams.make(network_group)

        # Get layer names and expected input dimensions
        input_name = hef.get_input_vstream_infos()[0].name
        output_name = hef.get_output_vstream_infos()[0].name
        input_info = hef.get_input_vstream_infos()[0]
        target_h, target_w = input_info.shape[:2]
        
        print(f"✅ Hailo Ready: {HEF_MODEL} (Input: {target_w}x{target_h})")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # 2. Camera Setup with FFMPEG
    frame_queues = [Queue(maxsize=1) for _ in CAMERAS]
    caps = []
    
    def reader(cap, q, cam_name):
        while not shutdown_requested:
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ Lost connection to {cam_name}. Retrying...")
                time.sleep(2)
                continue
            if q.full(): 
                try: q.get_nowait()
                except: pass
            q.put(frame)
        cap.release()

    for i, cam in enumerate(CAMERAS):
        # We use CAP_FFMPEG to fix the 'Unsupported pixel format' error
        url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{cam['ip']}:554/h264" 
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        
        if cap.isOpened():
            print(f"✅ Connected to {cam['name']}")
            threading.Thread(target=reader, args=(cap, frame_queues[i], cam['name']), daemon=True).start()
            caps.append(cap)
        else:
            print(f"❌ {cam['name']} failed. Check IP/Password.")

    # 3. Processing Loop
    last_detections = [[] for _ in CAMERAS]
    frame_counters = [0] * len(CAMERAS)

    try:
        # network_group.activate() powers up the Hailo chip
        with network_group.activate():
            # InferVStreams 'with' block prevents the AttributeError
            with InferVStreams(network_group, input_vparams, output_vparams) as infer_pipeline:
                print("🚀 Detecting People... Press 'Q' to quit.")
                
                while not shutdown_requested:
                    display_frames = []
                    
                    for i, cam in enumerate(CAMERAS):
                        if frame_queues[i].empty():
                            # Placeholder if camera feed is slow
                            display_frames.append(np.zeros((480, 640, 3), np.uint8))
                            continue
                            
                        frame = frame_queues[i].get()
                        frame_counters[i] += 1
                        
                        # --- INFERENCE ---
                        if frame_counters[i] % (FRAME_SKIP + 1) == 0:
                            # Resize to 640x640 and add Batch dimension (1, 640, 640, 3)
                            resized = cv2.resize(frame, (target_w, target_h))
                            input_dict = {input_name: np.expand_dims(resized, axis=0)}
                            
                            infer_results = infer_pipeline.infer(input_dict)
                            raw_boxes = infer_results[output_name][0] 
                            
                            current_people = []
                            for det in raw_boxes:
                                # Standard Hailo YOLO format: [ymin, xmin, ymax, xmax, score, class_id]
                                if len(det) < 6: continue
                                ymin, xmin, ymax, xmax, score, cls_id = det
                                
                                # Filter for Person (Class 0)
                                if int(cls_id) == 0 and score >= CONF_THRESH:
                                    h, w = frame.shape[:2]
                                    current_people.append({
                                        "box": [int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)],
                                        "score": score
                                    })
                            last_detections[i] = current_people

                        # --- DRAWING ---
                        for p in last_detections[i]:
                            x1, y1, x2, y2 = p["box"]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"PERSON {p['score']:.2f}", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Info Overlay
                        cv2.putText(frame, f"{cam['name']} | Count: {len(last_detections[i])}", (15, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
                        display_frames.append(cv2.resize(frame, (640, 480)))

                    # 4. Show Combined Video
                    if display_frames:
                        cv2.imshow("Hailo Multi-Cam Person Detection", cv2.hconcat(display_frames))
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        shutdown_requested = True

    finally:
        print("🧹 Cleaning up...")
        cv2.destroyAllWindows()
        device.release()
        print("👋 Finished.")

if __name__ == "__main__":
    main()
