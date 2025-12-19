import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
os.environ["QT_LOGGING_RULES"] = "*.debug=false"

import cv2
import numpy as np
import threading
from queue import Queue
import time
import signal
import sys

# ================= SIGNAL HANDLING =================
shutdown_requested = False
stop_threads = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\n⚠️ Shutdown requested")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ================= MAIN =================
def main():
    global shutdown_requested, stop_threads

    # -------- CAMERA CONFIG --------
    username = "admin"
    password = "" # Set your RTSP password here
    cameras = [
        {"ip": "192.168.18.2", "name": "Camera 1"},
        {"ip": "192.168.18.113", "name": "Camera 2"},
    ]

    frame_queues = [Queue(maxsize=1) for _ in cameras]

    # -------- MODEL CONFIG --------
    HEF_MODEL = "yolov8n_person.hef" # Standard YOLOv8n includes person class
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640
    CONF_THRESH = 0.5
    FRAME_SKIP = 1  

    if not os.path.exists(HEF_MODEL):
        print(f"❌ HEF file '{HEF_MODEL}' not found.")
        sys.exit(1)

    # ================= HAILO INIT =================
    try:
        from hailo_platform import HEF, VDevice, InferVStreams, InputVStreamParams, OutputVStreamParams, ConfigureParams, HailoStreamInterface
        print("✅ Hailo platform detected")
    except ImportError:
        print("❌ Hailo SDK not installed")
        sys.exit(1)

    try:
        hef = HEF(HEF_MODEL)
        device = VDevice()
        
        # Configure the network specifically for PCIe (Raspberry Pi 5)
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = device.configure(hef, configure_params)[0]
        
        # Create stream parameters
        input_vstreams_params = InputVStreamParams.make(network_group)
        output_vstreams_params = OutputVStreamParams.make(network_group)

        # Get the actual stream names from the HEF
        input_name = hef.get_input_vstream_infos()[0].name
        output_name = hef.get_output_vstream_infos()[0].name
        print(f"✅ Hailo initialized. Input: {input_name}, Output: {output_name}")

    except Exception as e:
        print(f"❌ Hailo hardware init failed: {e}")
        sys.exit(1)

    # ================= PREPROCESS =================
    def preprocess_frame(frame):
        img = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        return img.astype(np.uint8)

    # ================= CAMERA THREAD =================
    def camera_reader(cap, queue):
        global stop_threads
        while not stop_threads:
            ret, frame = cap.read()
            if ret:
                if queue.full():
                    try: queue.get_nowait()
                    except: pass
                queue.put(frame)
            else:
                time.sleep(0.01)

    # ================= OPEN CAMERAS =================
    caps = []
    threads = []
    for cam in cameras:
        rtsp_url = f"rtsp://{username}:{password}@{cam['ip']}:554/h264"
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            print(f"✅ Connected to {cam['name']}")
            t = threading.Thread(target=camera_reader, args=(cap, frame_queues[cameras.index(cam)]), daemon=True)
            t.start()
            threads.append(t)
        else:
            print(f"❌ Failed to connect {cam['name']}")
        caps.append(cap)

    # ================= MAIN LOOP =================
    last_frames = [None] * len(cameras)
    last_detections = [[] for _ in cameras]
    frame_count = [0] * len(cameras)

    try:
        # WRAP THE LOOP IN THE CONTEXT MANAGERS
        with network_group.activate():
            with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                print("🚀 Starting person detection (press Q to quit)")
                
                while not shutdown_requested:
                    frames_to_show = []

                    for i, cam in enumerate(cameras):
                        try:
                            frame = frame_queues[i].get_nowait()
                            last_frames[i] = frame
                        except:
                            frame = last_frames[i]

                        if frame is not None:
                            frame_count[i] += 1
                            
                            # Run Inference every X frames
                            if frame_count[i] % (FRAME_SKIP + 1) == 0:
                                processed = preprocess_frame(frame)
                                # Modern API requires dict input: {stream_name: data}
                                infer_results = infer_pipeline.infer({input_name: processed})
                                
                                # Parse detections
                                raw_detections = infer_results[output_name][0]
                                h, w, _ = frame.shape
                                current_dets = []
                                
                                for det in raw_detections:
                                    if len(det) >= 6:
                                        x1, y1, x2, y2, score, cls = det[:6]
                                        if int(cls) == 0 and score >= CONF_THRESH: # 0 = Person
                                            current_dets.append({
                                                "bbox": (int(x1*w), int(y1*h), int(x2*w), int(y2*h)),
                                                "conf": score
                                            })
                                last_detections[i] = current_dets

                            # Draw detections
                            for det in last_detections[i]:
                                x1, y1, x2, y2 = det["bbox"]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"Person {det['conf']:.2f}", (x1, y1-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            frames_to_show.append(cv2.resize(frame, (640, 480)))
                        else:
                            frames_to_show.append(np.zeros((480, 640, 3), dtype=np.uint8))

                    if frames_to_show:
                        cv2.imshow("Person Detection", cv2.hconcat(frames_to_show))

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    except Exception as e:
        print(f"⚠️ Runtime Error: {e}")
    finally:
        print("🧹 Cleaning up...")
        stop_threads = True
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
        device.release()
        print("👋 Exit complete")

if __name__ == "__main__":
    main()
