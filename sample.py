import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
os.environ["QT_LOGGING_RULES"] = "*.debug=false"

import cv2
import numpy as np
import threading
from queue import Queue
import sys
import time
import subprocess
import signal
import urllib.request

shutdown_requested = False
def signal_handler(sig, frame):
    global shutdown_requested
    print("\n⚠️ Shutdown requested...")
    shutdown_requested = True
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    global shutdown_requested, stop_threads

    # Camera config
    username = "admin"
    password = ""
    cameras = [
        {"ip": "192.168.18.2", "name": "Camera 1"},
        {"ip": "192.168.18.113", "name": "Camera 2"}
    ]
    frame_queues = [Queue(maxsize=1) for _ in cameras]
    stop_threads = False

    HEF_MODEL = "yolov8n_person.hef"
    HEF_URL = "https://github.com/hailo-ai/hailo-rpi5-examples/raw/main/resources/yolov8n.hef"
    INPUT_HEIGHT = 640
    INPUT_WIDTH = 640
    CONF_THRESH = 0.5
    FRAME_SKIP = 1  # Number of frames to skip between inferences (0 = no skip, 1 = infer every 2nd frame)

    # Download HEF if not present
    if not os.path.exists(HEF_MODEL):
        print(f"⬇️ Downloading HEF from {HEF_URL} ...")
        urllib.request.urlretrieve(HEF_URL, HEF_MODEL)
        print("✅ HEF downloaded.")

    # Hailo setup
    HAILO_AVAILABLE = False
    infer = None
    network = None
    try:
        from hailo_platform import HEF, VDevice, InferVStreams
        HAILO_AVAILABLE = True
        print("✅ Hailo platform detected")
    except ImportError:
        print("❌ Hailo platform library not found. This script requires Hailo. Exiting.")
        sys.exit(1)

    if HAILO_AVAILABLE:
        try:
            hef = HEF(HEF_MODEL)
            device = VDevice()
            network = device.configure(hef)[0]
            # --- FIX: Use create_vstreams_params() for input/output ---
            vstreams_params = network.create_vstreams_params()
            infer = InferVStreams(network, vstreams_params)
            print("✅ HEF loaded and Hailo device configured.")
        except Exception as e:
            print(f"❌ Failed to load HEF or configure Hailo: {e}")
            print("   This script requires a working Hailo device. Exiting.")
            sys.exit(1)

    def preprocess_frame(frame):
        img = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        img = np.expand_dims(img, axis=0)
        return img

    def run_hailo_inference(frame):
        img = preprocess_frame(frame)
        outputs = infer.infer(img)
        detections = outputs[list(outputs.keys())[0]][0]
        results = []
        h, w, _ = frame.shape
        for det in detections:
            x1, y1, x2, y2, score, cls = det
            if int(cls) == 0 and score > CONF_THRESH:
                x1 = int(x1 * w)
                x2 = int(x2 * w)
                y1 = int(y1 * h)
                y2 = int(y2 * h)
                results.append({'bbox': (x1, y1, x2, y2), 'conf': score, 'class_id': int(cls)})
        return results

    def run_inference(frame):
        return run_hailo_inference(frame)

    def camera_reader(cap, queue, cam_name):
        global stop_threads, shutdown_requested
        while not stop_threads and not shutdown_requested:
            try:
                ret, frame = cap.read()
                if ret:
                    if queue.full():
                        try: queue.get_nowait()
                        except: pass
                    queue.put(frame)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"⚠️ Camera {cam_name} read error: {e}")
                time.sleep(0.05)

    caps = []
    threads = []
    for i, cam in enumerate(cameras):
        rtsp_url = f"rtsp://{username}:{password}@{cam['ip']}:554/h264"
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        if not cap.isOpened():
            print(f"❌ Cannot connect to {cam['name']} at {cam['ip']}")
        else:
            print(f"✅ Connected to {cam['name']}")
            t = threading.Thread(target=camera_reader, args=(cap, frame_queues[i], cam['name']), daemon=True)
            t.start()
            threads.append(t)
        caps.append(cap)

    display_available = False
    try:
        cv2.namedWindow("Person Detection", cv2.WINDOW_NORMAL)
        display_available = True
        print("✅ Display window created")
    except Exception as e:
        print(f"⚠️ Cannot create display window: {e}")
        print("   Running in headless mode (no display)")

    last_frames = [None for _ in cameras]
    fps_times = [time.time() for _ in cameras]
    fps_counts = [0 for _ in cameras]
    fps_values = [0.0 for _ in cameras]
    last_detections = [{} for _ in cameras]  # Cache for last detections per camera
    frame_counters = [0 for _ in cameras]    # Frame counters for skipping

    print("🚀 Starting person detection...")
    print("   Press 'q' to quit (or Ctrl+C)")

    try:
        while not shutdown_requested:
            frames = []
            for i, cam in enumerate(cameras):
                try:
                    frame = frame_queues[i].get_nowait()
                    last_frames[i] = frame.copy()
                except:
                    frame = last_frames[i]
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, f"No Signal - {cam['name']}", (80, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    try:
                        frame_counters[i] += 1
                        run_infer = (frame_counters[i] % (FRAME_SKIP + 1) == 0)
                        start = time.time()
                        if run_infer:
                            detections = run_inference(frame)
                            last_detections[i] = detections
                        else:
                            detections = last_detections[i]
                        end = time.time()
                        fps_counts[i] += 1
                        elapsed = end - fps_times[i]
                        if elapsed >= 1.0:
                            fps_values[i] = fps_counts[i] / elapsed
                            fps_counts[i] = 0
                            fps_times[i] = end
                        for det in detections:
                            x1, y1, x2, y2 = det['bbox']
                            conf = det['conf']
                            label = "Person"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, cam['name'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(frame, f"FPS: {fps_values[i]:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    except Exception as e:
                        print(f"⚠️ Processing error: {e}")
                frames.append(frame)
            if display_available and frames and all(f is not None for f in frames):
                try:
                    target_h = 480
                    resized = [cv2.resize(f, (int(f.shape[1]*target_h/f.shape[0]), target_h)) for f in frames]
                    combined = cv2.hconcat(resized)
                    cv2.imshow("Person Detection", combined)
                except Exception as e:
                    print(f"⚠️ Display error: {e}")
                    display_available = False
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("👋 Quit requested...")
                break
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Main loop error: {e}")

    print("🧹 Cleaning up...")
    stop_threads = True
    shutdown_requested = True
    time.sleep(0.5)
    for cap in caps:
        try:
            if cap.isOpened():
                cap.release()
        except:
            pass
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except:
        pass
    print("👋 Cleanup complete")

if __name__ == "__main__":
    main()
