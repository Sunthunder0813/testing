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
    password = "" # Add password if needed
    cameras = [
        {"ip": "192.168.18.2", "name": "Camera 1"},
        {"ip": "192.168.18.113", "name": "Camera 2"},
    ]

    frame_queues = [Queue(maxsize=1) for _ in cameras]

    # -------- MODEL CONFIG --------
    # Use the filename we downloaded earlier
    HEF_MODEL = "yolov8n_person.hef" 
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640
    CONF_THRESH = 0.5
    FRAME_SKIP = 1  

    if not os.path.exists(HEF_MODEL):
        print(f"❌ HEF file '{HEF_MODEL}' not found. Please ensure it is in the same folder.")
        sys.exit(1)

    # ================= HAILO INIT (MODERN API) =================
    try:
        from hailo_platform import HEF, VDevice, InferVStreams, InputVStreamParams, OutputVStreamParams
        print("✅ Hailo platform detected")
    except ImportError:
        print("❌ Hailo SDK not installed")
        sys.exit(1)

    try:
        hef = HEF(HEF_MODEL)
        device = VDevice()
        
        # Configure network
        network_group = device.configure(hef)[0]
        
        # Prepare parameters for InferVStreams (Required for 4.23.0)
        input_vstreams_params = InputVStreamParams.make(network_group)
        output_vstreams_params = OutputVStreamParams.make(network_group)
        
        # Create inference context
        infer = InferVStreams(network_group, input_vstreams_params, output_vstreams_params)
        print("✅ Hailo initialized with modern API")

    except Exception as e:
        print(f"❌ Hailo init failed: {e}")
        sys.exit(1)

    # ================= PREPROCESS =================
    def preprocess_frame(frame):
        img = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        img = img.astype(np.uint8)
        # No need for expand_dims here as infer handles the list/batch
        return img

    # ================= INFERENCE =================
    def run_hailo_inference(frame):
        img = preprocess_frame(frame)

        # Updated inference call for modern API
        input_data = {hef.get_input_vstream_infos()[0].name: img}
        
        with network_group.activate():
            outputs = infer.infer(input_data)
        
        # YOLOv8 output handling depends on the specific HEF compilation.
        # Usually, the first output contains the detection tensors.
        output_name = hef.get_output_vstream_infos()[0].name
        detections_raw = outputs[output_name][0]

        h, w, _ = frame.shape
        detections = []

        for det in detections_raw:
            # Note: The indices (x, y, w, h, score, cls) may vary based on 
            # how your HEF was compiled. This assumes standard YOLO output.
            if len(det) >= 6:
                x1, y1, x2, y2, score, cls = det[:6]
                if int(cls) == 0 and score >= CONF_THRESH: # 0 is usually 'person'
                    detections.append({
                        "bbox": (
                            int(x1 * w),
                            int(y1 * h),
                            int(x2 * w),
                            int(y2 * h),
                        ),
                        "conf": float(score)
                    })
        return detections

    # ================= CAMERA THREAD =================
    def camera_reader(cap, queue, name):
        global stop_threads, shutdown_requested
        while not stop_threads and not shutdown_requested:
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

    for i, cam in enumerate(cameras):
        rtsp_url = f"rtsp://{username}:{password}@{cam['ip']}:554/h264"
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if cap.isOpened():
            print(f"✅ Connected to {cam['name']}")
            t = threading.Thread(
                target=camera_reader,
                args=(cap, frame_queues[i], cam['name']),
                daemon=True
            )
            t.start()
            threads.append(t)
        else:
            print(f"❌ Failed to connect {cam['name']}")

        caps.append(cap)

    # ================= DISPLAY SETUP =================
    cv2.namedWindow("Person Detection", cv2.WINDOW_NORMAL)
    display = True

    last_frames = [None] * len(cameras)
    last_detections = [[] for _ in cameras]
    frame_count = [0] * len(cameras)
    fps_time = [time.time()] * len(cameras)
    fps_val = [0] * len(cameras)
    fps_count = [0] * len(cameras)

    print("🚀 Starting person detection (press Q to quit)")

    # ================= MAIN LOOP =================
    try:
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
                    # Inference
                    if frame_count[i] % (FRAME_SKIP + 1) == 0:
                        last_detections[i] = run_hailo_inference(frame)

                    # Drawing
                    for det in last_detections[i]:
                        x1, y1, x2, y2 = det["bbox"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person {det['conf']:.2f}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # FPS Logic
                    fps_count[i] += 1
                    if time.time() - fps_time[i] >= 1:
                        fps_val[i] = fps_count[i]
                        fps_count[i] = 0
                        fps_time[i] = time.time()

                    cv2.putText(frame, f"{cam['name']} | FPS: {fps_val[i]}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    frames_to_show.append(cv2.resize(frame, (640, 480)))
                else:
                    # Placeholder if no frame
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "WAITING FOR SIGNAL", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    frames_to_show.append(blank)

            if frames_to_show:
                cv2.imshow("Person Detection", cv2.hconcat(frames_to_show))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("🧹 Cleaning up...")
        stop_threads = True
        shutdown_requested = True
        infer.__exit__(None, None, None) # Close the infer stream
        device.release()
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
        print("👋 Exit complete")

if __name__ == "__main__":
    main()
