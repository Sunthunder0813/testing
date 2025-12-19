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

# ---------------- SIGNAL HANDLING ----------------
shutdown_requested = False
stop_threads = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\n⚠️ Shutdown requested...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ---------------- MAIN ----------------
def main():
    global shutdown_requested, stop_threads

    # -------- CAMERA CONFIG --------
    username = "admin"
    password = ""
    cameras = [
        {"ip": "192.168.18.2", "name": "Camera 1"},
        {"ip": "192.168.18.113", "name": "Camera 2"},
    ]

    frame_queues = [Queue(maxsize=1) for _ in cameras]

    # -------- MODEL CONFIG --------
    HEF_MODEL = "yolov8n_person.hef"
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640
    CONF_THRESH = 0.5
    FRAME_SKIP = 1  # 0 = infer every frame, 1 = every 2nd frame

    if not os.path.exists(HEF_MODEL):
        print(f"❌ HEF file '{HEF_MODEL}' not found")
        sys.exit(1)

    # -------- HAILO SETUP --------
    try:
        from hailo_platform import HEF, VDevice, InferVStreams
        print("✅ Hailo platform detected")
    except ImportError:
        print("❌ Hailo SDK not installed")
        sys.exit(1)

    try:
        hef = HEF(HEF_MODEL)
        device = VDevice()
        network = device.configure(hef)[0]

        # 🔥 SDK-VERSION SAFE INIT
        try:
            vstreams_params = network.create_vstreams_params()
            infer = InferVStreams(network, vstreams_params)
            print("✅ Using unified vstreams params (new SDK)")
        except AttributeError:
            input_params = network.create_input_vstream_params()
            output_params = network.create_output_vstream_params()
            infer = InferVStreams(network, input_params, output_params)
            print("✅ Using input/output vstreams params (old SDK)")

        input_name = infer.get_input_names()[0]

    except Exception as e:
        print(f"❌ Hailo init failed: {e}")
        sys.exit(1)

    # -------- PREPROCESS --------
    def preprocess_frame(frame):
        img = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        img = img.astype(np.uint8)
        img = np.expand_dims(img, axis=0)
        return img

    # -------- INFERENCE --------
    def run_hailo_inference(frame):
        img = preprocess_frame(frame)

        # ✅ CORRECT HAILO INPUT FORMAT
        outputs = infer.infer({input_name: img})

        output_tensor = list(outputs.values())[0][0]
        h, w, _ = frame.shape
        detections = []

        for det in output_tensor:
            x1, y1, x2, y2, score, cls = det
            if int(cls) == 0 and score >= CONF_THRESH:
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

    # -------- CAMERA THREAD --------
    def camera_reader(cap, queue, name):
        global stop_threads, shutdown_requested
        while not stop_threads and not shutdown_requested:
            ret, frame = cap.read()
            if ret:
                if queue.full():
                    queue.get_nowait()
                queue.put(frame)
            else:
                time.sleep(0.01)

    # -------- OPEN CAMERAS --------
    caps = []
    threads = []

    for i, cam in enumerate(cameras):
        rtsp = f"rtsp://{username}:{password}@{cam['ip']}:554/h264"
        cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)

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

    # -------- DISPLAY --------
    try:
        cv2.namedWindow("Person Detection", cv2.WINDOW_NORMAL)
        display = True
    except:
        display = False
        print("⚠️ Headless mode")

    last_frames = [None] * len(cameras)
    last_detections = [[] for _ in cameras]
    frame_count = [0] * len(cameras)
    fps_time = [time.time()] * len(cameras)
    fps_val = [0] * len(cameras)
    fps_count = [0] * len(cameras)

    print("🚀 Starting detection (press Q to quit)")

    # -------- MAIN LOOP --------
    try:
        while not shutdown_requested:
            frames = []

            for i, cam in enumerate(cameras):
                try:
                    frame = frame_queues[i].get_nowait()
                    last_frames[i] = frame
                except:
                    frame = last_frames[i]

                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "NO SIGNAL", (150, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    frame_count[i] += 1
                    if frame_count[i] % (FRAME_SKIP + 1) == 0:
                        last_detections[i] = run_hailo_inference(frame)

                    for det in last_detections[i]:
                        x1, y1, x2, y2 = det["bbox"]
                        conf = det["conf"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person {conf:.2f}",
                                    (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 2)

                    fps_count[i] += 1
                    if time.time() - fps_time[i] >= 1:
                        fps_val[i] = fps_count[i]
                        fps_count[i] = 0
                        fps_time[i] = time.time()

                    cv2.putText(frame, cam["name"], (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, f"FPS: {fps_val[i]}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                frames.append(frame)

            if display and frames:
                resized = [cv2.resize(f, (640, 480)) for f in frames]
                cv2.imshow("Person Detection", cv2.hconcat(resized))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    # -------- CLEANUP --------
    print("🧹 Cleaning up...")
    stop_threads = True
    shutdown_requested = True
    time.sleep(0.5)

    for cap in caps:
        cap.release()

    cv2.destroyAllWindows()
    print("👋 Exit complete")

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()
