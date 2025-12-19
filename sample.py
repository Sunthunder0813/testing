import cv2
import numpy as np
import threading
import time
from hailo_platform import HEF, VDevice, InferVStreams

# ---------------- CONFIG ----------------
CAMERA_URLS = [
    "rtsp://user:password@192.168.18.2:554/stream1",
    "rtsp://user:password@192.168.18.71:554/stream1"
]
HEF_PATH = "yolov8n_person.hef"
INPUT_SIZE = 640
CONF_THRESH = 0.5
# ----------------------------------------

def run_camera(rtsp_url, window_name):
    gst_pipeline = (
        f"rtspsrc location={rtsp_url} latency=0 ! "
        "decodebin ! videoconvert ! appsink"
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(f"❌ Cannot open RTSP stream: {rtsp_url}")
        return

    hef = HEF(HEF_PATH)
    with VDevice() as device:
        network = device.configure(hef)[0]
        infer = InferVStreams(network)

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize for model
            img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            img = np.expand_dims(img, axis=0)

            # Inference
            outputs = infer.infer(img)

            # ---- SIMPLE YOLO PERSON FILTER ----
            detections = outputs[list(outputs.keys())[0]][0]

            for det in detections:
                x1, y1, x2, y2, score, cls = det
                if int(cls) == 0 and score > CONF_THRESH:  # person
                    h, w, _ = frame.shape
                    x1 = int(x1 * w)
                    x2 = int(x2 * w)
                    y1 = int(y1 * h)
                    y2 = int(y2 * h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Person",
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyWindow(window_name)

threads = []
for idx, url in enumerate(CAMERA_URLS):
    t = threading.Thread(target=run_camera, args=(url, f"Camera {idx+1}"))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

cv2.destroyAllWindows()
