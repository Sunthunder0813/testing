import cv2
import numpy as np
from hailo_platform import HEF, VDevice, InferVStreams

# ---------------- CONFIG ----------------
RTSP_URL = "rtsp://user:password@192.168.1.100:554/stream1"
HEF_PATH = "yolov8n_person.hef"
INPUT_SIZE = 640
CONF_THRESH = 0.5
# ----------------------------------------

# GStreamer pipeline (LOW LATENCY, HIGH FPS)
gst_pipeline = (
    f"rtspsrc location={RTSP_URL} latency=0 ! "
    "decodebin ! videoconvert ! appsink"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("❌ Cannot open RTSP stream")
    exit(1)

hef = HEF(HEF_PATH)

with VDevice() as device:
    network = device.configure(hef)[0]
    infer = InferVStreams(network)

    print("✅ Hailo inference started")

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
        # NOTE: output parsing depends on model
        # This is a simplified example structure
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

        cv2.imshow("Hailo IP Camera - Person Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
