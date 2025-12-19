import cv2
import numpy as np
from hailo_platform import HEF, VDevice, InferVStreams

# ---------------- CONFIG ----------------
RTSP_URL = "rtsp://admin@192.168.18.2:554/stream1"
LOCAL_ONNX_PATH = "/home/pi/hailo_models/yolov8n.onnx"  # <-- your local ONNX
HEF_PATH = "/home/pi/hailo_models/yolov8n_person.hef"   # <-- after converting to HEF
INPUT_SIZE = 640
CONF_THRESH = 0.5
# ----------------------------------------

print(f"➡️ Using local ONNX: {LOCAL_ONNX_PATH}")
print(f"➡️ Make sure HEF exists: {HEF_PATH}")

# ---------------- GStreamer RTSP pipeline ----------------
gst_pipeline = (
    f"rtspsrc location={RTSP_URL} latency=200 ! "
    "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("❌ Cannot open RTSP stream")
    exit(1)

# ---------------- LOAD HEF ----------------
hef = HEF(HEF_PATH)

with VDevice() as device:
    network = device.configure(hef)[0]
    infer = InferVStreams(network)

    print("✅ Hailo inference started")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to model input
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = np.expand_dims(img, axis=0)

        # Hailo inference
        outputs = infer.infer(img)

        # ---------------- SIMPLE PERSON FILTER ----------------
        # Adjust based on your HEF output format
        detections = outputs[list(outputs.keys())[0]][0]

        for det in detections:
            x1, y1, x2, y2, score, cls = det
            if int(cls) == 0 and score > CONF_THRESH:  # person class
                h, w, _ = frame.shape
                x1 = int(x1 * w)
                x2 = int(x2 * w)
                y1 = int(y1 * h)
                y2 = int(y2 * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Hailo IP Camera - Person Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
