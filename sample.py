import cv2
import numpy as np
import requests
import tempfile
from hailo_platform import VDevice, InferVStreams

# ---------------- CONFIG ----------------
RTSP_URL = "rtsp://admin@192.168.18.2:554/stream1"
MODEL_URL = "https://example.com/person_detection.tflite"  # Replace with actual URL
INPUT_SIZE = 640
CONF_THRESH = 0.5
# ----------------------------------------

# GStreamer pipeline (LOW LATENCY, HIGH FPS)
gst_pipeline = (
    f"rtspsrc location={RTSP_URL} latency=0 ! "
    "decodebin ! videoconvert ! appsink"
)

# Download model from internet
response = requests.get(MODEL_URL)
response.raise_for_status()
with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as tmp_model:
    tmp_model.write(response.content)
    model_path = tmp_model.name

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("❌ Cannot open RTSP stream")
    exit(1)

with VDevice() as device:
    # Load TFLite model directly (API may differ based on SDK)
    network = device.load_model(model_path)
    infer = InferVStreams(network)

    print("✅ Hailo inference started (TFLite model)")

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
