import cv2
import serial
import time
from ultralytics import YOLO

# Set SERIAL_PORT to your Bluetooth COM port (e.g., "COM3" on Windows)
SERIAL_PORT = "/dev/rfcomm0"
BAUD = 115200

# Use a real hand-detection model here:
MODEL_PATH = "yolov8n.pt"   # <- replace with your hand model filename

CONF_THRES = 0.35

def main():
    # Bluetooth
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
        print("Bluetooth: Connected to ESP32")
        time.sleep(1)
    except Exception as e:
        print(f"Bluetooth Error: {e}")
        return

    # Load model
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera Error: cannot open camera. Please check if your camera is connected and the index (0) is correct.")
        ser.close()
        return

    print("Vision Active: Detecting HANDS... (press q to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=640, conf=CONF_THRES, verbose=False)

        hand_found = False
        annotated = frame

        # If your hand model uses class 0 = hand, this is enough:
        # any detection => hand_found True
        if results and len(results[0].boxes) > 0:
            hand_found = True
            annotated = results[0].plot()

        if hand_found:
            ser.write(b'u')
            cv2.putText(annotated, "HAND DETECTED", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            ser.write(b'd')

        cv2.imshow("Hand Detection (CPU)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
