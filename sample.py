import cv2
import serial
import time

# Set SERIAL_PORT to your Bluetooth COM port (e.g., "COM3" on Windows)
SERIAL_PORT = "/dev/rfcomm0"
BAUD = 115200

def main():
    # Bluetooth
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
        print("Bluetooth: Connected to ESP32")
        time.sleep(1)
    except Exception as e:
        print(f"Bluetooth Error: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera Error: cannot open camera. Please check if your camera is connected and the index (0) is correct.")
        ser.close()
        return

    print("Vision Active: Press UP or DOWN arrow key (press q to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()
        cv2.imshow("Arrow Key Control", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == 82:  # Up arrow key
            ser.write(b'u')
            ser.write(b'UP SENT\n')
            cv2.putText(annotated, "UP SENT", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Arrow Key Control", annotated)
            cv2.waitKey(300)  # Briefly show feedback
        elif key == 84:  # Down arrow key
            ser.write(b'd')
            ser.write(b'DOWN SENT\n')
            cv2.putText(annotated, "DOWN SENT", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Arrow Key Control", annotated)
            cv2.waitKey(300)  # Briefly show feedback

    cap.release()
    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
