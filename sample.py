import cv2
import serial
import time
import numpy as np

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

    print("Press UP or DOWN arrow key (press q to quit)")

    # Create a blank window for key capture
    window_name = "Arrow Key Control"
    blank_img = 255 * np.ones((200, 400, 3), dtype=np.uint8)
    cv2.imshow(window_name, blank_img)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == 82:  # Up arrow key
            ser.write(b'u')
            ser.write(b'UP SENT\n')
            img = blank_img.copy()
            cv2.putText(img, "UP SENT", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, img)
            cv2.waitKey(300)
        elif key == 84:  # Down arrow key
            ser.write(b'd')
            ser.write(b'DOWN SENT\n')
            img = blank_img.copy()
            cv2.putText(img, "DOWN SENT", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(window_name, img)
            cv2.waitKey(300)

    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
