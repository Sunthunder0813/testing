import cv2
import numpy as np
import serial
import time
from hailo_platform import (PcieDevice, HEF, ConfigureParams, HailoStreamInterface, 
                            InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType)

# --- Bluetooth Setup ---
# Ensure you ran: sudo rfcomm bind 0 00:4B:12:30:06:FA
SERIAL_PORT = "/dev/rfcomm0"
try:
    ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
    print("Bluetooth: Connected to ESP32")
except:
    print("Bluetooth Error: Check RFCOMM binding.")
    exit()

# --- AI Configuration ---
# Download yolov8n.hef from Hailo Model Zoo if you don't have it
HEF_PATH = "yolov8n.hef" 

def run_ai_collector():
    # Initialize Hailo-8L device
    target = PcieDevice()
    hef = HEF(HEF_PATH)

    # Configure the network
    configure_params = ConfigureParams.get_default_config_params(hef, HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, format_type=FormatType.AUTO)
    output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, format_type=FormatType.AUTO)

    cap = cv2.VideoCapture(0) # Standard USB/Pi Camera
    
    print("Robot Vision Active. Detecting Bottles...")

    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Prepare image (Standard YOLOv8 input is 640x640)
            resized_frame = cv2.resize(frame, (640, 640))
            input_data = {hef.get_input_vstream_infos()[0].name: np.expand_dims(resized_frame, axis=0)}

            # Run AI Inference on the HAT
            with network_group.activate(network_group_params):
                infer_results = infer_pipeline.infer(input_data)
            
            # --- Simple Logic for ESP32 ---
            # If the inference produces an output (detection), we blink the LED faster
            # Note: Detailed parsing of HEF output depends on your specific model's output layer
            if len(infer_results) > 0:
                ser.write(b'u') # Send Speed Up command
                cv2.putText(frame, "BOTTLE DETECTED", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                ser.write(b'd') # Slow down if nothing seen
            
            cv2.imshow('Autonomous Collector Vision', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_ai_collector()