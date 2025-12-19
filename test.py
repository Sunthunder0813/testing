import numpy as np
import cv2
from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams

# --- CONFIGURATION ---
# Replace with your actual RTSP URL (e.g., rtsp://admin:password@192.168.1.10:554/stream)
RTSP_URL = "rtsp://admin:192.168.18.2:554/stream"
HEF_PATH = "yolov8n_person.hef"
CONF_THRESHOLD = 0.5

def run_inference():
    # 1. Initialize Hailo VDevice
    target = VDevice()
    hef = HEF(HEF_PATH)

    # 2. Configure the NPU
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    # 3. Get Input/Output stream information
    input_vstreams_params = InferVStreams.get_params(network_group)
    output_vstreams_params = InferVStreams.get_params(network_group)
    
    # Open the IP Camera
    cap = cv2.VideoCapture(RTSP_URL)
    
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Pre-process: Resize to model input (usually 640x640)
            # Replace 640 with your specific model's requirements
            resized_frame = cv2.resize(frame, (640, 640))
            input_data = {hef.get_input_vstream_infos()[0].name: np.expand_dims(resized_frame, axis=0)}

            # 4. RUN INFERENCE ON HAILO-8L
            with network_group.activate(network_group_params):
                raw_results = infer_pipeline.infer(input_data)

            # 5. POST-PROCESS (Draw boxes for 'Person' class)
            # Note: Hailo YOLO models usually return detections in a format like:
            # [x_min, y_min, x_max, y_max, confidence, class_id]
            detections = raw_results[hef.get_output_vstream_infos()[0].name][0]

            for det in detections:
                confidence = det[4]
                class_id = int(det[5])
                
                # Class 0 is usually 'person' in COCO-trained YOLO models
                if confidence > CONF_THRESHOLD and class_id == 0:
                    x1, y1, x2, y2 = det[:4]
                    # Draw on frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person: {confidence:.2f}", (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Hailo-8L Person Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()
