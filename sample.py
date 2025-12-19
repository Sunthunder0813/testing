import os
import cv2
import numpy as np
import threading
from queue import Queue
import time
import signal
import sys

# Forces FFMPEG and sets it to use TCP for stable, non-blinking streams
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp" 

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.45
FRAME_SKIP = 1  # 0 = every frame, 1 = every other frame

RTSP_USER = "admin"
RTSP_PASS = "" 
CAMERAS = [
    {"ip": "192.168.18.2", "name": "Cam 1"},
    {"ip": "192.168.18.113", "name": "Cam 2"},
]

# ================= SHUTDOWN HANDLER =================
shutdown_requested = False
def signal_handler(sig, frame):
    global shutdown_requested
    print("\n👋 Shutting down safely...")
    shutdown_requested = True
signal.signal(signal.SIGINT, signal_handler)

def main():
    global shutdown_requested

    # 1. Hailo Init
    try:
        from hailo_platform import HEF, VDevice, InferVStreams, InputVStreamParams, OutputVStreamParams, ConfigureParams, HailoStreamInterface
        hef = HEF(HEF_MODEL)
        device = VDevice()
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = device.configure(hef, configure_params)[0]
        input_vparams = InputVStreamParams.make(network_group)
        output_vparams = OutputVStreamParams.make(network_group)
        
        # Auto-detect layer names
        input_name = hef.get_input_vstream_infos()[0].name
        output_name = hef.get_output_vstream_infos()[0].name
        target_h, target_w = hef.get_input_vstream_infos()[0].shape[:2]
        print(f"✅ Hailo Ready: {HEF_MODEL}")
    except Exception as e:
        print(f"❌ Init failed: {e}")
        return

    # 2. Optimized Camera Threading
    frame_queues = [Queue(maxsize=1) for _ in CAMERAS]
    
    def reader(cap, q, cam_name):
        while not shutdown_requested:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                continue
            # Clear old frame to keep it 'Live'
            if not q.empty():
                try: q.get_nowait()
                except: pass
            q.put(frame)

    for i, cam in enumerate(CAMERAS):
        url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{cam['ip']}:554/h264"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            print(f"✅ {cam['name']} connected")
            threading.Thread(target=reader, args=(cap, frame_queues[i], cam['name']), daemon=True).start()
        else:
            print(f"❌ {cam['name']} failed")

    # 3. Main Processing
    last_detections = [[] for _ in CAMERAS]
    frame_count = [0] * len(CAMERAS)

    try:
        with network_group.activate():
            with InferVStreams(network_group, input_vparams, output_vparams) as infer_pipeline:
                while not shutdown_requested:
                    display_frames = []
                    
                    for i, cam in enumerate(CAMERAS):
                        if frame_queues[i].empty():
                            display_frames.append(np.zeros((480, 640, 3), np.uint8))
                            continue
                            
                        frame = frame_queues[i].get()
                        frame_count[i] += 1
                        
                        # AI Inference
                        if frame_count[i] % (FRAME_SKIP + 1) == 0:
                            try:
                                resized = cv2.resize(frame, (target_w, target_h))
                                results = infer_pipeline.infer({input_name: np.expand_dims(resized, axis=0)})
                                raw_boxes = results[output_name][0]
                                
                                current_people = []
                                # --- THE FIX: SAFETY CHECK ---
                                if raw_boxes is not None and len(raw_boxes) > 0:
                                    for det in raw_boxes:
                                        # Only attempt to unpack if the row has data
                                        if len(det) >= 6:
                                            ymin, xmin, ymax, xmax, score, cls_id = det[:6]
                                            if int(cls_id) == 0 and score >= CONF_THRESH:
                                                h, w = frame.shape[:2]
                                                current_people.append([int(xmin*w), int(ymin*h), int(xmax*w), int(ymax*h)])
                                last_detections[i] = current_people
                            except Exception as e:
                                print(f"⚠️ Inference skip on {cam['name']}: {e}")

                        # Drawing
                        for box in last_detections[i]:
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        
                        cv2.putText(frame, f"{cam['name']} People: {len(last_detections[i])}", (20, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        display_frames.append(cv2.resize(frame, (640, 480)))

                    if display_frames:
                        # Stack cameras side-by-side
                        cv2.imshow("Hailo Person Detection", cv2.hconcat(display_frames))
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        shutdown_requested = True
                        break
    finally:
        print("🧹 Cleaning up...")
        cv2.destroyAllWindows()
        device.release()

if __name__ == "__main__":
    main()
