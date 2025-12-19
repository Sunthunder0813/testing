import os
import cv2
import numpy as np
import threading
import time
import sys
from hailo_platform import (
    HEF, VDevice, ConfigureParams, InputVStreamParams, 
    OutputVStreamParams, HailoStreamInterface, InferVStreams
)

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.50
RTSP_USER = "admin"
RTSP_PASS = "" 

CAMERAS = [
    {"ip": "192.168.18.2", "name": "Zone A"},
    {"ip": "192.168.18.113", "name": "Zone B"},
]

# ================= ROBUST CAMERA WORKER =================
class CameraWorker:
    def __init__(self, ip, name):
        self.name = name
        # Force TCP transport to prevent UDP packet loss (solves POC 0 errors)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{ip}:554/h264"
        self.latest_frame = None
        self.detections = []
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._stream, daemon=True).start()

    def _stream(self):
        while self.running:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            # Crucial for smoothness: clear the buffer
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print(f"⚠️ {self.name} connection lost. Reconnecting...")
                    break
                
                with self.lock:
                    self.latest_frame = frame
            
            cap.release()
            time.sleep(2) # Wait before reconnecting

# ================= HAILO MANAGER WITH EXCEPTION HANDLING =================
class AIModelManager:
    def __init__(self, model_path):
        try:
            self.hef = HEF(model_path)
            self.device = VDevice()
            self.target_shape = self.hef.get_input_vstream_infos()[0].shape[:2]
            params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
            self.network_group = self.device.configure(self.hef, params)[0]
            self.input_name = self.hef.get_input_vstream_infos()[0].name
            self.output_name = self.hef.get_output_vstream_infos()[0].name
        except Exception as e:
            print(f"❌ Hailo Init Error: {e}")
            sys.exit(1)

    def run_inference(self, streams):
        try:
            with self.network_group.activate():
                with InferVStreams(self.network_group, 
                                   InputVStreamParams.make(self.network_group), 
                                   OutputVStreamParams.make(self.network_group)) as pipeline:
                    while True:
                        for s in streams:
                            img = None
                            with s.lock:
                                if s.latest_frame is not None:
                                    img = s.latest_frame.copy()
                            
                            if img is None: continue
                            
                            fh, fw = img.shape[:2]
                            resized = cv2.resize(img, (self.target_shape[1], self.target_shape[0]))
                            
                            # Perform actual inference
                            results = pipeline.infer({self.input_name: np.expand_dims(resized, axis=0)})
                            raw_data = results[self.output_name][0]
                            
                            valid_dets = []
                            if hasattr(raw_data, "__len__"):
                                for d in raw_data:
                                    if len(d) >= 6:
                                        ymin, xmin, ymax, xmax, conf, cls_id = d
                                        if float(conf) > CONF_THRESH and int(cls_id) == 0:
                                            valid_dets.append([int(xmin*fw), int(ymin*fh), int(xmax*fw), int(ymax*fh), float(conf)])
                            
                            with s.lock: s.detections = valid_dets
                        time.sleep(0.001)
        except Exception as e:
            print(f"🤖 Inference Thread Stopped: {e}")

# ================= MAIN LOOP =================
def main():
    workers = [CameraWorker(c['ip'], c['name']) for c in CAMERAS]
    ai = AIModelManager(HEF_MODEL)
    
    # Start inference in a daemon thread
    inf_thread = threading.Thread(target=ai.run_inference, args=(workers,), daemon=True)
    inf_thread.start()

    print("✅ System Online. Press 'q' to quit safely.")

    try:
        while True:
            canvases = []
            for s in workers:
                with s.lock:
                    if s.latest_frame is None: continue
                    frame = s.latest_frame.copy()
                    dets = list(s.detections)

                # Draw detections
                for d in dets:
                    cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (0, 255, 0), 2)
                
                cv2.putText(frame, f"{s.name} | Count: {len(dets)}", (20, 40), 1, 1.5, (255, 255, 255), 2)
                canvases.append(cv2.resize(frame, (640, 480)))

            if canvases:
                combined = cv2.hconcat(canvases) if len(canvases) > 1 else canvases[0]
                cv2.imshow("Hailo-8L Secure Feed", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Cleaning up resources...")
        for s in workers: s.running = False
        cv2.destroyAllWindows()
        # Essential: release the device to prevent "terminate called" on next run
        ai.device.release()

if __name__ == "__main__":
    main()
