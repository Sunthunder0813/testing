import os
import cv2
import numpy as np
import threading
import time
import signal
import sys

# 1. HAILO PLATFORM IMPORTS
try:
    from hailo_platform import (
        HEF, VDevice, ConfigureParams, InputVStreamParams, 
        OutputVStreamParams, HailoStreamInterface, InferVStreams
    )
except ImportError:
    print("❌ Error: hailo_platform not found.")
    sys.exit(1)

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.45  
GREEN_BRIGHT = (0, 255, 127)
SHADOW = (0, 40, 0)
CAMERAS = [
    {"ip": "192.168.18.2", "name": "Entrance"},
    {"ip": "192.168.18.113", "name": "Backyard"},
]
RTSP_USER = "admin"
RTSP_PASS = "" # ENTER YOUR PASSWORD

shutdown_requested = False
def signal_handler(sig, frame):
    global shutdown_requested
    shutdown_requested = True
signal.signal(signal.SIGINT, signal_handler)

# ================= ADVANCED DRAWING UI =================
def draw_pro_target(img, box, label):
    x, y, x2, y2 = map(int, box)
    w, h = x2 - x, y2 - y
    
    # Background Glow
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), GREEN_BRIGHT, -1)
    cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)

    # Main Border with Shadow
    cv2.rectangle(img, (x+1, y+1), (x2+1, y2+1), SHADOW, 2)
    cv2.rectangle(img, (x, y), (x2, y2), GREEN_BRIGHT, 2)

    # Targeting Corners
    t_len = int(w * 0.15)
    for pt1, pt2 in [((x,y),(x+t_len,y)), ((x,y),(x,y+t_len)), 
                     ((x2,y),(x2-t_len,y)), ((x2,y),(x2,y+t_len)),
                     ((x,y2),(x+t_len,y2)), ((x,y2),(x,y2-t_len)),
                     ((x2,y2),(x2-t_len,y2)), ((x2,y2),(x2,y2-t_len))]:
        cv2.line(img, pt1, pt2, GREEN_BRIGHT, 5)

    # Label Tag
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
    cv2.rectangle(img, (x, y - th - 15), (x + tw + 20, y), GREEN_BRIGHT, -1)
    cv2.putText(img, label, (x + 10, y - 8), font, 0.5, (0,0,0), 2, cv2.LINE_AA)

# ================= STREAM WORKER =================
class StreamWorker:
    def __init__(self, url, name):
        self.name = name
        self.url = url
        self.latest_frame = None
        self.detections = [] 
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        full_url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{self.url}:554/h264" if "192" in self.url else self.url
        cap = cv2.VideoCapture(full_url, cv2.CAP_FFMPEG)
        while self.running and not shutdown_requested:
            ret, frame = cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.1)
        cap.release()

# ================= AI ENGINE =================
class AIWorker:
    def __init__(self, model_path, streams):
        self.hef = HEF(model_path)
        self.device = VDevice()
        params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.device.configure(self.hef, params)[0]
        
        self.input_name = self.hef.get_input_vstream_infos()[0].name
        self.output_name = self.hef.get_output_vstream_infos()[0].name
        self.target_shape = self.hef.get_input_vstream_infos()[0].shape[:2]
        self.streams = streams

    def _run(self):
        input_vparams = InputVStreamParams.make(self.network_group)
        output_vparams = OutputVStreamParams.make(self.network_group)

        with self.network_group.activate():
            with InferVStreams(self.network_group, input_vparams, output_vparams) as infer_pipeline:
                while not shutdown_requested:
                    for s in self.streams:
                        with s.lock:
                            if s.latest_frame is None: continue
                            raw_frame = s.latest_frame.copy()
                        
                        fh, fw = raw_frame.shape[:2]
                        resized = cv2.resize(raw_frame, (self.target_shape[1], self.target_shape[0]))
                        
                        infer_res = infer_pipeline.infer({self.input_name: np.expand_dims(resized, axis=0)})
                        raw_out = np.array(infer_res[self.output_name][0])

                        new_dets = []
                        # Corrected Parsing: check if array is empty or 0-filled
                        if raw_out.size > 0:
                            for i in range(0, len(raw_out), 6):
                                # Slice carefully and check values
                                chunk = raw_out[i:i+6]
                                if chunk.size < 6: break
                                
                                ymin, xmin, ymax, xmax, conf, cls_id = chunk
                                
                                # FIX: Ensure we are comparing scalar values, not arrays
                                if float(conf) > CONF_THRESH and int(cls_id) == 0:
                                    new_dets.append([
                                        int(xmin * fw), int(ymin * fh), 
                                        int(xmax * fw), int(ymax * fh), float(conf)
                                    ])
                        
                        with s.lock:
                            s.detections = new_dets
                    time.sleep(0.001)

# ================= MAIN APP =================
def main():
    print("💎 Initializing Premium Hailo-8L Detection...")
    streams = [StreamWorker(c['ip'], c['name']) for c in CAMERAS]
    ai = AIWorker(HEF_MODEL, streams)
    
    ai_thread = threading.Thread(target=ai._run)
    ai_thread.start()

    while not shutdown_requested:
        views = []
        for s in streams:
            with s.lock:
                if s.latest_frame is None: continue
                frame = s.latest_frame.copy()
                dets = list(s.detections)

            for d in dets:
                draw_pro_target(frame, d[:4], f"PERSON {int(d[4]*100)}%")

            cv2.putText(frame, s.name, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            views.append(cv2.resize(frame, (800, 450)))

        if views:
            cv2.imshow("Hailo-8L Precision Guard", cv2.hconcat(views) if len(views) > 1 else views[0])
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    print("\nShutting down safely...")
    cv2.destroyAllWindows()
    # Device release must happen after the AI thread finishes
    ai_thread.join(timeout=2)
    ai.device.release()

if __name__ == "__main__":
    main()
