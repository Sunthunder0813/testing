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

# PERFORMANCE TWEAKS
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.35
MOTION_THRESHOLD = 1500 
GREEN = (0, 255, 0)      # AI Detection
BLUE = (255, 100, 0)     # Motion (Cyan-ish)
CAMERAS = [
    {"ip": "192.168.18.2", "name": "Cam 1"},
    {"ip": "192.168.18.113", "name": "Cam 2"},
]
RTSP_USER = "admin"
RTSP_PASS = "" # ENTER YOUR PASSWORD

shutdown_requested = False
def signal_handler(sig, frame):
    global shutdown_requested
    shutdown_requested = True
signal.signal(signal.SIGINT, signal_handler)

# ================= DRAWING UTILITY =================
def draw_fancy_bbox(img, box, label, color):
    x1, y1, x2, y2 = box
    
    # 1. Draw semi-transparent overlay inside the box
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

    # 2. Draw thick main border
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # 3. Draw "Corners" for extra visibility
    length = int((x2 - x1) * 0.1)
    # Top Left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, 4)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, 4)
    # Top Right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, 4)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, 4)
    # Bottom Left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, 4)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, 4)
    # Bottom Right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, 4)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, 4)

    # 4. Label with background contrast
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(img, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

# ================= CAMERA READER + MOTION DETECTION =================
class StreamWorker:
    def __init__(self, url, name):
        self.name = name
        self.url = url
        self.latest_frame = None
        self.detections = [] 
        self.motion_boxes = []
        self.has_motion = False
        self.lock = threading.Lock()
        self.running = True
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        while self.running and not shutdown_requested:
            if not cap.grab():
                time.sleep(0.01); continue
            ret, frame = cap.retrieve()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fgmask = self.fgbg.apply(gray)
                contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                temp_motion_boxes = []
                motion_detected = False
                for cnt in contours:
                    if cv2.contourArea(cnt) > MOTION_THRESHOLD:
                        motion_detected = True
                        x, y, w, h = cv2.boundingRect(cnt)
                        temp_motion_boxes.append((x, y, x+w, y+h))
                
                with self.lock:
                    self.latest_frame = frame
                    self.motion_boxes = temp_motion_boxes
                    self.has_motion = motion_detected
        cap.release()

# ================= HAILO AI ENGINE =================
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
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        input_vparams = InputVStreamParams.make(self.network_group)
        output_vparams = OutputVStreamParams.make(self.network_group)
        with self.network_group.activate():
            with InferVStreams(self.network_group, input_vparams, output_vparams) as infer_pipeline:
                while self.running and not shutdown_requested:
                    for s in self.streams:
                        if not s.has_motion:
                            with s.lock: s.detections = []
                            continue

                        with s.lock:
                            if s.latest_frame is None: continue
                            frame_to_ai = s.latest_frame.copy()
                        
                        h, w = self.target_shape
                        resized = cv2.resize(frame_to_ai, (w, h))
                        try:
                            results = infer_pipeline.infer({self.input_name: np.expand_dims(resized, axis=0)})
                            raw_out = results[self.output_name][0]
                            new_dets = []
                            for i in range(0, len(raw_out), 6):
                                det = raw_out[i:i+6]
                                if len(det) < 6 or det[4] < CONF_THRESH: continue
                                if int(det[5]) == 0: # Person
                                    fh, fw = frame_to_ai.shape[:2]
                                    new_dets.append((int(det[1]*fw), int(det[0]*fh), int(det[3]*fw), int(det[2]*fh), det[4]))
                            with s.lock: s.detections = new_dets
                        except: pass
                    time.sleep(0.005)

# ================= MAIN DISPLAY =================
def main():
    print("🚀 Running High-Visibility Detection System...")
    streams = [StreamWorker(f"rtsp://{RTSP_USER}:{RTSP_PASS}@{c['ip']}:554/h264", c['name']) for c in CAMERAS]
    ai = AIWorker(HEF_MODEL, streams)
    ai.thread.start()

    while not shutdown_requested:
        canvases = []
        for s in streams:
            with s.lock:
                frame = s.latest_frame.copy() if s.latest_frame is not None else np.zeros((480, 640, 3), np.uint8)
                ai_dets = s.detections
                mot_dets = s.motion_boxes
            
            # 1. Draw Motion (Thin boxes)
            for m_box in mot_dets:
                cv2.rectangle(frame, (m_box[0], m_box[1]), (m_box[2], m_box[3]), BLUE, 1)

            # 2. Draw AI Detections (Fancy Visible boxes)
            for (x1, y1, x2, y2, score) in ai_dets:
                label = f"PERSON: {int(score*100)}%"
                draw_fancy_bbox(frame, (x1, y1, x2, y2), label, GREEN)
            
            canvases.append(cv2.resize(frame, (640, 480)))

        cv2.imshow("Hailo-8L Pro View", cv2.hconcat(canvases))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    ai.running = False
    cv2.destroyAllWindows()
    ai.device.release()

if __name__ == "__main__":
    main()
