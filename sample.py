import os
import cv2
import numpy as np
import threading
import time
import sys
from datetime import datetime
from hailo_platform import (
    HEF, VDevice, ConfigureParams, InputVStreamParams, 
    OutputVStreamParams, HailoStreamInterface, InferVStreams
)

# ================= CONFIGURATION =================
HEF_MODEL = "yolov8n_person.hef"
CONF_THRESH = 0.55  
GREEN_TARGET = (0, 255, 127) 
RTSP_USER = "admin"
RTSP_PASS = "" # YOUR PASSWORD
SAVE_PATH = "detections"

# Create storage folder if it doesn't exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

CAMERAS = [
    {"ip": "192.168.18.2", "name": "Zone A"},
    {"ip": "192.168.18.113", "name": "Zone B"},
]

# ================= SIMPLE CENTROID TRACKER =================
class CentroidTracker:
    def __init__(self, max_disappeared=10):
        self.next_obj_id = 1
        self.objects = {} # ID -> Centroid
        self.disappeared = {} # ID -> Count
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_obj_id] = centroid
        self.disappeared[self.next_obj_id] = 0
        self.next_obj_id += 1
        return self.next_obj_id - 1

    def update(self, rects):
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            input_centroids[i] = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            obj_ids = list(self.objects.keys())
            obj_centroids = list(self.objects.values())
            D = np.linalg.norm(np.array(obj_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols: continue
                obj_id = obj_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

        return self.objects

# ================= UI & DRAWING =================
def draw_ui(img, box, obj_id, score):
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    label = f"ID:{obj_id} | {int(score*100)}%"
    
    # Target Brackets
    c_len = int(w * 0.18)
    t = 4
    cv2.rectangle(img, (x1, y1), (x2, y2), GREEN_TARGET, 1)
    cv2.line(img, (x1, y1), (x1+c_len, y1), GREEN_TARGET, t) # TL
    cv2.line(img, (x1, y1), (x1, y1+c_len), GREEN_TARGET, t)
    cv2.line(img, (x2, y2), (x2-c_len, y2), GREEN_TARGET, t) # BR
    cv2.line(img, (x2, y2), (x2, y2-c_len), GREEN_TARGET, t)

    # ID Tag
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 15), (x1 + tw + 15, y1), GREEN_TARGET, -1)
    cv2.putText(img, label, (x1 + 7, y1 - 8), font, 0.5, (0,0,0), 1, cv2.LINE_AA)

# ================= STREAM WORKER =================
class CameraWorker:
    def __init__(self, ip, name):
        self.name = name
        self.url = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{ip}:554/h264"
        self.latest_frame = None
        self.detections = [] # [x1, y1, x2, y2, conf]
        self.tracker = CentroidTracker(max_disappeared=15)
        self.logged_ids = set()
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._stream, daemon=True).start()

    def _stream(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        while self.running:
            ret, frame = cap.read()
            if ret:
                with self.lock: self.latest_frame = frame
            else:
                time.sleep(0.01)

# ================= AI MANAGER =================
class AIModelManager:
    def __init__(self, model_path):
        self.hef = HEF(model_path)
        self.device = VDevice()
        self.target_shape = self.hef.get_input_vstream_infos()[0].shape[:2]
        params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.device.configure(self.hef, params)[0]
        self.input_name = self.hef.get_input_vstream_infos()[0].name
        self.output_name = self.hef.get_output_vstream_infos()[0].name

    def run_inference(self, streams):
        with self.network_group.activate():
            with InferVStreams(self.network_group, 
                               InputVStreamParams.make(self.network_group), 
                               OutputVStreamParams.make(self.network_group)) as pipeline:
                while True:
                    for s in streams:
                        with s.lock:
                            if s.latest_frame is None: continue
                            img = s.latest_frame.copy()
                        fh, fw = img.shape[:2]
                        resized = cv2.resize(img, (self.target_shape[1], self.target_shape[0]))
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

# ================= MAIN =================
def main():
    print(f"🚀 AI Guard Active. Snapshots saving to: /{SAVE_PATH}")
    streams = [CameraWorker(c['ip'], c['name']) for c in CAMERAS]
    ai = AIModelManager(HEF_MODEL)
    threading.Thread(target=ai.run_inference, args=(streams,), daemon=True).start()

    while True:
        display_canvases = []
        for s in streams:
            with s.lock:
                if s.latest_frame is None: continue
                frame = s.latest_frame.copy()
                dets = list(s.detections)
            
            # Update Tracker
            rects = [d[:4] for d in dets]
            tracked_objects = s.tracker.update(rects)

            # Draw & Log
            for (obj_id, centroid) in tracked_objects.items():
                # Find matching detection for score and bounding box
                for d in dets:
                    # If centroid is inside this box, it's the right one
                    if d[0] <= centroid[0] <= d[2] and d[1] <= centroid[1] <= d[3]:
                        draw_ui(frame, d[:4], obj_id, d[4])
                        
                        # Snapshot Logic: Only log if this is a new ID
                        if obj_id not in s.logged_ids:
                            timestamp = datetime.now().strftime("%H-%M-%S")
                            filename = f"{SAVE_PATH}/{s.name}_ID{obj_id}_{timestamp}.jpg"
                            cv2.imwrite(filename, frame)
                            s.logged_ids.add(obj_id)
                        break

            # HUD Display
            count_text = f"{s.name} | PEOPLE: {len(tracked_objects)}"
            cv2.rectangle(frame, (10, 10), (450, 60), (0,0,0), -1)
            cv2.putText(frame, count_text, (25, 45), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            
            display_canvases.append(cv2.resize(frame, (854, 480)))

        if display_canvases:
            combined = cv2.hconcat(display_canvases) if len(display_canvases) > 1 else display_canvases[0]
            cv2.imshow("Hailo-8L Target Monitor", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
    ai.device.release()

if __name__ == "__main__":
    main()
