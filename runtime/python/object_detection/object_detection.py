#!/usr/bin/env python3

import os
import sys
from loguru import logger
import numpy as np
import cv2
from pathlib import Path
import urllib.request
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.tracker.byte_tracker import BYTETracker
from common.hailo_inference import HailoInfer
from object_detection_post_process import inference_result_handler
from common.toolbox import (
    get_labels,
    load_json_file,
    visualize,
    FrameRateTracker,
)

APP_NAME = Path(__file__).stem

def init_dual_ip_camera_source(ip1, ip2, camera_resolution):
    """
    Initialize two IP camera streams and yield merged frames.
    """
    cap1 = cv2.VideoCapture(f"rtsp://{ip1}/live")
    cap2 = cv2.VideoCapture(f"rtsp://{ip2}/live")
    if not cap1.isOpened() or not cap2.isOpened():
        raise RuntimeError(f"Could not open IP cameras: {ip1}, {ip2}")

    # Optionally set resolution
    if camera_resolution:
        res_map = {"sd": (640, 480), "hd": (1280, 720), "fhd": (1920, 1080)}
        w, h = res_map.get(camera_resolution, (640, 480))
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    return cap1, cap2

def preprocess_frame(frame, width, height):
    # Simple resize and normalization (customize as needed)
    resized = cv2.resize(frame, (width, height))
    img = resized.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def download_hef_model(url, dest_path):
    if not os.path.isfile(dest_path):
        logger.info(f"Downloading model from {url} to {dest_path} ...")
        urllib.request.urlretrieve(url, dest_path)
        logger.success(f"Downloaded model to {dest_path}")

def main():
    # --- Hardcoded config ---
    ip1 = "192.168.18.2"
    ip2 = "192.168.18.71"
    camera_resolution = "hd"
    hef_url = "https://example.com/path/to/yolo8vn.hef"  # <-- Replace with actual URL
    net = "./yolo8vn.hef"
    labels_path = str(Path(__file__).parent.parent / "common" / "coco.txt")
    config_data = load_json_file("config.json")
    labels = get_labels(labels_path)
    tracker = None
    draw_trail = False

    # --- Download model if not exists ---
    download_hef_model(hef_url, net)

    # --- Check model file exists ---
    if not os.path.isfile(net):
        logger.error(f"Model file not found: {net}\n"
                     f"Please provide a valid .hef model file path in the 'net' variable.")
        sys.exit(1)

    # --- Init model ---
    hailo_inference = HailoInfer(net, batch_size=1)
    height, width, _ = hailo_inference.get_input_shape()

    # --- Init tracker if needed ---
    if config_data.get("visualization_params", {}).get("tracker", {}):
        tracker_config = config_data.get("visualization_params", {}).get("tracker", {})
        from types import SimpleNamespace
        tracker = BYTETracker(SimpleNamespace(**tracker_config))

    # --- Init cameras ---
    cap1, cap2 = init_dual_ip_camera_source(ip1, ip2, camera_resolution)

    fps_tracker = FrameRateTracker()
    fps_tracker.start()

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            logger.error("Failed to read from one or both cameras.")
            break

        # Resize to same size if needed
        if frame1.shape != frame2.shape:
            h = min(frame1.shape[0], frame2.shape[0])
            w = min(frame1.shape[1], frame2.shape[1])
            frame1 = cv2.resize(frame1, (w, h))
            frame2 = cv2.resize(frame2, (w, h))
        merged = np.hstack((frame1, frame2))

        # Preprocess for inference
        preprocessed = preprocess_frame(merged, width, height)

        # Run inference
        result = None
        def inference_callback(completion_info, bindings_list):
            nonlocal result
            if completion_info.exception:
                logger.error(f'Inference error: {completion_info.exception}')
            else:
                bindings = bindings_list[0]
                if len(bindings._output_names) == 1:
                    result = bindings.output().get_buffer()
                else:
                    result = {
                        name: np.expand_dims(
                            bindings.output(name).get_buffer(), axis=0
                        )
                        for name in bindings._output_names
                    }
        hailo_inference.run(preprocessed, inference_callback)
        # Wait for inference to complete (simulate async)
        while result is None:
            cv2.waitKey(1)

        # Post-process and visualize
        vis_frame = merged.copy()
        vis_frame = inference_result_handler(
            vis_frame, result, labels=labels,
            config_data=config_data, tracker=tracker, draw_trail=draw_trail
        )

        # Show FPS
        fps = fps_tracker.update()
        cv2.putText(vis_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Dual IP Camera Detection", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    hailo_inference.close()
    logger.success("Streaming stopped.")

if __name__ == "__main__":
    main()