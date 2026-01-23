import os
import subprocess
from ultralytics import YOLO
from hailo_sdk_client import ClientRunner

def pt_to_hef(model_name="yolov8n", pt_path="yolov8n.pt"):
    # --- Configuration ---
    onnx_path = f"{model_name}.onnx"
    har_path = f"{model_name}.har"
    hef_path = f"{model_name}.hef"
    chosen_hw_arch = "hailo8l" # Target for Raspberry Pi 5 AI Kit

    print(f"--- Step 1: Exporting {pt_path} to ONNX ---")
    model = YOLO(pt_path)
    # Export with 640x640 resolution (standard for YOLOv8n)
    model.export(format="onnx", opset=11, imgsz=640, simplify=True)

    if not os.path.exists(onnx_path):
        print("Export to ONNX failed.")
        return

    print(f"--- Step 2: Translating ONNX to Hailo HAR ---")
    runner = ClientRunner(hw_arch=chosen_hw_arch)
    # Translate the ONNX model to Hailo's Internal Representation
    runner.translate_onnx_model(
        onnx_path,
        model_name,
        start_node_names=['images'],
        end_node_names=[
            '/model.22/cv2.2/cv2.2.2/Conv', 
            '/model.22/cv3.2/cv3.2.2/Conv',
            '/model.22/cv2.1/cv2.1.2/Conv',
            '/model.22/cv3.1/cv3.1.2/Conv',
            '/model.22/cv2.0/cv2.0.2/Conv',
            '/model.22/cv3.0/cv3.0.2/Conv'
        ]
    )
    runner.save_har(har_path)

    print(f"--- Step 3: Compiling to HEF (No Dataset Mode) ---")
    # optimize_full(None) performs basic quantization without external data
    runner.optimize_full(None)
    
    # Compile the model to the final binary format
    hef = runner.compile()

    with open(hef_path, "wb") as f:
        f.write(hef)

    print("-" * 30)
    print(f"SUCCESS!")
    print(f"Original: {pt_path}")
    print(f"Final:    {hef_path}")
    print("-" * 30)
    print("You can now move this .hef file to your Raspberry Pi 5.")

if __name__ == "__main__":
    # Ensure yolov8n.pt is in the same directory
    if os.path.exists("yolov8n.pt"):
        pt_to_hef()
    else:
        print("Error: yolov8n.pt not found in current directory.")