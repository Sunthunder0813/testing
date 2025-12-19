import cv2
from ultralytics import YOLO

# 1. Load the model (yolo11n.pt is fast and lightweight)
model = YOLO("yolo11n.pt") 

# 2. Define the target classes (0 is 'person' in the COCO dataset)
# If you want to detect EVERYTHING, remove the 'classes' argument in model.predict
target_classes = [0] # Add other IDs like 2 (car), 16 (dog) if needed

# 3. Open webcam (use a file path string here for videos)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 4. Run inference
    # We set 'persist=True' for smoother tracking if using video
    results = model.predict(frame, classes=target_classes, conf=0.5, verbose=False)

    # 5. Process results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class name and confidence
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            # 6. Draw the GREEN border (BGR format: 0, 255, 0)
            # Thickness is set to 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add a small label background and text
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 7. Display the frame
    cv2.imshow("YOLO Green Border Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
