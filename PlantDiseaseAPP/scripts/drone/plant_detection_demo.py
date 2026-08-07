from pathlib import Path

from ultralytics import YOLO
import cv2

# Use the trained weights shipped for the web app if present, otherwise fall
# back to a pretrained model (80 COCO classes).
DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent.parent / "plant_disease" / "best.pt"
model = YOLO(DEFAULT_WEIGHTS if DEFAULT_WEIGHTS.exists() else "yolov8n.pt")


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Run YOLOv8 inference on the frame
    results = model.predict(source=frame, show=False, conf=0.1, verbose=False)

    # Visualize results on frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()