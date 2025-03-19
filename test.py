from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # 'n' is nano, fastest but less accurate
# Open webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on the frame
    results = model(frame)

    # Visualize detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            label = model.names[int(box.cls[0])]  # Class name

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLO Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

for r in results:
    for obj in r.boxes.data:
        class_id = int(obj[5])  # Object class index
        label = model.names[class_id]  # Get object name

        if label == "person":
            print("Person detected! Sending alert...")
        elif label == "book":
            print("Book detected! Do you want to set a study reminder?")
