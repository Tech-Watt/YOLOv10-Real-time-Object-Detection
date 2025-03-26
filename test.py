from ultralytics import YOLO
import cvzone
import cv2
from collections import defaultdict

# Load YOLO model
model = YOLO('yolov10n.pt')

# Dictionary to count occurrences of each object category
object_counts = defaultdict(int)

# Set to track currently visible objects
active_objects = set()

# Live webcam
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    if not ret:
        break

    results = model(image)
    current_objects = set()  # Track objects in the current frame

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            confidence = box.conf[0].numpy().astype('float') * 100
            class_detected_number = int(box.cls[0])
            class_detected_name = results[0].names[class_detected_number]

            # Add detected object to current_objects set
            current_objects.add(class_detected_name)

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cvzone.putTextRect(image, f'{class_detected_name} ({object_counts[class_detected_name]})',
                               [x1 + 8, y1 - 12], thickness=2, scale=1.5)

    # Detect objects that have left the frame
    disappeared_objects = active_objects - current_objects

    # Detect new objects appearing in the frame
    new_objects = current_objects - active_objects

    # Update object count only for newly appearing objects
    for obj in new_objects:
        object_counts[obj] += 1

    # Update active objects for the next iteration
    active_objects = current_objects

    # Display object counts on the frame
    y_offset = 30
    for obj, count in object_counts.items():
        cv2.putText(image, f"{obj}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30

    cv2.imshow('frame', image)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print final object counts
print("Final Object Counts:")
for obj, count in object_counts.items():
    print(f"{obj}: {count}")
