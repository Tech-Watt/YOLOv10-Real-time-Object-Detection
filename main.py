from ultralytics import YOLO 
import cvzone
import cv2

# Detectiong on images
model = YOLO('yolov10n.pt')
results = model('birds.png')
results[0].show()

# Accessing important informations from detected objects

# print(results)
# print(results[0].boxes.xyxy.numpy().astype('int32'))
# class_detected = results[0].boxes.cls.numpy().astype('int')
# confidence = results[0].boxes.conf.numpy().astype('int')

# Live webcam
cap = cv2.VideoCapture(1)

while True:
    ret,image = cap.read()
    results = model(image)
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1,y1,x2,y2 = box.xyxy[0].numpy().astype('int')
            confidence = box.conf[0].numpy().astype('int')*100
            class_detected_number = box.cls[0]
            class_detected_number = int(class_detected_number)
            class_detected_name = results[0].names[class_detected_number]

            cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),3)
            cvzone.putTextRect(image,f'{class_detected_name}',[x1 + 8, y1 - 12], thickness=2,scale=1.5)
            

    cv2.imshow('frame',image)
    cv2.waitKey(1)