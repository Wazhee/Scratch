import cv2
from ultralytics import YOLO
import math 
import time
import random

x_thresh, y_thresh = 300, 300

model = YOLO("yolo_weights/yolo11s.pt")


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)

prev_time = time.time()

while True:
    ret, frame = cap.read()  
    if not ret:
        break
    
    # frame_resized = cv2.resize(frame, (640, 480))  # Resize frame to 640x480 to increase speed
    # results = model(frame_resized, stream=True)

    results = model(frame, stream=True)
    
    # Calculate FPS
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time))
    prev_time = curr_time
    
    # Draw FPS on frame
    cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # choose a random color
    ch1,ch2,ch3 = random.randint(0,255),random.randint(0,255),random.randint(0,255)
    # Process detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            # check if box is too big
            # if abs(x2 - x1) < x_thresh or abs(y2 - y1) < y_thresh:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (ch1,ch2,ch3), 3)
            cls = int(box.cls[0])
            cv2.putText(frame, classNames[cls], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow('Live Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
