import cv2

cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  
    # Process the frame using OpenCV functions
    cv2.imshow('Live Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break
        
cap.release()
cv2.destroyAllWindows()
