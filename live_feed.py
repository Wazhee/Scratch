from flask import Flask, Response, render_template
import cv2
import torch
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv11 model
model = YOLO("yolo_weights/yolo11s.pt")  # Ensure this is the correct path

def generate_frames():
    cap = cv2.VideoCapture(0)  # Use external camera if needed
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Run YOLO detection
        results = model(frame)
        
        # Draw results on frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                label = result.names[int(box.cls[0])]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)