async function setupCamera() {
    const video = document.getElementById('webcam');
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => resolve(video);
    });
}

async function setupCamera() {
    const video = document.getElementById('webcam');
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => resolve(video);
    });
}

// perform object detection with Yolo v11
async function detectObjects() {
    const video = await setupCamera();
    video.play();

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Load the YOLO model (ensure the model is converted to TensorFlow.js format)
    const model = await tf.loadGraphModel('model.json'); 

    async function runDetection() {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert video frame into a tensor for model inference
        const tensor = tf.browser.fromPixels(canvas)
            .resizeNearestNeighbor([416, 416])
            .expandDims(0)
            .toFloat();

        // Run inference
        const predictions = await model.executeAsync(tensor);

        // Process model output and draw bounding boxes
        drawBoundingBoxes(predictions);

        requestAnimationFrame(runDetection);
    }

    function drawBoundingBoxes(predictions) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const [boxes, scores, classes] = predictions; 

        for (let i = 0; i < scores.dataSync().length; i++) {
            const score = scores.dataSync()[i];
            if (score > 0.5) {  // Confidence threshold
                const [x1, y1, x2, y2] = boxes.dataSync().slice(i * 4, (i + 1) * 4);

                // Convert normalized coordinates back to canvas size
                const x = x1 * canvas.width;
                const y = y1 * canvas.height;
                const width = (x2 - x1) * canvas.width;
                const height = (y2 - y1) * canvas.height;

                // Draw bounding box
                ctx.strokeStyle = 'lime';
                ctx.lineWidth = 2;
                ctx.strokeRect(x, y, width, height);

                // Draw label
                ctx.fillStyle = 'lime';
                ctx.font = '16px Arial';
                ctx.fillText(`Object ${classes.dataSync()[i]}: ${score.toFixed(2)}`, x, y - 5);
            }
        }
    }

    runDetection();
}

detectObjects();
