from picamera2 import Picamera2
import cv2
from flask import Flask, Response, render_template
import threading

# Initialize Flask app
app = Flask(__name__)

# Define resolution
RESOLUTION = (640, 480)

# Initialize Picamera2
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": RESOLUTION, "format": "RGB888"})
picam2.framerate = 10
picam2.configure(camera_config)
picam2.start()

# Frame variable to store the latest frame
frame = None

# Capture frames in a separate thread
def capture_frames():
    global frame
    while True:
        frame = picam2.capture_array()

# Flask route to serve video feed
@app.route('/video_feed')
def video_feed():
    def generate():
        global frame
        while True:
            if frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                # Yield buffer as bytes
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start the Flask app in a separate thread
def start_server():
    app.run(host='0.0.0.0', port=8888, debug=True, use_reloader=False)

if __name__ == '__main__':
    # Start the capture thread
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()

    # Start the Flask server
    start_server()
