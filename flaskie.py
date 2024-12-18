#!/usr/bin/python3
from flask import Flask, Response, request, jsonify
from picamera2 import Picamera2, MappedArray
import threading
import os
import time
import cv2
print("server starting")
# start the server and tts engine
app = Flask(__name__)

# configure and start the camera
picam = Picamera2()
camera_config = picam.create_video_configuration(
    main={"size": (640, 480)},
    lores={"size": (320, 240)},
    controls={"FrameRate": 15},
    buffer_count=4
)
picam.configure(camera_config)
picam.start()

# self explanatory
def generate_video_feed():
    while True:
        frame = picam.capture_array("main")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # encode frame as JPEG
        _, jpeg = cv2.imencode(".jpg", frame)
        
        # yield encoded frame as needed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# video feed GET route
@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# tts route POST route
@app.route('/say', methods=['POST'])
def say():
    data = request.json
    # validate response
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid request, no 'text' field"}), 400
    
    text = data["text"] # get the text string
    def tts_task():
        os.system(f"espeak -a 200 '{text}'")
    threading.Thread(target=tts, args=(text,)).start()
    return jsonify({"message": "text being spoken..."}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888)

picam.stop()
picam.close()