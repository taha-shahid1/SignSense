from flask import Flask, Response, request, jsonify
from picamera2 import Picamera2, MappedArray
import pyttsx3
import threading
import cv2

# start the server and tts engine
app = Flask(__name__)
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)
tts_lock = threading.Lock()

# configure and start the camera
picam = PIcamera2()
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
        with MappedArray(picam, "lores") as m:
            frame = m.array # get frame as NumPy array
        
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
    data = response.json
    # validate response
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid request, no 'text' field"}), 400
    
    text = data["text"] # get the text string
    
    # run tts in seperate thread
    def tts_task():
        with tts_lock:
            engine.say(text)
            engine.runAndWait()
    threading.Thread(target=tts_task).start()
    
    return jsonify({"message": "text being spoken..."}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888)