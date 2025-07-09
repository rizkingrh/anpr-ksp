# save as stream_server.py
import cv2
from flask import Flask, Response

app = Flask(__name__)

RTSP_URL = "rtsp://admin:ZUHDGC@192.168.30.160"

def generate():
    cap = cv2.VideoCapture(RTSP_URL)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
