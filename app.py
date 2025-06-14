from flask import Flask, Response
import cv2
import torch
import requests
import time

app = Flask(__name__)

# === CONFIG ===
ESP32_CAM_URL = 'http://192.168.1.4/stream'  # Add trailing slash to ensure MJPEG works
TELEGRAM_BOT_TOKEN = '7251...................................' #Add your telegram BOT token
TELEGRAM_CHAT_ID = '15.......' #Add telegram chat token
DETECTION_INTERVAL = 10  # Minimum seconds between two alerts

# === Load YOLOv5 Model ===
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # Confidence threshold

last_alert_time = 0

def send_telegram_alert():
    global last_alert_time
    now = time.time()
    if now - last_alert_time >= DETECTION_INTERVAL:
        message = "🚨 Human detected by ESP32-CAM AI!"
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        try:
            response = requests.post(url, data=data)
            if response.status_code == 200:
                print("[INFO] Telegram alert sent")
            else:
                print(f"[ERROR] Telegram failed: {response.text}")
            last_alert_time = now
        except Exception as e:
            print(f"[ERROR] Telegram exception: {e}")

def gen_frames():
    global last_alert_time
    stream_url = f"{ESP32_CAM_URL}"
    print(f"[INFO] Connecting to {stream_url}")
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("[ERROR] Cannot open ESP32 stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed, retrying...")
            continue

        # Run YOLOv5 detection
        results = model(frame)
        labels = results.pandas().xyxy[0]['name'].tolist()
        if 'person' in labels:
            send_telegram_alert()

        # Draw boxes
        frame = results.render()[0]

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return '<h1>ESP32-CAM AI Human Detection</h1><img src="/video">'

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
