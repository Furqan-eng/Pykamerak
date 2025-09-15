from flask import Flask, Response, send_file
import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

app = Flask(__name__)

# Setup kontrol audio
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
min_vol, max_vol, _ = volume.GetVolumeRange()

# Setup Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # jempol & telunjuk
                    x1, y1 = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
                    x2, y2 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)

                    cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)
                    cv2.circle(frame, (x2, y2), 10, (255, 0, 0), -1)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    distance = np.hypot(x2 - x1, y2 - y1)

                    # kontrol volume
                    vol = np.interp(distance, [20, 200], [min_vol, max_vol])
                    volume.SetMasterVolumeLevel(vol, None)

                    # bar volume
                    vol_bar = np.interp(distance, [20, 200], [400, 150])
                    cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 2)
                    cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (0, 255, 0), -1)

                    vol_perc = np.interp(distance, [20, 200], [0, 100])
                    cv2.putText(frame, f'Vol: {int(vol_perc)} %', (40, 430),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Encode frame ke JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # ambil file html di folder yg sama dengan app.py
    return send_file("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run()
