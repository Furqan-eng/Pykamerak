from flask import Flask, request, jsonify, send_file
import cv2
import base64
import numpy as np
import re

app = Flask(__name__)

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/process_frame", methods=["POST"])
def process_frame():
    data = request.get_json()
    img_data = data["image"]

    # decode base64
    img_data = re.sub('^data:image/.+;base64,', '', img_data)
    img_bytes = base64.b64decode(img_data)
    img_np = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # contoh pemrosesan: konversi ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # balikin hasil (contoh jumlah pixel > 200 hitam)
    dark_pixels = int(np.sum(gray < 50))

    return jsonify({"message": f"Pixel gelap terdeteksi: {dark_pixels}"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
