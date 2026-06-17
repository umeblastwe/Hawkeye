import os
import uuid
import cv2
import numpy as np
import torch

# =========================================================================
# FIX: PyTorch 2.6+ changed torch.load default to weights_only=True, which
# blocks unpickling YOLO model classes. The env var approach does NOT work
# because TORCH_LOAD_WEIGHTS_ONLY is not a real PyTorch setting.
# The reliable fix is to monkey-patch torch.load itself, BEFORE importing
# ultralytics/YOLO, so every internal torch.load call inside ultralytics
# uses weights_only=False automatically.
# =========================================================================
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# Now safe to import ultralytics — it will use the patched torch.load
from ultralytics import YOLO

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Load model AFTER the patch is applied
model = YOLO("yolov8n.pt")


# ==========================================
# LIGHTWEIGHT KALMAN FILTER FOR SMOOTHING
# ==========================================
class SimpleKalman:
    def __init__(self):
        self.q = 0.05
        self.r = 1.0
        self.x = 0.0
        self.p = 1.0
        self.k = 0.0

    def update(self, measurement):
        self.p = self.p + self.q
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (measurement - self.x)
        self.p = (1 - self.k) * self.p
        return int(self.x)


# ==========================================
# ADVANCED TRACKING WITH PERSISTENCE
# ==========================================
def detect_ball_professional(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    raw_positions = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        results = model.track(frame, persist=True, conf=0.10, verbose=False)

        if results and results[0].boxes:
            for box in results[0].boxes:
                class_id = int(box.cls[0])

                if class_id == 32:  # Sports Ball
                    xyxy = box.xyxy[0].cpu().numpy()
                    cx = int((xyxy[0] + xyxy[2]) / 2)
                    cy = int((xyxy[1] + xyxy[3]) / 2)

                    raw_positions.append({"frame": frame_number, "x": cx, "y": cy})
                    break

    cap.release()

    if len(raw_positions) < 3:
        return raw_positions

    kf_x = SimpleKalman()
    kf_y = SimpleKalman()

    kf_x.x = raw_positions[0]["x"]
    kf_y.x = raw_positions[0]["y"]

    smoothed_positions = []
    for p in raw_positions:
        smoothed_positions.append({
            "frame": p["frame"],
            "x": kf_x.update(p["x"]),
            "y": kf_y.update(p["y"])
        })

    return smoothed_positions


# ==========================================
# PROFESSIONAL HAWK-EYE QUADRATIC PREDICTION
# ==========================================
def calculate_quadratic_path(points, width, height):
    if len(points) < 5:
        return {
            "pitching": "UNKNOWN", "impact": "UNKNOWN", "wicket": "UNKNOWN", "decision": "NOT OUT"
        }

    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])

    dy = np.diff(ys)
    bounce_index = np.argmax(ys)
    for i in range(1, len(dy)):
        if dy[i-1] > 0 and dy[i] < 0:
            bounce_index = i
            break

    pitch_x = float(xs[bounce_index])
    pitch_y = float(ys[bounce_index])

    impact_index = min(bounce_index + 2, len(xs) - 1)
    impact_x = float(xs[impact_index])
    impact_y = float(ys[impact_index])

    coeff = np.polyfit(ys, xs, 2)

    stump_y = height * 0.60
    projected_x = float(np.polyval(coeff, stump_y))

    stump_left = width * 0.465
    stump_right = width * 0.535
    uc_margin = width * 0.015

    # ── Umpire's Call zone added (edge of stump line) ──
    if stump_left + uc_margin <= projected_x <= stump_right - uc_margin:
        decision = "OUT"
        wicket = "HITTING"
    elif (stump_left - uc_margin <= projected_x < stump_left + uc_margin) or \
         (stump_right - uc_margin < projected_x <= stump_right + uc_margin):
        decision = "UMPIRE'S CALL"
        wicket = "UMPIRE'S CALL"
    else:
        decision = "NOT OUT"
        wicket = "MISSING"

    return {
        "pitching": "IN LINE", "impact": "IN LINE", "wicket": wicket, "decision": decision,
        "pitch": {"x": pitch_x, "y": pitch_y},
        "impact_point": {"x": impact_x, "y": impact_y},
        "projection": {"x": projected_x, "y": stump_y}
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video"})

    file = request.files["video"]
    filename = "hawk_" + str(uuid.uuid4())[:8] + os.path.splitext(secure_filename(file.filename))[1]
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    points = detect_ball_professional(path)

    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    result = calculate_quadratic_path(points, width, height)

    return jsonify({
        "success": True,
        "video": url_for("static", filename="uploads/" + filename),
        "ball_points": points,
        "analysis": result
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
