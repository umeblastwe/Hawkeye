import os
import uuid
import cv2
import numpy as np
import torch
import gc
import time

# =========================================================================
# MONKEY PATCH: PyTorch v2.6+ Weights-Only Loading Fix
# =========================================================================
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from ultralytics import YOLO
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

print(">>> Loading YOLO model...")
_t0 = time.time()
model = YOLO("yolov8n.pt")
print(f">>> Model loaded in {time.time() - _t0:.2f}s")

print(">>> Warming up model...")
_t0 = time.time()
_ = model.predict(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False, imgsz=320)
print(f">>> Warm-up done in {time.time() - _t0:.2f}s")


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
# FAST BALL TRACKING — with detailed timing instrumentation
# ==========================================
def detect_ball_professional(video_path, max_seconds_budget=40):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(">>> ERROR: could not open video file")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f">>> Video: {total_frames} frames @ {fps:.1f}fps")

    raw_positions = []
    frame_number = 0
    FRAME_SKIP = 2
    TARGET_WIDTH = 320

    start_time = time.time()
    inference_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        if frame_number % FRAME_SKIP != 0:
            continue

        if time.time() - start_time > max_seconds_budget:
            print(f">>> TIME BUDGET EXCEEDED at frame {frame_number}, stopping early")
            break

        h, w = frame.shape[:2]
        scale = TARGET_WIDTH / w
        small_frame = cv2.resize(frame, (TARGET_WIDTH, int(h * scale)))

        _t0 = time.time()
        results = model.predict(small_frame, conf=0.10, imgsz=320, verbose=False)
        inference_times.append(time.time() - _t0)

        if results and results[0].boxes and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                if class_id == 32:
                    xyxy = box.xyxy[0].cpu().numpy()
                    cx = (xyxy[0] + xyxy[2]) / 2
                    cy = (xyxy[1] + xyxy[3]) / 2
                    orig_x = int(cx / scale)
                    orig_y = int(cy / scale)
                    raw_positions.append({"frame": frame_number, "x": orig_x, "y": orig_y})
                    break

        del results

    cap.release()
    gc.collect()

    total_time = time.time() - start_time
    avg_inf = sum(inference_times) / len(inference_times) if inference_times else 0
    print(f">>> Processed {len(inference_times)} frames in {total_time:.2f}s "
          f"(avg {avg_inf*1000:.0f}ms/frame), found {len(raw_positions)} ball detections")

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
    if len(points) < 4:
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

    impact_index = min(bounce_index + 1, len(xs) - 1)
    impact_x = float(xs[impact_index])
    impact_y = float(ys[impact_index])

    coeff = np.polyfit(ys, xs, 2)

    stump_y = height * 0.60
    projected_x = float(np.polyval(coeff, stump_y))

    stump_left = width * 0.465
    stump_right = width * 0.535
    uc_margin = width * 0.015

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

    print(f">>> Received video: {filename}")
    _t_total = time.time()

    points = detect_ball_professional(path)

    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    result = calculate_quadratic_path(points, width, height)

    print(f">>> Total /analyze time: {time.time() - _t_total:.2f}s")

    return jsonify({
        "success": True,
        "video": url_for("static", filename="uploads/" + filename),
        "ball_points": points,
        "analysis": result
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
